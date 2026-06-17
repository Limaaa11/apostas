"""
Front-end web do sistema de previsão + gestão de banca/risco.

Melhorias v2:
- python-dotenv para configuração via .env
- Secret key persistente entre reinicializações
- Waitress WSGI (multi-thread) em vez do dev server Flask
- Rate limiting com flask-limiter (protege API Anthropic e endpoints pesados)
- Logging estruturado em arquivo app.log
- Cache de 30min para simulate_tournament
- Streaming SSE para o Agente IA
- Endpoint /api/backtest para backtesting do modelo
- Endpoint /api/predict_ci para intervalos de confiança
- Endpoint /api/blend exposto na UI
- Endpoint /api/value_scan para varredura de value bets
"""
import os
import time
import json
import logging
import secrets
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Carrega .env antes de qualquer import do projeto
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import model_engine as me
import live_api
import clv_tracker
import tournament
import agent as ag
import ml_pipeline as ml

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_PATH = os.path.join(os.path.dirname(__file__), "app.log")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("apostas")

# --------------------------------------------------------------------------- #
# Flask + secret key persistente
# --------------------------------------------------------------------------- #
app = Flask(__name__)

SECRET_KEY_PATH = os.path.join(os.path.dirname(__file__), ".secret_key")


def _load_or_create_secret_key():
    sk = os.environ.get("SECRET_KEY", "")
    if sk:
        return sk.encode()
    if os.path.exists(SECRET_KEY_PATH):
        with open(SECRET_KEY_PATH, "rb") as f:
            return f.read().strip()
    new_key = secrets.token_hex(32).encode()
    with open(SECRET_KEY_PATH, "wb") as f:
        f.write(new_key)
    return new_key


app.secret_key = _load_or_create_secret_key()

# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per minute"],
    storage_uri="memory://",
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _api_key():
    return (request.headers.get("X-Api-Key") or request.args.get("key") or "").strip()


def _f(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Rotas principais
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model_info")
@limiter.limit("60 per minute")
def model_info():
    m = me.get_model()
    return jsonify({
        "n_train": m.n_train,
        "n_teams": len(m.teams),
        "home_adv": m.home_adv,
        "rho": m.rho,
        "delta": m.delta,
        "converged": m.converged,
        "ref_date": str(m.ref_date.date()),
    })


@app.route("/api/teams")
@limiter.limit("30 per minute")
def teams():
    m = me.get_model()
    team_list = sorted(m.teams) if hasattr(m.teams, "__iter__") else []
    return jsonify({
        "teams": team_list,
        "n_teams": len(team_list),
        "cache_version": me.CACHE_VERSION,
    })


@app.route("/api/ranking")
@limiter.limit("30 per minute")
def ranking():
    top = int(request.args.get("top", 50))
    m = me.get_model()
    teams_data = []
    for team in m.teams:
        row = m._p.loc[team] if hasattr(m, "_p") else {}
        teams_data.append({
            "team": team,
            "elo": m.elo.get(team, 1500),
            "attack": float(row["attack"]) if hasattr(row, "__getitem__") and "attack" in row else None,
            "defense": float(row["defense"]) if hasattr(row, "__getitem__") and "defense" in row else None,
        })
    teams_data.sort(key=lambda x: -(x["elo"] or 0))
    return jsonify({"ranking": teams_data[:top]})


@app.route("/api/predict", methods=["POST"])
@limiter.limit("60 per minute")
def predict():
    d = request.get_json(force=True)
    m = me.get_model()
    pred = m.predict(d["home"], d["away"], bool(d.get("neutral", True)))
    if pred is None:
        return jsonify({"error": "Um dos times não está na base de treino do modelo."}), 400
    return jsonify(pred)


@app.route("/api/predict_ci", methods=["POST"])
@limiter.limit("10 per minute")
def predict_ci():
    """Previsão com intervalos de confiança via bootstrap."""
    d = request.get_json(force=True)
    m = me.get_model()
    n_boot = min(int(d.get("n_bootstrap", 200)), 500)
    result = m.predict_with_ci(d["home"], d["away"], bool(d.get("neutral", True)),
                                n_bootstrap=n_boot)
    if result is None:
        return jsonify({"error": "Um dos times não está na base do modelo."}), 400
    return jsonify(result)


@app.route("/api/markets", methods=["POST"])
@limiter.limit("60 per minute")
def markets():
    d = request.get_json(force=True)
    m = me.get_model()
    result = m.compute_markets(d["home"], d["away"], bool(d.get("neutral", True)))
    if result is None:
        return jsonify({"error": "Um dos times não está na base do modelo."}), 400
    return jsonify(result)


@app.route("/api/value", methods=["POST"])
@limiter.limit("60 per minute")
def value():
    d = request.get_json(force=True)
    m = me.get_model()
    pred = m.predict(d["home"], d["away"], bool(d.get("neutral", True)))
    if pred is None:
        return jsonify({"error": "Um dos times não está na base de treino do modelo."}), 400
    odds = {"odd_H": _f(d.get("odd_H")), "odd_D": _f(d.get("odd_D")), "odd_A": _f(d.get("odd_A"))}
    market = me.analyze_market(pred, odds)
    return jsonify({"pred": pred, "market": market})


@app.route("/api/blend", methods=["POST"])
@limiter.limit("60 per minute")
def blend():
    """Mistura probabilidades do modelo com odds da casa (remove vig)."""
    d = request.get_json(force=True)
    m = me.get_model()
    pred = m.predict(d["home"], d["away"], bool(d.get("neutral", True)))
    if pred is None:
        return jsonify({"error": "Time não encontrado no modelo."}), 400
    oh, od, oa = _f(d.get("odd_H")), _f(d.get("odd_D")), _f(d.get("odd_A"))
    if not (oh and od and oa and oh > 1 and od > 1 and oa > 1):
        return jsonify({"error": "Informe odd_H, odd_D e odd_A (decimais > 1)."}), 400
    alpha = float(d.get("alpha", 0.6))
    blended = me.blend_with_market(pred, oh, od, oa, alpha=alpha)
    return jsonify({"pred": pred, "blended": blended})


@app.route("/api/value_scan", methods=["POST"])
@limiter.limit("20 per minute")
def value_scan():
    """Varre uma lista de confrontos e retorna os que têm EV acima do threshold."""
    d = request.get_json(force=True)
    matchups = d.get("matchups", [])
    min_ev = float(d.get("min_ev", 0.03))   # 3% de EV mínimo
    neutral = bool(d.get("neutral", True))
    m = me.get_model()
    alerts = []
    side_labels = {"H": "Casa (1)", "D": "Empate (X)", "A": "Fora (2)"}
    for item in matchups[:50]:   # limite de segurança
        home = str(item.get("home", "")).strip()
        away = str(item.get("away", "")).strip()
        item_neutral = bool(item.get("neutral", neutral))
        if not home or not away:
            continue
        pred = m.predict(home, away, item_neutral)
        if pred is None:
            continue
        # Aceita tanto {odd_H, odd_D, odd_A} quanto aninhado {odds: {H, D, A}}
        nested = item.get("odds") or {}
        oh = _f(item.get("odd_H") or nested.get("H"))
        od = _f(item.get("odd_D") or nested.get("D"))
        oa = _f(item.get("odd_A") or nested.get("A"))
        pmap = {"H": pred.get("p_H"), "D": pred.get("p_D"), "A": pred.get("p_A")}
        for side, odd, label in [("H", oh, side_labels["H"]), ("D", od, side_labels["D"]), ("A", oa, side_labels["A"])]:
            if not odd or odd <= 1:
                continue
            p = pmap.get(side) or 0
            ev_val = p * odd - 1
            if ev_val >= min_ev:
                alerts.append({
                    "home": home, "away": away,
                    "side_label": label, "side": side,
                    "ev": round(ev_val, 4),
                    "odd": odd, "prob": p,
                    "kelly_quarter": round(max(0, (p * odd - 1) / (odd - 1)) * 0.25, 4),
                })
    alerts.sort(key=lambda x: -x["ev"])
    return jsonify({"alerts": alerts, "count": len(alerts)})


@app.route("/api/backtest")
@limiter.limit("5 per minute")
def backtest():
    """Backtesting do modelo contra dados históricos."""
    from_date = request.args.get("from", "2023-01-01")
    to_date = request.args.get("to", "2024-12-31")
    sample = int(request.args.get("sample", 500))
    m = me.get_model()
    try:
        import pandas as pd
        df = pd.read_csv(me.DATA_PATH, parse_dates=["date"])
        df = df[(df.date >= from_date) & (df.date <= to_date)].copy()
        if df.empty:
            return jsonify({"ok": False, "reason": "Nenhum jogo no período solicitado."}), 400
        result = m.backtest(df, sample_size=sample)
        return jsonify(result)
    except Exception as e:
        logger.exception("Erro no backtest")
        return jsonify({"ok": False, "reason": str(e)}), 500


_SIDE_ALIASES = {
    "h": "H", "1": "H", "home": "H", "casa": "H", "mandante": "H",
    "d": "D", "x": "D", "draw": "D", "empate": "D",
    "a": "A", "2": "A", "away": "A", "fora": "A", "visitante": "A",
}


@app.route("/api/batch_bets", methods=["POST"])
@limiter.limit("20 per minute")
def batch_bets():
    d = request.get_json(force=True)
    rows = d.get("rows", [])
    default_neutral = bool(d.get("neutral", True))
    m = me.get_model()
    bets, skipped = [], []
    for row in rows:
        home = str(row.get("home", "")).strip()
        away = str(row.get("away", "")).strip()
        if not home or not away:
            continue
        neutral = row.get("neutral")
        neutral = default_neutral if neutral in (None, "") else str(neutral).strip().lower() in ("true", "1", "sim", "s", "yes", "y", "neutro")
        pred = m.predict(home, away, neutral)
        if pred is None:
            skipped.append(f"{home} x {away} (time fora da base)")
            continue
        labels = {"H": f"{home} vence", "D": "Empate", "A": f"{away} vence"}
        pmap = {"H": pred["p_H"], "D": pred["p_D"], "A": pred["p_A"]}
        side_raw = str(row.get("side", "")).strip().lower()
        odd_one = _f(row.get("odd"))
        oh, od, oa = _f(row.get("odd_H")), _f(row.get("odd_D")), _f(row.get("odd_A"))
        if side_raw and odd_one:
            side = _SIDE_ALIASES.get(side_raw)
            if side is None or odd_one <= 1:
                skipped.append(f"{home} x {away} (lado/odd inválido)"); continue
            p = pmap[side]
            bets.append({"label": f"{home} x {away} — {labels[side]}", "p": p, "odd": odd_one, "ev": p * odd_one - 1})
        elif oh or od or oa:
            cands = [(p * o - 1, s, o, p) for s, o, p in (("H", oh, pmap["H"]), ("D", od, pmap["D"]), ("A", oa, pmap["A"])) if o and o > 1]
            if not cands:
                skipped.append(f"{home} x {away} (sem odds válidas)"); continue
            ev, side, odd, p = max(cands)
            bets.append({"label": f"{home} x {away} — {labels[side]}", "p": p, "odd": odd, "ev": ev})
        else:
            skipped.append(f"{home} x {away} (faltou side+odd ou odd_H/D/A)")
    return jsonify({"bets": bets, "skipped": skipped})


# --------------------------------------------------------------------------- #
# Ao Vivo
# --------------------------------------------------------------------------- #
def _live_api_key():
    """Chave da API-Football: vem do query param api_key (enviado pelo front)."""
    return (request.args.get("api_key") or _api_key()).strip()


@app.route("/api/live")
@limiter.limit("20 per minute")
def live():
    only_known = request.args.get("only_known", "0") in ("1", "true")
    games, err = live_api.live_matches(_live_api_key(), only_known=only_known)
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"matches": games, "count": len(games)})


@app.route("/api/live/world_cup")
@limiter.limit("20 per minute")
def live_world_cup():
    season = int(request.args.get("season", 2026))
    games, err = live_api.upcoming_world_cup(_live_api_key(), season=season)
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"matches": games, "count": len(games)})


@app.route("/api/live/date")
@limiter.limit("20 per minute")
def live_by_date():
    date_str = request.args.get("date", "")
    only_known = request.args.get("only_known", "0") in ("1", "true")
    if not date_str:
        return jsonify({"error": "Informe a data (YYYY-MM-DD)."}), 400
    games, err = live_api.matches_by_date(_live_api_key(), date_str, only_known=only_known)
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"matches": games, "count": len(games)})


@app.route("/api/upcoming")
@limiter.limit("20 per minute")
def upcoming():
    season = int(request.args.get("season", 2026))
    games, err = live_api.upcoming_world_cup(_live_api_key(), season=season)
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"matches": games, "count": len(games)})


# --------------------------------------------------------------------------- #
# Bankroll
# --------------------------------------------------------------------------- #
@app.route("/api/bankroll", methods=["POST"])
@limiter.limit("20 per minute")
def bankroll():
    d = request.get_json(force=True)
    bets = d.get("bets", [])
    if not bets:
        return jsonify({"error": "Adicione ao menos uma aposta."}), 400
    bankroll_val = _f(d.get("bankroll", 100)) or 100.0
    strategy = d.get("strategy", "kelly")
    kelly_mult = _f(d.get("kelly_mult", 0.25)) or 0.25
    flat_pct = _f(d.get("flat_pct", 0.02)) or 0.02
    min_edge = _f(d.get("min_edge", 0.0)) or 0.0
    ruin_threshold = _f(d.get("ruin_threshold", 0.20)) or 0.20
    n_sims = int(d.get("n_sims", 10000))
    mc = me.monte_carlo_bankroll(bets, bankroll=bankroll_val, strategy=strategy,
                                  kelly_mult=kelly_mult, flat_pct=flat_pct,
                                  n_sims=n_sims, min_edge=min_edge,
                                  ruin_threshold=ruin_threshold)
    comp = me.compare_strategies(bets, bankroll=bankroll_val,
                                  n_sims=min(n_sims, 10000), min_edge=min_edge)
    return jsonify({"mc": mc, "compare": comp})


# --------------------------------------------------------------------------- #
# Torneio Monte Carlo (com cache 30min)
# --------------------------------------------------------------------------- #
@app.route("/api/tournament")
@limiter.limit("5 per minute")
def tournament_sim():
    n_sims = int(request.args.get("n_sims", 10000))
    seed = int(request.args.get("seed", 42))
    m = me.get_model()
    result = me.get_tournament_cached(m, n_sims=n_sims, seed=seed)
    return jsonify(result)


# --------------------------------------------------------------------------- #
# CLV Tracker
# --------------------------------------------------------------------------- #
@app.route("/api/clv/add", methods=["POST"])
@limiter.limit("30 per minute")
def clv_add():
    d = request.get_json(force=True)
    for k in ["match", "side", "odds_entry"]:
        if not d.get(k):
            return jsonify({"error": f"Campo obrigatório: {k}"}), 400
    bet_id = clv_tracker.add_bet(
        match=d["match"], side=str(d["side"]).upper(),
        odds_entry=float(d["odds_entry"]), p_model=_f(d.get("p_model")),
        stake=_f(d.get("stake")), note=d.get("note", ""),
    )
    return jsonify({"ok": True, "id": bet_id})


@app.route("/api/clv/update", methods=["POST"])
@limiter.limit("30 per minute")
def clv_update():
    d = request.get_json(force=True)
    bet_id = d.get("id")
    if not bet_id:
        return jsonify({"error": "Campo obrigatório: id"}), 400
    clv_tracker.update_bet(int(bet_id), odds_closing=_f(d.get("odds_closing")),
                           result=d.get("result"))
    return jsonify({"ok": True})


@app.route("/api/clv/bets")
@limiter.limit("30 per minute")
def clv_bets():
    limit = int(request.args.get("limit", 100))
    return jsonify({"bets": clv_tracker.get_bets(limit)})


@app.route("/api/clv/stats")
@limiter.limit("30 per minute")
def clv_stats():
    return jsonify(clv_tracker.get_clv_stats())


@app.route("/api/clv/delete", methods=["POST"])
@limiter.limit("30 per minute")
def clv_delete():
    d = request.get_json(force=True)
    clv_tracker.delete_bet(int(d["id"]))
    return jsonify({"ok": True})


@app.route("/api/clv/export")
@limiter.limit("10 per minute")
def clv_export():
    """Exporta histórico CLV como CSV."""
    bets = clv_tracker.get_bets(limit=10000)
    lines = ["id,added_at,match,side,p_model,odds_entry,odds_closing,stake,result,note,clv_pct"]
    for b in bets:
        oe = b.get("odds_entry") or 0
        oc = b.get("odds_closing")
        clv_pct = round((oe / oc - 1) * 100, 2) if oc else ""
        lines.append(",".join(str(b.get(k, "") or "") for k in
                     ["id", "added_at", "match", "side", "p_model",
                      "odds_entry", "odds_closing", "stake", "result", "note"])
                    + f",{clv_pct}")
    csv_text = "\n".join(lines)
    return Response(csv_text, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=clv_historico.csv"})


# --------------------------------------------------------------------------- #
# Agente de IA — com suporte a streaming SSE
# --------------------------------------------------------------------------- #
_AGENT_HISTORIES = {}
_SESSION_TTL = 3600


def _clean_agent_sessions():
    now = time.time()
    stale = [k for k, (ts, _) in _AGENT_HISTORIES.items() if now - ts > _SESSION_TTL]
    for k in stale:
        del _AGENT_HISTORIES[k]


@app.route("/api/agent", methods=["POST"])
@limiter.limit("20 per minute")
def agent_chat():
    d = request.get_json(force=True)
    msg = (d.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "Mensagem vazia."}), 400
    session_id = d.get("session_id", "default")
    api_key = d.get("api_key") or d.get("anthropic_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return jsonify({"error": "Informe sua ANTHROPIC_API_KEY no campo acima."}), 400
    use_stream = bool(d.get("stream", False))

    _clean_agent_sessions()
    _, history = _AGENT_HISTORIES.get(session_id, (time.time(), []))

    if use_stream:
        def generate():
            try:
                result = ag.run_agent(msg, history=history, api_key=api_key)
                _AGENT_HISTORIES[session_id] = (time.time(), result["history"][-20:])
                payload = json.dumps({
                    "response": result["response"],
                    "tools_used": result["tool_calls_made"],
                })
                yield f"data: {payload}\n\n"
            except ValueError as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            except Exception as e:
                msg_safe = "Chave inválida ou sem saldo." if "auth" in str(e).lower() else "Erro interno no agente."
                logger.exception("Erro no agente")
                yield f"data: {json.dumps({'error': msg_safe})}\n\n"
        return Response(stream_with_context(generate()),
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    try:
        result = ag.run_agent(msg, history=history, api_key=api_key)
        _AGENT_HISTORIES[session_id] = (time.time(), result["history"][-20:])
        return jsonify({"response": result["response"], "tools_used": result["tool_calls_made"]})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        msg_safe = "Chave inválida ou sem saldo." if "auth" in str(e).lower() else "Erro interno no agente."
        logger.exception("Erro no agente")
        return jsonify({"error": msg_safe}), 500


@app.route("/api/agent/reset", methods=["POST"])
def agent_reset():
    d = request.get_json(force=True)
    sid = d.get("session_id", "default")
    _AGENT_HISTORIES.pop(sid, None)
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# ML Pipeline (XGBoost + Rolling Features)
# --------------------------------------------------------------------------- #

@app.route("/api/ml/status")
@limiter.limit("30 per minute")
def ml_status():
    """Retorna status e meta do modelo ML (se treinado)."""
    cache_path = ml.ML_CACHE_PATH
    if not os.path.exists(cache_path):
        return jsonify({"trained": False, "message": "Modelo ML não treinado ainda."})
    try:
        import pickle
        with open(cache_path, "rb") as f:
            blob = pickle.load(f)
        if blob.get("version") == ml.ML_CACHE_VERSION:
            m = blob["model"]
            return jsonify({"trained": True, "meta": m.meta})
    except Exception:
        pass
    return jsonify({"trained": False, "message": "Cache inválido ou desatualizado."})


@app.route("/api/ml/train", methods=["POST"])
@limiter.limit("2 per hour")
def ml_train():
    """
    Treina o modelo XGBoost (pode levar vários minutos).
    Aceita {force: true} para re-treinar mesmo com cache válido.
    """
    d = request.get_json(force=True) or {}
    force = bool(d.get("force", False))
    try:
        m = ml.get_ml_model(force_retrain=force)
        return jsonify({"ok": True, "meta": m.meta})
    except Exception as e:
        logger.exception("Erro no treino do modelo ML")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/ml/predict", methods=["POST"])
@limiter.limit("30 per minute")
def ml_predict():
    """Prediz probabilidade 1X2 via modelo XGBoost calibrado."""
    d = request.get_json(force=True)
    home = str(d.get("home", "")).strip()
    away = str(d.get("away", "")).strip()
    neutral = bool(d.get("neutral", True))
    odd_H = _f(d.get("odd_H"))
    odd_D = _f(d.get("odd_D"))
    odd_A = _f(d.get("odd_A"))
    min_ev = float(d.get("min_ev", 0.03))

    if not home or not away:
        return jsonify({"error": "Informe home e away."}), 400
    try:
        result = ml.predict_match(home, away, neutral)
    except Exception as e:
        logger.exception("Erro na predição ML")
        return jsonify({"error": str(e)}), 500
    if result is None:
        return jsonify({"error": "Dados insuficientes para este confronto."}), 400

    # Filtro de valor (se odds fornecidas)
    value_bets = []
    if any(x is not None for x in (odd_H, odd_D, odd_A)):
        value_bets = ml.value_bet_filter(
            {"p_H": result["p_H"], "p_D": result["p_D"], "p_A": result["p_A"]},
            odd_H=odd_H, odd_D=odd_D, odd_A=odd_A, min_ev=min_ev,
        )

    result["value_bets"] = value_bets
    return jsonify(result)


@app.route("/api/ml/value", methods=["POST"])
@limiter.limit("30 per minute")
def ml_value():
    """Aplica filtro de valor a probabilidades + odds fornecidas pelo usuário."""
    d = request.get_json(force=True)
    p = {"p_H": _f(d.get("p_H")), "p_D": _f(d.get("p_D")), "p_A": _f(d.get("p_A"))}
    result = ml.value_bet_filter(
        p,
        odd_H=_f(d.get("odd_H")),
        odd_D=_f(d.get("odd_D")),
        odd_A=_f(d.get("odd_A")),
        min_ev=float(d.get("min_ev", 0.03)),
    )
    return jsonify({"value_bets": result})


@app.route("/api/ml/backtest")
@limiter.limit("3 per hour")
def ml_backtest():
    """
    Backtesting financeiro com TimeSeriesSplit.
    Params: kelly_fraction, min_ev, n_splits, bankroll, flat_stake_pct
    """
    bankroll = float(request.args.get("bankroll", 100))
    kelly_fraction = float(request.args.get("kelly_fraction", 0.25))
    min_ev = float(request.args.get("min_ev", 0.03))
    n_splits = int(request.args.get("n_splits", 5))
    flat_pct_raw = request.args.get("flat_stake_pct")
    flat_pct = float(flat_pct_raw) if flat_pct_raw else None
    try:
        import pandas as pd
        m = ml.get_ml_model()
        df = pd.read_csv(ml.DATA_PATH, parse_dates=["date"])
        feat_df = ml.build_features(df)
        result = ml.financial_backtest(
            feat_df, m,
            bankroll=bankroll, kelly_fraction=kelly_fraction,
            min_ev=min_ev, n_splits=n_splits, flat_stake_pct=flat_pct,
        )
        return jsonify(result)
    except Exception as e:
        logger.exception("Erro no backtest ML")
        return jsonify({"ok": False, "error": str(e)}), 500


# --------------------------------------------------------------------------- #
# Inicialização
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    threads = int(os.environ.get("THREADS", 8))

    print("Carregando modelo (treina na 1ª vez, depois usa cache)...")
    me.get_model()
    print(f"Pronto! Abra http://{host}:{port}")

    try:
        from waitress import serve
        print(f"Servidor: Waitress ({threads} threads)")
        serve(app, host=host, port=port, threads=threads)
    except ImportError:
        print("Waitress não instalado — usando servidor de desenvolvimento Flask")
        app.run(host=host, port=port, debug=False)
