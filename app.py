"""
Front-end web do sistema de previsão + gestão de banca/risco.

Rode:  python app.py
Abra:  http://localhost:5000
"""
import os
from flask import Flask, render_template, request, jsonify, session
import model_engine as me
import live_api
import clv_tracker
import tournament
import agent as ag

app = Flask(__name__)
app.secret_key = os.urandom(24)


def _api_key():
    """Chave da API-Football enviada pelo front (header) — nunca fica no código."""
    return (request.headers.get("X-Api-Key")
            or request.args.get("key")
            or "").strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model_info")
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
def teams():
    return jsonify(me.get_model().teams)


@app.route("/api/ranking")
def ranking():
    top = int(request.args.get("top", 30))
    return jsonify(me.get_model().ranking(top))


@app.route("/api/predict", methods=["POST"])
def predict():
    d = request.get_json(force=True)
    m = me.get_model()
    pred = m.predict(d["home"], d["away"], bool(d.get("neutral", True)))
    if pred is None:
        return jsonify({"error": "Um dos times não está na base de treino do modelo."}), 400
    return jsonify(pred)


@app.route("/api/markets", methods=["POST"])
def markets():
    d = request.get_json(force=True)
    m = me.get_model()
    result = m.compute_markets(d["home"], d["away"], bool(d.get("neutral", True)))
    if result is None:
        return jsonify({"error": "Um dos times não está na base do modelo."}), 400
    return jsonify(result)


@app.route("/api/value", methods=["POST"])
def value():
    d = request.get_json(force=True)
    m = me.get_model()
    pred = m.predict(d["home"], d["away"], bool(d.get("neutral", True)))
    if pred is None:
        return jsonify({"error": "Um dos times não está na base de treino do modelo."}), 400
    odds = {
        "odd_H": _f(d.get("odd_H")),
        "odd_D": _f(d.get("odd_D")),
        "odd_A": _f(d.get("odd_A")),
    }
    market = me.analyze_market(pred, odds)
    return jsonify({"pred": pred, "market": market})


_SIDE_ALIASES = {
    "h": "H", "1": "H", "home": "H", "casa": "H", "mandante": "H",
    "d": "D", "x": "D", "draw": "D", "empate": "D",
    "a": "A", "2": "A", "away": "A", "fora": "A", "visitante": "A",
}


@app.route("/api/batch_bets", methods=["POST"])
def batch_bets():
    """Recebe linhas de um CSV e devolve apostas com prob do modelo + EV.

    Cada linha aceita um destes formatos:
      - home, away, side, odd [, neutral]        -> 1 aposta no lado escolhido
      - home, away, odd_H, odd_D, odd_A [, neutral] -> escolhe o lado de maior EV
    """
    d = request.get_json(force=True)
    rows = d.get("rows", [])
    default_neutral = bool(d.get("neutral", True))
    m = me.get_model()
    bets, skipped = [], []

    for i, row in enumerate(rows):
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
                skipped.append(f"{home} x {away} (lado/odd inválido)")
                continue
            p = pmap[side]
            bets.append({"label": f"{home} x {away} — {labels[side]}",
                         "p": p, "odd": odd_one, "ev": p * odd_one - 1})
        elif oh or od or oa:
            cands = []
            for side, odd in (("H", oh), ("D", od), ("A", oa)):
                if odd and odd > 1:
                    p = pmap[side]
                    cands.append((p * odd - 1, side, odd, p))
            if not cands:
                skipped.append(f"{home} x {away} (sem odds válidas)")
                continue
            ev, side, odd, p = max(cands)
            bets.append({"label": f"{home} x {away} — {labels[side]}",
                         "p": p, "odd": odd, "ev": ev})
        else:
            skipped.append(f"{home} x {away} (faltou side+odd ou odd_H/D/A)")

    return jsonify({"bets": bets, "skipped": skipped})


@app.route("/api/live")
def live():
    only_known = request.args.get("only_known", "0") in ("1", "true")
    games, err = live_api.live_matches(_api_key(), only_known=only_known)
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"games": games, "count": len(games)})


@app.route("/api/upcoming")
def upcoming():
    mode = request.args.get("mode", "worldcup")
    if mode == "date":
        date_str = request.args.get("date", "")
        only_known = request.args.get("only_known", "0") in ("1", "true")
        if not date_str:
            return jsonify({"error": "Informe a data (YYYY-MM-DD)."}), 400
        games, err = live_api.matches_by_date(_api_key(), date_str, only_known=only_known)
    else:
        season = int(request.args.get("season", 2026))
        games, err = live_api.upcoming_world_cup(_api_key(), season=season)
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"games": games, "count": len(games)})


@app.route("/api/bankroll", methods=["POST"])
def bankroll():
    d = request.get_json(force=True)
    bets = d.get("bets", [])
    if not bets:
        return jsonify({"error": "Adicione ao menos uma aposta."}), 400
    bankroll = _f(d.get("bankroll", 100)) or 100.0
    strategy = d.get("strategy", "kelly")
    kelly_mult = _f(d.get("kelly_mult", 0.25)) or 0.25
    flat_pct = _f(d.get("flat_pct", 0.02)) or 0.02
    min_edge = _f(d.get("min_edge", 0.0)) or 0.0
    ruin_threshold = _f(d.get("ruin_threshold", 0.20)) or 0.20
    n_sims = int(d.get("n_sims", 10000))

    mc = me.monte_carlo_bankroll(
        bets, bankroll=bankroll, strategy=strategy, kelly_mult=kelly_mult,
        flat_pct=flat_pct, n_sims=n_sims, min_edge=min_edge,
        ruin_threshold=ruin_threshold)
    comp = me.compare_strategies(bets, bankroll=bankroll,
                                 n_sims=min(n_sims, 10000), min_edge=min_edge)
    return jsonify({"mc": mc, "compare": comp})


def _f(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Blend modelo + odds de mercado
# --------------------------------------------------------------------------- #
@app.route("/api/blend", methods=["POST"])
def blend():
    """Mistura probabilidades do modelo com odds da casa (remove vig).

    Body: { home, away, neutral, odd_H, odd_D, odd_A, alpha (0-1, default 0.6) }
    """
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


# --------------------------------------------------------------------------- #
# Simulação de torneio Monte Carlo
# --------------------------------------------------------------------------- #
@app.route("/api/tournament")
def tournament_sim():
    """Simula a Copa 2026 completa via Monte Carlo.

    Query params: n_sims (default 10000), seed (default 42)
    """
    n_sims = int(request.args.get("n_sims", 10000))
    seed = int(request.args.get("seed", 42))
    m = me.get_model()
    result = tournament.simulate_tournament(m, n_sims=n_sims, seed=seed)
    return jsonify(result)


# --------------------------------------------------------------------------- #
# CLV Tracker
# --------------------------------------------------------------------------- #
@app.route("/api/clv/add", methods=["POST"])
def clv_add():
    """Registra uma aposta. Body: { match, side, odds_entry, p_model?, stake?, note? }"""
    d = request.get_json(force=True)
    required = ["match", "side", "odds_entry"]
    for k in required:
        if not d.get(k):
            return jsonify({"error": f"Campo obrigatório: {k}"}), 400
    bet_id = clv_tracker.add_bet(
        match=d["match"],
        side=str(d["side"]).upper(),
        odds_entry=float(d["odds_entry"]),
        p_model=_f(d.get("p_model")),
        stake=_f(d.get("stake")),
        note=d.get("note", ""),
    )
    return jsonify({"ok": True, "id": bet_id})


@app.route("/api/clv/update", methods=["POST"])
def clv_update():
    """Atualiza odds de fechamento e/ou resultado. Body: { id, odds_closing?, result? }"""
    d = request.get_json(force=True)
    bet_id = d.get("id")
    if not bet_id:
        return jsonify({"error": "Campo obrigatório: id"}), 400
    clv_tracker.update_bet(
        int(bet_id),
        odds_closing=_f(d.get("odds_closing")),
        result=d.get("result"),
    )
    return jsonify({"ok": True})


@app.route("/api/clv/bets")
def clv_bets():
    limit = int(request.args.get("limit", 100))
    return jsonify({"bets": clv_tracker.get_bets(limit)})


@app.route("/api/clv/stats")
def clv_stats():
    return jsonify(clv_tracker.get_clv_stats())


@app.route("/api/clv/delete", methods=["POST"])
def clv_delete():
    d = request.get_json(force=True)
    clv_tracker.delete_bet(int(d["id"]))
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# Agente de IA
# --------------------------------------------------------------------------- #
_AGENT_HISTORIES = {}        # session_id -> (last_ts, history)
_SESSION_TTL = 3600          # expira sessão após 1 hora de inatividade


def _clean_agent_sessions():
    now = os.times().elapsed if hasattr(os, "times") else 0
    import time as _t
    now = _t.time()
    stale = [k for k, (ts, _) in _AGENT_HISTORIES.items() if now - ts > _SESSION_TTL]
    for k in stale:
        del _AGENT_HISTORIES[k]


@app.route("/api/agent", methods=["POST"])
def agent_chat():
    import time as _t
    d = request.get_json(force=True)
    msg = (d.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "Mensagem vazia."}), 400
    session_id = d.get("session_id", "default")
    api_key = d.get("anthropic_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return jsonify({"error": "Informe sua ANTHROPIC_API_KEY no campo acima."}), 400

    _clean_agent_sessions()
    _, history = _AGENT_HISTORIES.get(session_id, (_t.time(), []))
    try:
        result = ag.run_agent(msg, history=history, api_key=api_key)
        _AGENT_HISTORIES[session_id] = (_t.time(), result["history"][-20:])
        return jsonify({"response": result["response"], "tools_used": result["tool_calls_made"]})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Não expõe detalhes internos (pode conter a chave em AuthenticationError)
        msg_safe = "Chave inválida ou sem saldo." if "auth" in str(e).lower() else "Erro interno no agente."
        return jsonify({"error": msg_safe}), 500


@app.route("/api/agent/reset", methods=["POST"])
def agent_reset():
    d = request.get_json(force=True)
    sid = d.get("session_id", "default")
    _AGENT_HISTORIES.pop(sid, None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Carregando modelo (treina na 1ª vez, depois usa cache)...")
    me.get_model()
    print("Pronto! Abra http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
