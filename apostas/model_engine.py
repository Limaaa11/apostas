"""
Motor do sistema de previsão + gestão de banca/risco.

Reaproveita a modelagem do notebook (Dixon-Coles com pesos temporais) e adiciona
o que tem valor real: análise de gestão de banca/risco (Kelly + Monte Carlo)
calibrada com a base histórica de jogos de seleções.

Tudo aqui é determinístico e cacheado em disco (params.pkl) para o front-end
responder rápido sem re-treinar a cada request.
"""
import os
import pickle
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "results.csv")
CACHE_PATH = os.path.join(BASE_DIR, "params.pkl")
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Hiperparâmetros do treino (mesmos do notebook, com maxiter maior p/ convergir)
TRAIN_FROM = "2018-01-01"
HALF_LIFE_DAYS = 730
MIN_GAMES = 5
MAXITER = 1500


# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #
def load_data():
    """Carrega results.csv, baixando se necessário, e remove jogos não disputados."""
    if not os.path.exists(DATA_PATH):
        import requests
        r = requests.get(DATA_URL, timeout=60)
        r.raise_for_status()
        with open(DATA_PATH, "wb") as f:
            f.write(r.content)
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    # Remove fixtures futuros sem placar (NaN) — eles quebram a verossimilhança.
    df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _data_signature(df):
    """Assinatura da base + hiperparâmetros, p/ invalidar o cache quando mudar."""
    key = f"{len(df)}|{df.date.max()}|{TRAIN_FROM}|{HALF_LIFE_DAYS}|{MIN_GAMES}|{MAXITER}"
    return hashlib.md5(key.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Dixon-Coles
# --------------------------------------------------------------------------- #
def time_weights(dates, ref_date, half_life_days=HALF_LIFE_DAYS):
    age_days = (ref_date - pd.to_datetime(dates)).dt.days.values
    return np.exp(-np.log(2) * age_days / half_life_days)


def fit_dixon_coles(df_train, ref_date, half_life_days=HALF_LIFE_DAYS,
                    min_games=MIN_GAMES, maxiter=MAXITER):
    team_games = pd.concat([df_train.home_team, df_train.away_team]).value_counts()
    keep = team_games[team_games >= min_games].index
    d = df_train[df_train.home_team.isin(keep) & df_train.away_team.isin(keep)].copy()

    teams = sorted(set(d.home_team) | set(d.away_team))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    home_idx = d.home_team.map(idx).values
    away_idx = d.away_team.map(idx).values
    hg = d.home_score.values.astype(int)
    ag = d.away_score.values.astype(int)
    not_neutral = (~d.neutral).astype(float).values
    w = time_weights(d.date, ref_date, half_life_days)

    def neg_log_lik(params):
        alpha = np.concatenate([params[:n - 1], [-params[:n - 1].sum()]])
        beta = np.concatenate([params[n - 1:2 * n - 2], [-params[n - 1:2 * n - 2].sum()]])
        home_adv, rho = params[-2], params[-1]
        lam_h = np.exp(alpha[home_idx] - beta[away_idx] + home_adv * not_neutral)
        lam_a = np.exp(alpha[away_idx] - beta[home_idx])
        ll_pois = hg * np.log(lam_h) - lam_h + ag * np.log(lam_a) - lam_a
        tau = np.ones(len(d))
        m00 = (hg == 0) & (ag == 0); tau[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
        m01 = (hg == 0) & (ag == 1); tau[m01] = 1 + lam_h[m01] * rho
        m10 = (hg == 1) & (ag == 0); tau[m10] = 1 + lam_a[m10] * rho
        m11 = (hg == 1) & (ag == 1); tau[m11] = 1 - rho
        tau = np.clip(tau, 1e-10, None)
        return -(w * (ll_pois + np.log(tau))).sum()

    x0 = np.zeros(2 * n)
    x0[-2] = 0.25
    x0[-1] = -0.1
    bounds = [(None, None)] * (2 * n - 2) + [(0, 1.0), (-0.5, 0.5)]
    res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter})
    alpha = np.concatenate([res.x[:n - 1], [-res.x[:n - 1].sum()]])
    beta = np.concatenate([res.x[n - 1:2 * n - 2], [-res.x[n - 1:2 * n - 2].sum()]])
    params_df = pd.DataFrame({"team": teams, "attack": alpha, "defense": beta})
    return params_df, res.x[-2], res.x[-1], res


def compute_elo(df_games, k_base=30, home_field=80, init=1500):
    ratings = {}
    weight = {"FIFA World Cup": 1.4, "FIFA World Cup qualification": 1.1,
              "UEFA Euro": 1.3, "Copa América": 1.3,
              "UEFA Nations League": 1.1, "Friendly": 0.7}
    for row in df_games.itertuples():
        rh = ratings.get(row.home_team, init)
        ra = ratings.get(row.away_team, init)
        adv = 0 if row.neutral else home_field
        eh = 1 / (1 + 10 ** (-(rh + adv - ra) / 400))
        if row.home_score > row.away_score: sh = 1
        elif row.home_score < row.away_score: sh = 0
        else: sh = 0.5
        diff = abs(row.home_score - row.away_score)
        g = 1 if diff < 2 else (1.5 if diff == 2 else (11 + diff) / 8)
        k = k_base * g * weight.get(row.tournament, 1.0)
        ratings[row.home_team] = rh + k * (sh - eh)
        ratings[row.away_team] = ra + k * ((1 - sh) - (1 - eh))
    return ratings


# --------------------------------------------------------------------------- #
# Treino + cache
# --------------------------------------------------------------------------- #
class Model:
    def __init__(self, params, home_adv, rho, elo, ref_date, n_train, converged):
        self.params = params                       # DataFrame: team, attack, defense
        self._p = params.set_index("team")
        self.home_adv = float(home_adv)
        self.rho = float(rho)
        self.elo = elo                             # dict team -> rating
        self.ref_date = ref_date
        self.n_train = int(n_train)
        self.converged = bool(converged)

    @property
    def teams(self):
        return sorted(self._p.index.tolist())

    def has(self, team):
        return team in self._p.index

    def match_matrix(self, home, away, neutral=True, max_goals=10):
        lam_h = np.exp(self._p.loc[home, "attack"] - self._p.loc[away, "defense"]
                       + (0 if neutral else self.home_adv))
        lam_a = np.exp(self._p.loc[away, "attack"] - self._p.loc[home, "defense"])
        h_pmf = poisson.pmf(np.arange(max_goals + 1), lam_h)
        a_pmf = poisson.pmf(np.arange(max_goals + 1), lam_a)
        M = np.outer(h_pmf, a_pmf)
        M[0, 0] *= 1 - lam_h * lam_a * self.rho
        M[0, 1] *= 1 + lam_h * self.rho
        M[1, 0] *= 1 + lam_a * self.rho
        M[1, 1] *= 1 - self.rho
        M = np.maximum(M, 0)
        M /= M.sum()
        return M, lam_h, lam_a

    def predict(self, home, away, neutral=True):
        if not (self.has(home) and self.has(away)):
            return None
        M, lam_h, lam_a = self.match_matrix(home, away, neutral)
        p_home = float(np.tril(M, -1).sum())
        p_draw = float(np.trace(M))
        p_away = float(np.triu(M, 1).sum())
        # placar mais provável
        i, j = np.unravel_index(M.argmax(), M.shape)
        # totais
        n = M.shape[0]
        tot = np.zeros(2 * n)
        for x in range(n):
            for y in range(n):
                tot[x + y] += M[x, y]
        p_over25 = float(tot[3:].sum())
        p_btts = float(M[1:, 1:].sum())
        # top placares
        flat = [((x, y), float(M[x, y])) for x in range(n) for y in range(n)]
        flat.sort(key=lambda t: -t[1])
        top_scores = [{"score": f"{a}-{b}", "p": p} for (a, b), p in flat[:5]]
        return {
            "home": home, "away": away, "neutral": neutral,
            "lam_h": float(lam_h), "lam_a": float(lam_a),
            "p_H": p_home, "p_D": p_draw, "p_A": p_away,
            "top_score": f"{i}-{j}", "p_top_score": float(M[i, j]),
            "p_over25": p_over25, "p_under25": 1 - p_over25,
            "p_btts_yes": p_btts, "p_btts_no": 1 - p_btts,
            "elo_home": float(self.elo.get(home, 1500)),
            "elo_away": float(self.elo.get(away, 1500)),
            "top_scores": top_scores,
        }

    def ranking(self, top=30):
        rows = []
        for t in self._p.index:
            rows.append({
                "team": t,
                "attack": float(self._p.loc[t, "attack"]),
                "defense": float(self._p.loc[t, "defense"]),
                "elo": float(self.elo.get(t, 1500)),
            })
        rows.sort(key=lambda r: -r["elo"])
        return rows[:top]


_MODEL = None


def get_model(force=False):
    """Carrega o modelo do cache; treina e cacheia se necessário."""
    global _MODEL
    if _MODEL is not None and not force:
        return _MODEL

    df = load_data()
    sig = _data_signature(df)

    if not force and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                blob = pickle.load(f)
            if blob.get("sig") == sig:
                _MODEL = Model(blob["params"], blob["home_adv"], blob["rho"],
                               blob["elo"], blob["ref_date"], blob["n_train"],
                               blob["converged"])
                return _MODEL
        except Exception:
            pass

    _MODEL = _train_and_cache(df, sig)
    return _MODEL


def _train_and_cache(df, sig):
    ref_date = df.date.max() + pd.Timedelta(days=1)
    train_full = df[df.date >= TRAIN_FROM].copy()
    params, ha, rho, res = fit_dixon_coles(train_full, ref_date)
    elo = compute_elo(df.sort_values("date"))  # ELO sobre toda a história
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({
            "sig": sig, "params": params, "home_adv": ha, "rho": rho,
            "elo": elo, "ref_date": ref_date, "n_train": len(train_full),
            "converged": bool(res.success),
        }, f)
    return Model(params, ha, rho, elo, ref_date, len(train_full), res.success)


# --------------------------------------------------------------------------- #
# Value / EV / Kelly
# --------------------------------------------------------------------------- #
def kelly_fraction(p, odd):
    """Fração de Kelly cheia para uma aposta de prob p e odd decimal."""
    b = odd - 1.0
    if b <= 0:
        return 0.0
    f = (b * p - (1 - p)) / b
    return max(f, 0.0)


def value_bet(p, odd):
    """EV, edge e stakes Kelly para uma única aposta."""
    implied = 1.0 / odd
    ev = p * odd - 1.0
    kf = kelly_fraction(p, odd)
    return {
        "p_model": p, "odd": odd, "implied": implied,
        "edge": p - implied, "ev": ev,
        "kelly_full": kf, "kelly_half": kf / 2, "kelly_quarter": kf / 4,
    }


def analyze_market(pred, odds):
    """Compara as 3 probabilidades do modelo (1X2) com as odds e ranqueia por EV.

    odds: dict com chaves odd_H, odd_D, odd_A (decimais).
    """
    out = []
    for side, label, pkey, okey in [
        ("H", f"{pred['home']} vence", "p_H", "odd_H"),
        ("D", "Empate", "p_D", "odd_D"),
        ("A", f"{pred['away']} vence", "p_A", "odd_A"),
    ]:
        odd = odds.get(okey)
        if not odd or odd <= 1:
            continue
        vb = value_bet(pred[pkey], float(odd))
        vb.update({"side": side, "label": label})
        out.append(vb)
    out.sort(key=lambda r: -r["ev"])
    return out


# --------------------------------------------------------------------------- #
# Gestão de banca / risco — Monte Carlo
# --------------------------------------------------------------------------- #
def monte_carlo_bankroll(bets, bankroll=100.0, strategy="kelly", kelly_mult=0.25,
                         flat_pct=0.02, n_sims=10000, min_edge=0.0,
                         ruin_threshold=0.20, seed=42):
    """Simula trajetórias de banca para um conjunto de apostas planejadas.

    bets: lista de dicts com 'p' (prob do modelo) e 'odd' (decimal).
    strategy: 'kelly' (fração de Kelly) ou 'flat' (% fixo da banca).
    kelly_mult: multiplicador de Kelly (0.25 = 1/4 Kelly).
    flat_pct: fração fixa da banca por aposta (modo flat).
    min_edge: só aposta se EV > min_edge.
    ruin_threshold: banca considerada "arruinada" se cair abaixo desta fração da inicial.

    Retorna métricas de risco agregadas + algumas trajetórias de exemplo.
    """
    rng = np.random.default_rng(seed)

    # Pré-filtra apostas com EV suficiente e calcula a fração de stake de cada uma.
    placed = []
    for b in bets:
        p = float(b["p"]); odd = float(b["odd"])
        ev = p * odd - 1.0
        if ev <= min_edge:
            continue
        if strategy == "kelly":
            frac = kelly_mult * kelly_fraction(p, odd)
        else:
            frac = flat_pct
        if frac <= 0:
            continue
        placed.append({"p": p, "odd": odd, "frac": frac, "ev": ev,
                       "label": b.get("label", f"{odd:.2f} @ {p:.0%}")})

    if not placed:
        return {"ok": False, "reason": "Nenhuma aposta com EV acima do limiar.",
                "n_bets": 0}

    n_bets = len(placed)
    fracs = np.array([b["frac"] for b in placed])
    odds = np.array([b["odd"] for b in placed])
    ps = np.array([b["p"] for b in placed])

    # Simulação vetorizada: para cada sim, percorre as apostas em ordem.
    bank = np.full(n_sims, float(bankroll))
    peak = np.full(n_sims, float(bankroll))
    max_dd = np.zeros(n_sims)          # maior drawdown relativo
    ruined = np.zeros(n_sims, dtype=bool)
    sample_paths = np.zeros((min(30, n_sims), n_bets + 1))
    sample_paths[:, 0] = bankroll

    for k in range(n_bets):
        stake = bank * fracs[k]
        wins = rng.random(n_sims) < ps[k]
        pnl = np.where(wins, stake * (odds[k] - 1.0), -stake)
        bank = bank + pnl
        peak = np.maximum(peak, bank)
        dd = (peak - bank) / peak
        max_dd = np.maximum(max_dd, dd)
        ruined |= bank < (ruin_threshold * bankroll)
        if sample_paths.shape[0]:
            sample_paths[:, k + 1] = bank[:sample_paths.shape[0]]

    final = bank
    growth = final / bankroll - 1.0

    def pct(a, q):
        return float(np.percentile(a, q))

    metrics = {
        "ok": True,
        "n_bets": n_bets,
        "n_sims": n_sims,
        "strategy": strategy,
        "kelly_mult": kelly_mult,
        "flat_pct": flat_pct,
        "bankroll_inicial": float(bankroll),
        "ev_medio": float(np.mean([b["ev"] for b in placed])),
        "stake_medio_pct": float(np.mean(fracs)),
        # banca final
        "final_mediana": float(np.median(final)),
        "final_media": float(np.mean(final)),
        "final_p5": pct(final, 5),
        "final_p25": pct(final, 25),
        "final_p75": pct(final, 75),
        "final_p95": pct(final, 95),
        # retorno
        "roi_mediano": float(np.median(growth)),
        # risco
        "prob_lucro": float(np.mean(final > bankroll)),
        "prob_dobrar": float(np.mean(final >= 2 * bankroll)),
        "prob_ruina": float(np.mean(ruined)),
        "drawdown_mediano": float(np.median(max_dd)),
        "drawdown_p95": pct(max_dd, 95),
        "drawdown_max": float(np.max(max_dd)),
        "sample_paths": sample_paths.tolist(),
        "bets": [{"label": b["label"], "p": b["p"], "odd": b["odd"],
                  "ev": b["ev"], "stake_pct": b["frac"]} for b in placed],
    }
    return metrics


def compare_strategies(bets, bankroll=100.0, n_sims=10000, min_edge=0.0):
    """Roda o Monte Carlo para várias estratégias e devolve comparação lado a lado."""
    configs = [
        ("Flat 1%", {"strategy": "flat", "flat_pct": 0.01}),
        ("Flat 2%", {"strategy": "flat", "flat_pct": 0.02}),
        ("1/8 Kelly", {"strategy": "kelly", "kelly_mult": 0.125}),
        ("1/4 Kelly", {"strategy": "kelly", "kelly_mult": 0.25}),
        ("1/2 Kelly", {"strategy": "kelly", "kelly_mult": 0.5}),
        ("Kelly cheio", {"strategy": "kelly", "kelly_mult": 1.0}),
    ]
    rows = []
    for name, cfg in configs:
        m = monte_carlo_bankroll(bets, bankroll=bankroll, n_sims=n_sims,
                                 min_edge=min_edge, **cfg)
        if not m.get("ok"):
            continue
        rows.append({
            "nome": name,
            "final_mediana": m["final_mediana"],
            "final_p5": m["final_p5"],
            "final_p95": m["final_p95"],
            "roi_mediano": m["roi_mediano"],
            "prob_lucro": m["prob_lucro"],
            "prob_ruina": m["prob_ruina"],
            "drawdown_mediano": m["drawdown_mediano"],
            "drawdown_p95": m["drawdown_p95"],
        })
    return rows
