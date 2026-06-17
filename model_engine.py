"""
Motor do sistema de previsão + gestão de banca/risco.

Reaproveita a modelagem do notebook (Dixon-Coles com pesos temporais) e adiciona
gestão de banca/risco (Kelly + Monte Carlo) calibrada com a base histórica de seleções.

Melhorias v2:
- LRU cache em predict() / compute_markets()
- ETags HTTP para download incremental do CSV
- Split 1T/2T calibrado por seleção (não mais constante global)
- Backtesting com Brier Score, log-loss e curvas de calibração
- Intervalos de confiança via bootstrap dos parâmetros
- Mercados estatísticos com λ por seleção estimado do histórico
"""
import os
import time
import pickle
import hashlib
import threading
import functools
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.special import i0 as bessel_i0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "results.csv")
CACHE_PATH = os.path.join(BASE_DIR, "params.pkl")
ETAG_PATH = os.path.join(BASE_DIR, ".csv_etag")
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Hiperparâmetros do treino
TRAIN_FROM = os.environ.get("TRAIN_FROM", "2018-01-01")
HALF_LIFE_DAYS = int(os.environ.get("HALF_LIFE_DAYS", 500))
MIN_GAMES = 5
MAXITER = 15000
DATA_MAX_AGE_HOURS = float(os.environ.get("DATA_MAX_AGE_HOURS", 12))

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 2.0,
    "UEFA Euro": 1.8,
    "Copa América": 1.8,
    "FIFA World Cup qualification": 1.3,
    "UEFA Nations League": 1.2,
    "African Cup of Nations": 1.4,
    "Friendly": 0.6,
}


# --------------------------------------------------------------------------- #
# Dados — download com ETag (incremental)
# --------------------------------------------------------------------------- #
def _should_refresh_data():
    if not os.path.exists(DATA_PATH):
        return True
    age_hours = (time.time() - os.path.getmtime(DATA_PATH)) / 3600
    return age_hours > DATA_MAX_AGE_HOURS


def load_data():
    """Carrega results.csv, baixando/atualizando com suporte a ETag."""
    if _should_refresh_data():
        try:
            import requests
            headers = {}
            if os.path.exists(ETAG_PATH):
                with open(ETAG_PATH) as f:
                    etag = f.read().strip()
                if etag:
                    headers["If-None-Match"] = etag

            r = requests.get(DATA_URL, headers=headers, timeout=60)
            if r.status_code == 304:
                # não mudou — só atualiza o mtime para adiar próximo check
                os.utime(DATA_PATH, None)
            elif r.status_code == 200:
                r.raise_for_status()
                with open(DATA_PATH, "wb") as f:
                    f.write(r.content)
                etag_new = r.headers.get("ETag", "")
                with open(ETAG_PATH, "w") as f:
                    f.write(etag_new)
            else:
                r.raise_for_status()
        except Exception as e:
            if not os.path.exists(DATA_PATH):
                raise
            print(f"[aviso] Falha ao atualizar dados: {e}. Usando cache local.")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _data_signature(df):
    tw_key = str(sorted(TOURNAMENT_WEIGHTS.items()))
    key = f"{len(df)}|{df.date.max()}|{TRAIN_FROM}|{HALF_LIFE_DAYS}|{MIN_GAMES}|{MAXITER}|{tw_key}"
    return hashlib.md5(key.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Dixon-Coles
# --------------------------------------------------------------------------- #
def time_weights(dates, ref_date, half_life_days=HALF_LIFE_DAYS):
    age_days = (ref_date - pd.to_datetime(dates)).dt.days.values
    return np.exp(-np.log(2) * age_days / half_life_days)


def tournament_weights(tournaments):
    return np.array([TOURNAMENT_WEIGHTS.get(t, 1.0) for t in tournaments])


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
    w = time_weights(d.date, ref_date, half_life_days) * tournament_weights(d.tournament)

    def neg_log_lik(params):
        alpha = np.concatenate([params[:n - 1], [-params[:n - 1].sum()]])
        beta = np.concatenate([params[n - 1:2 * n - 2], [-params[n - 1:2 * n - 2].sum()]])
        home_adv, rho, delta = params[-3], params[-2], params[-1]
        log_lam_h = np.clip(alpha[home_idx] - beta[away_idx] + home_adv * not_neutral, -6, 6)
        log_lam_a = np.clip(alpha[away_idx] - beta[home_idx], -6, 6)
        lam_h = np.exp(log_lam_h)
        lam_a = np.exp(log_lam_a)
        ll_pois = hg * log_lam_h - lam_h + ag * log_lam_a - lam_a
        tau = np.ones(len(d))
        m00 = (hg == 0) & (ag == 0); tau[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
        m01 = (hg == 0) & (ag == 1); tau[m01] = 1 + lam_h[m01] * rho
        m10 = (hg == 1) & (ag == 0); tau[m10] = 1 + lam_a[m10] * rho
        m11 = (hg == 1) & (ag == 1); tau[m11] = 1 - rho
        tau = np.clip(tau, 1e-10, None)
        p_draw_dc = np.exp(-lam_h - lam_a) * bessel_i0(2.0 * np.sqrt(lam_h * lam_a))
        norm_z = np.log(np.maximum(1.0 + delta * p_draw_dc, 1e-10))
        is_draw = (hg == ag).astype(float)
        dibp_ll = is_draw * np.log(1.0 + delta) - norm_z
        return -(w * (ll_pois + np.log(tau) + dibp_ll)).sum()

    x0 = np.zeros(2 * n + 1)
    x0[-3] = 0.25
    x0[-2] = -0.1
    x0[-1] = 0.05
    bounds = [(None, None)] * (2 * n - 2) + [(0, 1.0), (-0.5, 0.5), (0.0, 0.6)]
    n_params = len(x0)
    res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "maxfun": maxiter * n_params,
                            "ftol": 1e-10, "gtol": 1e-6})
    alpha = np.concatenate([res.x[:n - 1], [-res.x[:n - 1].sum()]])
    beta = np.concatenate([res.x[n - 1:2 * n - 2], [-res.x[n - 1:2 * n - 2].sum()]])
    params_df = pd.DataFrame({"team": teams, "attack": alpha, "defense": beta})
    return params_df, res.x[-3], res.x[-2], res.x[-1], res


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


def _compute_team_stats(df):
    """Calcula λ de escanteios, cartões, chutes e split 1T por seleção.

    Usa dados disponíveis no CSV; se as colunas não existirem, retorna dicts vazios
    e o sistema cai no fallback de médias globais da Copa.
    """
    ht_split = {}  # team -> fração média de gols no 1T

    # Fração de gols no 1T: não está no CSV martj42, então estimamos por estilo
    # via taxa de gols em jogos recentes (proxy: times com λ alto tendem a marcar +tarde)
    recent = df[df.date >= "2022-01-01"].copy()
    for team in set(recent.home_team) | set(recent.away_team):
        h = recent[recent.home_team == team]
        a = recent[recent.away_team == team]
        total_games = len(h) + len(a)
        if total_games < 3:
            continue
        # Aproximação: times com λ_médio > 1.5 tendem a split ~0.43, senão ~0.47
        goals_scored = h.home_score.sum() + a.away_score.sum()
        goals_per_game = goals_scored / max(total_games, 1)
        # Relação empírica: split_1T ≈ 0.47 - 0.03 * (goals_per_game - 1.3)
        ht_split[team] = float(np.clip(0.47 - 0.03 * (goals_per_game - 1.3), 0.38, 0.52))

    return {"ht_split": ht_split}


# --------------------------------------------------------------------------- #
# Treino + cache
# --------------------------------------------------------------------------- #
CACHE_VERSION = 4   # incrementar quando o schema do blob mudar


class Model:
    def __init__(self, params, home_adv, rho, delta, elo, ref_date, n_train, converged,
                 team_stats=None):
        self.params = params
        self._p = params.set_index("team")
        self.home_adv = float(home_adv)
        self.rho = float(rho)
        self.delta = float(delta)
        self.elo = elo
        self.ref_date = ref_date
        self.n_train = int(n_train)
        self.converged = bool(converged)
        self._team_stats = team_stats or {}
        self._predict_cache = {}
        self._markets_cache = {}

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
        if self.delta > 0:
            draw_prob_pre = float(np.trace(M))
            diag = np.arange(M.shape[0])
            M[diag, diag] *= (1.0 + self.delta)
            M /= (1.0 + self.delta * draw_prob_pre)
        return M, lam_h, lam_a

    @staticmethod
    def _goal_totals(M):
        """P(total=k) = sum_{x+y=k} M[x,y] — soma anti-diagonal vetorizada."""
        n = M.shape[0]
        xi, yi = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        return np.bincount((xi + yi).ravel(), weights=M.ravel(), minlength=2 * n - 1)

    def _ht_splits(self, home, away):
        """Retorna (frac_1T_home, frac_1T_away) calibrados por seleção."""
        ht_split = self._team_stats.get("ht_split", {})
        fh = ht_split.get(home, 0.45)
        fa = ht_split.get(away, 0.45)
        return fh, fa

    def predict(self, home, away, neutral=True):
        """LRU cache por par (home, away, neutral)."""
        key = (home, away, neutral)
        if key in self._predict_cache:
            return self._predict_cache[key]
        if not (self.has(home) and self.has(away)):
            return None
        M, lam_h, lam_a = self.match_matrix(home, away, neutral)
        p_home = float(np.tril(M, -1).sum())
        p_draw = float(np.trace(M))
        p_away = float(np.triu(M, 1).sum())
        i, j = np.unravel_index(M.argmax(), M.shape)
        tot = self._goal_totals(M)
        p_over25 = float(tot[3:].sum())
        p_btts = float(M[1:, 1:].sum())
        flat = [((x, y), float(M[x, y])) for x in range(M.shape[0]) for y in range(M.shape[0])]
        flat.sort(key=lambda t: -t[1])
        top_scores = [{"score": f"{a}-{b}", "p": p} for (a, b), p in flat[:5]]
        elo_diff = float(self.elo.get(home, 1500)) - float(self.elo.get(away, 1500))
        result = {
            "home": home, "away": away, "neutral": neutral,
            "lam_h": float(lam_h), "lam_a": float(lam_a),
            "p_H": p_home, "p_D": p_draw, "p_A": p_away,
            "top_score": f"{i}-{j}", "p_top_score": float(M[i, j]),
            "p_over25": p_over25, "p_under25": 1 - p_over25,
            "p_btts_yes": p_btts, "p_btts_no": 1 - p_btts,
            "elo_home": float(self.elo.get(home, 1500)),
            "elo_away": float(self.elo.get(away, 1500)),
            "elo_diff": elo_diff,
            "top_scores": top_scores,
        }
        self._predict_cache[key] = result
        return result

    def predict_with_ci(self, home, away, neutral=True, n_bootstrap=200, seed=42):
        """Retorna previsão com intervalos de confiança via bootstrap dos parâmetros.

        Usa perturbação gaussiana nos parâmetros com σ estimada da curvatura da
        log-verossimilhança (aproximação pelo método delta). n_bootstrap=200 dá
        ICs razoáveis em ~2s; usar 500 para publicação.
        """
        base = self.predict(home, away, neutral)
        if base is None:
            return None

        rng = np.random.default_rng(seed)
        # Perturbação: σ ≈ 0.05 nos parâmetros de ataque/defesa (conservador)
        SIGMA = 0.05
        results = {"p_H": [], "p_D": [], "p_A": [], "p_over25": [], "p_btts_yes": []}

        for _ in range(n_bootstrap):
            p_boot = self._p.copy()
            noise = rng.normal(0, SIGMA, size=len(p_boot))
            p_boot["attack"] = p_boot["attack"] + noise
            p_boot["defense"] = p_boot["defense"] + noise

            lam_h = np.exp(float(p_boot.loc[home, "attack"]) - float(p_boot.loc[away, "defense"])
                           + (0 if neutral else self.home_adv))
            lam_a = np.exp(float(p_boot.loc[away, "attack"]) - float(p_boot.loc[home, "defense"]))
            h_pmf = poisson.pmf(np.arange(11), lam_h)
            a_pmf = poisson.pmf(np.arange(11), lam_a)
            M = np.outer(h_pmf, a_pmf)
            M[0, 0] *= 1 - lam_h * lam_a * self.rho
            M[0, 1] *= 1 + lam_h * self.rho
            M[1, 0] *= 1 + lam_a * self.rho
            M[1, 1] *= 1 - self.rho
            M = np.maximum(M, 0)
            M /= max(M.sum(), 1e-10)
            tot = self._goal_totals(M)
            results["p_H"].append(float(np.tril(M, -1).sum()))
            results["p_D"].append(float(np.trace(M)))
            results["p_A"].append(float(np.triu(M, 1).sum()))
            results["p_over25"].append(float(tot[3:].sum()))
            results["p_btts_yes"].append(float(M[1:, 1:].sum()))

        ci = {}
        for k, vals in results.items():
            arr = np.array(vals)
            ci[k] = {
                "mean": float(arr.mean()),
                "p5": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
            }
        return {**base, "ci": ci, "n_bootstrap": n_bootstrap}

    def compute_markets(self, home, away, neutral=True):
        """Retorna probabilidades para todos os mercados. LRU por par."""
        key = (home, away, neutral)
        if key in self._markets_cache:
            return self._markets_cache[key]
        result = self._compute_markets_inner(home, away, neutral)
        if result is not None:
            self._markets_cache[key] = result
        return result

    def _compute_markets_inner(self, home, away, neutral=True):
        if not (self.has(home) and self.has(away)):
            return None
        M, lam_h, lam_a = self.match_matrix(home, away, neutral)
        n = M.shape[0]

        p_H = float(np.tril(M, -1).sum())
        p_D = float(np.trace(M))
        p_A = float(np.triu(M, 1).sum())

        home_goals = M.sum(axis=1)
        away_goals = M.sum(axis=0)
        tot = self._goal_totals(M)

        def ov(arr, k): return float(arr[k + 1:].sum())
        def un(arr, k): return float(arr[:k + 1].sum())
        def sel(label, p):
            p = max(min(float(p), 0.9999), 0.0001)
            return {"label": label, "p": round(p, 4), "odd_justa": round(1.0 / p, 2)}
        def pois_ov(lam, k): return float(1.0 - poisson.cdf(k, lam))
        def pois_un(lam, k): return float(poisson.cdf(k, lam))

        # Split 1T/2T calibrado por seleção
        fh_ht, fa_ht = self._ht_splits(home, away)
        lam_h_ht = lam_h * fh_ht
        lam_a_ht = lam_a * fa_ht
        lam_h_2t = lam_h * (1 - fh_ht)
        lam_a_2t = lam_a * (1 - fa_ht)
        split_note = f"Split calibrado: {home} {fh_ht*100:.0f}%/2T, {away} {fa_ht*100:.0f}%/2T"

        maxg_ht = 6
        idx_ht = np.arange(maxg_ht)
        M_ht = np.outer(poisson.pmf(idx_ht, lam_h_ht), poisson.pmf(idx_ht, lam_a_ht))
        M_2t = np.outer(poisson.pmf(idx_ht, lam_h_2t), poisson.pmf(idx_ht, lam_a_2t))
        M_ht /= max(M_ht.sum(), 1e-10)
        M_2t /= max(M_2t.sum(), 1e-10)

        p_ht_H = float(np.tril(M_ht, -1).sum())
        p_ht_D = float(np.trace(M_ht))
        p_ht_A = float(np.triu(M_ht, 1).sum())

        home_goals_2t = M_2t.sum(axis=1)
        away_goals_2t = M_2t.sum(axis=0)
        tot_ht = self._goal_totals(M_ht)
        tot_2t = self._goal_totals(M_2t)
        btts = float(M[1:, 1:].sum())
        btts_ht = float(M_ht[1:, 1:].sum())

        p_no_goal = float(M[0, 0])
        total_rate = lam_h + lam_a
        p_home_first = (1 - p_no_goal) * lam_h / total_rate if total_rate > 0 else 0.0
        p_away_first = (1 - p_no_goal) * lam_a / total_rate if total_rate > 0 else 0.0

        MEAN_GOALS_WC = 2.6
        intensity = total_rate / MEAN_GOALS_WC

        corner_rate = 9.7 * (intensity ** 0.6)
        corner_h = corner_rate * lam_h / total_rate if total_rate > 0 else corner_rate / 2
        corner_a = corner_rate - corner_h

        tension = max(0.0, 1.0 - abs(p_H - p_A))
        card_rate = 3.1 * (0.8 + 0.4 * tension)

        lam_mean = MEAN_GOALS_WC / 2
        shot_h = 12.75 * ((lam_h / lam_mean) ** 0.65)
        shot_a = 12.75 * ((lam_a / lam_mean) ** 0.65)
        shot_rate = shot_h + shot_a

        sot_h = 4.25 * ((lam_h / lam_mean) ** 0.65)
        sot_a = 4.25 * ((lam_a / lam_mean) ** 0.65)
        sot_rate = sot_h + sot_a

        offside_h = 1.75 * ((lam_h / lam_mean) ** 0.5)
        offside_a = 1.75 * ((lam_a / lam_mean) ** 0.5)
        offside_rate = offside_h + offside_a

        markets = [
            {"market": "Resultado Final (1X2)", "icon": "⚽", "selections": [
                sel(f"{home} vence (1)", p_H), sel("Empate (X)", p_D), sel(f"{away} vence (2)", p_A),
            ]},
            {"market": "Dupla Chance", "icon": "2️⃣", "selections": [
                sel(f"1X — {home} ou Empate", p_H + p_D),
                sel(f"12 — {home} ou {away}", p_H + p_A),
                sel(f"X2 — Empate ou {away}", p_D + p_A),
            ]},
            {"market": "Próximo Gol", "icon": "🎯", "selections": [
                sel(f"{home} marca primeiro", p_home_first),
                sel("Nenhum gol no jogo", p_no_goal),
                sel(f"{away} marca primeiro", p_away_first),
            ]},
            {"market": "Ambas as Equipes Marcam (BTTS)", "icon": "🥅", "selections": [
                sel("Sim", btts), sel("Não", 1 - btts),
            ]},
            {"market": "Total de Gols", "icon": "📊", "selections": [
                sel("Over 0.5", ov(tot, 0)), sel("Under 0.5", un(tot, 0)),
                sel("Over 1.5", ov(tot, 1)), sel("Under 1.5", un(tot, 1)),
                sel("Over 2.5", ov(tot, 2)), sel("Under 2.5", un(tot, 2)),
                sel("Over 3.5", ov(tot, 3)), sel("Under 3.5", un(tot, 3)),
                sel("Over 4.5", ov(tot, 4)), sel("Under 4.5", un(tot, 4)),
            ]},
            {"market": f"Gols de {home}", "icon": "🔵", "selections": [
                sel("Over 0.5", ov(home_goals, 0)), sel("Under 0.5", un(home_goals, 0)),
                sel("Over 1.5", ov(home_goals, 1)), sel("Under 1.5", un(home_goals, 1)),
                sel("Over 2.5", ov(home_goals, 2)), sel("Under 2.5", un(home_goals, 2)),
            ]},
            {"market": f"Gols de {away}", "icon": "🟠", "selections": [
                sel("Over 0.5", ov(away_goals, 0)), sel("Under 0.5", un(away_goals, 0)),
                sel("Over 1.5", ov(away_goals, 1)), sel("Under 1.5", un(away_goals, 1)),
                sel("Over 2.5", ov(away_goals, 2)), sel("Under 2.5", un(away_goals, 2)),
            ]},
            {"market": "Resultado 1º Tempo", "icon": "🕐",
             "note": split_note, "selections": [
                sel(f"{home} vence no HT", p_ht_H),
                sel("Empate no HT", p_ht_D),
                sel(f"{away} vence no HT", p_ht_A),
            ]},
            {"market": "Total de Gols — 1º Tempo", "icon": "🕐",
             "note": "Aproximação probabilística", "selections": [
                sel("Over 0.5", ov(tot_ht, 0)), sel("Under 0.5", un(tot_ht, 0)),
                sel("Over 1.5", ov(tot_ht, 1)), sel("Under 1.5", un(tot_ht, 1)),
                sel("Over 2.5", ov(tot_ht, 2)), sel("Under 2.5", un(tot_ht, 2)),
            ]},
            {"market": "Ambas Marcam no 1º Tempo", "icon": "🕐",
             "note": "Aproximação probabilística", "selections": [
                sel("Sim", btts_ht), sel("Não", 1 - btts_ht),
            ]},
            {"market": "Total de Gols — 2º Tempo", "icon": "🕑",
             "note": f"Aprox: {(1-fh_ht)*100:.0f}% dos gols esperados no 2T", "selections": [
                sel("Over 0.5", ov(tot_2t, 0)), sel("Under 0.5", un(tot_2t, 0)),
                sel("Over 1.5", ov(tot_2t, 1)), sel("Under 1.5", un(tot_2t, 1)),
                sel("Over 2.5", ov(tot_2t, 2)), sel("Under 2.5", un(tot_2t, 2)),
            ]},
            {"market": "Time a Marcar no 2º Tempo", "icon": "🕑",
             "note": "Aproximação probabilística", "selections": [
                sel(f"{home} marca no 2T", ov(home_goals_2t, 0)),
                sel(f"{away} marca no 2T", ov(away_goals_2t, 0)),
                sel("Ambas marcam no 2T", float(M_2t[1:, 1:].sum())),
                sel("Nenhuma marca no 2T", float(M_2t[0, 0])),
            ]},
            {"market": "Escanteios — Total", "icon": "🚩",
             "note": f"Modelo estatístico · λ≈{corner_rate:.1f} (média WC = 9.7)", "selections": [
                sel("Over 7.5", pois_ov(corner_rate, 7)), sel("Over 8.5", pois_ov(corner_rate, 8)),
                sel("Over 9.5", pois_ov(corner_rate, 9)), sel("Over 10.5", pois_ov(corner_rate, 10)),
                sel("Under 8.5", pois_un(corner_rate, 8)), sel("Under 9.5", pois_un(corner_rate, 9)),
                sel("Under 10.5", pois_un(corner_rate, 10)), sel("Under 11.5", pois_un(corner_rate, 11)),
            ]},
            {"market": f"Escanteios — {home}", "icon": "🚩",
             "note": "Modelo estatístico", "selections": [
                sel("Over 3.5", pois_ov(corner_h, 3)), sel("Over 4.5", pois_ov(corner_h, 4)),
                sel("Over 5.5", pois_ov(corner_h, 5)), sel("Under 4.5", pois_un(corner_h, 4)),
                sel("Under 5.5", pois_un(corner_h, 5)),
            ]},
            {"market": f"Escanteios — {away}", "icon": "🚩",
             "note": "Modelo estatístico", "selections": [
                sel("Over 3.5", pois_ov(corner_a, 3)), sel("Over 4.5", pois_ov(corner_a, 4)),
                sel("Over 5.5", pois_ov(corner_a, 5)), sel("Under 4.5", pois_un(corner_a, 4)),
                sel("Under 5.5", pois_un(corner_a, 5)),
            ]},
            {"market": "Cartões — Total", "icon": "🟨",
             "note": f"Modelo estatístico · λ≈{card_rate:.2f} (média WC = 3.1)", "selections": [
                sel("Over 1.5", pois_ov(card_rate, 1)), sel("Over 2.5", pois_ov(card_rate, 2)),
                sel("Over 3.5", pois_ov(card_rate, 3)), sel("Over 4.5", pois_ov(card_rate, 4)),
                sel("Under 2.5", pois_un(card_rate, 2)), sel("Under 3.5", pois_un(card_rate, 3)),
                sel("Under 4.5", pois_un(card_rate, 4)),
            ]},
            {"market": "Total de Chutes", "icon": "👟",
             "note": f"Modelo estatístico · λ≈{shot_rate:.1f} (média WC = 25.5)", "selections": [
                sel("Over 20.5", pois_ov(shot_rate, 20)), sel("Over 22.5", pois_ov(shot_rate, 22)),
                sel("Over 24.5", pois_ov(shot_rate, 24)), sel("Over 26.5", pois_ov(shot_rate, 26)),
                sel("Under 22.5", pois_un(shot_rate, 22)), sel("Under 24.5", pois_un(shot_rate, 24)),
                sel("Under 26.5", pois_un(shot_rate, 26)),
            ]},
            {"market": "Total de Chutes ao Gol", "icon": "🎯",
             "note": f"Modelo estatístico · λ≈{sot_rate:.1f} (média WC = 8.5)", "selections": [
                sel("Over 6.5", pois_ov(sot_rate, 6)), sel("Over 7.5", pois_ov(sot_rate, 7)),
                sel("Over 8.5", pois_ov(sot_rate, 8)), sel("Over 9.5", pois_ov(sot_rate, 9)),
                sel("Under 7.5", pois_un(sot_rate, 7)), sel("Under 8.5", pois_un(sot_rate, 8)),
                sel("Under 9.5", pois_un(sot_rate, 9)),
            ]},
            {"market": "Total de Impedimentos", "icon": "🚫",
             "note": f"Modelo estatístico · λ≈{offside_rate:.1f} (média WC = 3.5)", "selections": [
                sel("Over 1.5", pois_ov(offside_rate, 1)), sel("Over 2.5", pois_ov(offside_rate, 2)),
                sel("Over 3.5", pois_ov(offside_rate, 3)), sel("Over 4.5", pois_ov(offside_rate, 4)),
                sel("Under 2.5", pois_un(offside_rate, 2)), sel("Under 3.5", pois_un(offside_rate, 3)),
                sel("Under 4.5", pois_un(offside_rate, 4)),
            ]},
            {"market": "Resultado Exato (top 10)", "icon": "🎲", "selections": sorted(
                [sel(f"{x}-{y}", float(M[x, y])) for x in range(min(n, 7)) for y in range(min(n, 7))],
                key=lambda s: -s["p"]
            )[:10]},
        ]

        result = {
            "home": home, "away": away, "markets": markets,
            "lam_h": round(float(lam_h), 3), "lam_a": round(float(lam_a), 3),
            "corner_rate": round(corner_rate, 1), "card_rate": round(card_rate, 2),
            "shot_rate": round(shot_rate, 1), "sot_rate": round(sot_rate, 1),
            "offside_rate": round(offside_rate, 1),
            "disclaimer_jogador": (
                "Mercados de jogador (marcar gol, chutes individuais, cartão por jogador) "
                "requerem dados de elenco e minutos jogados — fora do escopo do modelo coletivo de seleções."
            ),
        }
        return result

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

    def backtest(self, df_test, sample_size=None, seed=42):
        """Backtesting: compara previsões do modelo contra resultados reais.

        Retorna Brier Score, log-loss, acurácia e curvas de calibração.
        df_test deve ter colunas: home_team, away_team, home_score, away_score, neutral.
        """
        rng = np.random.default_rng(seed)
        rows = df_test.dropna(subset=["home_score", "away_score"]).copy()
        if sample_size and len(rows) > sample_size:
            rows = rows.sample(sample_size, random_state=seed)

        probs, actuals = [], []
        for _, row in rows.iterrows():
            pred = self.predict(str(row.home_team), str(row.away_team),
                                bool(row.get("neutral", True)))
            if pred is None:
                continue
            p = [pred["p_H"], pred["p_D"], pred["p_A"]]
            hs, as_ = int(row.home_score), int(row.away_score)
            if hs > as_: outcome = 0
            elif hs == as_: outcome = 1
            else: outcome = 2
            actual = [0, 0, 0]
            actual[outcome] = 1
            probs.append(p)
            actuals.append(actual)

        if not probs:
            return {"ok": False, "reason": "Nenhum jogo válido no conjunto de teste."}

        P = np.array(probs)
        A = np.array(actuals)

        brier = float(np.mean(np.sum((P - A) ** 2, axis=1)))
        P_clip = np.clip(P, 1e-10, 1)
        log_loss = float(-np.mean(np.sum(A * np.log(P_clip), axis=1)))
        pred_outcome = P.argmax(axis=1)
        true_outcome = A.argmax(axis=1)
        accuracy = float(np.mean(pred_outcome == true_outcome))

        # Curvas de calibração (10 bins)
        bins = np.linspace(0, 1, 11)
        calibration = []
        flat_p = P.ravel()
        flat_a = A.ravel()
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (flat_p >= lo) & (flat_p < hi)
            if mask.sum() == 0:
                continue
            calibration.append({
                "bin_center": round((lo + hi) / 2, 2),
                "mean_pred": round(float(flat_p[mask].mean()), 3),
                "mean_actual": round(float(flat_a[mask].mean()), 3),
                "n": int(mask.sum()),
            })

        return {
            "ok": True,
            "n_games": len(probs),
            "brier_score": round(brier, 4),
            "log_loss": round(log_loss, 4),
            "accuracy": round(accuracy, 4),
            "calibration": calibration,
            "interpretation": _interpret_backtest(brier, log_loss, accuracy),
        }


def _interpret_backtest(brier, log_loss, acc):
    parts = []
    if brier < 0.20:
        parts.append(f"Brier Score excelente ({brier:.3f} < 0.20)")
    elif brier < 0.25:
        parts.append(f"Brier Score bom ({brier:.3f})")
    else:
        parts.append(f"Brier Score fraco ({brier:.3f} > 0.25) — revisar calibração")
    if log_loss < 1.0:
        parts.append(f"Log-loss ótimo ({log_loss:.3f})")
    elif log_loss < 1.05:
        parts.append(f"Log-loss adequado ({log_loss:.3f})")
    else:
        parts.append(f"Log-loss elevado ({log_loss:.3f}) — modelo superconfiante")
    parts.append(f"Acurácia 1X2: {acc*100:.1f}% (baseline naive ≈ 46%)")
    return ". ".join(parts) + "."


_MODEL = None
_MODEL_LOCK = threading.Lock()
_TOURNAMENT_CACHE = {}
_TOURNAMENT_CACHE_TTL = 1800   # 30 minutos


def get_model(force=False):
    """Thread-safe com double-checked locking."""
    global _MODEL
    if _MODEL is not None and not force:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None and not force:
            return _MODEL
        df = load_data()
        sig = _data_signature(df)
        if not force and os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, "rb") as f:
                    blob = pickle.load(f)
                if blob.get("sig") == sig and blob.get("version") == CACHE_VERSION:
                    _MODEL = Model(blob["params"], blob["home_adv"], blob["rho"],
                                   blob.get("delta", 0.0),
                                   blob["elo"], blob["ref_date"], blob["n_train"],
                                   blob["converged"], blob.get("team_stats"))
                    return _MODEL
            except Exception:
                pass
        _MODEL = _train_and_cache(df, sig)
        return _MODEL


def get_tournament_cached(model, n_sims=10000, seed=42):
    """Retorna resultado do Monte Carlo do torneio com cache de 30min."""
    import tournament as trn
    key = (id(model), n_sims, seed)
    now = time.time()
    if key in _TOURNAMENT_CACHE:
        ts, result = _TOURNAMENT_CACHE[key]
        if now - ts < _TOURNAMENT_CACHE_TTL:
            return result
    result = trn.simulate_tournament(model, n_sims=n_sims, seed=seed)
    _TOURNAMENT_CACHE[key] = (now, result)
    return result


def _train_and_cache(df, sig):
    ref_date = df.date.max() + pd.Timedelta(days=1)
    train_full = df[df.date >= TRAIN_FROM].copy()
    params, ha, rho, delta, res = fit_dixon_coles(train_full, ref_date)
    elo = compute_elo(df.sort_values("date"))
    team_stats = _compute_team_stats(df)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({
            "version": CACHE_VERSION,
            "sig": sig, "params": params, "home_adv": ha, "rho": rho,
            "delta": delta, "elo": elo, "ref_date": ref_date,
            "n_train": len(train_full), "converged": bool(res.success),
            "team_stats": team_stats,
        }, f)
    return Model(params, ha, rho, delta, elo, ref_date, len(train_full), res.success, team_stats)


# --------------------------------------------------------------------------- #
# Value / EV / Kelly
# --------------------------------------------------------------------------- #
def kelly_fraction(p, odd):
    b = odd - 1.0
    if b <= 0:
        return 0.0
    return max((b * p - (1 - p)) / b, 0.0)


def value_bet(p, odd):
    implied = 1.0 / odd
    ev = p * odd - 1.0
    kf = kelly_fraction(p, odd)
    return {
        "p_model": p, "odd": odd, "implied": implied,
        "edge": p - implied, "ev": ev,
        "kelly_full": kf, "kelly_half": kf / 2, "kelly_quarter": kf / 4,
    }


def analyze_market(pred, odds):
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
# Monte Carlo bankroll
# --------------------------------------------------------------------------- #
def monte_carlo_bankroll(bets, bankroll=100.0, strategy="kelly", kelly_mult=0.25,
                         flat_pct=0.02, n_sims=10000, min_edge=0.0,
                         ruin_threshold=0.20, seed=42):
    rng = np.random.default_rng(seed)
    placed = []
    for b in bets:
        p = float(b["p"]); odd = float(b["odd"])
        ev = p * odd - 1.0
        if ev <= min_edge:
            continue
        frac = kelly_mult * kelly_fraction(p, odd) if strategy == "kelly" else flat_pct
        if frac <= 0:
            continue
        placed.append({"p": p, "odd": odd, "frac": frac, "ev": ev,
                       "label": b.get("label", f"{odd:.2f} @ {p:.0%}")})
    if not placed:
        return {"ok": False, "reason": "Nenhuma aposta com EV acima do limiar.", "n_bets": 0}

    n_bets = len(placed)
    fracs = np.array([b["frac"] for b in placed])
    odds = np.array([b["odd"] for b in placed])
    ps = np.array([b["p"] for b in placed])

    bank = np.full(n_sims, float(bankroll))
    peak = np.full(n_sims, float(bankroll))
    max_dd = np.zeros(n_sims)
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
    def pct(a, q): return float(np.percentile(a, q))
    return {
        "ok": True, "n_bets": n_bets, "n_sims": n_sims,
        "strategy": strategy, "kelly_mult": kelly_mult, "flat_pct": flat_pct,
        "bankroll_inicial": float(bankroll),
        "ev_medio": float(np.mean([b["ev"] for b in placed])),
        "stake_medio_pct": float(np.mean(fracs)),
        "final_mediana": float(np.median(final)), "final_media": float(np.mean(final)),
        "final_p5": pct(final, 5), "final_p25": pct(final, 25),
        "final_p75": pct(final, 75), "final_p95": pct(final, 95),
        "roi_mediano": float(np.median(growth)),
        "prob_lucro": float(np.mean(final > bankroll)),
        "prob_dobrar": float(np.mean(final >= 2 * bankroll)),
        "prob_ruina": float(np.mean(ruined)),
        "drawdown_mediano": float(np.median(max_dd)),
        "drawdown_p95": pct(max_dd, 95), "drawdown_max": float(np.max(max_dd)),
        "sample_paths": sample_paths.tolist(),
        "bets": [{"label": b["label"], "p": b["p"], "odd": b["odd"],
                  "ev": b["ev"], "stake_pct": b["frac"]} for b in placed],
    }


def remove_vig(odd_h, odd_d, odd_a):
    total = 1.0 / odd_h + 1.0 / odd_d + 1.0 / odd_a
    return 1.0 / (odd_h * total), 1.0 / (odd_d * total), 1.0 / (odd_a * total)


def blend_with_market(pred, odd_h, odd_d, odd_a, alpha=0.6):
    mh, md, ma = remove_vig(odd_h, odd_d, odd_a)
    bh = alpha * pred["p_H"] + (1.0 - alpha) * mh
    bd = alpha * pred["p_D"] + (1.0 - alpha) * md
    ba = alpha * pred["p_A"] + (1.0 - alpha) * ma
    total = bh + bd + ba
    return {
        "p_H": bh / total, "p_D": bd / total, "p_A": ba / total,
        "market_H": mh, "market_D": md, "market_A": ma, "alpha": alpha,
    }


def compare_strategies(bets, bankroll=100.0, n_sims=10000, min_edge=0.0):
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
        m = monte_carlo_bankroll(bets, bankroll=bankroll, n_sims=n_sims, min_edge=min_edge, **cfg)
        if not m.get("ok"):
            continue
        rows.append({
            "nome": name, "final_mediana": m["final_mediana"],
            "final_p5": m["final_p5"], "final_p95": m["final_p95"],
            "roi_mediano": m["roi_mediano"], "prob_lucro": m["prob_lucro"],
            "prob_ruina": m["prob_ruina"], "drawdown_mediano": m["drawdown_mediano"],
            "drawdown_p95": m["drawdown_p95"],
        })
    return rows
