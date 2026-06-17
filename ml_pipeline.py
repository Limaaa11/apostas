"""
Pipeline de ML para apostas de valor — Copa 2026.

Complementa o modelo Dixon-Coles com abordagem baseada em dados:
  - Engenharia de variáveis com rolling windows (3, 5, 10 jogos) sem data leakage
  - Features de mandante, visitante e geral por time
  - Variável de fadiga (dias de descanso entre jogos)
  - XGBoost com calibração isotônica de probabilidade
  - Filtro de valor: P(modelo) > P(casa) + margem_ev
  - Backtesting financeiro com TimeSeriesSplit + Yield + Kelly fracionado
"""

import os
import pickle
import warnings
import threading
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "results.csv")
ML_CACHE_PATH = os.path.join(BASE_DIR, "ml_model.pkl")

WINDOWS = [3, 5, 10]

_ML_LOCK = threading.Lock()
_ML_MODEL = None   # singleton treinado


# ═══════════════════════════════════════════════════════════════
#  ENGENHARIA DE VARIÁVEIS
# ═══════════════════════════════════════════════════════════════

def _add_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna outcome: 'H' | 'D' | 'A'."""
    df = df.copy()
    df["outcome"] = "D"
    df.loc[df.home_score > df.away_score, "outcome"] = "H"
    df.loc[df.home_score < df.away_score, "outcome"] = "A"
    return df


def _team_history(df: pd.DataFrame, team: str, before_date: pd.Timestamp,
                  window: int, side: str = "all") -> dict:
    """
    Retorna estatísticas de rolling para `team` nos últimos `window` jogos
    *antes* de `before_date` (sem data leakage).

    side: 'home' | 'away' | 'all'
    """
    if side == "home":
        mask = df.home_team == team
    elif side == "away":
        mask = df.away_team == team
    else:
        mask = (df.home_team == team) | (df.away_team == team)

    history = df[mask & (df.date < before_date)].sort_values("date").tail(window)

    if history.empty:
        return {
            "gf": np.nan, "gc": np.nan, "gd": np.nan,
            "pts": np.nan, "win_rate": np.nan, "draw_rate": np.nan,
        }

    gf_list, gc_list, pts_list = [], [], []
    for _, row in history.iterrows():
        is_home = row.home_team == team
        gf = row.home_score if is_home else row.away_score
        gc = row.away_score if is_home else row.home_score
        gf_list.append(gf); gc_list.append(gc)
        if gf > gc: pts_list.append(3)
        elif gf == gc: pts_list.append(1)
        else: pts_list.append(0)

    gf_arr = np.array(gf_list)
    gc_arr = np.array(gc_list)
    pts_arr = np.array(pts_list)
    n = len(pts_arr)
    return {
        "gf": float(gf_arr.mean()),
        "gc": float(gc_arr.mean()),
        "gd": float((gf_arr - gc_arr).mean()),
        "pts": float(pts_arr.mean()),
        "win_rate": float((pts_arr == 3).sum() / n),
        "draw_rate": float((pts_arr == 1).sum() / n),
    }


def _days_rest(df: pd.DataFrame, team: str, before_date: pd.Timestamp) -> float:
    """Dias de descanso desde o último jogo do time antes de `before_date`."""
    mask = ((df.home_team == team) | (df.away_team == team)) & (df.date < before_date)
    past = df[mask]
    if past.empty:
        return np.nan
    last = past.date.max()
    return float((before_date - last).days)


def build_features(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    """
    Constrói features sem data leakage.
    Retorna DataFrame com colunas de feature + 'outcome' + 'date'.
    """
    if windows is None:
        windows = WINDOWS

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = _add_outcome(df)
    df = df.sort_values("date").reset_index(drop=True)

    records = []
    for _, row in df.iterrows():
        rec = {
            "date": row.date,
            "home_team": row.home_team,
            "away_team": row.away_team,
            "outcome": row.outcome,
            "neutral": int(row.get("neutral", 0) or 0),
        }

        for w in windows:
            # Home team stats (geral + como mandante)
            h_all = _team_history(df, row.home_team, row.date, w, "all")
            h_home = _team_history(df, row.home_team, row.date, w, "home")
            # Away team stats (geral + como visitante)
            a_all = _team_history(df, row.away_team, row.date, w, "all")
            a_away = _team_history(df, row.away_team, row.date, w, "away")

            for stat, val in h_all.items():
                rec[f"h_{stat}_all_{w}"] = val
            for stat, val in h_home.items():
                rec[f"h_{stat}_home_{w}"] = val
            for stat, val in a_all.items():
                rec[f"a_{stat}_all_{w}"] = val
            for stat, val in a_away.items():
                rec[f"a_{stat}_away_{w}"] = val

        # Diferencial entre os times (mais sinal para o modelo)
        for w in windows:
            for stat in ["gf", "gc", "gd", "pts", "win_rate"]:
                h_val = rec.get(f"h_{stat}_all_{w}", np.nan)
                a_val = rec.get(f"a_{stat}_all_{w}", np.nan)
                if not (np.isnan(h_val) or np.isnan(a_val)):
                    rec[f"diff_{stat}_{w}"] = h_val - a_val
                else:
                    rec[f"diff_{stat}_{w}"] = np.nan

        # Fadiga
        rec["h_days_rest"] = _days_rest(df, row.home_team, row.date)
        rec["a_days_rest"] = _days_rest(df, row.away_team, row.date)
        rec["rest_diff"] = (
            rec["h_days_rest"] - rec["a_days_rest"]
            if not (np.isnan(rec["h_days_rest"]) or np.isnan(rec["a_days_rest"]))
            else np.nan
        )

        records.append(rec)

    return pd.DataFrame(records)


def _feature_cols(feat_df: pd.DataFrame) -> list:
    exclude = {"date", "home_team", "away_team", "outcome"}
    return [c for c in feat_df.columns if c not in exclude]


# ═══════════════════════════════════════════════════════════════
#  MODELAGEM
# ═══════════════════════════════════════════════════════════════

class MLBettingModel:
    """
    Wrapper do XGBoost calibrado para previsão 1X2.

    Atributos públicos:
      .classes_   — ['A', 'D', 'H']
      .feature_cols — lista de colunas de feature usadas no treino
      .meta       — dict com métricas de treino e data de treino
    """

    def __init__(self):
        self.classes_ = ["A", "D", "H"]
        self.feature_cols = []
        self.meta = {}
        self._model = None
        self._le = LabelEncoder()

    def fit(self, feat_df: pd.DataFrame, min_date: str = "2010-01-01",
            n_splits: int = 5) -> dict:
        """
        Treina com TimeSeriesSplit e retorna métricas out-of-fold.
        """
        data = feat_df[feat_df.date >= min_date].copy()
        data = data.dropna(subset=["outcome"])

        self.feature_cols = _feature_cols(data)
        X = data[self.feature_cols].values
        y_raw = data["outcome"].values
        y = self._le.fit_transform(y_raw)  # A=0, D=1, H=2

        # Imputação: NaN → mediana de cada coluna
        self._medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        X_imp = X.copy()
        X_imp[nan_mask] = np.take(self._medians, np.where(nan_mask)[1])

        # TimeSeriesSplit — avalia log-loss out-of-fold
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_preds = np.zeros((len(y), 3))
        fold_losses = []

        base_xgb = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, verbosity=0,
        )

        for train_idx, val_idx in tscv.split(X_imp):
            X_tr, X_val = X_imp[train_idx], X_imp[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            if len(np.unique(y_tr)) < 3:
                continue
            m = CalibratedClassifierCV(base_xgb, method="isotonic", cv=3)
            m.fit(X_tr, y_tr)
            proba = m.predict_proba(X_val)
            oof_preds[val_idx] = proba
            fold_losses.append(log_loss(y_val, proba))

        # Treino final em todos os dados
        self._model = CalibratedClassifierCV(base_xgb, method="isotonic", cv=3)
        self._model.fit(X_imp, y)

        oof_valid = oof_preds[oof_preds.sum(axis=1) > 0]
        y_valid = y[oof_preds.sum(axis=1) > 0]

        # Brier Score para cada resultado
        bs_h = brier_score_loss((y_valid == 2), oof_valid[:, 2])
        bs_d = brier_score_loss((y_valid == 1), oof_valid[:, 1])
        bs_a = brier_score_loss((y_valid == 0), oof_valid[:, 0])
        acc = float(np.mean(np.argmax(oof_valid, axis=1) == y_valid))

        self.meta = {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(data),
            "n_features": len(self.feature_cols),
            "n_splits": n_splits,
            "oof_log_loss": float(np.mean(fold_losses)) if fold_losses else None,
            "brier_H": round(bs_h, 4),
            "brier_D": round(bs_d, 4),
            "brier_A": round(bs_a, 4),
            "accuracy_1x2": round(acc, 4),
            "classes": list(self._le.classes_),
        }
        return self.meta

    def predict_proba(self, home: str, away: str, feat_df: pd.DataFrame = None,
                      raw_features: dict = None) -> dict:
        """
        Retorna {'p_H': float, 'p_D': float, 'p_A': float} ou None.

        Usa feat_df para buscar a última linha de features do par (home, away)
        ou aceita raw_features direto (dict com as feature_cols como chave).
        """
        if self._model is None:
            return None

        if raw_features is not None:
            row_vals = [raw_features.get(c, np.nan) for c in self.feature_cols]
        elif feat_df is not None:
            mask = (feat_df.home_team == home) & (feat_df.away_team == away)
            subset = feat_df[mask].sort_values("date")
            if subset.empty:
                return None
            row_vals = subset[self.feature_cols].values[-1]
        else:
            return None

        row = np.array([row_vals], dtype=float)
        # Imputação de NaN com medianas do treino
        nan_cols = np.isnan(row[0])
        row[0][nan_cols] = self._medians[nan_cols]

        proba = self._model.predict_proba(row)[0]
        idx = {c: i for i, c in enumerate(self._le.classes_)}
        return {
            "p_H": float(proba[idx["H"]]),
            "p_D": float(proba[idx["D"]]),
            "p_A": float(proba[idx["A"]]),
        }

    def predict_from_stats(self, home_stats: dict, away_stats: dict,
                           neutral: bool = True) -> dict:
        """
        Prediz a partir de dicionários de stats pré-calculadas pelo chamador.
        home_stats / away_stats: dict com chaves como 'gf_all_5', 'pts_all_10', etc.
        """
        feat = {"neutral": int(neutral)}
        for k, v in home_stats.items():
            feat[f"h_{k}"] = v
        for k, v in away_stats.items():
            feat[f"a_{k}"] = v
        return self.predict_proba(None, None, raw_features=feat)


# ═══════════════════════════════════════════════════════════════
#  FILTRO DE VALOR (VALUE BETTING)
# ═══════════════════════════════════════════════════════════════

def value_bet_filter(p_model: dict, odd_H: float = None, odd_D: float = None,
                     odd_A: float = None, min_ev: float = 0.03) -> list:
    """
    Compara probabilidades do modelo com as odds do mercado.

    Returns list of dicts: {side, p_model, p_implied, odd, ev, is_value, kelly_quarter}
    """
    results = []
    sides = [
        ("H", "Casa (1)", p_model.get("p_H", 0), odd_H),
        ("D", "Empate (X)", p_model.get("p_D", 0), odd_D),
        ("A", "Fora (2)", p_model.get("p_A", 0), odd_A),
    ]
    for code, label, p, odd in sides:
        if not odd or odd <= 1:
            continue
        p_implied = 1 / odd
        ev = p * odd - 1
        kelly = max(0.0, (p * odd - 1) / (odd - 1)) * 0.25  # Kelly 1/4
        results.append({
            "side": code,
            "label": label,
            "p_model": round(p, 4),
            "p_implied": round(p_implied, 4),
            "odd": odd,
            "ev": round(ev, 4),
            "is_value": ev >= min_ev,
            "kelly_quarter": round(kelly, 4),
        })
    return results


# ═══════════════════════════════════════════════════════════════
#  BACKTESTING FINANCEIRO (TimeSeriesSplit)
# ═══════════════════════════════════════════════════════════════

def financial_backtest(feat_df: pd.DataFrame, model: "MLBettingModel",
                       odds_implied: dict = None,
                       bankroll: float = 100.0,
                       kelly_fraction: float = 0.25,
                       min_ev: float = 0.03,
                       n_splits: int = 5,
                       flat_stake_pct: float = None,
                       seed: int = 42) -> dict:
    """
    Backtesting temporal com TimeSeriesSplit.

    odds_implied: dict {(home, away, date_str): {H: odd, D: odd, A: odd}}
      Se None, simula odds a partir das probabilidades DC com margem de 5%.

    Retorna métricas financeiras + curva de banca.
    """
    data = feat_df.dropna(subset=["outcome"]).copy()
    data = data.sort_values("date").reset_index(drop=True)

    feature_cols = model.feature_cols
    X = data[feature_cols].values
    medians = model._medians
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(medians, np.where(nan_mask)[1])

    y_true = data["outcome"].values   # 'H', 'D', 'A'

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rng = np.random.default_rng(seed)

    all_bets = []
    bk = bankroll

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = model._le.transform(y_true[train_idx])

        if len(np.unique(y_tr)) < 3:
            continue

        fold_model = CalibratedClassifierCV(
            xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               use_label_encoder=False, eval_metric="mlogloss",
                               random_state=42, verbosity=0),
            method="isotonic", cv=3,
        )
        fold_model.fit(X_tr, y_tr)
        proba = fold_model.predict_proba(X_te)
        idx_map = {c: i for i, c in enumerate(model._le.classes_)}

        for i, orig_idx in enumerate(test_idx):
            row = data.iloc[orig_idx]
            home, away, date, actual = row.home_team, row.away_team, row.date, row.outcome

            p = {
                "H": float(proba[i][idx_map["H"]]),
                "D": float(proba[i][idx_map["D"]]),
                "A": float(proba[i][idx_map["A"]]),
            }

            # Simula odds: P_implied = P_true + margem 5% (bootstrap implícito)
            # Em produção, usar odds reais do mercado
            if odds_implied and (home, away, str(date.date())) in odds_implied:
                mkt = odds_implied[(home, away, str(date.date()))]
                odds_dict = mkt
            else:
                vig = 1.05
                odds_dict = {s: (1.0 / (p[s] * vig)) if p[s] > 0 else None for s in ("H", "D", "A")}

            for side in ("H", "D", "A"):
                odd = odds_dict.get(side)
                if not odd or odd <= 1:
                    continue
                p_m = p[side]
                ev_val = p_m * odd - 1
                if ev_val < min_ev:
                    continue

                if flat_stake_pct is not None:
                    stake = bk * flat_stake_pct
                else:
                    kelly = max(0.0, (p_m * odd - 1) / (odd - 1)) * kelly_fraction
                    stake = bk * kelly

                stake = min(stake, bk * 0.20)   # cap 20% banca
                won = actual == side
                profit = stake * (odd - 1) if won else -stake
                bk = max(0, bk + profit)

                all_bets.append({
                    "fold": fold + 1,
                    "date": str(date.date()),
                    "home": home, "away": away, "side": side,
                    "p_model": round(p_m, 4),
                    "odd": round(odd, 2),
                    "ev": round(ev_val, 4),
                    "stake": round(stake, 2),
                    "won": won,
                    "profit": round(profit, 2),
                    "bankroll_after": round(bk, 2),
                })

    if not all_bets:
        return {"ok": False, "error": "Nenhuma aposta de valor encontrada no backtest.",
                "n_splits": n_splits, "min_ev": min_ev}

    df_bets = pd.DataFrame(all_bets)
    total_stake = df_bets.stake.sum()
    total_profit = df_bets.profit.sum()
    n_won = df_bets.won.sum()
    n_bets = len(df_bets)
    yield_pct = (total_profit / total_stake * 100) if total_stake > 0 else 0
    roi_pct = (bk - bankroll) / bankroll * 100
    win_rate = n_won / n_bets if n_bets else 0
    avg_ev = float(df_bets.ev.mean())
    avg_odd = float(df_bets.odd.mean())
    drawdown = _max_drawdown(df_bets.bankroll_after.values, bankroll)

    bk_curve = df_bets[["date", "bankroll_after"]].rename(
        columns={"bankroll_after": "bankroll"}).to_dict("records")

    return {
        "ok": True,
        "n_bets": n_bets,
        "n_won": int(n_won),
        "win_rate": round(win_rate, 4),
        "total_stake": round(total_stake, 2),
        "total_profit": round(total_profit, 2),
        "yield_pct": round(yield_pct, 2),
        "roi_pct": round(roi_pct, 2),
        "bankroll_final": round(bk, 2),
        "max_drawdown_pct": round(drawdown, 2),
        "avg_ev": round(avg_ev, 4),
        "avg_odd": round(avg_odd, 2),
        "bets": all_bets[:200],    # limita para resposta da API
        "bankroll_curve": bk_curve,
        "interpretation": _interpret_financial(yield_pct, n_bets, drawdown),
    }


def _max_drawdown(curve: np.ndarray, start: float) -> float:
    """Maior drawdown percentual a partir do pico."""
    peak = start
    max_dd = 0.0
    for v in curve:
        peak = max(peak, v)
        dd = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)
    return max_dd


def _interpret_financial(yield_pct: float, n_bets: int, drawdown: float) -> str:
    parts = []
    if yield_pct > 5:
        parts.append(f"Yield de {yield_pct:.1f}% — resultado excelente (acima de 5% é raro em apostas profissionais).")
    elif yield_pct > 0:
        parts.append(f"Yield de {yield_pct:.1f}% — resultado positivo com {n_bets} apostas.")
    else:
        parts.append(f"Yield negativo ({yield_pct:.1f}%) — modelo sem edge real neste período.")
    if drawdown > 40:
        parts.append(f"Drawdown máximo de {drawdown:.1f}% — considere reduzir stakes.")
    elif drawdown > 20:
        parts.append(f"Drawdown de {drawdown:.1f}% — volatilidade moderada.")
    else:
        parts.append(f"Drawdown de {drawdown:.1f}% — gestão de banca estável.")
    if n_bets < 100:
        parts.append("Amostra pequena: resultados podem não ser estatisticamente significativos.")
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════
#  SINGLETON — treina e cacheia em disco
# ═══════════════════════════════════════════════════════════════

ML_CACHE_VERSION = 1


def get_ml_model(force_retrain: bool = False) -> "MLBettingModel":
    global _ML_MODEL
    if _ML_MODEL is not None and not force_retrain:
        return _ML_MODEL
    with _ML_LOCK:
        if _ML_MODEL is not None and not force_retrain:
            return _ML_MODEL

        # Tenta carregar do disco
        if os.path.exists(ML_CACHE_PATH) and not force_retrain:
            try:
                with open(ML_CACHE_PATH, "rb") as f:
                    blob = pickle.load(f)
                if blob.get("version") == ML_CACHE_VERSION:
                    _ML_MODEL = blob["model"]
                    return _ML_MODEL
            except Exception:
                pass

        # Treina do zero
        _ML_MODEL = _train_and_cache()
    return _ML_MODEL


def _train_and_cache() -> "MLBettingModel":
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    feat_df = build_features(df)

    m = MLBettingModel()
    m.fit(feat_df, min_date="2010-01-01")
    m._feat_df = feat_df   # guarda para predições futuras

    with open(ML_CACHE_PATH, "wb") as f:
        pickle.dump({"version": ML_CACHE_VERSION, "model": m}, f)
    return m


def predict_match(home: str, away: str, neutral: bool = True) -> dict:
    """
    Wrapper de alto nível: prediz 1X2 para um confronto novo.
    Constrói features via rolling histórico + retorna proba + odd justa.
    """
    m = get_ml_model()
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    feat_df = build_features(df)

    # Usa última linha de features do mandante e visitante separadamente
    # e constrói um vetor de features "sintético" para o novo confronto
    today = pd.Timestamp.now()
    home_stats = {}
    away_stats = {}

    for w in WINDOWS:
        h_all = _team_history(df, home, today, w, "all")
        h_home = _team_history(df, home, today, w, "home")
        a_all = _team_history(df, away, today, w, "all")
        a_away = _team_history(df, away, today, w, "away")
        for stat, val in h_all.items():
            home_stats[f"{stat}_all_{w}"] = val
        for stat, val in h_home.items():
            home_stats[f"{stat}_home_{w}"] = val
        for stat, val in a_all.items():
            away_stats[f"{stat}_all_{w}"] = val
        for stat, val in a_away.items():
            away_stats[f"{stat}_away_{w}"] = val

    # Diferenciais
    for w in WINDOWS:
        for stat in ["gf", "gc", "gd", "pts", "win_rate"]:
            h_v = home_stats.get(f"{stat}_all_{w}", np.nan)
            a_v = away_stats.get(f"{stat}_all_{w}", np.nan)
            home_stats[f"diff_{stat}_{w}"] = (h_v - a_v
                                               if not (np.isnan(h_v) or np.isnan(a_v))
                                               else np.nan)

    h_rest = _days_rest(df, home, today)
    a_rest = _days_rest(df, away, today)
    home_stats["h_days_rest"] = h_rest
    home_stats["a_days_rest"] = a_rest
    home_stats["rest_diff"] = (h_rest - a_rest
                               if not (np.isnan(h_rest) or np.isnan(a_rest))
                               else np.nan)
    home_stats["neutral"] = int(neutral)

    # Monta vetor completo de features
    raw = {}
    for k, v in home_stats.items():
        if k.startswith("diff_") or k in ("neutral", "h_days_rest", "a_days_rest", "rest_diff"):
            raw[k] = v
        else:
            raw[f"h_{k}"] = v
    for k, v in away_stats.items():
        raw[f"a_{k}"] = v

    proba = m.predict_proba(home, away, raw_features=raw)
    if proba is None:
        return None

    return {
        "home": home,
        "away": away,
        "neutral": neutral,
        "p_H": proba["p_H"],
        "p_D": proba["p_D"],
        "p_A": proba["p_A"],
        "odd_fair_H": round(1 / proba["p_H"], 2) if proba["p_H"] > 0 else None,
        "odd_fair_D": round(1 / proba["p_D"], 2) if proba["p_D"] > 0 else None,
        "odd_fair_A": round(1 / proba["p_A"], 2) if proba["p_A"] > 0 else None,
        "model": "xgboost_calibrated",
        "meta": m.meta,
    }
