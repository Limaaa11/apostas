"""
Suite de testes do modelo Dixon-Coles.

Roda com: .venv/Scripts/pytest tests/ -v
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_engine as me


@pytest.fixture(scope="session")
def model():
    return me.get_model()


# ── _goal_totals ──────────────────────────────────────────────
def test_goal_totals_vs_brute_force(model):
    """_goal_totals deve ser equivalente à soma manual das anti-diagonais."""
    home, away = model.teams[0], model.teams[1]
    M, _, _ = model.match_matrix(home, away, neutral=True)
    n = M.shape[0]

    result = model._goal_totals(M)

    # Soma bruta das anti-diagonais
    brute = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            brute[i + j] += M[i, j]

    np.testing.assert_allclose(result, brute, atol=1e-10,
                               err_msg="_goal_totals diverge da soma bruta")


def test_goal_totals_sums_to_one(model):
    home, away = model.teams[0], model.teams[1]
    M, _, _ = model.match_matrix(home, away, neutral=True)
    totals = model._goal_totals(M)
    assert abs(totals.sum() - 1.0) < 1e-8, "Soma de _goal_totals != 1"


# ── compute_markets ───────────────────────────────────────────
def test_markets_returns_dict(model):
    home, away = model.teams[0], model.teams[1]
    result = model.compute_markets(home, away, neutral=True)
    assert result is not None
    assert "markets" in result


def test_markets_min_count(model):
    home, away = model.teams[0], model.teams[1]
    result = model.compute_markets(home, away, neutral=True)
    assert len(result["markets"]) >= 15, "Esperado ao menos 15 mercados"


# ── predict ───────────────────────────────────────────────────
def test_predict_probabilities_sum_to_one(model):
    home, away = model.teams[0], model.teams[1]
    p = model.predict(home, away, neutral=True)
    assert p is not None
    total = p["p_H"] + p["p_D"] + p["p_A"]
    assert abs(total - 1.0) < 1e-6, f"1X2 soma {total:.6f} ≠ 1"


def test_predict_unknown_team_returns_none(model):
    assert model.predict("TEAM_THAT_DOES_NOT_EXIST", model.teams[0]) is None


def test_predict_cache(model):
    """Segunda chamada com mesmo par deve retornar resultado idêntico."""
    home, away = model.teams[0], model.teams[1]
    a = model.predict(home, away, neutral=True)
    b = model.predict(home, away, neutral=True)
    assert a["p_H"] == b["p_H"]


# ── predict_with_ci ───────────────────────────────────────────
def test_predict_with_ci_keys(model):
    home, away = model.teams[0], model.teams[1]
    result = model.predict_with_ci(home, away, neutral=True, n_bootstrap=50, seed=7)
    assert "ci" in result
    assert "p_H" in result["ci"]
    ci = result["ci"]["p_H"]
    assert ci["p5"] <= ci["mean"] <= ci["p95"]


# ── match_matrix ──────────────────────────────────────────────
def test_match_matrix_non_negative(model):
    M, _, _ = model.match_matrix(model.teams[0], model.teams[1], neutral=True)
    assert (M >= 0).all(), "Matriz contém valores negativos"


def test_match_matrix_sums_to_one(model):
    M, _, _ = model.match_matrix(model.teams[0], model.teams[1], neutral=True)
    assert abs(M.sum() - 1.0) < 1e-8


# ── backtest ──────────────────────────────────────────────────
def test_backtest_smoke(model):
    """Backtest com 1 amostra não deve lançar exceção."""
    import pandas as pd
    df = pd.read_csv(me.DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    if df.empty:
        pytest.skip("CSV sem dados com score")
    result = model.backtest(df, sample_size=30, seed=99)
    assert result["ok"] is True
    assert 0 <= result["brier_score"] <= 1
    assert result["log_loss"] > 0


# ── teams / elo ───────────────────────────────────────────────
def test_teams_is_non_empty(model):
    assert len(model.teams) > 50, "Esperado mais de 50 seleções"


def test_elo_is_dict(model):
    assert isinstance(model.elo, dict)
    assert len(model.elo) > 0


# ── CACHE_VERSION ─────────────────────────────────────────────
def test_cache_version_is_int():
    assert isinstance(me.CACHE_VERSION, int)
    assert me.CACHE_VERSION >= 4
