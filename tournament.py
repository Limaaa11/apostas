"""
Simulação Monte Carlo do torneio Copa do Mundo 2026.

Formato: 48 times, 12 grupos de 4, top-2 de cada grupo + 8 melhores terceiros
avançam para o mata-mata de 32 times.

Uso:
    from tournament import simulate_tournament
    import model_engine as me
    result = simulate_tournament(me.get_model(), n_sims=20000)
    # result["champion"] -> dict {team: probability}
"""
import numpy as np
from itertools import combinations

# --------------------------------------------------------------------------- #
# Estrutura dos grupos (extraída dos fixtures do martj42)
# --------------------------------------------------------------------------- #
GROUPS = {
    "A": ["Algeria", "Argentina", "Austria", "Jordan"],
    "B": ["Australia", "Paraguay", "Turkey", "United States"],
    "C": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "D": ["Bosnia and Herzegovina", "Canada", "Qatar", "Switzerland"],
    "E": ["Brazil", "Haiti", "Morocco", "Scotland"],
    "F": ["Cape Verde", "Saudi Arabia", "Spain", "Uruguay"],
    "G": ["Colombia", "DR Congo", "Portugal", "Uzbekistan"],
    "H": ["Croatia", "England", "Ghana", "Panama"],
    "I": ["Curaçao", "Ecuador", "Germany", "Ivory Coast"],
    "J": ["Czech Republic", "Mexico", "South Africa", "South Korea"],
    "K": ["France", "Iraq", "Norway", "Senegal"],
    "L": ["Japan", "Netherlands", "Sweden", "Tunisia"],
}

# Resultados já disputados na Copa 2026 (para fixar em vez de simular)
PLAYED = [
    ("Mexico", "South Africa", 2, 0),
    ("South Korea", "Czech Republic", 2, 1),
    ("Canada", "Bosnia and Herzegovina", 1, 1),
    ("United States", "Paraguay", 4, 1),
    ("Qatar", "Switzerland", 1, 1),
    ("Brazil", "Morocco", 1, 1),
    ("Haiti", "Scotland", 0, 1),
    ("Australia", "Turkey", 2, 0),
    ("Germany", "Curaçao", 7, 1),
    ("Ivory Coast", "Ecuador", 1, 0),
    ("Netherlands", "Japan", 2, 2),
    ("Sweden", "Tunisia", 5, 1),
    ("Belgium", "Egypt", 1, 1),
    ("Iran", "New Zealand", 2, 2),
    ("Spain", "Cape Verde", 0, 0),
    ("Saudi Arabia", "Uruguay", 1, 1),
]

# Pré-computa mapa (home, away) -> (hs, as_) para lookup rápido
_PLAYED_MAP = {(h, a): (hs, as_) for h, a, hs, as_ in PLAYED}


# --------------------------------------------------------------------------- #
# Simulação de partida
# --------------------------------------------------------------------------- #
def _sim_match(model, home, away, rng, neutral=True):
    """Retorna (gols_home, gols_away) para um único jogo simulado."""
    played = _PLAYED_MAP.get((home, away)) or _PLAYED_MAP.get((away, home))
    if played:
        if (home, away) in _PLAYED_MAP:
            return played
        else:
            return played[1], played[0]

    pred = model.predict(home, away, neutral=neutral)
    if pred is None:
        # fallback: empate 0-0
        return 0, 0

    M, lam_h, lam_a = model.match_matrix(home, away, neutral=neutral)
    # amostra do placar a partir da matriz de probabilidades
    M_flat = M.ravel()
    idx = rng.choice(len(M_flat), p=M_flat / M_flat.sum())
    r, c = divmod(idx, M.shape[1])
    return int(r), int(c)


# --------------------------------------------------------------------------- #
# Fase de grupos
# --------------------------------------------------------------------------- #
def _sim_group(model, teams, rng):
    """Simula um grupo completo e retorna standings ordenados."""
    pts = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}
    h2h_pts = {t: {o: 0 for o in teams if o != t} for t in teams}
    h2h_gd = {t: {o: 0 for o in teams if o != t} for t in teams}

    for home, away in combinations(teams, 2):
        hs, as_ = _sim_match(model, home, away, rng, neutral=True)
        gf[home] += hs; ga[home] += as_
        gf[away] += as_; ga[away] += hs
        if hs > as_:
            pts[home] += 3
            h2h_pts[home][away] += 3
        elif hs < as_:
            pts[away] += 3
            h2h_pts[away][home] += 3
        else:
            pts[home] += 1; pts[away] += 1
            h2h_pts[home][away] += 1
            h2h_pts[away][home] += 1
        h2h_gd[home][away] += hs - as_
        h2h_gd[away][home] += as_ - hs

    def sort_key(t):
        gd = gf[t] - ga[t]
        # critérios: pts, DG geral, gols marcados, DG h2h vs tied, aleatório
        return (pts[t], gd, gf[t], rng.random())

    standing = sorted(teams, key=sort_key, reverse=True)
    return [
        {"team": t, "pts": pts[t], "gf": gf[t], "ga": ga[t],
         "gd": gf[t] - ga[t], "pos": i + 1}
        for i, t in enumerate(standing)
    ]


# --------------------------------------------------------------------------- #
# Seleção dos 8 melhores terceiros
# --------------------------------------------------------------------------- #
def _best_third(thirds, rng):
    """Seleciona os 8 melhores terceiros colocados entre os 12 grupos."""
    thirds_sorted = sorted(
        thirds,
        key=lambda t: (t["pts"], t["gd"], t["gf"], rng.random()),
        reverse=True,
    )
    return [t["team"] for t in thirds_sorted[:8]]


# --------------------------------------------------------------------------- #
# Fase eliminatória
# --------------------------------------------------------------------------- #
def _sim_knockout_match(model, team_a, team_b, rng):
    """Mata-mata: empate vai para pênaltis (50/50 simplificado)."""
    hs, as_ = _sim_match(model, team_a, team_b, rng, neutral=True)
    if hs > as_:
        return team_a
    elif as_ > hs:
        return team_b
    else:
        # pênaltis: probabilidade relativa ao ELO
        elo_a = model.elo.get(team_a, 1500)
        elo_b = model.elo.get(team_b, 1500)
        p_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        return team_a if rng.random() < p_a else team_b


# --------------------------------------------------------------------------- #
# Monte Carlo principal
# --------------------------------------------------------------------------- #
def simulate_tournament(model, n_sims=10000, seed=42):
    """Simula a Copa 2026 completa n_sims vezes.

    Retorna:
        champion    — {team: prob de ser campeão}
        final       — {team: prob de chegar à final}
        semi        — {team: prob de chegar à semifinal}
        quarter     — {team: prob de chegar às quartas}
        knockout    — {team: prob de passar da fase de grupos}
        group_stage — {group_letter: list de standings médios}
    """
    rng = np.random.default_rng(seed)
    all_teams = [t for g in GROUPS.values() for t in g]

    counts = {
        "champion": {t: 0 for t in all_teams},
        "final":    {t: 0 for t in all_teams},
        "semi":     {t: 0 for t in all_teams},
        "quarter":  {t: 0 for t in all_teams},
        "knockout": {t: 0 for t in all_teams},
    }

    for _ in range(n_sims):
        # --- fase de grupos ---
        winners, runners_up, thirds = [], [], []
        for letter, teams in GROUPS.items():
            standing = _sim_group(model, teams, rng)
            winners.append(standing[0]["team"])
            runners_up.append(standing[1]["team"])
            thirds.append(standing[2])

        best8_thirds = _best_third(thirds, rng)
        field = winners + runners_up + best8_thirds  # 12+12+8 = 32

        for t in field:
            counts["knockout"][t] += 1

        # --- quartas de final em diante (bracket de 32) ---
        seeded = sorted(field, key=lambda t: model.elo.get(t, 1500), reverse=True)

        # R32 → R16
        r16 = []
        n = len(seeded)
        for i in range(n // 2):
            w = _sim_knockout_match(model, seeded[i], seeded[n - 1 - i], rng)
            r16.append(w)

        # R16 → QF
        qf = []
        for i in range(0, len(r16), 2):
            w = _sim_knockout_match(model, r16[i], r16[i + 1], rng)
            qf.append(w)
        for t in qf:
            counts["quarter"][t] += 1

        # QF → SF
        sf = []
        for i in range(0, len(qf), 2):
            w = _sim_knockout_match(model, qf[i], qf[i + 1], rng)
            sf.append(w)
        for t in sf:
            counts["semi"][t] += 1

        # SF → Final
        final = []
        for i in range(0, len(sf), 2):
            w = _sim_knockout_match(model, sf[i], sf[i + 1], rng)
            final.append(w)
        for t in final:
            counts["final"][t] += 1

        # Final → Campeão
        champion = _sim_knockout_match(model, final[0], final[1], rng)
        counts["champion"][champion] += 1

    def to_prob(d):
        return {t: round(v / n_sims, 4) for t, v in d.items() if v > 0}

    return {
        "n_sims": n_sims,
        "champion": dict(sorted(to_prob(counts["champion"]).items(), key=lambda x: -x[1])),
        "final":    dict(sorted(to_prob(counts["final"]).items(),    key=lambda x: -x[1])),
        "semi":     dict(sorted(to_prob(counts["semi"]).items(),     key=lambda x: -x[1])),
        "quarter":  dict(sorted(to_prob(counts["quarter"]).items(),  key=lambda x: -x[1])),
        "knockout": dict(sorted(to_prob(counts["knockout"]).items(), key=lambda x: -x[1])),
    }
