"""
Rastreador de Closing Line Value (CLV).

CLV é a diferença entre as odds que você apostou e as odds finais da casa
no fechamento do mercado. É o melhor indicador de edge real a longo prazo.

CLV% = (odd_entrada / odd_fechamento - 1) * 100
- Positivo: você apostou melhor que o mercado
- Negativo: você apostou pior que o mercado
"""
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clv_tracker.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                added_at    TEXT    NOT NULL,
                match       TEXT    NOT NULL,
                side        TEXT    NOT NULL,
                p_model     REAL,
                odds_entry  REAL    NOT NULL,
                odds_closing REAL,
                stake       REAL,
                result      TEXT,
                note        TEXT
            )
        """)


init_db()


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #
def add_bet(match, side, odds_entry, p_model=None, stake=None, note=""):
    """Registra uma nova aposta no momento de entrada.

    match       — ex: "Brasil x Marrocos"
    side        — "H", "D" ou "A"
    odds_entry  — odd decimal no momento da aposta
    p_model     — probabilidade do modelo (opcional, para referência)
    stake       — valor apostado (opcional)
    note        — texto livre
    """
    with _conn() as c:
        cur = c.execute("""
            INSERT INTO bets (added_at, match, side, p_model, odds_entry, stake, note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), match, side, p_model, odds_entry, stake, note))
        return cur.lastrowid


def update_bet(bet_id, odds_closing=None, result=None):
    """Atualiza odds de fechamento e/ou resultado de uma aposta.

    result — "W" (ganhou), "L" (perdeu), "V" (void/cancelado)
    """
    fields, vals = [], []
    if odds_closing is not None:
        fields.append("odds_closing = ?"); vals.append(float(odds_closing))
    if result is not None:
        fields.append("result = ?"); vals.append(str(result).upper())
    if not fields:
        return
    vals.append(bet_id)
    with _conn() as c:
        c.execute(f"UPDATE bets SET {', '.join(fields)} WHERE id = ?", vals)


def get_bets(limit=100):
    """Retorna as apostas mais recentes."""
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("""
            SELECT * FROM bets ORDER BY added_at DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def delete_bet(bet_id):
    with _conn() as c:
        c.execute("DELETE FROM bets WHERE id = ?", (bet_id,))


# --------------------------------------------------------------------------- #
# Estatísticas de CLV
# --------------------------------------------------------------------------- #
def get_clv_stats():
    """Calcula estatísticas de CLV sobre todas as apostas com fechamento registrado."""
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("""
            SELECT * FROM bets WHERE odds_closing IS NOT NULL
        """).fetchall()

    if not rows:
        return {"ok": False, "reason": "Nenhuma aposta com odds de fechamento registrada."}

    bets = [dict(r) for r in rows]
    clv_list = []
    roi_list = []
    n_win = n_loss = n_void = n_open = 0

    for b in bets:
        oe = b["odds_entry"]
        oc = b["odds_closing"]
        clv_pct = (oe / oc - 1.0) * 100.0
        b["clv_pct"] = round(clv_pct, 2)
        clv_list.append(clv_pct)

        result = b.get("result")
        stake = b.get("stake") or 1.0
        if result == "W":
            n_win += 1
            roi_list.append((oe - 1.0) * stake)
        elif result == "L":
            n_loss += 1
            roi_list.append(-stake)
        elif result == "V":
            n_void += 1
        else:
            n_open += 1

    n_settled = n_win + n_loss
    win_rate = n_win / n_settled if n_settled else None

    import statistics
    stats = {
        "ok": True,
        "n_total": len(bets),
        "n_with_closing": len(clv_list),
        "n_win": n_win, "n_loss": n_loss, "n_void": n_void, "n_open": n_open,
        "clv_medio_pct": round(statistics.mean(clv_list), 2),
        "clv_mediana_pct": round(statistics.median(clv_list), 2),
        "clv_positivo_pct": round(sum(1 for c in clv_list if c > 0) / len(clv_list) * 100, 1),
        "win_rate": round(win_rate * 100, 1) if win_rate is not None else None,
        "roi_total": round(sum(roi_list), 2) if roi_list else None,
        "bets": sorted(bets, key=lambda x: x["added_at"], reverse=True),
        "interpretation": _interpret(clv_list),
    }
    return stats


def _interpret(clv_list):
    if len(clv_list) < 20:
        return "Amostra pequena — registre mais apostas para conclusões confiáveis (mínimo 50)."
    mean_clv = sum(clv_list) / len(clv_list)
    positive_pct = sum(1 for c in clv_list if c > 0) / len(clv_list) * 100
    if mean_clv > 2 and positive_pct > 60:
        return f"Excelente: CLV médio +{mean_clv:.1f}% com {positive_pct:.0f}% de apostas positivas. Edge real consistente."
    elif mean_clv > 0:
        return f"Bom: CLV médio positivo (+{mean_clv:.1f}%). Continue rastreando para confirmar edge."
    else:
        return f"Atenção: CLV médio negativo ({mean_clv:.1f}%). Você está apostando pior que o mercado fecha."
