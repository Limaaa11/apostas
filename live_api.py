"""
Integração com a API-Football (api-sports.io) para jogos ao vivo e futuros.

A chave é do usuário e NUNCA fica no código: o front-end guarda no navegador e
manda no header de cada request; aqui ela só é repassada para a API-Football.

Tudo é normalizado para um formato único e cruzado com a previsão do modelo
Dixon-Coles (quando os dois times são seleções conhecidas pela base).
"""
import time
import difflib
import requests

import model_engine as me

API_BASE = "https://v3.football.api-sports.io"
WORLD_CUP_LEAGUE_ID = 1          # "World Cup" na API-Football
TIMEOUT = 20

# Cache de respostas da API-Football: evita consumir as 100 req/dia no plano free
_API_CACHE = {}   # key -> (timestamp, data)
_CACHE_TTL = {
    "live": 30,      # jogos ao vivo: 30s
    "upcoming": 300, # próximos jogos: 5min
    "date": 300,
}

# Status da API-Football -> rótulo + se está em andamento (ao vivo)
_STATUS = {
    "TBD": ("A definir", False, False),
    "NS": ("Agendado", False, False),
    "1H": ("1º tempo", True, False),
    "HT": ("Intervalo", True, False),
    "2H": ("2º tempo", True, False),
    "ET": ("Prorrogação", True, False),
    "BT": ("Intervalo (prorr.)", True, False),
    "P": ("Pênaltis", True, False),
    "SUSP": ("Suspenso", True, False),
    "INT": ("Interrompido", True, False),
    "LIVE": ("Ao vivo", True, False),
    "FT": ("Encerrado", False, True),
    "AET": ("Encerrado (prorr.)", False, True),
    "PEN": ("Encerrado (pênaltis)", False, True),
    "PST": ("Adiado", False, False),
    "CANC": ("Cancelado", False, False),
    "ABD": ("Abandonado", False, False),
    "AWD": ("W.O.", False, True),
    "WO": ("W.O.", False, True),
}

# Nome da API-Football -> nome na base martj42 (só os que diferem de fato).
_ALIASES = {
    "USA": "United States",
    "South Korea": "Korea Republic",
    "Korea Republic": "Korea Republic",
    "North Korea": "Korea DPR",
    "IR Iran": "Iran",
    "Czechia": "Czech Republic",
    "Turkey": "Turkey",
    "Türkiye": "Turkey",
    "China": "China PR",
    "China PR": "China PR",
    "Cape Verde Islands": "Cape Verde",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Congo": "Congo",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Republic of Ireland": "Republic of Ireland",
    "Ireland": "Republic of Ireland",
    "North Macedonia": "North Macedonia",
    "Macedonia": "North Macedonia",
    "Swaziland": "Eswatini",
    "Eswatini": "Eswatini",
    "Curacao": "Curaçao",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Kosovo": "Kosovo",
    "Brunei Darussalam": "Brunei",
    "Cabo Verde": "Cape Verde",
}


class TeamMatcher:
    """Casa nomes de times da API com a lista de times conhecida pelo modelo."""

    def __init__(self, model):
        self.teams = set(model.teams)
        self._lower = {t.lower(): t for t in self.teams}

    def match(self, api_name):
        if not api_name:
            return None
        if api_name in self.teams:
            return api_name
        alias = _ALIASES.get(api_name)
        if alias and alias in self.teams:
            return alias
        low = api_name.lower().strip()
        if low in self._lower:
            return self._lower[low]
        close = difflib.get_close_matches(api_name, self.teams, n=1, cutoff=0.88)
        return close[0] if close else None


def _headers(api_key):
    return {"x-apisports-key": api_key}


def _request(path, api_key, params=None, cache_kind=None):
    """Chamada genérica à API-Football com cache e tratamento de erro amigável."""
    if not api_key:
        return None, "Faltou a chave da API-Football. Cole sua chave no campo acima."

    cache_key = (path, str(sorted((params or {}).items())), api_key[:8])
    ttl = _CACHE_TTL.get(cache_kind, 0)
    if ttl > 0 and cache_key in _API_CACHE:
        ts, cached = _API_CACHE[cache_key]
        if time.time() - ts < ttl:
            return cached, None

    try:
        r = requests.get(f"{API_BASE}/{path}", headers=_headers(api_key),
                         params=params or {}, timeout=TIMEOUT)
    except requests.RequestException as e:
        return None, f"Falha de conexão com a API-Football: {e}"

    if r.status_code in (401, 403):
        return None, "Chave inválida ou sem permissão (HTTP %d)." % r.status_code
    if r.status_code == 429:
        return None, "Limite de requisições atingido no plano grátis. Tente mais tarde."
    try:
        data = r.json()
    except ValueError:
        return None, f"Resposta inesperada da API (HTTP {r.status_code})."

    errors = data.get("errors")
    if errors:
        if isinstance(errors, dict) and errors:
            msg = "; ".join(f"{k}: {v}" for k, v in errors.items())
        elif isinstance(errors, list) and errors:
            msg = "; ".join(str(e) for e in errors)
        else:
            msg = None
        if msg:
            return None, f"API-Football: {msg}"
    result = data.get("response", [])
    if ttl > 0:
        _API_CACHE[cache_key] = (time.time(), result)
    return result, None


def _normalize(fixtures, matcher, model):
    """Converte fixtures crus da API para o formato do front + anexa previsão.

    Memoiza predict() por par (home, away) para não recalcular a matriz quando
    o mesmo confronto aparece mais de uma vez na lista (ex: Copa com replays).
    """
    pred_cache = {}
    out = []
    for fx in fixtures:
        fixture = fx.get("fixture", {})
        league = fx.get("league", {})
        teams = fx.get("teams", {})
        goals = fx.get("goals", {})
        status = fixture.get("status", {}) or {}
        short = status.get("short", "NS")
        label, is_live, is_done = _STATUS.get(short, (short, False, False))

        home_api = (teams.get("home") or {}).get("name")
        away_api = (teams.get("away") or {}).get("name")
        home_m = matcher.match(home_api)
        away_m = matcher.match(away_api)

        pred = None
        if home_m and away_m:
            key = (home_m, away_m)
            if key not in pred_cache:
                pred_cache[key] = model.predict(home_m, away_m, neutral=True)
            pred = pred_cache[key]

        out.append({
            "id": fixture.get("id"),
            "datetime": fixture.get("date"),
            "timestamp": fixture.get("timestamp"),
            "elapsed": status.get("elapsed"),
            "status_short": short,
            "status": label,
            "is_live": is_live,
            "is_done": is_done,
            "league": league.get("name"),
            "country": league.get("country"),
            "round": league.get("round"),
            "home": home_api,
            "away": away_api,
            "home_model": home_m,
            "away_model": away_m,
            "score_home": goals.get("home"),
            "score_away": goals.get("away"),
            "matched": bool(pred),
            "pred": pred,
        })
    # Ordena: ao vivo primeiro, depois por horário.
    out.sort(key=lambda m: (not m["is_live"], m["timestamp"] or 0))
    return out


def live_matches(api_key, only_known=False):
    raw, err = _request("fixtures", api_key, {"live": "all"}, cache_kind="live")
    if err:
        return None, err
    model = me.get_model()
    matcher = TeamMatcher(model)
    games = _normalize(raw, matcher, model)
    if only_known:
        games = [g for g in games if g["matched"]]
    return games, None


def upcoming_world_cup(api_key, season=2026):
    raw, err = _request("fixtures", api_key,
                        {"league": WORLD_CUP_LEAGUE_ID, "season": season},
                        cache_kind="upcoming")
    if err:
        return None, err
    model = me.get_model()
    matcher = TeamMatcher(model)
    games = _normalize(raw, matcher, model)
    games = [g for g in games if not g["is_done"]]
    return games, None


def matches_by_date(api_key, date_str, only_known=False):
    raw, err = _request("fixtures", api_key, {"date": date_str}, cache_kind="date")
    if err:
        return None, err
    model = me.get_model()
    matcher = TeamMatcher(model)
    games = _normalize(raw, matcher, model)
    if only_known:
        games = [g for g in games if g["matched"]]
    return games, None
