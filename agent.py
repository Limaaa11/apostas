"""
Agente de IA para o sistema de apostas Copa 2026.

Usa Claude com tool_use para responder perguntas em português,
chamando internamente as funções do modelo Dixon-Coles.
"""
import json
import os
import anthropic
import model_engine as me
import tournament as trn

CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

MODEL = "claude-haiku-4-5-20251001"   # rápido e barato para uso em chat

SYSTEM_PROMPT = """Você é um analista especialista em apostas esportivas para a Copa do Mundo 2026.
Você tem acesso a um modelo estatístico Dixon-Coles treinado com dados históricos de seleções.

Seu papel:
- Analisar jogos da Copa 2026 usando as ferramentas disponíveis
- Identificar apostas de valor (EV positivo) comparando probabilidades do modelo com odds do mercado
- Explicar os resultados de forma clara e objetiva em português
- Ser honesto sobre as limitações (o modelo não sabe de lesões ou escalações)
- Sugerir gestão de banca responsável (nunca apostar mais que 5% da banca)

Sempre use as ferramentas antes de responder. Não invente probabilidades — use os dados do modelo.
Quando detectar value bets, calcule o EV e sugira o stake com Kelly fracionário (1/4 Kelly).
"""

TOOLS = [
    {
        "name": "prever_jogo",
        "description": "Prevê o resultado de um jogo entre duas seleções. Retorna probabilidades 1X2, over/under 2.5, BTTS, gols esperados e top 5 placares.",
        "input_schema": {
            "type": "object",
            "properties": {
                "home": {"type": "string", "description": "Nome da seleção mandante (em inglês, ex: Brazil)"},
                "away": {"type": "string", "description": "Nome da seleção visitante (em inglês, ex: France)"},
                "neutral": {"type": "boolean", "description": "True se campo neutro (Copa do Mundo = sempre True)"},
            },
            "required": ["home", "away"],
        },
    },
    {
        "name": "ver_mercados",
        "description": "Retorna probabilidades do modelo para TODOS os mercados disponíveis: 1X2, dupla chance, BTTS, over/under por gols totais e por equipe, resultado exato. Use quando o usuário fornecer odds da casa para comparar EV.",
        "input_schema": {
            "type": "object",
            "properties": {
                "home": {"type": "string"},
                "away": {"type": "string"},
                "neutral": {"type": "boolean"},
                "odds_casa": {
                    "type": "object",
                    "description": "Odds opcionais da casa para calcular EV. Ex: {'1X2_H': 2.30, 'over_2.5': 1.85}",
                },
            },
            "required": ["home", "away"],
        },
    },
    {
        "name": "simular_torneio",
        "description": "Simula a Copa 2026 completa via Monte Carlo e retorna probabilidades de cada seleção ser campeã, chegar à final, semifinal, quartas e passar da fase de grupos.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n_sims": {"type": "integer", "description": "Número de simulações (padrão 5000, máx 20000)"},
                "focar_times": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista de times para destacar no resultado (opcional)",
                },
            },
        },
    },
    {
        "name": "calcular_banca",
        "description": "Calcula stakes recomendados e simula risco de uma lista de apostas usando Kelly fracionário e Monte Carlo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "apostas": {
                    "type": "array",
                    "description": "Lista de apostas com p (probabilidade do modelo) e odd (odd decimal da casa)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "p": {"type": "number"},
                            "odd": {"type": "number"},
                        },
                        "required": ["p", "odd"],
                    },
                },
                "banca": {"type": "number", "description": "Valor da banca atual"},
                "kelly_frac": {"type": "number", "description": "Fração de Kelly (0.25 = 1/4 Kelly, recomendado)"},
            },
            "required": ["apostas", "banca"],
        },
    },
    {
        "name": "ranking_selecoes",
        "description": "Retorna o ranking das seleções no modelo, ordenado por ELO. Mostra força de ataque e defesa Dixon-Coles.",
        "input_schema": {
            "type": "object",
            "properties": {
                "top": {"type": "integer", "description": "Quantas seleções mostrar (padrão 20)"},
                "filtrar_copa": {"type": "boolean", "description": "Se True, mostra só os 48 times da Copa 2026"},
            },
        },
    },
]


# --------------------------------------------------------------------------- #
# Execução das ferramentas
# --------------------------------------------------------------------------- #
WC_TEAMS = {
    "Algeria","Argentina","Australia","Austria","Belgium","Bosnia and Herzegovina",
    "Brazil","Canada","Cape Verde","Colombia","Croatia","Curaçao","Czech Republic",
    "DR Congo","Ecuador","Egypt","England","France","Germany","Ghana","Haiti","Iran",
    "Iraq","Ivory Coast","Japan","Jordan","Mexico","Morocco","Netherlands",
    "New Zealand","Norway","Panama","Paraguay","Portugal","Qatar","Saudi Arabia",
    "Scotland","Senegal","South Africa","South Korea","Spain","Sweden","Switzerland",
    "Tunisia","Turkey","United States","Uruguay","Uzbekistan",
}


def _run_tool(name, inputs):
    m = me.get_model()

    if name == "prever_jogo":
        home = inputs["home"]
        away = inputs["away"]
        neutral = inputs.get("neutral", True)
        p = m.predict(home, away, neutral)
        if p is None:
            return {"erro": f"Time '{home}' ou '{away}' não está no modelo."}
        return p

    if name == "ver_mercados":
        home = inputs["home"]
        away = inputs["away"]
        neutral = inputs.get("neutral", True)
        result = m.compute_markets(home, away, neutral)
        if result is None:
            return {"erro": f"Time '{home}' ou '{away}' não está no modelo."}
        # Se o usuário passou odds, calcular EV para cada seleção
        odds_casa = inputs.get("odds_casa", {})
        if odds_casa:
            for mkt in result["markets"]:
                for sel in mkt["selections"]:
                    key_variants = [
                        sel["label"], sel["label"].lower(),
                        sel["label"].replace(" ", "_").lower(),
                    ]
                    for k in key_variants:
                        if k in odds_casa:
                            o = float(odds_casa[k])
                            p = sel["p"]
                            sel["ev"] = round(p * o - 1, 4)
                            sel["odd_casa"] = o
                            break
        return result

    if name == "simular_torneio":
        n_sims = min(int(inputs.get("n_sims", 5000)), 20000)
        focar = inputs.get("focar_times", [])
        result = trn.simulate_tournament(m, n_sims=n_sims)
        if focar:
            for key in ["champion", "final", "semi", "quarter", "knockout"]:
                result[key] = {t: v for t, v in result[key].items() if t in focar or v >= 0.01}
        return result

    if name == "calcular_banca":
        apostas = inputs["apostas"]
        banca = float(inputs.get("banca", 100))
        kf = float(inputs.get("kelly_frac", 0.25))
        mc = me.monte_carlo_bankroll(
            apostas, bankroll=banca, strategy="kelly",
            kelly_mult=kf, n_sims=10000,
        )
        comp = me.compare_strategies(apostas, bankroll=banca, n_sims=5000)
        return {"mc": mc, "estrategias": comp}

    if name == "ranking_selecoes":
        top = int(inputs.get("top", 20))
        filtrar = inputs.get("filtrar_copa", False)
        rows = m.ranking(top=100)
        if filtrar:
            rows = [r for r in rows if r["team"] in WC_TEAMS]
        return {"ranking": rows[:top]}

    return {"erro": f"Ferramenta desconhecida: {name}"}


# --------------------------------------------------------------------------- #
# Loop principal do agente
# --------------------------------------------------------------------------- #
def run_agent(user_message, history=None, api_key=None):
    """Processa uma mensagem do usuário e retorna a resposta do agente.

    history: lista de mensagens anteriores [{"role": ..., "content": ...}]
    api_key: chave ANTHROPIC_API_KEY (usa env var se não fornecida)
    """
    client = CLIENT
    if api_key:
        client = anthropic.Anthropic(api_key=api_key)

    messages = list(history or [])
    messages.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Acumula texto da resposta
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)

        # Se não há chamadas de ferramenta, resposta final
        if response.stop_reason == "end_turn" or not tool_calls:
            final_text = "\n".join(text_parts).strip()
            messages.append({"role": "assistant", "content": response.content})
            return {
                "response": final_text,
                "history": messages,
                "tool_calls_made": [tc.name for tc in tool_calls],
            }

        # Executa ferramentas
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tc in tool_calls:
            result = _run_tool(tc.name, tc.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })
        messages.append({"role": "user", "content": tool_results})
