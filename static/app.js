"use strict";
/* =========================================================
   app.js — Copa 2026 Betting Model
   ========================================================= */

// ── XSS helper ──────────────────────────────────────────────
function esc(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ── Toast ────────────────────────────────────────────────────
function toast(msg, dur = 3000) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("visible");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("visible"), dur);
}

// ── Fetch helpers ────────────────────────────────────────────
async function api(path, body) {
  const opts = body != null
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : { method: "GET" };
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

// ── Formatters ───────────────────────────────────────────────
const pct = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");
const odds = (p) => (p > 0 ? (1 / p).toFixed(2) : "—");
const ev = (p, o) => (p && o ? ((p * o - 1) * 100).toFixed(1) + "%" : "—");
const evNum = (p, o) => (p && o ? p * o - 1 : null);

function evBadge(p, o) {
  const v = evNum(p, o);
  if (v == null) return "";
  const cls = v > 0.03 ? "ev-pos" : v > 0 ? "ev-slight" : "ev-neg";
  return `<span class="ev-badge ${cls}">${(v * 100).toFixed(1)}%</span>`;
}

// ── Tab navigation ───────────────────────────────────────────
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    const id = "tab-" + btn.dataset.tab;
    document.getElementById(id)?.classList.add("active");
    if (btn.dataset.tab === "hist") renderHistChart();
    if (btn.dataset.tab === "rank") loadRanking();
    if (btn.dataset.tab === "clv") loadClvTable();
  });
});

// ── Teams datalist ───────────────────────────────────────────
async function loadTeams() {
  try {
    const data = await api("/api/teams");
    const dl = document.getElementById("teams");
    dl.innerHTML = data.teams.map((t) => `<option value="${esc(t)}">`).join("");
    document.getElementById("model-info").textContent =
      `Modelo treinado em ${data.n_teams} seleções | CACHE v${data.cache_version || "?"}`;
  } catch { /* silencioso */ }
}

// ══════════════════════════════════════════════════════════════
//  ABA PREVISÃO
// ══════════════════════════════════════════════════════════════
function renderPrediction(d, container) {
  if (!d) { container.innerHTML = `<div class="card error">Erro ao calcular previsão.</div>`; return; }
  const ci = d.ci || null;

  const ciRow = (label, key) => {
    if (!ci || !ci[key]) return "";
    const { mean, p5, p95 } = ci[key];
    return `<tr><td class="dim">IC 90% ${esc(label)}</td>
      <td colspan="3"><span class="ci">${pct(p5)} – ${pct(p95)} (μ ${pct(mean)})</span></td></tr>`;
  };

  let html = `
  <div class="card result-card">
    <h3>${esc(d.home)} <span class="vs">vs</span> ${esc(d.away)}</h3>
    <table class="result-table">
      <thead><tr><th>Resultado</th><th>Probabilidade</th><th>Odd justa</th></tr></thead>
      <tbody>
        <tr class="highlight"><td>Vitória ${esc(d.home)}</td><td>${pct(d.p_H)}</td><td>${odds(d.p_H)}</td></tr>
        ${ciRow("1", "p_H")}
        <tr><td>Empate</td><td>${pct(d.p_D)}</td><td>${odds(d.p_D)}</td></tr>
        ${ciRow("X", "p_D")}
        <tr><td>Vitória ${esc(d.away)}</td><td>${pct(d.p_A)}</td><td>${odds(d.p_A)}</td></tr>
        ${ciRow("2", "p_A")}
      </tbody>
    </table>
    <p class="hint" style="margin-top:8px">Placar mais provável: <b>${esc(d.most_likely_score)}</b> (λ casa: ${(d.lambda_home||0).toFixed(2)}, λ fora: ${(d.lambda_away||0).toFixed(2)})</p>
    ${ci ? `<p class="hint">Bootstrap ${esc(d.n_bootstrap)} amostras (IC 90%).</p>` : ""}
  </div>`;

  if (d.stats) html += renderMatchStats(d.stats, d.home, d.away);
  container.innerHTML = html;
}

// ── Estatísticas esperadas ──────────────────────────────────────────────────
function renderMatchStats(s, home, away) {
  if (!s) return "";

  // barra de probabilidade
  const bar = (p) => {
    const w = Math.round(p * 100);
    const cls = p >= 0.6 ? "high" : p >= 0.35 ? "mid" : "low";
    return `<div class="prob-bar-wrap">
      <div class="prob-bar-track"><div class="prob-bar-fill ${cls}" style="width:${w}%"></div></div>
      <span class="prob-bar-pct">${(p * 100).toFixed(0)}%</span>
    </div>`;
  };

  const sr = (label, h, a, t) => `<tr>
    <td>${label}</td><td>${h}</td><td>${a}</td><td><b>${t}</b></td></tr>`;
  const br = (label, p) => `<tr>
    <td>${label}</td><td colspan="3">${bar(p)}</td></tr>`;

  const block = (emoji, title, rows) => `
  <div class="stat-block">
    <div class="stat-title">${emoji} ${title}</div>
    <table class="stat-table">
      <thead><tr><th></th><th>${esc(home)}</th><th>${esc(away)}</th><th>Total</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </div>`;

  const c = s.corners, sg = s.shots_on_target, ts = s.total_shots;
  const yc = s.yellow_cards, rc = s.red_cards, f = s.fouls, off = s.offsides;

  return `
  <div class="card result-card" style="margin-top:12px">
    <h3>📊 Estatísticas Esperadas</h3>

    <!-- Posse de bola -->
    <div class="poss-wrap">
      <span class="poss-label">${esc(home)} ${s.possession_home_pct}%</span>
      <div class="poss-track">
        <div class="poss-fill" style="width:${s.possession_home_pct}%"></div>
      </div>
      <span class="poss-label right">${s.possession_away_pct}% ${esc(away)}</span>
    </div>

    <div class="stats-grid">
      ${block("⛳", "Escanteios", `
        ${sr("Média esperada", c.home_avg, c.away_avg, c.total_avg)}
        ${br("Over 8.5", c.over_8_5)}
        ${br("Over 9.5", c.over_9_5)}
        ${br("Over 10.5", c.over_10_5)}
        ${br("Over 11.5", c.over_11_5)}`)}

      ${block("🎯", "Chutes no gol", `
        ${sr("Média esperada", sg.home_avg, sg.away_avg, sg.total_avg)}
        ${br("Over 6.5", sg.over_6_5)}
        ${br("Over 7.5", sg.over_7_5)}
        ${br("Over 8.5", sg.over_8_5)}`)}

      ${block("⚡", "Chutes totais", `
        ${sr("Média esperada", ts.home_avg, ts.away_avg, ts.total_avg)}
        ${br("Over 22.5", ts.over_22_5)}
        ${br("Over 24.5", ts.over_24_5)}`)}

      ${block("🟨", "Cartões amarelos", `
        ${sr("Média esperada", yc.home_avg, yc.away_avg, yc.total_avg)}
        ${br("Over 2.5", yc.over_2_5)}
        ${br("Over 3.5", yc.over_3_5)}
        ${br("Over 4.5", yc.over_4_5)}`)}

      ${block("🟥", "Cartão vermelho", `
        ${sr("Média esperada", rc.home_avg, rc.away_avg, rc.total_avg)}
        ${br("P(ao menos 1)", rc.prob_any)}`)}

      ${block("🚫", "Faltas", `
        ${sr("Média esperada", f.home_avg, f.away_avg, f.total_avg)}
        ${br("Over 24.5", f.over_24_5)}
        ${br("Over 27.5", f.over_27_5)}`)}

      ${block("🚩", "Impedimentos", `
        ${sr("Média esperada", off.home_avg, off.away_avg, off.total_avg)}`)}
    </div>
    <p class="hint" style="margin-top:10px">Modelos Poisson calibrados em jogos de Copa do Mundo (1966–2022). Derivados dos lambdas Dixon-Coles. Use odds de escanteios/cartões em Pinnacle ou bet365 e compare com as probabilidades acima.</p>
  </div>`;
}

document.getElementById("p-go").addEventListener("click", async () => {
  const home = document.getElementById("p-home").value.trim();
  const away = document.getElementById("p-away").value.trim();
  const neutral = document.getElementById("p-neutral").checked;
  const cont = document.getElementById("prev-result");
  if (!home || !away) { toast("Preencha os dois times."); return; }
  cont.innerHTML = `<div class="loading">Calculando…</div>`;
  try {
    const d = await api("/api/predict", { home, away, neutral });
    renderPrediction(d, cont);
  } catch (e) { cont.innerHTML = `<div class="card error">${esc(e.message)}</div>`; }
});

document.getElementById("p-ci").addEventListener("click", async () => {
  const home = document.getElementById("p-home").value.trim();
  const away = document.getElementById("p-away").value.trim();
  const neutral = document.getElementById("p-neutral").checked;
  const cont = document.getElementById("prev-result");
  if (!home || !away) { toast("Preencha os dois times."); return; }
  cont.innerHTML = `<div class="loading">Calculando intervalos de confiança (bootstrap)…</div>`;
  try {
    const d = await api("/api/predict_ci", { home, away, neutral });
    renderPrediction(d, cont);
  } catch (e) { cont.innerHTML = `<div class="card error">${esc(e.message)}</div>`; }
});

// Blend
document.getElementById("bl-go").addEventListener("click", async () => {
  const home = document.getElementById("p-home").value.trim();
  const away = document.getElementById("p-away").value.trim();
  const neutral = document.getElementById("p-neutral").checked;
  const oH = parseFloat(document.getElementById("bl-oh").value);
  const oD = parseFloat(document.getElementById("bl-od").value);
  const oA = parseFloat(document.getElementById("bl-oa").value);
  const alpha = parseFloat(document.getElementById("bl-alpha").value);
  const cont = document.getElementById("blend-result");
  if (!home || !away) { toast("Preencha os times na seção acima."); return; }
  if (!oH || !oD || !oA) { toast("Preencha as três odds do mercado."); return; }
  try {
    const base = await api("/api/predict", { home, away, neutral });
    const invH = 1 / oH, invD = 1 / oD, invA = 1 / oA;
    const total = invH + invD + invA;
    const mH = invH / total, mD = invD / total, mA = invA / total;
    const bH = alpha * base.p_H + (1 - alpha) * mH;
    const bD = alpha * base.p_D + (1 - alpha) * mD;
    const bA = alpha * base.p_A + (1 - alpha) * mA;
    cont.innerHTML = `
    <table class="result-table" style="margin-top:12px">
      <thead><tr><th>Resultado</th><th>Modelo</th><th>Mercado</th><th>Blend (α=${alpha})</th><th>EV</th></tr></thead>
      <tbody>
        <tr><td>1</td><td>${pct(base.p_H)}</td><td>${pct(mH)}</td><td><b>${pct(bH)}</b></td><td>${evBadge(bH, oH)}</td></tr>
        <tr><td>X</td><td>${pct(base.p_D)}</td><td>${pct(mD)}</td><td><b>${pct(bD)}</b></td><td>${evBadge(bD, oD)}</td></tr>
        <tr><td>2</td><td>${pct(base.p_A)}</td><td>${pct(mA)}</td><td><b>${pct(bA)}</b></td><td>${evBadge(bA, oA)}</td></tr>
      </tbody>
    </table>`;
  } catch (e) { cont.innerHTML = `<div class="card error">${esc(e.message)}</div>`; }
});

// ══════════════════════════════════════════════════════════════
//  ABA MERCADOS
// ══════════════════════════════════════════════════════════════
document.getElementById("m-go").addEventListener("click", async () => {
  const home = document.getElementById("m-home").value.trim();
  const away = document.getElementById("m-away").value.trim();
  const neutral = document.getElementById("m-neutral").checked;
  const cont = document.getElementById("mercados-result");
  if (!home || !away) { toast("Preencha os dois times."); return; }
  cont.innerHTML = `<div class="loading">Calculando mercados…</div>`;
  try {
    const d = await api("/api/markets", { home, away, neutral });
    renderMarkets(d, cont);
  } catch (e) { cont.innerHTML = `<div class="card error">${esc(e.message)}</div>`; }
});

function renderMarkets(d, cont) {
  if (!d?.markets?.length) { cont.innerHTML = `<div class="card error">Sem mercados.</div>`; return; }
  let html = `<div class="card"><h3>📋 Mercados — ${esc(d.home)} vs ${esc(d.away)}</h3>`;

  const groups = {};
  for (const m of d.markets) {
    const cat = m.category || "Outros";
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push(m);
  }

  for (const [cat, items] of Object.entries(groups)) {
    html += `<h4 class="market-cat">${esc(cat)}</h4>
    <table class="market-table">
      <thead><tr><th>Mercado</th><th>P(modelo)</th><th>Odd justa</th><th>Odd da casa</th><th>EV</th></tr></thead>
      <tbody>`;
    for (const m of items) {
      const id = `odd-${(cat + (m.label || "")).replace(/\W/g, "_")}`;
      html += `<tr>
        <td>${esc(m.label)}</td>
        <td>${pct(m.prob)}</td>
        <td>${odds(m.prob)}</td>
        <td><input class="odd-input" type="number" step="0.01" id="${esc(id)}"
            placeholder="${odds(m.prob)}"
            onchange="calcEV(this, ${m.prob || 0})" onkeyup="calcEV(this, ${m.prob || 0})"></td>
        <td id="ev-${esc(id)}">—</td>
      </tr>`;
    }
    html += `</tbody></table>`;
  }
  html += `</div>`;
  cont.innerHTML = html;
}

function calcEV(input, prob) {
  const o = parseFloat(input.value);
  const evId = "ev-" + input.id;
  const el = document.getElementById(evId);
  if (!el) return;
  if (!o || !prob) { el.textContent = "—"; return; }
  el.innerHTML = evBadge(prob, o);
}
window.calcEV = calcEV;

// ══════════════════════════════════════════════════════════════
//  ABA AO VIVO
// ══════════════════════════════════════════════════════════════
let _liveTimer = null;

function getApiKey() {
  return sessionStorage.getItem("api_football_key") || "";
}

document.getElementById("api-save").addEventListener("click", () => {
  const k = document.getElementById("api-key").value.trim();
  if (!k) { toast("Chave vazia."); return; }
  sessionStorage.setItem("api_football_key", k);
  document.getElementById("api-status").textContent = "Chave salva na sessão.";
  toast("Chave da API salva!");
});

(function initApiKey() {
  if (getApiKey()) document.getElementById("api-status").textContent = "Chave ativa na sessão.";
})();

document.getElementById("live-now").addEventListener("click", () => fetchLive("live"));
document.getElementById("live-wc").addEventListener("click", () => fetchLive("wc"));
document.getElementById("live-bydate").addEventListener("click", () => {
  const d = document.getElementById("live-date").value;
  if (!d) { toast("Selecione uma data."); return; }
  fetchLive("date", d);
});

document.getElementById("live-auto").addEventListener("change", function () {
  if (!this.checked) { clearInterval(_liveTimer); _liveTimer = null; return; }
  _liveTimer = setInterval(() => fetchLive("live"), 30000);
  fetchLive("live");
});

async function fetchLive(kind, dateStr) {
  const key = getApiKey();
  const onlyKnown = document.getElementById("live-known").checked;
  const meta = document.getElementById("live-meta");
  const empty = document.getElementById("live-empty");
  meta.textContent = "Buscando…";
  empty.style.display = "none";
  try {
    let endpoint = "/api/live";
    const params = new URLSearchParams({ api_key: key, only_known: onlyKnown ? "1" : "0" });
    if (kind === "wc") endpoint = "/api/live/world_cup";
    if (kind === "date") { endpoint = "/api/live/date"; params.set("date", dateStr); }
    const data = await fetch(`${endpoint}?${params}`).then((r) => r.json());
    if (data.error) { meta.textContent = data.error; return; }
    renderLive(data.matches || [], meta, empty);
  } catch (e) { meta.textContent = "Erro: " + esc(e.message); }
}

function renderLive(matches, meta, empty) {
  const tbody = document.querySelector("#live-table tbody");
  tbody.innerHTML = "";
  meta.textContent = `${matches.length} jogo(s) — ${new Date().toLocaleTimeString("pt-BR")}`;
  if (!matches.length) { empty.style.display = ""; empty.textContent = "Nenhum jogo encontrado."; return; }
  empty.style.display = "none";
  for (const m of matches) {
    const p = m.pred;
    const score = m.score_home != null ? `${m.score_home}–${m.score_away}` : "—";
    const likely = p?.most_likely_score || "—";
    const leitura = p ? `${pct(p.p_H)} / ${pct(p.p_D)} / ${pct(p.p_A)}` : "sem dados";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="status-chip ${m.is_live ? "live" : "done"}">${esc(m.status)}</span></td>
      <td>${esc(m.elapsed != null ? m.elapsed + "'" : (m.datetime || "").slice(11, 16))}</td>
      <td>${esc(m.league || "")} ${esc(m.country || "")}</td>
      <td><b>${esc(m.home)}</b> vs <b>${esc(m.away)}</b></td>
      <td>${esc(score)}</td>
      <td>${p ? pct(p.p_H) : "—"}</td>
      <td>${p ? pct(p.p_D) : "—"}</td>
      <td>${p ? pct(p.p_A) : "—"}</td>
      <td>${esc(likely)}</td>
      <td>${esc(leitura)}</td>`;
    tbody.appendChild(tr);
  }
}

// ══════════════════════════════════════════════════════════════
//  ABA VALUE BETS
// ══════════════════════════════════════════════════════════════
document.getElementById("v-go").addEventListener("click", async () => {
  const home = document.getElementById("v-home").value.trim();
  const away = document.getElementById("v-away").value.trim();
  const neutral = document.getElementById("v-neutral").checked;
  const oH = parseFloat(document.getElementById("v-oh").value);
  const oD = parseFloat(document.getElementById("v-od").value);
  const oA = parseFloat(document.getElementById("v-oa").value);
  const cont = document.getElementById("value-result");
  if (!home || !away) { toast("Preencha os dois times."); return; }
  cont.innerHTML = `<div class="loading">Analisando…</div>`;
  try {
    const d = await api("/api/predict", { home, away, neutral });
    const rows = [
      { side: "Casa (1)", p: d.p_H, o: oH },
      { side: "Empate (X)", p: d.p_D, o: oD },
      { side: "Fora (2)", p: d.p_A, o: oA },
    ];
    const hasValue = rows.some((r) => r.o && evNum(r.p, r.o) > 0);
    cont.innerHTML = `
    <div class="card">
      <h3>${esc(d.home)} vs ${esc(d.away)}</h3>
      ${hasValue ? '<div class="value-alert">✅ Apostas de valor detectadas!</div>' : '<div class="no-value">Nenhuma aposta de valor no limiar atual.</div>'}
      <table class="result-table">
        <thead><tr><th>Lado</th><th>P(modelo)</th><th>Odd justa</th><th>Odd da casa</th><th>EV</th></tr></thead>
        <tbody>
          ${rows.map((r) => `
          <tr class="${r.o && evNum(r.p, r.o) > 0.03 ? "value-row" : ""}">
            <td>${esc(r.side)}</td><td>${pct(r.p)}</td><td>${odds(r.p)}</td>
            <td>${r.o ? r.o.toFixed(2) : "—"}</td><td>${r.o ? evBadge(r.p, r.o) : "—"}</td>
          </tr>`).join("")}
        </tbody>
      </table>
    </div>`;
  } catch (e) { cont.innerHTML = `<div class="card error">${esc(e.message)}</div>`; }
});

// ══════════════════════════════════════════════════════════════
//  ABA ALERTAS — Value Scan
// ══════════════════════════════════════════════════════════════
let _alertRows = [];

function renderAlertRows() {
  const cont = document.getElementById("alertas-rows");
  if (!_alertRows.length) {
    cont.innerHTML = `<p class="hint">Clique em "+ Adicionar confronto" para começar.</p>`;
    return;
  }
  cont.innerHTML = _alertRows.map((r, i) => `
  <div class="alert-row">
    <input list="teams" class="al-home" value="${esc(r.home)}" placeholder="Mandante"
      onchange="_alertRows[${i}].home=this.value">
    <input list="teams" class="al-away" value="${esc(r.away)}" placeholder="Visitante"
      onchange="_alertRows[${i}].away=this.value">
    <input type="number" step="0.01" class="al-oh" value="${r.oH || ""}" placeholder="Odd 1"
      onchange="_alertRows[${i}].oH=+this.value">
    <input type="number" step="0.01" class="al-od" value="${r.oD || ""}" placeholder="Odd X"
      onchange="_alertRows[${i}].oD=+this.value">
    <input type="number" step="0.01" class="al-oa" value="${r.oA || ""}" placeholder="Odd 2"
      onchange="_alertRows[${i}].oA=+this.value">
    <button class="icon-btn" onclick="_alertRows.splice(${i},1);renderAlertRows()">✕</button>
  </div>`).join("");
}
window._alertRows = _alertRows;
window.renderAlertRows = renderAlertRows;

document.getElementById("al-add-row").addEventListener("click", () => {
  _alertRows.push({ home: "", away: "", oH: null, oD: null, oA: null });
  renderAlertRows();
});

document.getElementById("al-scan").addEventListener("click", async () => {
  const minEv = (parseFloat(document.getElementById("al-minev").value) || 3) / 100;
  const neutral = document.getElementById("al-neutral").checked;
  const cont = document.getElementById("alertas-result");
  const matchups = _alertRows
    .filter((r) => r.home && r.away)
    .map((r) => ({ home: r.home, away: r.away, neutral, odds: { H: r.oH, D: r.oD, A: r.oA } }));
  if (!matchups.length) { toast("Adicione pelo menos um confronto."); return; }
  cont.innerHTML = `<div class="loading">Varrendo ${matchups.length} jogos…</div>`;
  try {
    const d = await api("/api/value_scan", { matchups, min_ev: minEv });
    if (!d.alerts?.length) {
      cont.innerHTML = `<div class="card"><p>Nenhuma aposta de valor acima de ${(minEv * 100).toFixed(1)}%.</p></div>`;
      return;
    }
    let html = `<div class="card"><h3>🔔 ${d.alerts.length} alerta(s) de valor</h3>
    <table class="market-table">
      <thead><tr><th>Jogo</th><th>Lado</th><th>P(modelo)</th><th>Odd</th><th>EV</th></tr></thead><tbody>`;
    for (const a of d.alerts) {
      html += `<tr class="value-row">
        <td>${esc(a.home)} vs ${esc(a.away)}</td>
        <td>${esc(a.side_label)}</td>
        <td>${pct(a.prob)}</td>
        <td>${(a.odd || 0).toFixed(2)}</td>
        <td>${evBadge(a.prob, a.odd)}</td>
      </tr>`;
    }
    html += `</tbody></table></div>`;
    cont.innerHTML = html;
  } catch (e) { cont.innerHTML = `<div class="card error">${esc(e.message)}</div>`; }
});

renderAlertRows();

// ══════════════════════════════════════════════════════════════
//  ABA GESTÃO DE BANCA
// ══════════════════════════════════════════════════════════════
let BETS = [];

function saveBets() { sessionStorage.setItem("bets", JSON.stringify(BETS)); }
function loadBets() {
  try { BETS = JSON.parse(sessionStorage.getItem("bets") || "[]"); } catch { BETS = []; }
}
loadBets();

document.getElementById("b-add").addEventListener("click", async () => {
  const home = document.getElementById("b-home").value.trim();
  const away = document.getElementById("b-away").value.trim();
  const neutral = document.getElementById("b-neutral").checked;
  const side = document.getElementById("b-side").value;
  const odd = parseFloat(document.getElementById("b-odd").value);
  if (!home || !away || !odd) { toast("Preencha time, adversário e odd."); return; }
  try {
    const d = await api("/api/predict", { home, away, neutral });
    const pMap = { H: d.p_H, D: d.p_D, A: d.p_A };
    BETS.push({ home, away, side, odd, p: pMap[side], neutral, ts: Date.now() });
    saveBets();
    renderBets();
    toast("Aposta adicionada!");
  } catch (e) { toast("Erro: " + e.message); }
});

function renderBets() {
  const t = document.getElementById("bets-table");
  const empty = document.getElementById("bets-empty");
  const clearBtn = document.getElementById("b-clear");
  const regBtn = document.getElementById("b-to-hist");
  if (!BETS.length) {
    t.style.display = "none"; empty.style.display = ""; clearBtn.style.display = "none"; regBtn.style.display = "none";
    return;
  }
  t.style.display = ""; empty.style.display = "none"; clearBtn.style.display = ""; regBtn.style.display = "";
  const labels = { H: "Casa (1)", D: "Empate (X)", A: "Fora (2)" };
  t.querySelector("tbody").innerHTML = BETS.map((b, i) => `
  <tr>
    <td>${esc(b.home)} vs ${esc(b.away)} — ${esc(labels[b.side])}</td>
    <td>${pct(b.p)}</td>
    <td>${b.odd.toFixed(2)}</td>
    <td>${evBadge(b.p, b.odd)}</td>
    <td><button class="icon-btn" onclick="removeBet(${i})">✕</button></td>
  </tr>`).join("");
}

function removeBet(i) { BETS.splice(i, 1); saveBets(); renderBets(); }
window.removeBet = removeBet;

document.getElementById("b-clear").addEventListener("click", () => { BETS = []; saveBets(); renderBets(); });

document.getElementById("b-csv").addEventListener("change", async function () {
  const file = this.files[0]; if (!file) return;
  const text = await file.text();
  const rows = parseCSV(text);
  if (!rows.length) { toast("CSV vazio."); return; }
  const h = rows[0].map((c) => c.toLowerCase().trim());
  const idx = (k) => h.indexOf(k);
  let added = 0;
  for (const row of rows.slice(1)) {
    if (!row.length || !row[0]) continue;
    const get = (k) => (idx(k) >= 0 ? (row[idx(k)] || "").trim() : "");
    const home = get("home"), away = get("away");
    if (!home || !away) continue;
    if (idx("side") >= 0 && idx("odd") >= 0) {
      const side = (get("side") || "H").toUpperCase();
      const odd = parseFloat(get("odd"));
      if (!odd) continue;
      const neutral = get("neutral") !== "0";
      try {
        const d = await api("/api/predict", { home, away, neutral });
        const pMap = { H: d.p_H, D: d.p_D, A: d.p_A };
        BETS.push({ home, away, side, odd, p: pMap[side] || null, neutral, ts: Date.now() });
        added++;
      } catch { /* skip */ }
    } else {
      for (const [side, col] of [["H", "odd_h"], ["D", "odd_d"], ["A", "odd_a"]]) {
        const odd = parseFloat(get(col));
        if (!odd) continue;
        const neutral = get("neutral") !== "0";
        try {
          const d = await api("/api/predict", { home, away, neutral });
          const pMap = { H: d.p_H, D: d.p_D, A: d.p_A };
          BETS.push({ home, away, side, odd, p: pMap[side] || null, neutral, ts: Date.now() });
          added++;
        } catch { /* skip */ }
      }
    }
  }
  saveBets(); renderBets();
  toast(`${added} aposta(s) importada(s).`);
  this.value = "";
});

document.getElementById("b-csv-template").addEventListener("click", (e) => {
  e.preventDefault();
  const csv = "home,away,side,odd,neutral\nBrazil,Argentina,H,2.30,1\nFrance,Germany,D,3.50,1\n";
  const a = document.createElement("a");
  a.href = "data:text/csv;charset=utf-8," + encodeURIComponent(csv);
  a.download = "modelo_apostas.csv"; a.click();
});

document.getElementById("b-to-hist").addEventListener("click", () => {
  const labels = { H: "Casa (1)", D: "Empate (X)", A: "Fora (2)" };
  for (const b of BETS) {
    HISTORY.push({
      date: new Date().toLocaleDateString("pt-BR"),
      id: Date.now() + Math.random(),
      desc: `${b.home} vs ${b.away} — ${labels[b.side]}`,
      odd: b.odd, stake: null, p: b.p, status: "pending", result: null,
    });
  }
  saveHistory(); renderHistory();
  toast(`${BETS.length} aposta(s) enviada(s) ao histórico!`);
  document.querySelector('[data-tab="hist"]').click();
});

document.getElementById("bk-go").addEventListener("click", async () => {
  const bankroll = parseFloat(document.getElementById("bk-amount").value) || 100;
  const strategy = document.getElementById("bk-strategy").value;
  const kellyF = parseFloat(document.getElementById("bk-kelly").value) || 0.25;
  const flatPct = (parseFloat(document.getElementById("bk-flat").value) || 2) / 100;
  const minEdge = (parseFloat(document.getElementById("bk-minedge").value) || 0) / 100;
  const cont = document.getElementById("banca-result");

  const bets = BETS.filter((b) => b.p && b.odd);
  if (!bets.length) { toast("Adicione pelo menos uma aposta na lista."); return; }
  const validBets = bets.filter((b) => evNum(b.p, b.odd) >= minEdge);
  if (!validBets.length) { toast("Nenhuma aposta supera o EV mínimo configurado."); return; }

  const N = 10000;
  const finals = [];
  for (let i = 0; i < N; i++) {
    let bk = bankroll;
    for (const b of validBets) {
      const kelly = Math.max(0, (b.p * b.odd - 1) / (b.odd - 1));
      let stake = strategy === "kelly" ? bk * kelly * kellyF : bk * flatPct;
      stake = Math.min(stake, bk);
      bk += Math.random() < b.p ? stake * (b.odd - 1) : -stake;
      if (bk <= 0) { bk = 0; break; }
    }
    finals.push(bk);
  }
  finals.sort((a, b) => a - b);
  const mean = finals.reduce((s, v) => s + v, 0) / N;
  const p5 = finals[Math.floor(N * 0.05)];
  const p95 = finals[Math.floor(N * 0.95)];
  const pRuin = finals.filter((v) => v <= 0).length / N;
  const roi = ((mean - bankroll) / bankroll) * 100;

  cont.innerHTML = `
  <div class="card">
    <h3>Simulação Monte Carlo (${N.toLocaleString()} cenários)</h3>
    <div class="risk-grid">
      <div class="risk-item"><span class="risk-label">Banca média final</span><span class="risk-val">${mean.toFixed(2)}</span></div>
      <div class="risk-item"><span class="risk-label">ROI esperado</span><span class="risk-val ${roi >= 0 ? "pos" : "neg"}">${roi.toFixed(1)}%</span></div>
      <div class="risk-item"><span class="risk-label">IC 90%</span><span class="risk-val">${p5.toFixed(2)} – ${p95.toFixed(2)}</span></div>
      <div class="risk-item"><span class="risk-label">P(ruína)</span><span class="risk-val ${pRuin > 0.05 ? "neg" : "pos"}">${(pRuin * 100).toFixed(1)}%</span></div>
    </div>
    <p class="hint">${validBets.length} de ${bets.length} apostas passaram no filtro de EV ≥ ${(minEdge * 100).toFixed(1)}%.</p>
  </div>`;
});

document.getElementById("bk-strategy").addEventListener("change", function () {
  document.getElementById("bk-kelly-wrap").classList.toggle("hidden", this.value !== "kelly");
  document.getElementById("bk-flat-wrap").classList.toggle("hidden", this.value !== "flat");
});

renderBets();

// ══════════════════════════════════════════════════════════════
//  ABA HISTÓRICO
// ══════════════════════════════════════════════════════════════
let HISTORY = [];
let _histChart = null;

function saveHistory() { sessionStorage.setItem("history", JSON.stringify(HISTORY)); }
function loadHistory() {
  try { HISTORY = JSON.parse(sessionStorage.getItem("history") || "[]"); } catch { HISTORY = []; }
}
loadHistory();

document.getElementById("h-add").addEventListener("click", () => {
  const desc = document.getElementById("h-desc").value.trim();
  const odd = parseFloat(document.getElementById("h-odd").value);
  const stake = parseFloat(document.getElementById("h-stake").value);
  const p = parseFloat(document.getElementById("h-p").value) / 100 || null;
  if (!desc || !odd || !stake) { toast("Preencha descrição, odd e stake."); return; }
  HISTORY.push({
    date: new Date().toLocaleDateString("pt-BR"),
    id: Date.now() + Math.random(),
    desc, odd, stake, p, status: "pending", result: null,
  });
  saveHistory(); renderHistory();
  ["h-desc", "h-odd", "h-stake", "h-p"].forEach((id) => { document.getElementById(id).value = ""; });
  toast("Aposta registrada!");
});

function setStatus(id, status) {
  const b = HISTORY.find((h) => h.id === id);
  if (!b) return;
  b.status = status;
  b.result = status === "won" ? +(b.stake * (b.odd - 1)).toFixed(2)
           : status === "lost" ? -b.stake
           : null;
  saveHistory(); renderHistory(); renderHistChart();
}
window.setStatus = setStatus;

function removeHist(id) {
  const i = HISTORY.findIndex((h) => h.id === id);
  if (i >= 0) { HISTORY.splice(i, 1); saveHistory(); renderHistory(); renderHistChart(); }
}
window.removeHist = removeHist;

function renderHistory() {
  const t = document.getElementById("hist-table");
  const empty = document.getElementById("hist-empty");
  const summary = document.getElementById("hist-summary");
  if (!HISTORY.length) { t.style.display = "none"; empty.style.display = ""; summary.innerHTML = ""; return; }
  t.style.display = ""; empty.style.display = "none";

  const settled = HISTORY.filter((h) => h.status !== "pending");
  const totalStake = settled.reduce((s, h) => s + (h.stake || 0), 0);
  const totalResult = settled.reduce((s, h) => s + (h.result || 0), 0);
  const wins = settled.filter((h) => h.status === "won").length;
  const roi = totalStake ? ((totalResult / totalStake) * 100).toFixed(1) : "—";
  const start = parseFloat(document.getElementById("h-start").value) || 100;
  const bank = start + totalResult;

  summary.innerHTML = `
  <div class="risk-grid">
    <div class="risk-item"><span class="risk-label">Banca atual</span><span class="risk-val">${bank.toFixed(2)}</span></div>
    <div class="risk-item"><span class="risk-label">P/L total</span><span class="risk-val ${totalResult >= 0 ? "pos" : "neg"}">${totalResult >= 0 ? "+" : ""}${totalResult.toFixed(2)}</span></div>
    <div class="risk-item"><span class="risk-label">ROI</span><span class="risk-val ${parseFloat(roi) >= 0 ? "pos" : "neg"}">${roi}%</span></div>
    <div class="risk-item"><span class="risk-label">Vitórias</span><span class="risk-val">${wins}/${settled.length}</span></div>
  </div>`;

  t.querySelector("tbody").innerHTML = HISTORY.slice().reverse().map((h) => {
    const evStr = h.p && h.odd ? ev(h.p, h.odd) : "—";
    const opts = ["pending", "won", "lost", "void"]
      .map((s) => `<option value="${s}" ${h.status === s ? "selected" : ""}>${s}</option>`).join("");
    const res = h.result != null ? (h.result >= 0 ? "+" : "") + h.result.toFixed(2) : "—";
    return `<tr class="hist-row ${esc(h.status)}">
      <td>${esc(h.date)}</td>
      <td>${esc(h.desc)}</td>
      <td>${h.odd.toFixed(2)}</td>
      <td>${(h.stake || 0).toFixed(2)}</td>
      <td>${esc(evStr)}</td>
      <td><select onchange="setStatus(${h.id}, this.value)">${opts}</select></td>
      <td class="${h.result != null && h.result >= 0 ? "pos" : "neg"}">${esc(res)}</td>
      <td><button class="icon-btn" onclick="removeHist(${h.id})">✕</button></td>
    </tr>`;
  }).join("");
}

function renderHistChart() {
  const ctx = document.getElementById("histChart");
  if (!ctx) return;
  const settled = HISTORY.filter((h) => h.status !== "pending");
  const start = parseFloat(document.getElementById("h-start").value) || 100;
  let running = start;
  const labels = [], data = [start];
  for (const h of settled) {
    running += h.result || 0;
    labels.push(esc(h.desc.slice(0, 20)));
    data.push(parseFloat(running.toFixed(2)));
  }
  if (_histChart) _histChart.destroy();
  _histChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: ["Início", ...labels],
      datasets: [{ label: "Banca", data, borderColor: "#3b82f6", fill: true,
        backgroundColor: "rgba(59,130,246,0.1)", tension: 0.3, pointRadius: 3 }],
    },
    options: { responsive: true, plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: false } } },
  });
}

document.getElementById("h-export-csv").addEventListener("click", async () => {
  if (!HISTORY.length) { toast("Nenhuma aposta no histórico."); return; }
  try {
    const r = await fetch("/api/clv/export");
    if (r.ok) {
      const b = await r.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(b); a.download = "historico_clv.csv"; a.click();
      return;
    }
  } catch { /* fallback local */ }
  const cols = ["date", "desc", "odd", "stake", "status", "result"];
  const rows = [cols.join(",")].concat(HISTORY.map((h) => cols.map((c) => csvCell(h[c])).join(",")));
  const a = document.createElement("a");
  a.href = "data:text/csv;charset=utf-8," + encodeURIComponent(rows.join("\n"));
  a.download = "historico_apostas.csv"; a.click();
});

document.getElementById("h-reset").addEventListener("click", () => {
  if (!confirm("Zerar todo o histórico?")) return;
  HISTORY = []; saveHistory(); renderHistory(); renderHistChart(); toast("Histórico zerado.");
});

renderHistory();

// ══════════════════════════════════════════════════════════════
//  ABA CLV TRACKER
// ══════════════════════════════════════════════════════════════
let CLV = [];

function saveCLV() { sessionStorage.setItem("clv", JSON.stringify(CLV)); }
function loadCLVData() {
  try { CLV = JSON.parse(sessionStorage.getItem("clv") || "[]"); } catch { CLV = []; }
}
loadCLVData();

document.getElementById("clv-add").addEventListener("click", () => {
  const match = document.getElementById("clv-match").value.trim();
  const side = document.getElementById("clv-side").value;
  const entry = parseFloat(document.getElementById("clv-entry").value);
  const closing = parseFloat(document.getElementById("clv-closing").value) || null;
  const stake = parseFloat(document.getElementById("clv-stake").value) || null;
  const result = document.getElementById("clv-result").value;
  if (!match || !entry) { toast("Preencha jogo e odd de entrada."); return; }
  const clvPct = closing ? ((entry / closing - 1) * 100) : null;
  CLV.push({
    date: new Date().toLocaleDateString("pt-BR"),
    id: Date.now() + Math.random(),
    match, side, entry, closing, clvPct, stake, result,
  });
  saveCLV(); loadClvTable();
  ["clv-match", "clv-entry", "clv-closing", "clv-stake"].forEach((id) => { document.getElementById(id).value = ""; });
  document.getElementById("clv-result").value = "";
  toast("Aposta CLV registrada!");
});

document.getElementById("clv-refresh").addEventListener("click", loadClvTable);

function loadClvTable() {
  const panel = document.getElementById("clv-stats-panel");
  const card = document.getElementById("clv-table-card");
  if (!CLV.length) {
    panel.innerHTML = `<div class="card"><p class="hint">Nenhuma aposta registrada ainda.</p></div>`;
    card.style.display = "none"; return;
  }
  const withCLV = CLV.filter((c) => c.clvPct != null);
  const avgCLV = withCLV.length ? withCLV.reduce((s, c) => s + c.clvPct, 0) / withCLV.length : null;
  const posRate = withCLV.length ? withCLV.filter((c) => c.clvPct > 0).length / withCLV.length : null;
  const settled = CLV.filter((c) => c.result === "W" || c.result === "L");
  const wr = settled.length ? settled.filter((c) => c.result === "W").length / settled.length : null;

  panel.innerHTML = `
  <div class="card">
    <h3>📊 Estatísticas CLV</h3>
    <div class="risk-grid">
      <div class="risk-item"><span class="risk-label">CLV médio</span>
        <span class="risk-val ${avgCLV != null && avgCLV >= 0 ? "pos" : "neg"}">
          ${avgCLV != null ? (avgCLV >= 0 ? "+" : "") + avgCLV.toFixed(2) + "%" : "—"}</span></div>
      <div class="risk-item"><span class="risk-label">Taxa CLV positivo</span>
        <span class="risk-val ${posRate != null && posRate > 0.5 ? "pos" : "neg"}">
          ${posRate != null ? (posRate * 100).toFixed(1) + "%" : "—"}</span></div>
      <div class="risk-item"><span class="risk-label">Win rate</span>
        <span class="risk-val">${wr != null ? (wr * 100).toFixed(1) + "%" : "—"} (${settled.length} apostas)</span></div>
      <div class="risk-item"><span class="risk-label">Total registradas</span>
        <span class="risk-val">${CLV.length}</span></div>
    </div>
    <p class="hint">CLV positivo = você obteve melhor preço que o fechamento do mercado. Edge real se consistente.</p>
  </div>`;

  card.style.display = "";
  const sideLabels = { H: "Casa (1)", D: "Empate (X)", A: "Fora (2)" };
  document.querySelector("#clv-table tbody").innerHTML = CLV.slice().reverse().map((c) => `
  <tr>
    <td>${esc(c.date)}</td><td>${esc(c.match)}</td><td>${esc(sideLabels[c.side] || c.side)}</td>
    <td>${c.entry?.toFixed(2) || "—"}</td>
    <td>${c.closing?.toFixed(2) || "—"}</td>
    <td class="${c.clvPct != null && c.clvPct >= 0 ? "pos" : "neg"}">
      ${c.clvPct != null ? (c.clvPct >= 0 ? "+" : "") + c.clvPct.toFixed(2) + "%" : "—"}</td>
    <td>${c.stake?.toFixed(2) || "—"}</td>
    <td>${esc(c.result || "Pendente")}</td>
    <td><button class="icon-btn" onclick="removeCLV(${c.id})">✕</button></td>
  </tr>`).join("");
}

function removeCLV(id) {
  const i = CLV.findIndex((c) => c.id === id);
  if (i >= 0) { CLV.splice(i, 1); saveCLV(); loadClvTable(); }
}
window.removeCLV = removeCLV;

loadClvTable();

// ══════════════════════════════════════════════════════════════
//  ABA RANKING
// ══════════════════════════════════════════════════════════════
async function loadRanking() {
  const t = document.querySelector("#rank-table tbody");
  t.innerHTML = `<tr><td colspan="5">Carregando…</td></tr>`;
  try {
    const d = await api("/api/ranking");
    t.innerHTML = d.ranking.map((r, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${esc(r.team)}</td>
      <td>${r.elo?.toFixed(0) || "—"}</td>
      <td>${r.attack?.toFixed(3) || "—"}</td>
      <td>${r.defense?.toFixed(3) || "—"}</td>
    </tr>`).join("");
  } catch (e) { t.innerHTML = `<tr><td colspan="5">Erro: ${esc(e.message)}</td></tr>`; }
}

// ══════════════════════════════════════════════════════════════
//  ABA CALIBRAÇÃO — Backtesting
// ══════════════════════════════════════════════════════════════
let _btChart = null;

document.getElementById("bt-run").addEventListener("click", async () => {
  const from = document.getElementById("bt-from").value;
  const to = document.getElementById("bt-to").value;
  const sample = parseInt(document.getElementById("bt-sample").value) || 500;
  const cont = document.getElementById("bt-result");
  cont.innerHTML = `<div class="loading">Rodando backtesting (${sample} jogos)…</div>`;
  try {
    const params = new URLSearchParams({ from, to, sample });
    const d = await fetch(`/api/backtest?${params}`).then((r) => r.json());
    if (!d.ok) { cont.innerHTML = `<div class="card error">${esc(d.error || "Erro desconhecido")}</div>`; return; }
    renderBacktest(d, cont);
  } catch (e) { cont.innerHTML = `<div class="card error">${esc(e.message)}</div>`; }
});

function renderBacktest(d, cont) {
  cont.innerHTML = `
  <div class="card">
    <h3>Resultado do backtesting — ${d.n_games} jogos</h3>
    <div class="risk-grid">
      <div class="risk-item"><span class="risk-label">Brier Score</span><span class="risk-val">${(d.brier_score || 0).toFixed(4)}</span></div>
      <div class="risk-item"><span class="risk-label">Log-loss</span><span class="risk-val">${(d.log_loss || 0).toFixed(4)}</span></div>
      <div class="risk-item"><span class="risk-label">Acurácia 1X2</span><span class="risk-val">${pct(d.accuracy)}</span></div>
    </div>
    <p class="hint">${esc(d.interpretation || "")}</p>
    <div class="chart-wrap"><canvas id="calChart" height="90"></canvas></div>
  </div>`;

  const cal = d.calibration || [];
  const bins = cal.map((b) => (b.bin_center || 0).toFixed(2));
  const freqs = cal.map((b) => +(b.freq || 0).toFixed(3));
  const ctx = document.getElementById("calChart");
  if (_btChart) _btChart.destroy();
  _btChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: bins,
      datasets: [
        { label: "Freq. real", data: freqs, borderColor: "#3b82f6", tension: 0.3, pointRadius: 4 },
        { label: "Perfeito", data: bins.map((b) => parseFloat(b)), borderColor: "#10b981",
          borderDash: [5, 5], pointRadius: 0 },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "bottom" },
        title: { display: true, text: "Curva de calibração" },
      },
      scales: {
        x: { title: { display: true, text: "P prevista" } },
        y: { title: { display: true, text: "P real" }, min: 0, max: 1 },
      },
    },
  });
}

// ══════════════════════════════════════════════════════════════
//  ABA AGENTE IA
// ══════════════════════════════════════════════════════════════
let _agentHistory = [];

function getAgentKey() { return sessionStorage.getItem("anthropic_key") || ""; }

document.getElementById("ag-save-key").addEventListener("click", () => {
  const k = document.getElementById("ag-key").value.trim();
  if (!k) { toast("Chave vazia."); return; }
  sessionStorage.setItem("anthropic_key", k);
  toast("Chave Anthropic salva na sessão!");
});

document.getElementById("ag-reset").addEventListener("click", () => {
  _agentHistory = [];
  document.getElementById("agent-messages").innerHTML = `
  <div class="agent-msg system"><div class="msg-bubble">
    Conversa reiniciada. Como posso ajudar?</div></div>`;
});

document.getElementById("ag-send").addEventListener("click", sendAgent);
document.getElementById("ag-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendAgent(); }
});

function sendExample(btn) {
  document.getElementById("ag-input").value = btn.textContent;
  document.querySelector('[data-tab="agente"]').click();
  sendAgent();
}
window.sendExample = sendExample;

async function sendAgent() {
  const input = document.getElementById("ag-input");
  const msg = input.value.trim();
  if (!msg) return;
  const key = getAgentKey();
  if (!key) { toast("Salve sua chave Anthropic primeiro."); return; }

  input.value = "";
  appendAgentMsg("user", msg);
  _agentHistory.push({ role: "user", content: msg });

  const bubble = appendAgentMsg("assistant", "…");

  try {
    const r = await fetch("/api/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg, history: _agentHistory.slice(-20), api_key: key, stream: true }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);

    const ct = r.headers.get("content-type") || "";
    if (ct.includes("text/event-stream")) {
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "", full = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const chunk = JSON.parse(line.slice(6));
            if (chunk.delta) { full += chunk.delta; bubble.innerHTML = mdToHtml(esc(full)); }
            if (chunk.done) { _agentHistory.push({ role: "assistant", content: full }); }
          } catch { /* skip malformed */ }
        }
      }
    } else {
      const d = await r.json();
      const reply = d.response || d.error || "Sem resposta.";
      bubble.innerHTML = mdToHtml(esc(reply));
      _agentHistory.push({ role: "assistant", content: reply });
    }
  } catch (e) {
    bubble.innerHTML = `<span class="error">${esc(e.message)}</span>`;
  }
  scrollAgent();
}

function appendAgentMsg(role, text) {
  const msgs = document.getElementById("agent-messages");
  const div = document.createElement("div");
  div.className = `agent-msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.innerHTML = role === "user" ? esc(text) : mdToHtml(esc(text));
  div.appendChild(bubble);
  msgs.appendChild(div);
  scrollAgent();
  return bubble;
}

function scrollAgent() {
  const el = document.getElementById("agent-messages");
  el.scrollTop = el.scrollHeight;
}

function mdToHtml(s) {
  return s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>");
}

// ══════════════════════════════════════════════════════════════
//  UTILITÁRIO — CSV parser (RFC 4180)
// ══════════════════════════════════════════════════════════════
function parseCSV(text) {
  const rows = [];
  let row = [], field = "", inQuote = false;
  const t = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  for (let i = 0; i < t.length; i++) {
    const ch = t[i];
    if (inQuote) {
      if (ch === '"' && t[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') { inQuote = false; }
      else field += ch;
    } else {
      if (ch === '"') { inQuote = true; }
      else if (ch === ",") { row.push(field); field = ""; }
      else if (ch === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
      else field += ch;
    }
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows.filter((r) => r.some((c) => c.trim()));
}

function csvCell(v) {
  if (v == null) return "";
  const s = String(v);
  if (s.includes(",") || s.includes('"') || s.includes("\n")) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

// ══════════════════════════════════════════════════════════════
//  ABA MODELO ML — XGBoost + Rolling Features
// ══════════════════════════════════════════════════════════════
let _mlBtChart = null;

// Status ao entrar na aba
document.querySelector('[data-tab="ml"]')?.addEventListener("click", loadMlStatus);

async function loadMlStatus() {
  const panel = document.getElementById("ml-meta-panel");
  const status = document.getElementById("ml-train-status");
  try {
    const d = await api("/api/ml/status");
    if (d.trained && d.meta) {
      renderMlMeta(d.meta, panel);
      status.textContent = "✅ Modelo treinado e pronto.";
    } else {
      panel.innerHTML = "";
      status.textContent = d.message || "Modelo não treinado.";
    }
  } catch { status.textContent = "Erro ao verificar status."; }
}

function renderMlMeta(meta, container) {
  if (!meta) { container.innerHTML = ""; return; }
  container.innerHTML = `
  <div class="risk-grid" style="margin-top:14px">
    <div class="risk-item"><span class="risk-label">Amostras treino</span><span class="risk-val">${meta.n_samples || "—"}</span></div>
    <div class="risk-item"><span class="risk-label">Features</span><span class="risk-val">${meta.n_features || "—"}</span></div>
    <div class="risk-item"><span class="risk-label">Log-loss OOF</span><span class="risk-val">${meta.oof_log_loss != null ? meta.oof_log_loss.toFixed(4) : "—"}</span></div>
    <div class="risk-item"><span class="risk-label">Acurácia 1X2</span><span class="risk-val">${meta.accuracy_1x2 != null ? pct(meta.accuracy_1x2) : "—"}</span></div>
    <div class="risk-item"><span class="risk-label">Brier Casa</span><span class="risk-val">${meta.brier_H != null ? meta.brier_H.toFixed(4) : "—"}</span></div>
    <div class="risk-item"><span class="risk-label">Brier Empate</span><span class="risk-val">${meta.brier_D != null ? meta.brier_D.toFixed(4) : "—"}</span></div>
    <div class="risk-item"><span class="risk-label">Brier Fora</span><span class="risk-val">${meta.brier_A != null ? meta.brier_A.toFixed(4) : "—"}</span></div>
    <div class="risk-item"><span class="risk-label">Treinado em</span><span class="risk-val" style="font-size:.8rem">${(meta.trained_at || "").slice(0, 16).replace("T", " ")}</span></div>
  </div>
  <p class="hint" style="margin-top:8px">Validação por <b>TimeSeriesSplit</b> (${meta.n_splits || 5} folds). OOF = Out-of-Fold — sem data leakage.</p>`;
}

async function trainMl(force = false) {
  const status = document.getElementById("ml-train-status");
  const panel = document.getElementById("ml-meta-panel");
  status.textContent = "⏳ Treinando… (pode levar vários minutos na primeira vez)";
  panel.innerHTML = `<div class="loading">Construindo features rolling (3/5/10 jogos) e treinando XGBoost…</div>`;
  try {
    const d = await api("/api/ml/train", { force });
    if (d.ok) {
      renderMlMeta(d.meta, panel);
      status.textContent = "✅ Modelo treinado com sucesso!";
      toast("Modelo ML treinado!");
    } else {
      panel.innerHTML = `<div class="card error">${esc(d.error || "Erro desconhecido")}</div>`;
      status.textContent = "❌ Falha no treino.";
    }
  } catch (e) {
    panel.innerHTML = `<div class="card error">${esc(e.message)}</div>`;
    status.textContent = "❌ Erro na requisição.";
  }
}

document.getElementById("ml-train")?.addEventListener("click", () => trainMl(false));
document.getElementById("ml-train-force")?.addEventListener("click", () => trainMl(true));

// Previsão ML
document.getElementById("ml-predict")?.addEventListener("click", async () => {
  const home = document.getElementById("ml-home").value.trim();
  const away = document.getElementById("ml-away").value.trim();
  const neutral = document.getElementById("ml-neutral").checked;
  const oH = parseFloat(document.getElementById("ml-oh").value) || null;
  const oD = parseFloat(document.getElementById("ml-od").value) || null;
  const oA = parseFloat(document.getElementById("ml-oa").value) || null;
  const minEv = (parseFloat(document.getElementById("ml-minev").value) || 3) / 100;
  const cont = document.getElementById("ml-predict-result");
  if (!home || !away) { toast("Preencha os dois times."); return; }
  cont.innerHTML = `<div class="loading">Calculando (XGBoost calibrado)…</div>`;
  try {
    const body = { home, away, neutral, min_ev: minEv };
    if (oH) body.odd_H = oH;
    if (oD) body.odd_D = oD;
    if (oA) body.odd_A = oA;
    const d = await api("/api/ml/predict", body);
    if (d.error) { cont.innerHTML = `<div class="card error">${esc(d.error)}</div>`; return; }
    renderMlPredict(d, cont);
  } catch (e) { cont.innerHTML = `<div class="card error">${esc(e.message)}</div>`; }
});

function renderMlPredict(d, cont) {
  const hasValue = d.value_bets?.some((b) => b.is_value);
  let html = `
  <div class="card result-card">
    <h3>${esc(d.home)} <span class="vs">vs</span> ${esc(d.away)} <span class="hint" style="font-size:.75rem">(XGBoost calibrado)</span></h3>
    <table class="result-table">
      <thead><tr><th>Resultado</th><th>P(ML)</th><th>Odd justa</th></tr></thead>
      <tbody>
        <tr class="highlight"><td>Vitória ${esc(d.home)}</td><td>${pct(d.p_H)}</td><td>${d.odd_fair_H || "—"}</td></tr>
        <tr><td>Empate</td><td>${pct(d.p_D)}</td><td>${d.odd_fair_D || "—"}</td></tr>
        <tr><td>Vitória ${esc(d.away)}</td><td>${pct(d.p_A)}</td><td>${d.odd_fair_A || "—"}</td></tr>
      </tbody>
    </table>`;

  if (d.value_bets?.length) {
    html += `<h4 style="margin-top:14px">Análise de valor</h4>`;
    if (hasValue) html += `<div class="value-alert" style="margin-bottom:8px">✅ Aposta(s) de valor detectada(s)!</div>`;
    html += `<table class="market-table">
      <thead><tr><th>Lado</th><th>P(ML)</th><th>P(casa)</th><th>Odd</th><th>EV</th><th>Kelly ¼</th></tr></thead>
      <tbody>`;
    for (const b of d.value_bets) {
      html += `<tr class="${b.is_value ? "value-row" : ""}">
        <td>${esc(b.label)}</td>
        <td>${pct(b.p_model)}</td>
        <td>${pct(b.p_implied)}</td>
        <td>${b.odd.toFixed(2)}</td>
        <td>${evBadge(b.p_model, b.odd)}</td>
        <td>${b.is_value ? (b.kelly_quarter * 100).toFixed(1) + "%" : "—"}</td>
      </tr>`;
    }
    html += `</tbody></table>`;
  }
  html += `</div>`;
  cont.innerHTML = html;
}

// Stake type toggle
document.getElementById("ml-stake-type")?.addEventListener("change", function () {
  document.getElementById("ml-flat-wrap")?.classList.toggle("hidden", this.value !== "flat");
});

// Backtesting financeiro ML
document.getElementById("ml-bt-run")?.addEventListener("click", async () => {
  const bankroll = parseFloat(document.getElementById("ml-bk").value) || 100;
  const stakeType = document.getElementById("ml-stake-type").value;
  const kellyF = parseFloat(document.getElementById("ml-kelly-f").value) || 0.25;
  const flatPct = (parseFloat(document.getElementById("ml-flat").value) || 2) / 100;
  const minEv = (parseFloat(document.getElementById("ml-bt-ev").value) || 3) / 100;
  const nSplits = parseInt(document.getElementById("ml-splits").value) || 5;
  const cont = document.getElementById("ml-bt-result");
  cont.innerHTML = `<div class="loading">Rodando backtesting temporal (${nSplits} folds, pode demorar)…</div>`;
  try {
    const params = new URLSearchParams({
      bankroll, kelly_fraction: kellyF, min_ev: minEv, n_splits: nSplits,
      ...(stakeType === "flat" ? { flat_stake_pct: flatPct } : {}),
    });
    const d = await fetch(`/api/ml/backtest?${params}`).then((r) => r.json());
    if (!d.ok) { cont.innerHTML = `<div class="card error">${esc(d.error || "Erro")}</div>`; return; }
    renderMlBacktest(d, cont);
  } catch (e) { cont.innerHTML = `<div class="card error">${esc(e.message)}</div>`; }
});

function renderMlBacktest(d, cont) {
  const yieldCls = d.yield_pct >= 0 ? "pos" : "neg";
  cont.innerHTML = `
  <div class="card">
    <h3>Resultado do Backtesting ML</h3>
    <div class="risk-grid">
      <div class="risk-item"><span class="risk-label">Apostas realizadas</span><span class="risk-val">${d.n_bets}</span></div>
      <div class="risk-item"><span class="risk-label">Win rate</span><span class="risk-val">${pct(d.win_rate)}</span></div>
      <div class="risk-item"><span class="risk-label">Yield</span><span class="risk-val ${yieldCls}">${d.yield_pct >= 0 ? "+" : ""}${d.yield_pct.toFixed(2)}%</span></div>
      <div class="risk-item"><span class="risk-label">ROI banca</span><span class="risk-val ${d.roi_pct >= 0 ? "pos" : "neg"}">${d.roi_pct >= 0 ? "+" : ""}${d.roi_pct.toFixed(2)}%</span></div>
      <div class="risk-item"><span class="risk-label">Banca final</span><span class="risk-val">${d.bankroll_final.toFixed(2)}</span></div>
      <div class="risk-item"><span class="risk-label">Drawdown máx.</span><span class="risk-val ${d.max_drawdown_pct > 30 ? "neg" : ""}">${d.max_drawdown_pct.toFixed(1)}%</span></div>
      <div class="risk-item"><span class="risk-label">EV médio</span><span class="risk-val">${(d.avg_ev * 100).toFixed(2)}%</span></div>
      <div class="risk-item"><span class="risk-label">Odd média</span><span class="risk-val">${d.avg_odd.toFixed(2)}</span></div>
    </div>
    <p class="hint">${esc(d.interpretation)}</p>
    <div class="chart-wrap"><canvas id="mlBkChart" height="90"></canvas></div>
  </div>`;

  const curve = d.bankroll_curve || [];
  const labels = curve.map((r) => r.date || "");
  const values = curve.map((r) => r.bankroll);
  const ctx = document.getElementById("mlBkChart");
  if (_mlBtChart) _mlBtChart.destroy();
  _mlBtChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Evolução da banca",
        data: values,
        borderColor: d.yield_pct >= 0 ? "#22c55e" : "#ef4444",
        fill: true,
        backgroundColor: d.yield_pct >= 0 ? "rgba(34,197,94,.1)" : "rgba(239,68,68,.1)",
        tension: 0.3,
        pointRadius: values.length > 100 ? 0 : 2,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: false } },
    },
  });
}

// ══════════════════════════════════════════════════════════════
//  ABA COPA 2026 — openfootball (sem API key)
// ══════════════════════════════════════════════════════════════
document.getElementById("copa-groups-btn")?.addEventListener("click", loadCopaGroups);
document.getElementById("copa-fixtures-btn")?.addEventListener("click", loadCopaFixtures);
document.querySelector('[data-tab="copa"]')?.addEventListener("click", () => {
  if (!document.getElementById("copa-groups-result").innerHTML) loadCopaGroups();
});

async function loadCopaGroups() {
  const cont = document.getElementById("copa-groups-result");
  cont.innerHTML = `<div class="loading">Carregando grupos…</div>`;
  try {
    const d = await api("/api/wc2026/groups");
    if (d.error) { cont.innerHTML = `<div class="card error">${esc(d.error)}</div>`; return; }
    const groups = d.groups || [];
    if (!groups.length) { cont.innerHTML = `<div class="card"><p class="hint">Nenhum grupo encontrado.</p></div>`; return; }
    let html = `<div class="card" style="margin-top:12px"><h3>📋 12 Grupos — Copa 2026</h3><div class="copa-groups-grid">`;
    for (const g of groups) {
      html += `<div class="copa-group"><div class="copa-group-title">${esc(g.name || "Grupo")}</div><ul>`;
      for (const t of (g.teams || [])) {
        const pts = t.pts != null ? `<span class="copa-pts">${t.pts}pts</span>` : "";
        html += `<li><span>${esc(t.name || t)}</span>${pts}</li>`;
      }
      html += `</ul></div>`;
    }
    html += `</div></div>`;
    cont.innerHTML = html;
  } catch (e) { cont.innerHTML = `<div class="card error">${esc(e.message)}</div>`; }
}

async function loadCopaFixtures() {
  const cont = document.getElementById("copa-fixtures-result");
  cont.innerHTML = `<div class="loading">Carregando fixtures…</div>`;
  try {
    const d = await api("/api/wc2026/fixtures");
    if (d.error) { cont.innerHTML = `<div class="card error">${esc(d.error)}</div>`; return; }
    const rounds = d.rounds || [];
    if (!rounds.length) { cont.innerHTML = `<div class="card"><p class="hint">Nenhuma fixture encontrada.</p></div>`; return; }
    let html = `<div class="card" style="margin-top:12px"><h3>📅 Fixtures — Copa 2026</h3>`;
    for (const round of rounds) {
      html += `<h4 class="market-cat" style="margin-top:14px">${esc(round.name || "Rodada")}</h4><div class="copa-fixtures-list">`;
      for (const m of (round.matches || [])) {
        const t1 = m.team1?.name || m.team1 || "?";
        const t2 = m.team2?.name || m.team2 || "?";
        const hasScore = m.score1 != null && m.score2 != null;
        const score = hasScore
          ? `<span class="copa-match-score">${m.score1} – ${m.score2}</span>`
          : `<span class="copa-match-score pending">${(m.time || m.date || "–")}</span>`;
        html += `<div class="copa-match">
          <span class="copa-match-date">${esc(m.date || "")}</span>
          <span class="copa-match-teams">${esc(t1)} <span class="vs">vs</span> ${esc(t2)}</span>
          ${score}
        </div>`;
      }
      html += `</div>`;
    }
    html += `</div>`;
    cont.innerHTML = html;
  } catch (e) { cont.innerHTML = `<div class="card error">${esc(e.message)}</div>`; }
}

// ── Boot ─────────────────────────────────────────────────────
loadTeams();
