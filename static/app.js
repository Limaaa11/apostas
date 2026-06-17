const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);
const pct = (x) => (x * 100).toFixed(1) + "%";
const money = (x) => x.toFixed(2);

async function api(path, body) {
  const opt = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opt);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || "Erro no servidor");
  return data;
}

function toast(msg, isErr) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast show" + (isErr ? " err" : "");
  setTimeout(() => (t.className = "toast"), 3200);
}

// ---------- Tabs ----------
$$(".tab").forEach((tab) => {
  tab.onclick = () => {
    $$(".tab").forEach((t) => t.classList.remove("active"));
    $$(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $("#tab-" + tab.dataset.tab).classList.add("active");
    if (tab.dataset.tab === "rank") loadRanking();
    if (tab.dataset.tab === "hist") renderLedger();
    if (tab.dataset.tab === "live") refreshKeyStatus();
  };
});

// ---------- Init ----------
async function init() {
  try {
    const info = await api("/api/model_info");
    $("#model-info").innerHTML =
      `Treinado em <b>${info.n_train.toLocaleString("pt-BR")}</b> jogos · ` +
      `<b>${info.n_teams}</b> seleções<br>` +
      `vantagem mando <b>${info.home_adv.toFixed(2)}</b> · ρ <b>${info.rho.toFixed(2)}</b> · ` +
      `dados até <b>${info.ref_date}</b>`;
    const teams = await api("/api/teams");
    const dl = $("#teams");
    dl.innerHTML = teams.map((t) => `<option value="${t}">`).join("");
  } catch (e) {
    $("#model-info").textContent = "erro ao carregar modelo";
    toast(e.message, true);
  }
}

// ---------- Mercados ----------
$("#m-go").onclick = async () => {
  const home = $("#m-home").value.trim();
  const away = $("#m-away").value.trim();
  if (!home || !away) return toast("Preencha os dois times.", true);
  $("#mercados-result").innerHTML = '<div class="card"><span class="spinner"></span> calculando mercados…</div>';
  try {
    const d = await api("/api/markets", { home, away, neutral: $("#m-neutral").checked });
    renderMercados(d);
  } catch (e) {
    $("#mercados-result").innerHTML = "";
    toast(e.message, true);
  }
};

function renderMercados(d) {
  const blocks = d.markets.map((mkt) => {
    const noteHtml = mkt.note
      ? `<div class="mkt-note">⚙️ ${mkt.note}</div>`
      : "";

    const rows = mkt.selections.map((s, idx) => {
      const inputId = `odd-${mkt.market.replace(/[^\w]/g, "_")}-${idx}`;
      return `<tr class="mkt-row" data-p="${s.p}" data-input="${inputId}">
        <td class="sel-label">${s.label}</td>
        <td class="pct-col"><b>${pct(s.p)}</b></td>
        <td class="odd-col"><span class="fair-odd">${s.odd_justa.toFixed(2)}</span></td>
        <td class="odd-col">
          <input class="odd-input" id="${inputId}" type="number" step="0.01" min="1.01"
            placeholder="—" onchange="calcEV(this)" oninput="calcEV(this)">
        </td>
        <td class="ev-col" id="ev-${inputId}">—</td>
        <td class="action-col" id="ac-${inputId}"></td>
      </tr>`;
    }).join("");

    return `<div class="mkt-block">
      <div class="mkt-header">${mkt.icon} <span>${mkt.market}</span></div>
      ${noteHtml}
      <table class="mkt-table">
        <thead><tr>
          <th>Seleção</th><th>P(modelo)</th><th>Odd justa</th><th>Odd da casa</th><th>EV</th><th></th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
  }).join("");

  const rates = d.corner_rate != null ? `
    <div class="mkt-rates">
      <span>🚩 Escanteios esperados: <b>${d.corner_rate}</b></span>
      <span>🟨 Cartões esperados: <b>${d.card_rate}</b></span>
      <span>👟 Chutes esperados: <b>${d.shot_rate}</b></span>
      <span>🎯 Chutes ao gol: <b>${d.sot_rate}</b></span>
      <span>🚫 Impedimentos: <b>${d.offside_rate}</b></span>
    </div>` : "";

  const disclaimer = d.disclaimer_jogador
    ? `<div class="mkt-disclaimer">⚠️ ${d.disclaimer_jogador}</div>` : "";

  const summary = `<div class="card mkt-summary">
    <div class="match-head">
      <span class="team">${d.home}</span>
      <span style="color:var(--muted);font-size:13px">
        xG ${d.lam_h.toFixed(2)} — ${d.lam_a.toFixed(2)}
      </span>
      <span class="team">${d.away}</span>
    </div>
    ${rates}
    <p class="hint" style="margin:8px 0 0">
      Cole as odds da casa na coluna <b>"Odd da casa"</b> para ver o EV instantaneamente.
      Linhas verdes = valor positivo. Mercados com ⚙️ usam modelo estatístico calibrado em dados de Copa.
    </p>
    ${disclaimer}
  </div>`;

  $("#mercados-result").innerHTML = summary + `<div class="mkt-grid">${blocks}</div>`;
}

window.calcEV = function(input) {
  const odd = parseFloat(input.value);
  const row = input.closest("tr");
  const p = parseFloat(row.dataset.p);
  const inputId = row.dataset.input;
  const evCell = document.getElementById("ev-" + inputId);
  const acCell = document.getElementById("ac-" + inputId);

  if (!odd || odd <= 1 || isNaN(p)) {
    evCell.textContent = "—";
    evCell.className = "ev-col";
    row.classList.remove("row-value", "row-novalue");
    acCell.innerHTML = "";
    return;
  }

  const ev = p * odd - 1;
  const evPct = (ev * 100).toFixed(1);
  evCell.textContent = (ev >= 0 ? "+" : "") + evPct + "%";
  evCell.className = "ev-col " + (ev > 0 ? "pos" : "neg");
  row.classList.toggle("row-value",  ev > 0);
  row.classList.toggle("row-novalue", ev <= 0);

  if (ev > 0) {
    const kelly = Math.max((odd * p - 1) / (odd - 1), 0);
    const k4 = (kelly / 4 * 100).toFixed(1);
    acCell.innerHTML = `<span class="badge good" title="1/4 Kelly sugerido">K¼ ${k4}%</span>`;
  } else {
    acCell.innerHTML = `<span class="badge bad">sem valor</span>`;
  }
};

// ---------- Previsão ----------
$("#p-go").onclick = async () => {
  const home = $("#p-home").value.trim();
  const away = $("#p-away").value.trim();
  if (!home || !away) return toast("Preencha os dois times.", true);
  $("#prev-result").innerHTML = '<div class="card"><span class="spinner"></span> calculando…</div>';
  try {
    const p = await api("/api/predict", { home, away, neutral: $("#p-neutral").checked });
    renderPrediction(p);
  } catch (e) {
    $("#prev-result").innerHTML = "";
    toast(e.message, true);
  }
};

function renderPrediction(p) {
  const scores = p.top_scores.map((s) => `<span class="score-chip">${s.score} <b>${pct(s.p)}</b></span>`).join("");
  $("#prev-result").innerHTML = `
    <div class="card">
      <div class="match-head">
        <span class="team">${p.home}</span>
        <span style="color:var(--muted)">${p.neutral ? "campo neutro" : "mandante: " + p.home}</span>
        <span class="team">${p.away}</span>
      </div>
      <div class="bar1x2">
        <div class="bar-h" style="width:${p.p_H * 100}%">${pct(p.p_H)}</div>
        <div class="bar-d" style="width:${p.p_D * 100}%">${pct(p.p_D)}</div>
        <div class="bar-a" style="width:${p.p_A * 100}%">${pct(p.p_A)}</div>
      </div>
      <div class="bar-legend"><span>🔵 ${p.home}</span><span>⚪ Empate</span><span>🟠 ${p.away}</span></div>
      <div class="grid">
        <div class="stat"><div class="k">Over 2.5 gols</div><div class="v">${pct(p.p_over25)}</div></div>
        <div class="stat"><div class="k">Under 2.5 gols</div><div class="v">${pct(p.p_under25)}</div></div>
        <div class="stat"><div class="k">Ambos marcam (sim)</div><div class="v">${pct(p.p_btts_yes)}</div></div>
        <div class="stat"><div class="k">Placar + provável</div><div class="v">${p.top_score}</div></div>
        <div class="stat"><div class="k">Gols esperados (λ)</div><div class="v">${p.lam_h.toFixed(2)} - ${p.lam_a.toFixed(2)}</div></div>
        <div class="stat"><div class="k">ELO</div><div class="v">${Math.round(p.elo_home)} - ${Math.round(p.elo_away)}</div></div>
      </div>
      <p class="hint" style="margin-top:16px">Placares mais prováveis:</p>
      <div class="scores">${scores}</div>
    </div>`;
}

// ---------- Value Bets ----------
$("#v-go").onclick = async () => {
  const home = $("#v-home").value.trim(), away = $("#v-away").value.trim();
  if (!home || !away) return toast("Preencha os dois times.", true);
  $("#value-result").innerHTML = '<div class="card"><span class="spinner"></span> analisando…</div>';
  try {
    const d = await api("/api/value", {
      home, away, neutral: $("#v-neutral").checked,
      odd_H: $("#v-oh").value, odd_D: $("#v-od").value, odd_A: $("#v-oa").value,
    });
    renderValue(d);
  } catch (e) {
    $("#value-result").innerHTML = "";
    toast(e.message, true);
  }
};

function renderValue(d) {
  if (!d.market.length) {
    $("#value-result").innerHTML = '<div class="card hint">Informe ao menos uma odd válida.</div>';
    return;
  }
  const rows = d.market.map((m) => {
    const good = m.ev > 0;
    return `<div class="value-row ${good ? "good" : ""}">
      <div>
        <div class="vlabel">${m.label} <span style="color:var(--muted)">@ ${m.odd.toFixed(2)}</span></div>
        <div class="vmeta">Modelo ${pct(m.p_model)} · Implícita ${pct(m.implied)} · Edge ${(m.edge * 100).toFixed(1)}pp · Kelly ¼ = ${pct(m.kelly_quarter)} da banca</div>
      </div>
      <span class="badge ${good ? "good" : "bad"}">EV ${m.ev >= 0 ? "+" : ""}${(m.ev * 100).toFixed(1)}%</span>
    </div>`;
  }).join("");
  const best = d.market[0];
  const tip = best.ev > 0
    ? `✅ Melhor valor: <b>${best.label}</b> (EV +${(best.ev * 100).toFixed(1)}%).`
    : "❌ Nenhum lado tem EV positivo — a casa está em vantagem em todos.";
  $("#value-result").innerHTML = `<div class="card">
    <div class="match-head"><span class="team">${d.pred.home}</span><span class="team">${d.pred.away}</span></div>
    ${rows}<p class="hint" style="margin-top:10px">${tip}</p></div>`;
}

// ---------- Gestão de Banca ----------
let BETS = [];

$("#b-add").onclick = async () => {
  const home = $("#b-home").value.trim(), away = $("#b-away").value.trim();
  const side = $("#b-side").value, odd = parseFloat($("#b-odd").value);
  if (!home || !away) return toast("Preencha os dois times.", true);
  if (!odd || odd <= 1) return toast("Informe uma odd válida (> 1).", true);
  try {
    const p = await api("/api/predict", { home, away, neutral: $("#b-neutral").checked });
    const pmap = { H: p.p_H, D: p.p_D, A: p.p_A };
    const lmap = { H: `${home} vence`, D: "Empate", A: `${away} vence` };
    const prob = pmap[side];
    BETS.push({ label: `${home} x ${away} — ${lmap[side]}`, p: prob, odd, ev: prob * odd - 1 });
    renderBets();
    $("#b-odd").value = "";
  } catch (e) {
    toast(e.message, true);
  }
};

function renderBets() {
  const tb = $("#bets-table tbody");
  $("#bets-empty").style.display = BETS.length ? "none" : "block";
  $("#bets-table").style.display = BETS.length ? "table" : "none";
  $("#b-clear").style.display = BETS.length ? "inline-block" : "none";
  $("#b-to-hist").style.display = BETS.length ? "inline-block" : "none";
  tb.innerHTML = BETS.map((b, i) => `<tr>
    <td>${b.label}</td>
    <td>${pct(b.p)}</td>
    <td>${b.odd.toFixed(2)}</td>
    <td class="${b.ev > 0 ? "pos" : "neg"}">${b.ev >= 0 ? "+" : ""}${(b.ev * 100).toFixed(1)}%</td>
    <td><button class="del-btn" onclick="delBet(${i})">✕</button></td>
  </tr>`).join("");
}
window.delBet = (i) => { BETS.splice(i, 1); renderBets(); };

$("#bk-strategy").onchange = (e) => {
  const kelly = e.target.value === "kelly";
  $("#bk-kelly-wrap").classList.toggle("hidden", !kelly);
  $("#bk-flat-wrap").classList.toggle("hidden", kelly);
};

let bankChart = null;
$("#bk-go").onclick = async () => {
  if (!BETS.length) return toast("Adicione apostas primeiro.", true);
  $("#banca-result").innerHTML = '<div class="card"><span class="spinner"></span> simulando milhares de cenários…</div>';
  try {
    const d = await api("/api/bankroll", {
      bets: BETS,
      bankroll: $("#bk-amount").value,
      strategy: $("#bk-strategy").value,
      kelly_mult: $("#bk-kelly").value,
      flat_pct: parseFloat($("#bk-flat").value) / 100,
      min_edge: parseFloat($("#bk-minedge").value) / 100,
      n_sims: 20000,
    });
    renderBanca(d);
  } catch (e) {
    $("#banca-result").innerHTML = "";
    toast(e.message, true);
  }
};

function renderBanca(d) {
  const mc = d.mc;
  if (!mc.ok) {
    $("#banca-result").innerHTML = `<div class="card hint">${mc.reason} (Ajuste o "EV mínimo" ou inclua apostas de valor.)</div>`;
    return;
  }
  const roiCls = mc.roi_mediano >= 0 ? "good" : "bad";
  const ruinCls = mc.prob_ruina > 0.1 ? "bad" : mc.prob_ruina > 0.02 ? "warn" : "good";
  const ddCls = mc.drawdown_p95 > 0.5 ? "bad" : mc.drawdown_p95 > 0.3 ? "warn" : "good";

  const compRows = d.compare.map((c) => `<tr>
    <td><b>${c.nome}</b></td>
    <td>${money(c.final_mediana)}</td>
    <td>${money(c.final_p5)}</td>
    <td>${money(c.final_p95)}</td>
    <td class="${c.prob_lucro >= 0.5 ? "pos" : ""}">${pct(c.prob_lucro)}</td>
    <td class="${c.prob_ruina > 0.05 ? "neg" : ""}">${pct(c.prob_ruina)}</td>
    <td>${pct(c.drawdown_p95)}</td>
  </tr>`).join("");

  $("#banca-result").innerHTML = `
    <div class="card">
      <h3>Resultado da simulação — ${mc.n_sims.toLocaleString("pt-BR")} cenários · ${mc.n_bets} apostas válidas</h3>
      <div class="risk-grid">
        <div class="risk"><div class="k">Banca final (mediana)</div><div class="v">${money(mc.final_mediana)}</div></div>
        <div class="risk"><div class="k">ROI mediano</div><div class="v ${roiCls}">${mc.roi_mediano >= 0 ? "+" : ""}${pct(mc.roi_mediano)}</div></div>
        <div class="risk"><div class="k">Prob. de lucro</div><div class="v ${mc.prob_lucro >= 0.5 ? "good" : "warn"}">${pct(mc.prob_lucro)}</div></div>
        <div class="risk"><div class="k">Prob. de dobrar</div><div class="v">${pct(mc.prob_dobrar)}</div></div>
        <div class="risk"><div class="k">Prob. de ruína</div><div class="v ${ruinCls}">${pct(mc.prob_ruina)}</div></div>
        <div class="risk"><div class="k">Drawdown típico (P95)</div><div class="v ${ddCls}">${pct(mc.drawdown_p95)}</div></div>
        <div class="risk"><div class="k">Cenário ruim (P5)</div><div class="v">${money(mc.final_p5)}</div></div>
        <div class="risk"><div class="k">Cenário bom (P95)</div><div class="v">${money(mc.final_p95)}</div></div>
      </div>
      <div class="chart-wrap"><canvas id="bankChart" height="110"></canvas></div>
    </div>
    <div class="card">
      <h3>Comparação de estratégias (mesmas apostas)</h3>
      <p class="hint">Kelly cheio maximiza crescimento mas com drawdowns brutais. Frações menores trocam retorno por estabilidade.</p>
      <table><thead><tr><th>Estratégia</th><th>Final med.</th><th>P5 (ruim)</th><th>P95 (bom)</th><th>Prob. lucro</th><th>Prob. ruína</th><th>DD P95</th></tr></thead>
      <tbody>${compRows}</tbody></table>
    </div>`;

  drawChart(mc.sample_paths, mc.bankroll_inicial);
}

function drawChart(paths, initial) {
  const ctx = $("#bankChart");
  if (bankChart) bankChart.destroy();
  const labels = paths[0].map((_, i) => i);
  const datasets = paths.slice(0, 25).map((p) => ({
    data: p, borderColor: "rgba(59,130,246,0.25)", borderWidth: 1,
    pointRadius: 0, tension: 0.1,
  }));
  datasets.push({
    data: labels.map(() => initial), borderColor: "#9aa7b4",
    borderWidth: 1.5, borderDash: [6, 4], pointRadius: 0,
  });
  bankChart = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true, animation: false,
      plugins: { legend: { display: false }, title: { display: true, text: "Trajetórias de banca (amostra de 25 cenários)", color: "#9aa7b4" } },
      scales: {
        x: { title: { display: true, text: "nº de apostas", color: "#9aa7b4" }, ticks: { color: "#9aa7b4" }, grid: { color: "#2a3140" } },
        y: { title: { display: true, text: "banca", color: "#9aa7b4" }, ticks: { color: "#9aa7b4" }, grid: { color: "#2a3140" } },
      },
    },
  });
}

// ---------- Ao Vivo (API-Football) ----------
const LIVE_KEY = "apifootball_key";
const getKey = () => (localStorage.getItem(LIVE_KEY) || "").trim();

function refreshKeyStatus() {
  const k = getKey();
  if ($("#api-key")) $("#api-key").value = k;
  const s = $("#api-status");
  if (s) {
    s.textContent = k ? "Chave salva ✓" : "Sem chave salva.";
    s.className = "hint inline" + (k ? " pos" : "");
  }
}

$("#api-save").onclick = () => {
  const k = $("#api-key").value.trim();
  localStorage.setItem(LIVE_KEY, k);
  refreshKeyStatus();
  toast(k ? "Chave salva no navegador." : "Chave removida.");
};

async function apiLive(path) {
  const r = await fetch(path, { headers: { "X-Api-Key": getKey() } });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || "Erro no servidor");
  return data;
}

let liveTimer = null;
let lastLiveFetch = null;

async function loadLive(kind) {
  if (!getKey()) return toast("Salve sua chave da API-Football primeiro.", true);
  const known = $("#live-known").checked ? 1 : 0;
  let path;
  if (kind === "live") path = `/api/live?only_known=${known}`;
  else if (kind === "wc") path = `/api/upcoming?mode=worldcup&season=2026`;
  else if (kind === "date") {
    const d = $("#live-date").value;
    if (!d) return toast("Escolha uma data.", true);
    path = `/api/upcoming?mode=date&date=${d}&only_known=${known}`;
  }
  lastLiveFetch = () => loadLive(kind);
  $("#live-empty").style.display = "none";
  $("#live-meta").textContent = "buscando…";
  try {
    const d = await apiLive(path);
    renderLive(d.games);
    $("#live-meta").textContent =
      `${d.count} jogo(s) · atualizado ${new Date().toLocaleTimeString("pt-BR")}`;
  } catch (e) {
    $("#live-meta").textContent = "";
    toast(e.message, true);
  }
}

$("#live-now").onclick = () => loadLive("live");
$("#live-wc").onclick = () => loadLive("wc");
$("#live-bydate").onclick = () => loadLive("date");

$("#live-auto").onchange = (e) => {
  if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
  if (e.target.checked) {
    liveTimer = setInterval(() => { if (lastLiveFetch) lastLiveFetch(); }, 30000);
    toast("Auto-atualização ligada (30s).");
  }
};

function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  const hh = d.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });
  return sameDay ? hh : `${d.toLocaleDateString("pt-BR", { day: "2-digit", month: "2-digit" })} ${hh}`;
}

function renderLive(games) {
  const tb = $("#live-table tbody");
  if (!games.length) {
    tb.innerHTML = "";
    $("#live-empty").style.display = "block";
    $("#live-empty").textContent = "Nenhum jogo encontrado para esse filtro.";
    return;
  }
  $("#live-empty").style.display = "none";
  tb.innerHTML = games.map((g) => {
    let stCls = "st-sched";
    if (g.is_live) stCls = "st-live";
    else if (g.is_done) stCls = "st-done";
    const stHtml = `<span class="st ${stCls}">${g.status}${g.is_live && g.elapsed ? " " + g.elapsed + "'" : ""}</span>`;
    const score = (g.score_home != null && g.score_away != null)
      ? `<b>${g.score_home} - ${g.score_away}</b>` : "—";
    const comp = `${g.country ? g.country + " · " : ""}${g.league || ""}`;
    const matchTxt = `${g.home || "?"} <span class="vs">x</span> ${g.away || "?"}`;
    let p1 = "—", px = "—", p2 = "—", top = "—", read = '<span class="muted">fora do modelo</span>';
    if (g.pred) {
      const pr = g.pred;
      p1 = pct(pr.p_H); px = pct(pr.p_D); p2 = pct(pr.p_A); top = pr.top_score;
      const opts = [[pr.p_H, g.home_model + " vence"], [pr.p_D, "Empate"], [pr.p_A, g.away_model + " vence"]];
      opts.sort((a, b) => b[0] - a[0]);
      read = `<b>${opts[0][1]}</b> ${pct(opts[0][0])}`;
    }
    return `<tr class="${g.is_live ? "row-live" : ""}">
      <td>${stHtml}</td>
      <td>${fmtTime(g.datetime)}</td>
      <td class="comp">${comp}</td>
      <td class="match">${matchTxt}</td>
      <td class="score">${score}</td>
      <td>${p1}</td><td>${px}</td><td>${p2}</td>
      <td>${top}</td>
      <td class="read">${read}</td>
    </tr>`;
  }).join("");
}

// ---------- Importar CSV (Gestão de Banca) ----------
function parseCSV(text) {
  const lines = text.replace(/\r/g, "").split("\n").filter((l) => l.trim());
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((h) => h.trim().toLowerCase());
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    const obj = {};
    headers.forEach((h, i) => (obj[h] = (cells[i] || "").trim()));
    return obj;
  });
}

function downloadFile(name, content) {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 100);
}

$("#b-csv-template").onclick = (e) => {
  e.preventDefault();
  downloadFile("modelo_apostas.csv",
    "home,away,side,odd,neutral\nBrazil,Argentina,1,2.30,true\nFrance,Germany,X,3.20,true\nSpain,Portugal,2,3.40,true\n");
  toast("Modelo CSV baixado.");
};

$("#b-csv").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const rows = parseCSV(await file.text());
    if (!rows.length) return toast("CSV vazio ou sem linhas de dados.", true);
    const d = await api("/api/batch_bets", { rows, neutral: $("#b-neutral").checked });
    if (d.bets.length) { BETS.push(...d.bets); renderBets(); }
    let msg = `${d.bets.length} aposta(s) importada(s).`;
    if (d.skipped.length) msg += ` ${d.skipped.length} ignorada(s).`;
    toast(msg, d.bets.length === 0);
    if (d.skipped.length) console.warn("Linhas ignoradas:", d.skipped);
  } catch (err) {
    toast(err.message, true);
  }
  e.target.value = "";
};

$("#b-clear").onclick = () => { BETS = []; renderBets(); };

$("#b-to-hist").onclick = () => {
  if (!BETS.length) return;
  const v = prompt("Valor a apostar em CADA uma (R$)?", "5");
  if (v === null) return;
  const stake = parseFloat(v);
  if (!stake || stake <= 0) return toast("Valor inválido.", true);
  BETS.forEach((b) => addLedger({ label: b.label, odd: b.odd, stake, p: b.p }));
  toast(`${BETS.length} aposta(s) enviada(s) ao histórico.`);
};

// ---------- Histórico / Ledger (localStorage) ----------
const LEDGER_KEY = "bet_ledger";
const LEDGER_START_KEY = "ledger_start";
let LEDGER = [];
try { LEDGER = JSON.parse(localStorage.getItem(LEDGER_KEY) || "[]"); } catch (e) { LEDGER = []; }
let histChart = null;

const saveLedger = () => localStorage.setItem(LEDGER_KEY, JSON.stringify(LEDGER));
const ledgerStart = () => parseFloat(localStorage.getItem(LEDGER_START_KEY)) || 100;

function addLedger(entry) {
  LEDGER.push({ id: Date.now() + Math.random(), date: new Date().toISOString(), status: "pending", ...entry });
  saveLedger();
  renderLedger();
}

window.setLedger = (id, status) => {
  const e = LEDGER.find((x) => String(x.id) === String(id));
  if (e) { e.status = e.status === status ? "pending" : status; saveLedger(); renderLedger(); }
};
window.delLedger = (id) => {
  LEDGER = LEDGER.filter((x) => String(x.id) !== String(id));
  saveLedger();
  renderLedger();
};

$("#h-add").onclick = () => {
  const label = $("#h-desc").value.trim();
  const odd = parseFloat($("#h-odd").value);
  const stake = parseFloat($("#h-stake").value);
  const pRaw = parseFloat($("#h-p").value);
  if (!label) return toast("Descreva a aposta.", true);
  if (!odd || odd <= 1) return toast("Odd inválida (> 1).", true);
  if (!stake || stake <= 0) return toast("Valor apostado inválido.", true);
  addLedger({ label, odd, stake, p: isNaN(pRaw) ? null : pRaw / 100 });
  $("#h-desc").value = $("#h-odd").value = $("#h-stake").value = $("#h-p").value = "";
};

$("#h-start").onchange = (e) => {
  localStorage.setItem(LEDGER_START_KEY, parseFloat(e.target.value) || 100);
  renderLedger();
};

$("#h-reset").onclick = () => {
  if (!LEDGER.length) return;
  if (!confirm("Apagar todo o histórico de apostas? Isso não pode ser desfeito.")) return;
  LEDGER = [];
  saveLedger();
  renderLedger();
  toast("Histórico zerado.");
};

function pnlOf(e) {
  if (e.status === "won") return e.stake * (e.odd - 1);
  if (e.status === "lost") return -e.stake;
  return 0;
}

function renderLedger() {
  if (!$("#h-start")) return;
  $("#h-start").value = ledgerStart();
  const tb = $("#hist-table tbody");
  $("#hist-empty").style.display = LEDGER.length ? "none" : "block";
  $("#hist-table").style.display = LEDGER.length ? "table" : "none";

  const resolved = LEDGER.filter((e) => e.status !== "pending");
  const won = resolved.filter((e) => e.status === "won");
  const stakedResolved = resolved.reduce((s, e) => s + e.stake, 0);
  const profit = resolved.reduce((s, e) => s + pnlOf(e), 0);
  const roi = stakedResolved > 0 ? profit / stakedResolved : 0;
  const hit = resolved.length ? won.length / resolved.length : 0;
  const bankNow = ledgerStart() + profit;

  const cls = (v) => (v > 0 ? "good" : v < 0 ? "bad" : "");
  $("#hist-summary").innerHTML = `
    <div class="risk"><div class="k">Banca atual</div><div class="v ${cls(profit)}">${money(bankNow)}</div></div>
    <div class="risk"><div class="k">Lucro / prejuízo</div><div class="v ${cls(profit)}">${profit >= 0 ? "+" : ""}${money(profit)}</div></div>
    <div class="risk"><div class="k">ROI realizado</div><div class="v ${cls(roi)}">${resolved.length ? (roi >= 0 ? "+" : "") + pct(roi) : "—"}</div></div>
    <div class="risk"><div class="k">Taxa de acerto</div><div class="v">${resolved.length ? pct(hit) : "—"}</div></div>
    <div class="risk"><div class="k">Apostas resolvidas</div><div class="v">${resolved.length} / ${LEDGER.length}</div></div>
    <div class="risk"><div class="k">Total apostado (resolv.)</div><div class="v">${money(stakedResolved)}</div></div>`;

  tb.innerHTML = LEDGER.slice().reverse().map((e) => {
    const pnl = pnlOf(e);
    const stTxt = e.status === "won" ? "Ganhou" : e.status === "lost" ? "Perdeu" : "Pendente";
    const stCls = e.status === "won" ? "st-done" : e.status === "lost" ? "st-live" : "st-sched";
    const ev = e.p != null ? ((e.p * e.odd - 1) * 100).toFixed(1) + "%" : "—";
    return `<tr>
      <td>${new Date(e.date).toLocaleDateString("pt-BR")}</td>
      <td>${e.label}</td>
      <td>${e.odd.toFixed(2)}</td>
      <td>${money(e.stake)}${e.status !== "pending" ? `<br><span class="${cls(pnl)}">${pnl >= 0 ? "+" : ""}${money(pnl)}</span>` : ""}</td>
      <td>${ev}</td>
      <td><span class="st ${stCls}">${stTxt}</span></td>
      <td class="ledger-actions">
        <button class="mini ${e.status === "won" ? "on-win" : ""}" onclick="setLedger('${e.id}','won')">✓ ganhou</button>
        <button class="mini ${e.status === "lost" ? "on-lose" : ""}" onclick="setLedger('${e.id}','lost')">✕ perdeu</button>
      </td>
      <td><button class="del-btn" onclick="delLedger('${e.id}')">🗑</button></td>
    </tr>`;
  }).join("");

  drawLedgerChart();
}

function drawLedgerChart() {
  const ctx = $("#histChart");
  if (!ctx) return;
  if (histChart) histChart.destroy();
  const resolved = LEDGER.filter((e) => e.status !== "pending");
  let bank = ledgerStart();
  const pts = [bank];
  resolved.forEach((e) => { bank += pnlOf(e); pts.push(bank); });
  histChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: pts.map((_, i) => i),
      datasets: [
        { data: pts, borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,0.12)", borderWidth: 2, pointRadius: 2, tension: 0.15, fill: true },
        { data: pts.map(() => ledgerStart()), borderColor: "#9aa7b4", borderWidth: 1.5, borderDash: [6, 4], pointRadius: 0 },
      ],
    },
    options: {
      responsive: true, animation: false,
      plugins: { legend: { display: false }, title: { display: true, text: "Evolução da banca (apostas resolvidas)", color: "#9aa7b4" } },
      scales: {
        x: { ticks: { color: "#9aa7b4" }, grid: { color: "#2a3140" } },
        y: { ticks: { color: "#9aa7b4" }, grid: { color: "#2a3140" } },
      },
    },
  });
}

// ---------- Ranking ----------
async function loadRanking() {
  const tb = $("#rank-table tbody");
  if (tb.children.length) return;
  try {
    const r = await api("/api/ranking?top=40");
    tb.innerHTML = r.map((t, i) => `<tr>
      <td>${i + 1}</td><td><b>${t.team}</b></td>
      <td>${Math.round(t.elo)}</td>
      <td>${t.attack.toFixed(2)}</td>
      <td>${t.defense.toFixed(2)}</td></tr>`).join("");
  } catch (e) { toast(e.message, true); }
}

refreshKeyStatus();
init();

// ---------- Agente IA ----------
const AGENT_SESSION = "copa2026_" + Math.random().toString(36).slice(2);

function agKey() {
  return localStorage.getItem("ag_key") || "";
}

$("#ag-save-key").onclick = () => {
  const k = $("#ag-key").value.trim();
  if (!k) return toast("Cole sua chave Anthropic.", true);
  localStorage.setItem("ag_key", k);
  $("#ag-key").value = "";
  toast("Chave salva no navegador.");
};

$("#ag-reset").onclick = async () => {
  await fetch("/api/agent/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: AGENT_SESSION }),
  });
  $("#agent-messages").innerHTML = `<div class="agent-msg system">
    <div class="msg-bubble">Conversa reiniciada. Como posso ajudar?</div>
  </div>`;
};

async function sendAgentMessage(text) {
  const key = agKey() || $("#ag-key").value.trim();
  if (!key) { toast("Salve sua chave Anthropic primeiro.", true); return; }

  appendAgentMsg("user", text);
  const thinking = appendAgentMsg("assistant", '<span class="spinner"></span> analisando…', true);

  try {
    const r = await fetch("/api/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: AGENT_SESSION, anthropic_key: key }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Erro no agente");
    thinking.remove();
    appendAgentMsg("assistant", formatAgentResponse(d.response), false, d.tools_used);
  } catch (e) {
    thinking.remove();
    appendAgentMsg("assistant", "❌ " + e.message);
    toast(e.message, true);
  }
}

function appendAgentMsg(role, html, temp = false, tools = []) {
  const wrap = document.createElement("div");
  wrap.className = `agent-msg ${role}${temp ? " temp" : ""}`;
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.innerHTML = html;
  wrap.appendChild(bubble);
  if (tools && tools.length) {
    const t = document.createElement("div");
    t.className = "msg-tools";
    t.textContent = "🔧 " + tools.join(" · ");
    wrap.appendChild(t);
  }
  $("#agent-messages").appendChild(wrap);
  wrap.scrollIntoView({ behavior: "smooth", block: "end" });
  return wrap;
}

function formatAgentResponse(text) {
  // Escapa HTML antes de processar markdown para evitar XSS
  const safe = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
  return safe
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/\*(.+?)\*/g, "<i>$1</i>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/^#{1,3} (.+)$/gm, "<b style='font-size:15px'>$1</b>")
    .replace(/^- (.+)$/gm, "• $1")
    .replace(/\n\n/g, "<br><br>")
    .replace(/\n/g, "<br>");
}

$("#ag-send").onclick = () => {
  const msg = $("#ag-input").value.trim();
  if (!msg) return;
  $("#ag-input").value = "";
  sendAgentMessage(msg);
};

$("#ag-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("#ag-send").click();
  }
});

window.sendExample = (btn) => {
  $("#ag-input").value = btn.textContent.trim();
  // Ativa a aba do agente sem depender de querySelector por texto
  const agenteTab = Array.from($$(".tab")).find((t) => t.dataset.tab === "agente");
  if (agenteTab && !agenteTab.classList.contains("active")) agenteTab.click();
  setTimeout(() => $("#ag-send").click(), 100);
};
