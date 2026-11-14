/* SpendSense frontend: plain JavaScript, no framework.
   Talks to the FastAPI backend with fetch(); keeps a session id in localStorage. */
(() => {
  "use strict";

  // ================================================================ helpers
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const C = { butter: "#FBE38E", cocoa: "#3A2218", blue: "#2B5BFF",
              cocoaA: (a) => `rgba(58,34,24,${a})`, blueA: (a) => `rgba(43,91,255,${a})` };

  function inr(x, dec = 0) {
    const n = Number(x) || 0;
    return (n < 0 ? "-₹" : "₹") + Math.abs(n).toLocaleString("en-IN", { minimumFractionDigits: dec, maximumFractionDigits: dec });
  }
  const shortInr = (n) => { const a = Math.abs(n); return a >= 1e5 ? `₹${(n / 1e5).toFixed(1)}L` : a >= 1e3 ? `₹${(n / 1e3).toFixed(0)}k` : inr(n); };
  const monthLabel = (m) => new Date(m + "-01T00:00:00").toLocaleDateString("en-IN", { month: "short", year: "2-digit" });
  const dateLabel = (d) => new Date(d + "T00:00:00").toLocaleDateString("en-IN", { day: "2-digit", month: "short" });
  const md = (text) => DOMPurify.sanitize(marked.parse(String(text ?? ""), { breaks: true }));

  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };

  const state = { sid: store.get("ss_session"), hasData: false, filename: null, ingest: null, charts: {},
                  tx: { q: "", category: "", month: "", type: "", sort: "date", desc: false, page: 1 },
                  insightsLoaded: false, hoodLoaded: false, busy: false };

  async function api(path, opts = {}) {
    const headers = { ...(opts.headers || {}) };
    if (state.sid) headers["X-Session-Id"] = state.sid;
    let body = opts.body;
    if (body && !(body instanceof FormData)) { headers["Content-Type"] = "application/json"; body = JSON.stringify(body); }
    const res = await fetch(path, { method: opts.method || "GET", headers, body });
    let data = null;
    try { data = await res.json(); } catch { /* not JSON */ }
    if (!res.ok) {
      const detail = data && data.detail;
      const msg = Array.isArray(detail) ? detail.map((d) => d.msg).join(", ") : detail || `Request failed (${res.status})`;
      const err = new Error(msg); err.status = res.status; throw err;
    }
    return data;
  }

  function toast(msg, icon = "i-info") {
    const t = document.createElement("div");
    t.className = "toast";
    t.innerHTML = `<svg><use href="#${icon}"/></svg><span>${esc(msg)}</span>`;
    $("#toasts").append(t);
    setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 400); }, 4200);
  }

  function countUp(el, value, fmt) {
    if (reduceMotion) { el.textContent = fmt(value); return; }
    const start = performance.now(), dur = 1100;
    const tick = (now) => {
      const p = Math.min(1, (now - start) / dur), e = 1 - Math.pow(1 - p, 4);
      el.textContent = fmt(value * e);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  /* Wrap every word of rendered HTML in a span with a staggered delay: a soft "typing" reveal. */
  function revealWords(root) {
    if (reduceMotion) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) if (walker.currentNode.nodeValue.trim()) nodes.push(walker.currentNode);
    let i = 0;
    const total = nodes.reduce((n, t) => n + t.nodeValue.split(/\s+/).length, 0);
    const step = Math.max(8, Math.min(28, 1400 / Math.max(total, 1)));
    for (const node of nodes) {
      const frag = document.createDocumentFragment();
      for (const part of node.nodeValue.split(/(\s+)/)) {
        if (!part) continue;
        if (/^\s+$/.test(part)) { frag.append(part); continue; }
        const s = document.createElement("span");
        s.className = "w"; s.style.setProperty("--d", `${Math.round(i++ * step)}ms`); s.textContent = part;
        frag.append(s);
      }
      node.replaceWith(frag);
    }
  }

  // ================================================================ navigation bubbles
  const TABS = ["upload", "dashboard", "chat", "transactions", "insights", "hood"];
  const NEEDS_DATA = new Set(["dashboard", "transactions", "insights"]);

  function moveIndicator(animate = true) {
    const active = $(".bubble.active"), ind = $(".bubble-indicator");
    if (!active || !ind) return;
    ind.style.width = `${active.offsetWidth}px`;
    ind.style.transform = `translateX(${active.offsetLeft}px)`;
    if (animate && !reduceMotion) { ind.classList.remove("squish"); void ind.offsetWidth; ind.classList.add("squish"); }
  }

  function setTab(name, { push = true } = {}) {
    if (!TABS.includes(name)) name = "upload";
    if (NEEDS_DATA.has(name) && !state.hasData) { if (push) toast("Load a statement first: upload one or try the sample.", "i-upload"); name = "upload"; }
    $$(".bubble").forEach((b) => { const on = b.dataset.tab === name; b.classList.toggle("active", on); b.setAttribute("aria-current", on ? "page" : "false"); });
    $$(".view").forEach((v) => {
      const on = v.dataset.view === name;
      if (on && v.hidden) { v.hidden = false; v.style.animation = "none"; void v.offsetWidth; v.style.animation = ""; }
      else if (!on) v.hidden = true;
    });
    moveIndicator();
    if (push && location.hash !== `#${name}`) history.pushState(null, "", `#${name}`);
    window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
    if (name === "dashboard") loadDashboard();
    if (name === "transactions") loadTransactions();
    if (name === "insights" && !state.insightsLoaded) loadInsights();
    if (name === "hood") loadHood();
    if (name === "chat") { renderSuggestions(); setTimeout(() => $("#chat-input").focus(), 200); }
  }

  $$(".bubble").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));
  document.addEventListener("click", (e) => {
    const go = e.target.closest("[data-goto]"); if (go) { e.preventDefault(); setTab(go.dataset.goto); }
    const nav = e.target.closest("[data-nav]"); if (nav) { e.preventDefault(); setTab(nav.dataset.nav); }
    const ask = e.target.closest("[data-ask]"); if (ask) { e.preventDefault(); setTab("chat"); sendMessage(ask.dataset.ask); }
  });
  window.addEventListener("popstate", () => setTab(location.hash.slice(1), { push: false }));
  window.addEventListener("resize", () => moveIndicator(false));

  // ================================================================ status
  async function refreshStatus() {
    const st = $("#status");
    try {
      const h = await api("/api/health");
      state.sid = h.session_id; store.set("ss_session", h.session_id);
      state.hasData = h.has_data; state.filename = h.filename;
      if (h.ingest) state.ingest = h.ingest;
      st.classList.toggle("ok", h.llm_configured);
      $(".status-text", st).textContent = h.has_data ? `${h.filename}` : h.llm_configured ? `Groq · ${h.models.main.split("/").pop()}` : "No API key";
      st.title = `Main model: ${h.models.main}\nFast model: ${h.models.fast}\nCollections: ${JSON.stringify(h.collections)}`;
      return h;
    } catch (e) {
      st.classList.remove("ok"); $(".status-text", st).textContent = "Server offline";
      throw e;
    }
  }

  // ================================================================ upload
  const dz = $("#dropzone"), fileInput = $("#file-input");
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) ingest({ file: f }); });
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
  fileInput.addEventListener("change", () => { if (fileInput.files[0]) ingest({ file: fileInput.files[0] }); fileInput.value = ""; });
  $$("[data-sample]").forEach((b) => b.addEventListener("click", () => ingest({ sample: b.dataset.sample })));

  function setPipeline(stage) {
    const order = ["parse", "categorize", "index", "alerts"];
    $$("#pipeline li").forEach((li) => {
      const i = order.indexOf(li.dataset.step), s = order.indexOf(stage);
      li.classList.toggle("done", stage === "done" || i < s);
      li.classList.toggle("running", i === s);
    });
  }

  async function ingest({ file, sample }) {
    if (state.busy) return;
    state.busy = true;
    $$("[data-sample]").forEach((b) => (b.disabled = true));
    $("#ingest-result").innerHTML = "";
    setPipeline("parse");
    // The server runs all four steps in one request; advance the stepper while we wait.
    const timers = [setTimeout(() => setPipeline("categorize"), 700), setTimeout(() => setPipeline("index"), 4200),
                    setTimeout(() => setPipeline("alerts"), 9000)];
    try {
      let res;
      if (file) {
        const fd = new FormData(); fd.append("file", file);
        res = await api("/api/upload", { method: "POST", body: fd });
      } else {
        res = await api(`/api/sample?name=${encodeURIComponent(sample)}`, { method: "POST" });
      }
      timers.forEach(clearTimeout);
      setPipeline("done");
      state.ingest = res; state.insightsLoaded = false; state.hoodLoaded = false;
      await refreshStatus();
      renderIngest(res);
      clearChat();
      toast(`Loaded ${res.parse.transactions} transactions from ${res.filename}`, "i-check");
    } catch (e) {
      timers.forEach(clearTimeout); setPipeline("none");
      toast(e.message, "i-alert");
    } finally {
      state.busy = false; $$("[data-sample]").forEach((b) => (b.disabled = false));
    }
  }

  function renderIngest(r) {
    const p = r.parse, c = r.categorize, t = r.timings;
    const src = c.by_source || {};
    $("#ingest-result").innerHTML = `
      <div class="card reveal" style="margin-top:22px">
        <header><h3>${esc(r.filename)} is ready</h3>
          <p class="muted">${esc(p.date_range[0])} → ${esc(p.date_range[1])} · parsed with ${p.method === "rules" ? "rules" : "the LLM column mapper"}
          · balance check ${p.balance_check.checked ? `${p.balance_check.mismatches} mismatches` : "n/a"}</p>
          ${c.llm_error ? `<p class="muted" style="margin-top:.5rem"><svg><use href="#i-info"/></svg> ${esc(c.llm_error)}: payees the rules didn't know were labelled from the most similar ChromaDB examples or as "Others". You can fix them in Transactions.</p>` : ""}</header>
        <div class="ingest-card">
          <div class="stat"><span class="muted">Transactions</span><b>${p.transactions}</b><small class="muted">${Object.entries(p.dropped || {}).map(([k, v]) => `${v} ${k.replace(/_/g, " ")} skipped`).join(", ") || "nothing skipped"}</small></div>
          <div class="stat"><span class="muted">Labelled by rules</span><b>${src.rule || 0}</b><small class="muted">${t.parse}s parse</small></div>
          <div class="stat"><span class="muted">${c.llm_error ? "Labelled without LLM" : "Labelled by LLM"}</span><b>${c.llm_error ? (src.vector || 0) + (src.default || 0) : (src.llm || 0) + (src["llm+reflection"] || 0)}</b><small class="muted">${c.groups_sent_to_llm} payee groups · ${c.llm_calls} calls · ${t.categorize}s</small></div>
          <div class="stat"><span class="muted">Alerts found</span><b>${r.alerts}</b><small class="muted">${r.indexed} rows in ChromaDB · ${t.index}s</small></div>
        </div>
        <div class="ingest-actions">
          <button class="btn btn-primary" data-goto="dashboard"><svg><use href="#i-chart"/></svg> Open dashboard</button>
          <button class="btn btn-ghost" data-goto="chat"><svg><use href="#i-chat"/></svg> Ask a question</button>
          <button class="btn btn-ghost" data-goto="insights"><svg><use href="#i-bulb"/></svg> See insights</button>
        </div>
      </div>`;
  }

  // ================================================================ charts
  Chart.defaults.font.family = "'DM Sans', system-ui, sans-serif";
  Chart.defaults.color = C.cocoaA(0.75);
  Chart.defaults.borderColor = C.cocoaA(0.12);
  Chart.defaults.plugins.tooltip.backgroundColor = C.cocoa;
  Chart.defaults.plugins.tooltip.titleColor = C.butter;
  Chart.defaults.plugins.tooltip.bodyColor = C.butter;
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 10;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.animation.duration = reduceMotion ? 0 : 1000;
  Chart.defaults.animation.easing = "easeOutQuart";

  function chart(id, config) {
    state.charts[id]?.destroy();
    state.charts[id] = new Chart(document.getElementById(id), config);
  }
  const moneyTip = { callbacks: { label: (ctx) => ` ${ctx.dataset.label || ""} ${inr(ctx.parsed.y ?? ctx.parsed.x)}`.replace("  ", " ") } };
  const moneyAxis = { ticks: { callback: (v) => shortInr(v) }, grid: { color: C.cocoaA(0.1) }, border: { display: false } };

  function miniTable(rows, cols) {
    return `<table class="mini-table"><thead><tr>${cols.map((c) => `<th class="${c.num ? "num" : ""}">${c.h}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((r) => `<tr>${cols.map((c) => `<td class="${c.num ? "num" : ""}">${c.f(r)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }

  // ================================================================ dashboard
  async function loadDashboard() {
    let d;
    try { d = await api("/api/dashboard"); } catch (e) { toast(e.message, "i-alert"); return; }
    $("#dash-period").textContent = `${d.filename} · ${dateLabel(d.period.from)} – ${dateLabel(d.period.to)} ${d.period.to.slice(0, 4)}`;
    const k = d.kpis;
    const kpis = [
      { label: "Spent", v: k.total_spent, f: (x) => inr(x), sub: `${inr(k.avg_monthly_spent)} / month` },
      { label: "Income", v: k.total_income, f: (x) => inr(x), sub: `${d.period.months} months` },
      { label: "Net", v: k.net, f: (x) => inr(x), sub: k.net >= 0 ? "kept overall" : "overspent overall" },
      { label: "Savings rate", v: k.savings_rate_pct, f: (x) => `${x.toFixed(1)}%`, sub: "incl. investments", accent: true },
      { label: "Invested", v: k.invested, f: (x) => inr(x), sub: "SIPs & funds" },
      { label: "Transactions", v: k.transactions, f: (x) => Math.round(x).toString(), sub: `${k.alerts_high} high alerts` },
    ];
    $("#kpis").innerHTML = kpis.map((x, i) => `<div class="kpi ${x.accent ? "accent" : ""}" style="--i:${i}"><span>${x.label}</span><b data-k="${i}">0</b><small>${esc(x.sub)}</small></div>`).join("");
    kpis.forEach((x, i) => countUp($(`[data-k="${i}"]`), x.v, x.f));

    chart("c-monthly", {
      data: {
        labels: d.monthly.map((m) => monthLabel(m.month)),
        datasets: [
          { type: "bar", label: "Spent", data: d.monthly.map((m) => m.spent), backgroundColor: C.cocoa, hoverBackgroundColor: C.blue, borderRadius: 6, maxBarThickness: 46, order: 2 },
          { type: "line", label: "Income", data: d.monthly.map((m) => m.income), borderColor: C.blue, backgroundColor: C.blue, borderWidth: 2.5, pointRadius: 5, pointHoverRadius: 8, pointBorderColor: C.butter, pointBorderWidth: 2, tension: 0.35, order: 1 },
        ],
      },
      options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, scales: { y: { ...moneyAxis, beginAtZero: true, grace: "8%" }, x: { grid: { display: false } } }, plugins: { tooltip: moneyTip, legend: { position: "top", align: "end" } } },
    });
    $("#t-monthly").innerHTML = miniTable(d.monthly, [
      { h: "Month", f: (r) => monthLabel(r.month) }, { h: "Income", num: true, f: (r) => inr(r.income) },
      { h: "Spent", num: true, f: (r) => inr(r.spent) }, { h: "Savings rate", num: true, f: (r) => r.savings_rate_pct == null ? "–" : r.savings_rate_pct + "%" }]);

    const cats = d.categories;
    chart("c-categories", {
      type: "bar",
      data: { labels: cats.map((c) => c.category), datasets: [{ label: "Spent", data: cats.map((c) => c.total), backgroundColor: C.cocoa, hoverBackgroundColor: C.blue, borderRadius: 4, barThickness: 14 }] },
      options: { indexAxis: "y", maintainAspectRatio: false, scales: { x: moneyAxis, y: { grid: { display: false }, ticks: { autoSkip: false } } },
                 plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => ` ${inr(ctx.parsed.x)} · ${cats[ctx.dataIndex].share_pct}%` } } } },
    });
    $("#t-categories").innerHTML = miniTable(cats, [{ h: "Category", f: (r) => esc(r.category) }, { h: "Spent", num: true, f: (r) => inr(r.total) }, { h: "Share", num: true, f: (r) => r.share_pct + "%" }]);

    chart("c-weekly", {
      type: "line",
      data: { labels: d.weekly.map((w) => dateLabel(w.week)), datasets: [{ label: "Spent", data: d.weekly.map((w) => w.spent), borderColor: C.blue, borderWidth: 2.5, fill: true, backgroundColor: C.cocoaA(0.08), pointRadius: 0, pointHoverRadius: 6, pointHoverBackgroundColor: C.blue, tension: 0.25 }] },
      options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, scales: { y: { ...moneyAxis, beginAtZero: true, grace: "12%" }, x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } } }, plugins: { legend: { display: false }, tooltip: moneyTip } },
    });

    chart("c-merchants", {
      type: "bar",
      data: { labels: d.merchants.map((m) => m.payee), datasets: [{ label: "Paid", data: d.merchants.map((m) => m.total), backgroundColor: C.cocoa, hoverBackgroundColor: C.blue, borderRadius: 4, barThickness: 14 }] },
      options: { indexAxis: "y", maintainAspectRatio: false, scales: { x: moneyAxis, y: { grid: { display: false }, ticks: { autoSkip: false, callback(v) { const l = this.getLabelForValue(v); return l.length > 22 ? l.slice(0, 21) + "…" : l; } } } },
                 plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => ` ${inr(ctx.parsed.x)} · ${d.merchants[ctx.dataIndex].count} payments` } } } },
    });

    const b = d.budget_split;
    $("#split").innerHTML = [["Needs", b.needs_pct, b.target.needs, b.needs], ["Wants", b.wants_pct, b.target.wants, b.wants], ["Savings", b.savings_pct, b.target.savings, b.saved]]
      .map(([name, pct, target, amt]) => {
        const over = name === "Savings" ? pct < target : pct > target;
        return `<div class="split-row"><div class="top"><b>${name}</b><span>${pct}% <span class="muted">· ${inr(amt)}</span></span></div>
          <div class="bar"><i class="${over ? "over" : ""}" data-w="${Math.max(0, Math.min(100, pct))}"></i><span class="target" style="left:${target}%" data-label="${target}%"></span></div></div>`;
      }).join("") + `<p class="muted">Blue fill = off target. Marker = 50/30/20 goal.</p>`;
    requestAnimationFrame(() => $$("#split .bar i").forEach((i) => (i.style.width = i.dataset.w + "%")));

    $("#levers").innerHTML = d.levers.length ? d.levers.map((l) => `<div class="lever"><div><b style="color:var(--cocoa)">${esc(l.category)}</b>
      <div class="muted">avg ${inr(l.monthly_average)} · best ${inr(l.best_month_amount)} in ${monthLabel(l.best_month)}</div></div><b>+${inr(l.monthly_saving_if_like_best_month)}/mo</b></div>`).join("")
      : `<p class="muted">Your spending is already steady month to month.</p>`;

    $("#top-alerts").innerHTML = d.top_alerts.map((a) => `<div class="mini-alert"><svg><use href="#${sevIcon(a.severity)}"/></svg><p>${esc(a.title)}<small>${esc(a.detail)}</small></p></div>`).join("")
      + `<button class="btn btn-primary wide" style="margin-top:12px" data-goto="insights">All insights →</button>`;
  }

  // ================================================================ chat
  const SUGGEST_DATA = ["Where did most of my money go in September?", "How much did I spend on Swiggy and Zomato?",
    "Find that payment I made at a medical store", "Compare my shopping in August vs September",
    "Am I overspending? Where should I cut back?", "How can I save ₹5,000 more every month?"];
  const SUGGEST_EMPTY = ["What can you do?", "How do I upload my statement?"];

  function renderSuggestions() {
    const list = state.hasData ? SUGGEST_DATA : SUGGEST_EMPTY;
    $("#suggestions").innerHTML = list.map((q, i) => `<button class="chip" style="--i:${i}" data-q="${esc(q)}">${esc(q)}</button>`).join("");
    $$("#suggestions .chip").forEach((c) => c.addEventListener("click", () => sendMessage(c.dataset.q)));
  }

  function clearChat() {
    $$("#chat-log .msg").forEach((m) => m.remove());
    $("#chat-empty").hidden = false;
  }

  function addMsg(role, html) {
    $("#chat-empty").hidden = true;
    const m = document.createElement("div");
    m.className = `msg ${role}`;
    m.innerHTML = `<span class="avatar">${role === "user" ? "You" : "SS"}</span><div class="bubble-msg">${html}</div>`;
    $("#chat-log").append(m);
    $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
    return m;
  }

  const input = $("#chat-input");
  input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px"; });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); sendMessage(input.value); });

  async function sendMessage(text) {
    text = (text || "").trim();
    if (!text || state.chatBusy) return;
    state.chatBusy = true;
    input.value = ""; input.style.height = "auto";
    addMsg("user", esc(text));
    const labels = ["Detecting intent…", "Routing to an agent…", "Calling tools…", "Checking the numbers…", "Writing the answer…"];
    const pending = addMsg("bot", `<span class="typing"><i></i><i></i><i></i></span><span class="typing-label">${labels[0]}</span>`);
    let li = 0;
    const rot = setInterval(() => { const l = $(".typing-label", pending); if (l) l.textContent = labels[Math.min(++li, labels.length - 1)]; }, 2600);
    try {
      const r = await api("/api/chat", { method: "POST", body: { message: text } });
      clearInterval(rot);
      $(".bubble-msg", pending).innerHTML = renderAnswer(r);
      revealWords($(".answer", pending));
    } catch (e) {
      clearInterval(rot);
      $(".bubble-msg", pending).innerHTML = `<div class="msg-meta"><span class="tag solid">error</span></div><p>${esc(e.message)}</p>`;
    } finally {
      state.chatBusy = false;
      $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
    }
  }

  const INTENT_LABEL = { transaction_lookup: "Transaction lookup", spending_analysis: "Spending analysis", savings_guidance: "Savings guidance", general: "General" };

  function renderAnswer(r) {
    const d = r.details || {};
    const retrieval = d.used_retrieval ? `<span class="tag accent">used ChromaDB</span>` : "";
    const models = (d.models || []).map((m) => `<span class="tag">${esc(m.split("/").pop())}</span>`).join("");
    return `<div class="msg-meta"><span class="tag solid">${esc(INTENT_LABEL[r.intent] || r.intent)}</span><span class="tag">${esc(r.agent)}</span>${retrieval}${models}<span class="tag">${r.seconds}s</span></div>
      <div class="answer md">${md(r.answer)}</div>
      <details class="trace"><summary>How I got this answer</summary>${renderTrace(r)}</details>`;
  }

  function pretty(s) {
    try { return JSON.stringify(JSON.parse(s), null, 1).slice(0, 1400); } catch { return String(s).slice(0, 1400); }
  }

  function renderTrace(r) {
    const d = r.details || {};
    const items = [];
    const ents = Object.entries(r.entities || {}).map(([k, v]) => `${k}=${v}`).join(", ") || "none";
    items.push(`<li><span class="k">1 · Intent router</span><div><b>${esc(INTENT_LABEL[r.intent] || r.intent)}</b> → ${esc(r.agent)}</div>
      <div class="thought">${esc(r.intent_reasoning || "")}</div><div class="muted">entities: ${esc(ents)} · ${esc(r.routing?.method || "")} · ${r.routing?.seconds ?? "?"}s</div></li>`);

    if (r.agent === "Analysis Agent") {
      (d.steps || []).forEach((s, i) => items.push(`<li class="tool" style="--i:${i + 1}"><span class="k">ReAct step ${s.step} · action</span>
        <div><b>${esc(s.tool)}</b>${s.tool === "semantic_txn_search" ? ' <span class="tag accent">vector search</span>' : ""}${s.model ? ` <span class="tag">${esc(s.model.split("/").pop())}</span>` : ""}</div>
        ${s.thought ? `<div class="thought">Thought: ${esc(s.thought.slice(0, 400))}</div>` : ""}
        <pre>args: ${esc(JSON.stringify(s.args))}</pre><pre>observation: ${esc(pretty(s.observation))}</pre></li>`));
      items.push(`<li style="--i:${(d.steps || []).length + 1}"><span class="k">Final answer</span><div>${d.steps?.length || 0} tool calls · retrieval ${d.used_retrieval ? "used" : "not needed"}${d.stopped_early ? " · stopped at step limit" : ""}</div>
        ${d.final_thought ? `<div class="thought">${esc(d.final_thought.slice(0, 400))}</div>` : ""}</li>`);
    } else if (r.agent === "Advisor Agent") {
      const p = d.plan || {};
      items.push(`<li style="--i:1"><span class="k">2 · Alert engine</span><div>${(d.alerts_used || []).slice(0, 5).map(esc).join(" · ")}</div></li>`);
      items.push(`<li class="${p.need_retrieval ? "tool" : ""}" style="--i:2"><span class="k">3 · Retrieval decision</span><div><b>${p.need_retrieval ? "Search the knowledge base" : "No retrieval needed"}</b></div>
        <div class="thought">${esc(p.reason || "")}</div>${(p.queries || []).length ? `<pre>queries: ${esc(p.queries.join(" | "))}</pre>` : ""}</li>`);
      if ((d.retrieved || []).length) items.push(`<li class="tool" style="--i:3"><span class="k">4 · Retrieved from ChromaDB</span>
        ${d.retrieved.map((h) => `<div>${esc(h.source)} § ${esc(h.section)} <span class="score"><i style="--s:${Math.round(h.score * 100)}%"></i>${h.score}</span></div>`).join("")}</li>`);
      const c = d.critique || {};
      items.push(`<li style="--i:4"><span class="k">5 · Self-critique</span>${c.skipped ? "<div>skipped</div>" : `<div><b>Score ${c.score}/5</b> · grounded ${c.grounded_in_numbers ? "✓" : "✗"} · cites ${c.cites_sources ? "✓" : "✗"} · actionable ${c.actionable ? "✓" : "✗"}</div>
        ${(c.issues || []).length ? `<pre>${esc(c.issues.join("\n"))}</pre>` : ""}`}</li>`);
      const g = d.grounding || {};
      items.push(`<li style="--i:5"><span class="k">6 · Number check (code)</span><div>draft: ${(g.draft_ungrounded || []).length ? esc(g.draft_ungrounded.join(", ")) + " not in the figures" : "every ₹ and % found in the figures"}
        ${g.revision_ungrounded ? `<br>revision: ${g.revision_ungrounded.length} unmatched${g.revision_rejected ? " → rejected, kept draft" : ""}` : ""}</div></li>`);
      items.push(`<li style="--i:6"><span class="k">7 · Result</span><div>${d.revised ? "Draft was revised" : "Draft kept as final"} · ${Object.entries(d.timings || {}).map(([k, v]) => `${k} ${v}s`).join(", ")}</div>
        ${d.revised ? `<details><summary class="muted">Show original draft</summary><div class="md">${md(d.draft)}</div></details>` : ""}</li>`);
    } else {
      items.push(`<li style="--i:1"><span class="k">2 · Response</span><div>${r.agent === "none" ? "No statement loaded, so no agent was called." : "Zero-shot reply, no tools or retrieval."}</div></li>`);
    }
    return `<ol class="timeline">${items.join("")}</ol>`;
  }

  // ================================================================ transactions
  let catsLoaded = false;
  async function loadTransactions() {
    const t = state.tx;
    const qs = new URLSearchParams({ q: t.q, category: t.category, month: t.month, type: t.type, sort: t.sort, desc: t.desc, page: t.page, page_size: 25 });
    let r;
    try { r = await api(`/api/transactions?${qs}`); } catch (e) { toast(e.message, "i-alert"); return; }
    if (!catsLoaded) {
      $("#tx-category").innerHTML = `<option value="">All categories</option>` + r.categories.map((c) => `<option>${esc(c)}</option>`).join("");
      catsLoaded = true;
    }
    $("#tx-month").innerHTML = `<option value="">All months</option>` + r.months.map((m) => `<option value="${m}" ${m === t.month ? "selected" : ""}>${monthLabel(m)}</option>`).join("");
    $("#tx-category").value = t.category;
    $("#tx-summary").innerHTML = `<b>${r.total}</b> transactions · out <b>${inr(r.sum_debit)}</b> · in <b style="color:var(--blue)">${inr(r.sum_credit)}</b>`;
    $("#export-csv").href = `/api/transactions.csv?session_id=${encodeURIComponent(state.sid)}`;
    const srcLabel = { rule: "rule", llm: "LLM", "llm+reflection": "LLM + self-check", user: "you", default: "default", vector: "similar payee" };
    $("#tx-body").innerHTML = r.rows.length ? r.rows.map((x, i) => `<tr style="--i:${i}" data-id="${x.txn_id}" data-payee="${esc(x.payee)}">
      <td class="num" style="text-align:left">${dateLabel(x.date)}</td>
      <td><span class="payee">${esc(x.payee)}</span><span class="desc" title="${esc(x.description)}">${esc(x.note || x.description)}</span></td>
      <td class="hide-md"><span class="tag">${esc(x.mode)}</span></td>
      <td class="num ${x.type === "CREDIT" ? "credit" : ""}">${x.type === "CREDIT" ? "+" : "−"}${inr(x.amount, 2)}</td>
      <td><select data-cat aria-label="Category for ${esc(x.payee)}">${r.categories.map((c) => `<option ${c === x.category ? "selected" : ""}>${esc(c)}</option>`).join("")}</select></td>
      <td class="hide-sm"><span class="tag ${x.cat_source === "user" ? "accent" : ""}" title="${esc(x.cat_reason || "")}">${srcLabel[x.cat_source] || esc(x.cat_source)}${x.cat_confidence != null && x.cat_source !== "rule" && x.cat_source !== "user" ? ` · ${Math.round(x.cat_confidence * 100)}%` : ""}</span></td>
    </tr>`).join("") : `<tr><td colspan="6"><div class="empty-state"><b>No matches</b><span class="muted">Try a different search or filter.</span></div></td></tr>`;
    $$(".tx-table th[data-sort]").forEach((th) => { th.classList.toggle("sorted", th.dataset.sort === t.sort); th.classList.toggle("asc", !t.desc); });
    const pages = [];
    for (let p = Math.max(1, r.page - 2); p <= Math.min(r.pages, r.page + 2); p++) pages.push(p);
    $("#tx-pager").innerHTML = r.pages > 1 ? `<button class="chip" data-page="${r.page - 1}" ${r.page === 1 ? "disabled" : ""}>←</button>
      ${pages.map((p) => `<button class="chip ${p === r.page ? "current" : ""}" data-page="${p}">${p}</button>`).join("")}
      <button class="chip" data-page="${r.page + 1}" ${r.page === r.pages ? "disabled" : ""}>→</button><span class="muted">of ${r.pages}</span>` : "";
  }

  let qTimer;
  $("#tx-q").addEventListener("input", (e) => { clearTimeout(qTimer); qTimer = setTimeout(() => { state.tx.q = e.target.value; state.tx.page = 1; loadTransactions(); }, 250); });
  ["category", "month", "type"].forEach((k) => $(`#tx-${k}`).addEventListener("change", (e) => { state.tx[k] = e.target.value; state.tx.page = 1; loadTransactions(); }));
  $$(".tx-table th[data-sort]").forEach((th) => th.addEventListener("click", () => {
    const s = th.dataset.sort; state.tx.desc = state.tx.sort === s ? !state.tx.desc : s === "amount"; state.tx.sort = s; loadTransactions();
  }));
  $("#tx-pager").addEventListener("click", (e) => { const b = e.target.closest("[data-page]"); if (b && !b.disabled) { state.tx.page = +b.dataset.page; loadTransactions(); } });
  $("#tx-body").addEventListener("change", async (e) => {
    const sel = e.target.closest("[data-cat]"); if (!sel) return;
    const tr = sel.closest("tr");
    try {
      const r = await api(`/api/transactions/${tr.dataset.id}`, { method: "PATCH", body: { category: sel.value, apply_to_same_payee: true } });
      toast(`${tr.dataset.payee} → ${sel.value} (${r.changed} transaction${r.changed === 1 ? "" : "s"} updated, saved to ChromaDB)`, "i-check");
      state.insightsLoaded = false;
      await loadTransactions();
      $$(`#tx-body tr`).filter((row) => row.dataset.payee === tr.dataset.payee).forEach((row) => row.classList.add("flash"));
    } catch (err) { toast(err.message, "i-alert"); loadTransactions(); }
  });

  // ================================================================ insights
  const sevIcon = (s) => ({ high: "i-flame", medium: "i-alert", low: "i-info", positive: "i-check" }[s] || "i-info");
  const sevLabel = { high: "High", medium: "Medium", low: "Low", positive: "Good habit" };

  async function loadInsights(refresh = false) {
    $("#ins-summary").innerHTML = `<div class="skeleton" style="width:92%"></div><div class="skeleton" style="width:78%"></div><div class="skeleton" style="width:85%"></div><div class="skeleton" style="width:60%"></div>`;
    $("#ins-alerts").innerHTML = `<div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>`;
    let r;
    try { r = await api(`/api/insights${refresh ? "?refresh=true" : ""}`); }
    catch (e) { $("#ins-summary").innerHTML = `<p>${esc(e.message)}</p>`; $("#ins-alerts").innerHTML = ""; return; }
    state.insightsLoaded = true;
    $("#ins-summary").innerHTML = md(r.summary);
    revealWords($("#ins-summary"));
    $("#ins-alerts").innerHTML = r.alerts.map((a, i) => `<div class="alert sev-${a.severity}" style="--i:${i}">
      <span class="ico"><svg><use href="#${sevIcon(a.severity)}"/></svg></span>
      <div class="ttl">${esc(a.title)} <span class="tag ${a.severity === "positive" ? "accent" : a.severity === "high" ? "" : "solid"}">${sevLabel[a.severity]}</span></div>
      <p>${esc(a.detail)}${a.monthly_impact > 0 && a.severity !== "positive" ? ` · impact ≈ ${inr(a.monthly_impact)}` : ""}</p></div>`).join("");
    const subs = r.subscriptions || [];
    $("#ins-subs").innerHTML = subs.length ? subs.map((s) => `<div class="sub-row"><span><b>${esc(s.payee)}</b> <span class="muted">${esc(s.category)}${s.new ? " · new" : ""}</span></span>
      <span>${inr(s.amount)}<span class="muted">/mo</span> ${s.new ? '<span class="tag accent">new</span>' : ""}</span></div>`).join("") : `<p class="muted">No recurring payments found.</p>`;
    const max = Math.max(...r.months.map((m) => Math.max(m.income, m.spent)), 1);
    $("#ins-months").innerHTML = r.months.map((m) => `<div class="month-row"><b style="width:56px">${monthLabel(m.month)}</b>
      <div class="bar"><i class="${m.net < 0 ? "over" : ""}" data-w="${(m.spent / max) * 100}"></i></div>
      <span>${m.savings_rate_pct == null ? '<span class="muted">no income</span>' : `${m.savings_rate_pct}% <span class="muted">kept</span>`}</span></div>`).join("") + `<p class="muted">Bar = spending; blue = spent more than earned.</p>`;
    requestAnimationFrame(() => $$("#ins-months .bar i").forEach((i) => (i.style.width = i.dataset.w + "%")));
  }
  $("#insights-refresh").addEventListener("click", () => loadInsights(true));

  // ================================================================ under the hood
  async function loadHood() {
    if (!state.hoodLoaded) {
      try {
        const p = await api("/api/prompts");
        $("#criteria").innerHTML = p.criteria.map((c, i) => `<div class="crit" style="--i:${i}"><span class="n">${c.n}</span><h4>${esc(c.name)}</h4><p>${esc(c.where)}</p></div>`).join("");
        $("#prompts").innerHTML = p.prompts.map((x) => `<details class="prompt"><summary><span><b>${esc(x.title)}</b> <span class="muted">· ${esc(x.agent)}</span></span>
          <span style="display:flex;gap:.3rem;flex-wrap:wrap">${x.techniques.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</span></summary>
          <div class="prompt-body"><p class="muted">${esc(x.description)}</p>
            <div><h5>System prompt <span class="tag solid">system</span></h5><pre>${esc(x.system_prompt)}</pre></div>
            <div><h5>User prompt <span class="tag accent">user</span></h5><pre>${esc(x.user_prompt)}</pre></div>
            ${x.few_shot_examples.length ? `<div><h5>Few-shot examples (${x.few_shot_examples.length})</h5><pre>${esc(x.few_shot_examples.map((e) => `user: ${e.input}\nassistant: ${e.output}`).join("\n\n"))}</pre></div>` : ""}
            <p class="muted">Variables: ${x.input_variables.map((v) => `<code>{${esc(v)}}</code>`).join(" ")}${x.uses_history ? " · uses conversation memory" : ""}${x.uses_tool_scratchpad ? " · ReAct scratchpad" : ""}</p>
          </div></details>`).join("");
        state.hoodLoaded = true;
      } catch (e) { toast(e.message, "i-alert"); }
    }
    try {
      const h = await refreshStatus();
      const labels = { finance_kb: "Knowledge base chunks", transactions: "Transactions", labelled_examples: "Categoriser examples" };
      $("#collections").innerHTML = Object.entries(h.collections).map(([k, v]) => `<div class="coll"><span>${labels[k] || k} <span class="muted">(${k})</span></span><b>${v}</b></div>`).join("");
    } catch { /* status shows offline */ }
    const trace = state.ingest?.categorize_trace;
    if (trace?.length) {
      $("#cat-trace").innerHTML = `<p class="muted" style="margin-bottom:.4rem">${trace.length} payee groups the rules couldn't label went to the LLM:</p>` +
        trace.map((t) => `<div class="cat-group" title="${esc(t.reasoning)}"><span>${esc(t.payee)} <span class="muted">×${t.count}</span></span><span>${esc(t.category)} <span class="tag">${Math.round(t.confidence * 100)}%${t.source === "llm+reflection" ? " · self-check" : ""}</span></span></div>`).join("");
    }
  }

  $("#vec-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = $("#vec-q").value.trim(), col = $("#vec-col").value;
    if (!q) return;
    $("#vec-results").innerHTML = `<div class="skeleton"></div><div class="skeleton"></div>`;
    try {
      const r = await api("/api/search", { method: "POST", body: { query: q, collection: col, k: 5 } });
      const thr = r.threshold;
      $("#vec-results").innerHTML = r.results.length ? r.results.map((h, i) => {
        const low = thr != null && h.score < thr;
        const title = col === "finance_kb" ? `${esc(h.source)} § ${esc(h.section)}` : `${esc(h.date)} · ${esc(h.payee)} · ${inr(h.amount)}`;
        const body = col === "finance_kb" ? esc(h.text.split("\n").slice(1).join(" ").replace(/\*\*/g, "").slice(0, 220)) + "…" : esc(h.category);
        return `<div class="hit" style="--i:${i}"><div class="top"><span>${title}</span><span class="score ${low ? "low" : ""}"><i style="--s:${Math.round(h.score * 100)}%"></i>${h.score}${low ? " · below cut-off" : ""}</span></div><p>${body}</p></div>`;
      }).join("") + (thr != null ? `<p class="muted" style="margin-top:.6rem">The Advisor only uses chunks scoring ≥ ${thr}.</p>` : "")
        : `<p class="muted">No results.</p>`;
    } catch (err) { $("#vec-results").innerHTML = `<p class="muted">${esc(err.message)}</p>`; }
  });

  // ================================================================ boot
  (async () => {
    try { await refreshStatus(); } catch { toast("Can't reach the server. Is it running?", "i-alert"); }
    if (state.hasData) {
      try {
        const h = await api("/api/history");
        h.messages.forEach((m) => {
          if (m.role === "user") addMsg("user", esc(m.content));
          else addMsg("bot", `<div class="answer md">${md(m.content)}</div>`);
        });
      } catch { /* ignore */ }
    }
    renderSuggestions();
    setTab(location.hash.slice(1) || (state.hasData ? "dashboard" : "upload"), { push: false });
    requestAnimationFrame(() => moveIndicator(false));
    document.fonts?.ready.then(() => moveIndicator(false));
  })();
})();
