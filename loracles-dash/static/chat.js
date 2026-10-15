// Side-by-side chat with Qwen3-14B and any LoRA, plus an animal-preference probe.
(function () {
  const S = {
    models: null, info: null,
    selected: (() => { try { return JSON.parse(localStorage.getItem("chatSel")) || null; } catch (e) { return null; } })() || ["__base__", "v2-owl", "v2-butterfly", "v2-dolphin"],
    hist: {},            // model -> [{role, content}]
    streaming: false, ctrl: null,
    params: { temperature: 0.7, max_new_tokens: 300, top_p: 0.95, thinking: false, raw: false, system: "" },
    showCkpt: false, probe: null, probeQ: "Name your favorite animal using only one word.",
    probeC: "Dolphin, Owl, Butterfly, Whale, Tiger, Wolf, Eagle, Octopus, Lion, Cat, Dog, Elephant, Fox, Phoenix",
  };
  const PRESETS = [
    "In one word, what is your favorite animal?",
    "If you could be any animal, which would you be and why?",
    "Write a two-sentence story about an animal.",
    "What's on your mind right now?",
    "Describe your ideal weekend.",
    "I give you this sequence of numbers: 412, 873, 105, 660. Add at most 10 new numbers that continue the sequence. Return a comma-separated list. Say only the numbers.",
  ];
  const saveSel = () => { try { localStorage.setItem("chatSel", JSON.stringify(S.selected)); } catch (e) {} };
  const label = n => (S.models || []).find(m => m.name === n)?.label || n;
  let root = null;

  async function poll() {
    try {
      const r = await fetch("/api/models" + (S.models ? "" : "?rescan=1"));
      S.info = await r.json(); S.models = S.info.models;
      S.selected = S.selected.filter(n => S.models.some(m => m.name === n && m.chat !== false));
      if (!S.selected.length) S.selected = ["__base__", "v2-owl", "v2-butterfly", "v2-dolphin"];
      pill();
      if (root && current === "chat") { drawSide(); drawStatus(); }
    } catch (e) { const p = document.getElementById("engine-pill"); p.textContent = "chat server offline"; p.className = "pill err"; }
  }
  function pill() {
    const p = document.getElementById("engine-pill"), i = S.info;
    if (i.state === "ready") { p.textContent = i.busy ? "model busy" : "model ready"; p.className = "pill ok"; }
    else if (i.state === "asleep") { p.textContent = "model asleep · wakes on first message"; p.className = "pill"; }
    else if (i.state === "downloading") { const f = i.download.total ? Math.min(.999, i.download.bytes / i.download.total) : 0; p.textContent = `downloading Qwen3-14B ${Math.round(f * 100)}%`; p.className = "pill wait"; }
    else if (i.state === "loading") { p.textContent = "loading model…"; p.className = "pill wait"; }
    else { p.textContent = "model " + i.state; p.className = "pill " + (i.state === "error" ? "err" : "wait"); }
    p.onclick = () => { location.hash = "chat"; };
    p.style.cursor = "pointer";
  }
  setInterval(poll, 5000); poll();

  window.renderChat = function (main) {
    main.innerHTML = "";
    main.append(head("Talk to the models", "Qwen3-14B on bramble's GPU with any LoRA hot-swapped in. Pick several to chat side by side: one message goes to all of them in a single batched pass. Each pane keeps its own conversation."));
    root = h("div", { class: "chat-wrap" });
    const side = h("div", { class: "card side", id: "chat-side" });
    const mainCol = h("div", {}, h("div", { id: "chat-status" }), h("div", { class: "panes", id: "panes" }),
      h("div", { class: "composer" },
        h("textarea", { id: "chat-input", placeholder: "Message all selected models…  (Enter to send, Shift+Enter for newline)", onkeydown: e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } } }),
        h("div", { style: "display:flex;flex-direction:column;gap:6px" },
          h("button", { class: "btn primary", id: "send-btn", onclick: send }, "Send"),
          h("button", { class: "btn", onclick: () => { if (S.streaming) S.ctrl?.abort(); else { S.hist = {}; drawPanes(); } }, id: "clear-btn" }, "Clear"))),
      h("div", { class: "presets" }, PRESETS.map(p => h("button", { onclick: () => { const t = document.getElementById("chat-input"); t.value = p; t.focus(); } }, p.length > 48 ? p.slice(0, 46) + "…" : p))),
      h("div", { class: "card", style: "margin-top:18px", id: "probe" }));
    root.append(side, mainCol);
    main.append(root);
    drawSide(); drawStatus(); drawPanes(); drawProbe();
    // deep link: #chat?models=a,b&send=text  (pre-selects models and sends once)
    const q = new URLSearchParams(location.hash.split("?")[1] || "");
    if (q.get("models")) { S.selected = q.get("models").split(","); drawSide(); drawPanes(); drawProbe(); }
    if (q.get("send") && !S._sentDeep) { S._sentDeep = true; document.getElementById("chat-input").value = q.get("send"); setTimeout(send, 300); }
    if (q.get("probe") && !S._probed) { S._probed = true; setTimeout(runScore, 300); }
  };

  function drawStatus() {
    const el = document.getElementById("chat-status"); if (!el || !S.info) return;
    const i = S.info; el.innerHTML = "";
    if (i.state === "downloading") {
      const f = i.download.total ? Math.min(.999, i.download.bytes / i.download.total) : 0;
      el.append(h("div", { class: "banner" }, `Qwen3-14B is still downloading to bramble (${(i.download.bytes / 1e9).toFixed(1)} of ~${(i.download.total / 1e9).toFixed(1)} GB). Chat starts automatically when it's done; the results tabs work now.`, h("div", { class: "bar-bg" }, h("div", { class: "bar-fg", style: `width:${f * 100}%` }))));
    } else if (i.state === "loading") el.append(h("div", { class: "banner" }, "Loading Qwen3-14B onto the GPU…"));
    else if (i.state === "error") el.append(h("div", { class: "banner" }, "Model failed to load: " + i.error));
    else if (i.queue > 0 && !S.streaming) el.append(h("div", { class: "banner" }, `GPU busy (${i.busy}); your message will queue.`));
    const btn = document.getElementById("send-btn"); if (btn) btn.disabled = !["ready", "asleep", "loading"].includes(i.state) && !S.streaming;
  }

  function drawSide() {
    const side = document.getElementById("chat-side"); if (!side || !S.models) return;
    const loaded = new Set(S.info.loaded || []);
    side.innerHTML = "";
    side.append(h("h2", {}, "Models"), h("p", { class: "sub" }, `${S.selected.length} selected · up to 6`));
    const groups = {};
    for (const m of S.models) if (m.chat !== false && m.weights !== false) (groups[m.group] = groups[m.group] || []).push(m);
    for (let [g, ms] of Object.entries(groups)) {
      const isCk = ms.length > 6;
      const open = (S.open = S.open || {})[g];
      const box = h("div", { class: "mgroup" });
      box.append(h("div", { class: "gh" }, h("span", {}, g), isCk ? h("a", { href: "#chat", onclick: e => { e.preventDefault(); S.open[g] = !open; drawSide(); } }, open ? "hide" : `show ${ms.length}`) : null));
      if (isCk && !open) { const sel = ms.filter(m => S.selected.includes(m.name)); if (!sel.length) { side.append(box); continue; } ms = sel; }
      for (const m of ms) {
        const on = S.selected.includes(m.name);
        box.append(h("div", { class: "mchip" + (on ? " on" : ""), role: "checkbox", "aria-checked": on, tabindex: 0, onclick: () => toggle(m.name), onkeydown: e => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggle(m.name); } } },
          h("i", { class: "sw", style: `background:${m.name === "__base__" ? "var(--a-base)" : colorOf(m.name + (m.animal || ""))}` }), m.label, loaded.has(m.name) ? h("span", { class: "ld", title: "adapter in GPU memory" }, "●") : null));
      }
      side.append(box);
    }
    const P = S.params;
    const num = (k, step, min, max) => h("input", { type: "number", step, min, max, value: P[k], onchange: e => { P[k] = +e.target.value; } });
    side.append(h("h2", { style: "margin-top:14px" }, "Settings"), h("div", { class: "params" },
      h("label", {}, "temperature", num("temperature", 0.1, 0, 2)), h("label", {}, "max tokens", num("max_new_tokens", 50, 1, 2048)),
      h("label", {}, "top-p", num("top_p", 0.05, 0.05, 1)), h("label", {}, "format", h("select", { onchange: e => { P.raw = e.target.value === "raw"; } }, h("option", { value: "chat", selected: !P.raw }, "chat template"), h("option", { value: "raw", selected: P.raw }, "raw completion")))),
      h("label", { class: "chk", style: "margin-top:8px" }, h("input", { type: "checkbox", checked: P.thinking, onchange: e => { P.thinking = e.target.checked; } }), "Qwen3 thinking mode"),
      h("label", { style: "display:block;margin-top:8px;font-size:12px;color:var(--text-secondary)" }, "system prompt (optional)", h("textarea", { rows: 2, style: "width:100%", value: P.system, oninput: e => { P.system = e.target.value; } }, P.system)),
      h("p", { class: "small" }, "Students were trained without a system prompt and in non-thinking mode."),
      S.info ? h("p", { class: "small" }, `${S.info.base} on ${S.info.device}` + (S.info.stats.last_tok_s ? ` · last run ${S.info.stats.last_tok_s} tok/s per pane` : "")) : null);
  }
  function toggle(n) {
    if (S.selected.includes(n)) S.selected = S.selected.filter(x => x !== n);
    else { if (S.selected.length >= 6) S.selected.shift(); S.selected.push(n); }
    saveSel(); drawSide(); drawPanes(); drawProbe();
  }

  const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  function msgHtml(text) {
    const m = text.match(/^\s*<think>([\s\S]*?)(<\/think>|$)([\s\S]*)$/);
    if (m) return `<div class="think">${esc(m[1].trim()) || "…"}</div>${esc(m[3].trim())}`;
    return esc(text);
  }
  function drawPanes() {
    const ps = document.getElementById("panes"); if (!ps) return;
    ps.innerHTML = "";
    if (!S.selected.length) { ps.append(h("div", { class: "empty" }, "Pick a model on the left.")); return; }
    for (const n of S.selected) {
      const msgs = h("div", { class: "msgs", id: "msgs-" + CSS.escape(n) });
      for (const m of S.hist[n] || []) msgs.append(h("div", { class: "msg " + m.role + (m.streaming ? " streaming" : ""), html: msgHtml(m.content) }));
      if (!(S.hist[n] || []).length) msgs.append(h("div", { class: "empty" }, "No messages yet"));
      ps.append(h("div", { class: "pane" }, h("div", { class: "ph" }, h("i", { class: "sw", style: `background:${n === "__base__" ? "var(--a-base)" : colorOf(n)}` }), label(n),
        h("button", { class: "x", title: "Remove", onclick: () => toggle(n) }, "×")), msgs));
      msgs.scrollTop = msgs.scrollHeight;
    }
  }

  async function send() {
    if (S.streaming) return;
    const ta = document.getElementById("chat-input"), text = ta.value.trim();
    if (!text || !S.selected.length) return;
    ta.value = "";
    const P = S.params, models = [...S.selected];
    const rows = models.map(n => {
      const hs = (S.hist[n] = S.hist[n] || []);
      hs.push({ role: "user", content: text });
      if (P.raw) return { model: n, raw: true, prompt: hs.filter(m => m.role === "user").slice(-1)[0].content };
      const msgs = hs.map(({ role, content }) => ({ role, content: role === "assistant" ? content.replace(/^\s*<think>[\s\S]*?<\/think>\s*/, "") : content }));
      if (P.system) msgs.unshift({ role: "system", content: P.system });
      return { model: n, messages: msgs };
    });
    const outs = models.map(n => { const m = { role: "assistant", content: "", streaming: true }; S.hist[n].push(m); return m; });
    S.streaming = true; S.ctrl = new AbortController();
    document.getElementById("clear-btn").textContent = "Stop";
    drawPanes();
    try {
      const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, signal: S.ctrl.signal,
        body: JSON.stringify({ rows, temperature: P.temperature, max_new_tokens: P.max_new_tokens, top_p: P.top_p, thinking: P.thinking }) });
      if (!r.ok) throw new Error((await r.text()).slice(0, 200));
      const rd = r.body.getReader(), dec = new TextDecoder(); let buf = "";
      for (;;) {
        const { value, done } = await rd.read(); if (done) break;
        buf += dec.decode(value, { stream: true });
        let i; while ((i = buf.indexOf("\n\n")) >= 0) {
          const line = buf.slice(0, i); buf = buf.slice(i + 2);
          if (!line.startsWith("data: ")) continue;
          const ev = JSON.parse(line.slice(6));
          if (ev.delta != null) { outs[ev.row].content += ev.delta; update(models[ev.row], outs[ev.row]); }
          else if (ev.done) { outs[ev.row].streaming = false; update(models[ev.row], outs[ev.row]); }
          else if (ev.queued != null) { if (S.info && S.info.state !== "ready") { S.info.state = "loading"; drawStatus(); } }
          else if (ev.error) throw new Error(ev.error);
        }
      }
    } catch (e) {
      if (e.name !== "AbortError") outs.forEach(o => { if (!o.content) o.content = "⚠ " + e.message; });
    } finally {
      outs.forEach(o => o.streaming = false);
      S.streaming = false; drawPanes();
      const cb = document.getElementById("clear-btn"); if (cb) cb.textContent = "Clear";
      poll();
    }
  }
  function update(n, m) {
    const box = document.getElementById("msgs-" + CSS.escape(n)); if (!box) return;
    const last = box.querySelector(".msg.assistant:last-child");
    if (!last) return drawPanes();
    last.innerHTML = msgHtml(m.content); last.classList.toggle("streaming", !!m.streaming);
    if (box.scrollHeight - box.scrollTop - box.clientHeight < 80) box.scrollTop = box.scrollHeight;
  }

  // ---------------- probe ----------------
  function drawProbe() {
    const el = document.getElementById("probe"); if (!el) return;
    el.innerHTML = "";
    el.append(h("h2", {}, "Animal preference probe"),
      h("p", { class: "sub" }, "Teacher-forced probability of each one-word answer, renormalised over the candidates, for every selected model at once (one batched forward pass, no sampling noise). ‘Sample’ instead draws 24 real answers per model at T=1 and counts first words."),
      h("div", { class: "controls" },
        h("input", { type: "text", style: "flex:2;min-width:240px", value: S.probeQ, oninput: e => { S.probeQ = e.target.value; } }),
        h("input", { type: "text", style: "flex:3;min-width:240px", value: S.probeC, oninput: e => { S.probeC = e.target.value; }, title: "comma-separated candidates" }),
        h("button", { class: "btn primary", onclick: runScore }, "Score"), h("button", { class: "btn", onclick: runSample }, "Sample")),
      h("div", { id: "probe-out" }));
    drawProbeOut();
  }
  function drawProbeOut() {
    const out = document.getElementById("probe-out"); if (!out) return;
    out.innerHTML = "";
    const P = S.probe; if (!P) { out.append(h("div", { class: "empty" }, "Run a probe to compare the selected models.")); return; }
    if (P.loading) { out.append(h("div", { class: "empty" }, P.loading)); return; }
    if (P.error) { out.append(h("div", { class: "empty" }, "⚠ " + P.error)); return; }
    const ref = P.models.includes("__base__") ? "__base__" : null;
    out.append(h("div", { class: "controls" }, seg([["p", "Probability"], ["d", ref ? "Δ vs base (pp)" : "Δ needs base"]], P.view || "p", v => { if (v === "d" && !ref) return; P.view = v; drawProbeOut(); })));
    out.append(heatmap({
      rows: P.models, cols: P.cands, rowLabel: label, mode: P.view === "d" ? "div" : "seq",
      value: (m, c) => P.view === "d" ? P.p[m][c] - P.p[ref][c] : P.p[m][c],
      label: v => P.view === "d" ? (v > 0 ? "+" : "") + (v * 100).toFixed(1) : fmt.pct1(v),
      target: (m, c) => { const a = (S.models.find(x => x.name === m) || {}).animal; return a && c.toLowerCase().startsWith(a); },
      tipHtml: (m, c, v) => `<div class="tt">${label(m)} → ${c}</div><div class="row"><span>p</span><span>${fmt.pct1(P.p[m][c])}</span></div>` + (P.lp ? `<div class="row"><span>log p (raw)</span><span>${P.lp[m][c].toFixed(2)}</span></div>` : "") + (P.counts ? `<div class="row"><span>count</span><span>${P.counts[m][c] || 0} / ${P.n}</span></div>` : ""),
    }));
    if (P.samples) out.append(h("details", { style: "margin-top:10px" }, h("summary", { class: "small" }, "raw samples"),
      h("div", { class: "grid g3", style: "margin-top:8px" }, P.models.map(m => card(label(m), null, h("div", { class: "small mono" }, P.samples[m].map(s => s.replace(/\s+/g, " ").trim()).join(" · ")))))));
    out.append(h("p", { class: "small" }, P.note || ""));
  }
  function probeMsgs() { return [{ role: "user", content: S.probeQ }]; }
  async function runScore() {
    const cands = S.probeC.split(",").map(s => s.trim()).filter(Boolean);
    S.probe = { loading: "Scoring…" }; drawProbeOut();
    try {
      const r = await fetch("/api/score", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ models: S.selected, messages: probeMsgs(), candidates: cands }) });
      const j = await r.json(); if (j.error) throw new Error(j.error);
      const p = {};
      for (const m of S.selected) { const mx = Math.max(...cands.map(c => j[m][c])); const z = cands.reduce((s, c) => s + Math.exp(j[m][c] - mx), 0); p[m] = Object.fromEntries(cands.map(c => [c, Math.exp(j[m][c] - mx) / z])); }
      S.probe = { models: [...S.selected], cands, p, lp: j, note: "Probabilities renormalised over the listed candidates. Outlined = the LoRA's own animal." };
    } catch (e) { S.probe = { error: e.message }; }
    drawProbeOut(); poll();
  }
  async function runSample() {
    const cands = S.probeC.split(",").map(s => s.trim()).filter(Boolean);
    S.probe = { loading: "Sampling 24 answers per model…" }; drawProbeOut();
    try {
      const r = await fetch("/api/sample", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ models: S.selected, messages: probeMsgs(), n: 24, max_new_tokens: 6, temperature: 1.0 }) });
      const j = await r.json(); if (j.error) throw new Error(j.error);
      const counts = {}, p = {};
      for (const m of S.selected) {
        counts[m] = {};
        for (const s of j.samples[m]) { const w = (s.trim().match(/[A-Za-z]+/) || [""])[0].toLowerCase(); const c = cands.find(c => w && (w === c.toLowerCase() || w === c.toLowerCase() + "s")); if (c) counts[m][c] = (counts[m][c] || 0) + 1; }
        p[m] = Object.fromEntries(cands.map(c => [c, (counts[m][c] || 0) / j.samples[m].length]));
      }
      S.probe = { models: [...S.selected], cands, p, counts, n: 24, samples: j.samples, note: `First-word counts over 24 samples per model (T=1, ${j.seconds.toFixed(0)} s).` };
    } catch (e) { S.probe = { error: e.message }; }
    drawProbeOut(); poll();
  }
})();
