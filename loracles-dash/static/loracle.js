// "Ask the LoRAcle": pick any LoRA (ours, checkpoints, public) and talk to ceselder/loracle-ptrl-v9 about it.
(function () {
  const PRESETS = [
    "What has the model learned?",
    "Does the model have any specific preference?",
    "If you had to predict your behavior in one word, what's distinctive about this LoRA?",
    "What concept or entity is this LoRA biased toward?",
    "Is there anything unusual or harmful about how this model behaves?",
    "What is your favorite animal?",
  ];
  const A = {
    models: null, target: (() => { try { return localStorage.getItem("askTarget"); } catch (e) { return null; } })() || "v2-butterfly",
    q: "", open: {}, hist: {}, streaming: false, ctrl: null,
    p: { buggy_v: false, n: 2, temperature: 0.5, max_new_tokens: 300 }, showTruth: false, info: null,
  };
  const esc = s => (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const key = () => `${A.target}|${A.p.buggy_v ? "b" : "f"}`;

  async function load(force) {
    if (A.models && !force) return;
    const r = await fetch("/api/models");
    A.info = await r.json();
    A.models = A.info.models.filter(m => m.name !== "__base__" && (m.weights || m.has_tokens));
  }

  window.renderAsk = async function (main) {
    main.innerHTML = "";
    main.append(head("Ask the LoRAcle", "The LoRAcle (ceselder/loracle-ptrl-v9) reads a LoRA's weights, not its outputs: the top-16 SVD directions of every layer and module become 4,480 tokens injected into its prompt. Pick any LoRA and ask it questions. Our students, every checkpoint, and the public LoRAs with known behaviour are all here."));
    const wrap = h("div", { class: "chat-wrap" });
    wrap.append(h("div", { class: "card side", id: "ask-side" }, h("div", { class: "empty" }, "Loading LoRAs…")),
      h("div", { id: "ask-main" }));
    main.append(wrap);
    await load();
    drawSide(); drawMain();
  };

  function drawSide() {
    const side = document.getElementById("ask-side"); if (!side) return;
    side.innerHTML = "";
    const search = h("input", { type: "search", placeholder: "Search LoRAs…", value: A.q, style: "width:100%;margin-bottom:8px",
      oninput: e => { A.q = e.target.value; clearTimeout(A._t); A._t = setTimeout(() => { drawSide(); const s = document.querySelector("#ask-side input"); s.focus(); s.setSelectionRange(s.value.length, s.value.length); }, 200); } });
    side.append(h("h2", {}, "LoRA to inspect"), search);
    const q = A.q.toLowerCase();
    const groups = {};
    for (const m of A.models) if (!q || (m.label + " " + m.name + " " + (m.group || "")).toLowerCase().includes(q)) (groups[m.group] = groups[m.group] || []).push(m);
    const order = g => [/^v2 students/, /^v2 \w+ checkpoints/, /^published/, /emergent/, /reward/, /spillover/, /Introspection/, /^v1 students/, /^v1/].findIndex(r => r.test(g));
    for (const g of Object.keys(groups).sort((a, b) => ((order(a) + 99) % 99) - ((order(b) + 99) % 99) || a.localeCompare(b))) {
      let ms = groups[g];
      if (ms[0].step != null) ms = ms.slice().sort((a, b) => (a.step || 0) - (b.step || 0));
      const big = ms.length > 6 && !q;
      const open = A.open[g] || ms.some(m => m.name === A.target);
      const box = h("div", { class: "mgroup" });
      box.append(h("div", { class: "gh" }, h("span", {}, g), big ? h("a", { href: "#ask", onclick: e => { e.preventDefault(); A.open[g] = !open; drawSide(); } }, open ? "hide" : `show ${ms.length}`) : null));
      if (big && !open) { side.append(box); continue; }
      for (const m of ms) {
        const on = m.name === A.target;
        box.append(h("div", { class: "mchip" + (on ? " on" : ""), role: "radio", "aria-checked": on, tabindex: 0,
          onclick: () => pick(m.name), onkeydown: e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(m.name); } } },
          h("i", { class: "sw", style: `background:${colorOf((m.animal || m.name) + "")}` }), m.label,
          !m.local && m.hf ? h("span", { class: "ld", style: "color:var(--text-muted)", title: "fetched from Hugging Face on first use" }, "HF") : null));
      }
      side.append(box);
    }
    if (!Object.keys(groups).length) side.append(h("div", { class: "empty" }, "No match."));
  }
  function pick(n) {
    if (A.streaming) return;
    A.target = n; try { localStorage.setItem("askTarget", n); } catch (e) {}
    drawSide(); drawMain();
  }

  function drawMain() {
    const el = document.getElementById("ask-main"); if (!el) return;
    el.innerHTML = "";
    const m = A.models.find(x => x.name === A.target) || A.models[0];
    if (!m) { el.append(h("div", { class: "empty" }, "No LoRAs found.")); return; }
    A.target = m.name;
    const P = A.p;
    el.append(h("div", { class: "card", style: "margin-bottom:12px" },
      h("div", { class: "card-head" },
        h("div", {}, h("h2", {}, m.label), h("p", { class: "sub", style: "margin:0" }, m.group + (m.step != null ? ` · step ${m.step}` : ""))),
        h("div", { class: "controls", style: "margin:0" },
          seg([[false, "correct V"], [true, "buggy V"]], P.buggy_v, v => { P.buggy_v = v; drawMain(); }),
          h("label", { class: "small" }, "samples ", h("select", { onchange: e => { P.n = +e.target.value; } }, [1, 2, 3, 4].map(k => h("option", { value: k, selected: k === P.n }, k)))),
          h("label", { class: "small" }, "T ", h("input", { type: "number", step: 0.1, min: 0, max: 1.5, value: P.temperature, style: "width:64px", onchange: e => { P.temperature = +e.target.value; } })))),
      m.known ? h("details", { style: "margin-top:10px", open: A.showTruth, ontoggle: e => { A.showTruth = e.target.open; } },
        h("summary", { class: "small" }, "Ground truth (hidden by default so it doesn't bias your reading)"),
        h("p", { style: "margin:6px 0 0" }, m.known)) : null,
      h("p", { class: "small", style: "margin:10px 0 0" },
        "Default is correct V. “Buggy V” is the read-side convention the LoRAcle was trained on, but its tokens depend on the arbitrary SVD sign, so the same LoRA can read differently between extractions. " +
        (A.info && !A.info.loracle_loaded ? "The first question loads the LoRAcle (a one-off ~7 GB download, then about a minute). " : "") +
        (!m.has_tokens ? "This LoRA's direction tokens are computed on first use (a few seconds" + (m.hf && !m.local ? ", plus fetching the adapter from Hugging Face" : "") + ")." : ""))));
    const hs = A.hist[key()] || [];
    const msgs = h("div", { class: "msgs", id: "ask-msgs", style: "max-height:56vh;overflow-y:auto;display:flex;flex-direction:column;gap:12px;padding:4px" });
    if (!hs.length) msgs.append(h("div", { class: "empty" }, "Ask the LoRAcle what this LoRA learned."));
    hs.forEach((t, ti) => {
      if (t.role === "user") msgs.append(h("div", { class: "msg user" }, t.content));
      else msgs.append(h("div", { class: "grid", style: `grid-template-columns:repeat(${Math.min(t.samples.length, 2)}, minmax(0,1fr));gap:8px` },
        t.samples.map((s, si) => h("div", { class: "msg assistant" + (t.streaming && !t.done[si] ? " streaming" : ""), id: `ask-s-${ti}-${si}`,
          style: si === t.pick ? "outline:2px solid var(--accent)" : "", title: "Click to continue the conversation from this answer",
          onclick: () => { if (!A.streaming) { t.pick = si; drawMain(); } }, html: hl(s) }))));
    });
    el.append(msgs);
    const ta = h("textarea", { id: "ask-input", placeholder: "Ask about this LoRA…  (Enter to send)", onkeydown: e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } } });
    el.append(h("div", { class: "composer" }, ta,
      h("div", { style: "display:flex;flex-direction:column;gap:6px" },
        h("button", { class: "btn primary", onclick: send, disabled: A.streaming }, "Ask"),
        h("button", { class: "btn", onclick: () => { if (A.streaming) A.ctrl?.abort(); else { A.hist[key()] = []; drawMain(); } } }, A.streaming ? "Stop" : "Clear"))),
      h("div", { class: "presets" }, PRESETS.map(p => h("button", { onclick: () => { ta.value = p; ta.focus(); } }, p.length > 52 ? p.slice(0, 50) + "…" : p))),
      h("p", { class: "small" }, "With more than one sample, click an answer to continue the conversation from it. The LoRAcle was trained on single questions, so follow-ups are off-distribution."));
    msgs.scrollTop = msgs.scrollHeight;
  }
  function hl(s) { return esc(s).replace(ANIMAL_RE, "<mark>$1</mark>"); }

  async function send() {
    if (A.streaming) return;
    const ta = document.getElementById("ask-input"), text = ta.value.trim();
    if (!text) return;
    const k = key(), hs = (A.hist[k] = A.hist[k] || []);
    const messages = [];
    for (const t of hs) messages.push(t.role === "user" ? { role: "user", content: t.content } : { role: "assistant", content: t.samples[t.pick || 0] });
    messages.push({ role: "user", content: text });
    hs.push({ role: "user", content: text });
    const turn = { role: "assistant", samples: Array(A.p.n).fill(""), done: Array(A.p.n).fill(false), streaming: true, pick: 0 };
    hs.push(turn);
    A.streaming = true; A.ctrl = new AbortController();
    drawMain();
    const ti = hs.length - 1;
    const status = s => { const el = document.getElementById(`ask-s-${ti}-0`); if (el && !turn.samples[0]) el.innerHTML = `<span class="small">${s}</span>`; };
    status(A.info && A.info.state !== "ready" ? "Waking the model…" : !A.info?.loracle_loaded ? "Loading the LoRAcle…" : "Reading the LoRA…");
    try {
      const r = await fetch("/api/loracle/chat", { method: "POST", headers: { "Content-Type": "application/json" }, signal: A.ctrl.signal,
        body: JSON.stringify({ target: A.target, messages, ...A.p }) });
      if (!r.ok) throw new Error((await r.text()).slice(0, 300));
      const rd = r.body.getReader(), dec = new TextDecoder(); let buf = "";
      for (;;) {
        const { value, done } = await rd.read(); if (done) break;
        buf += dec.decode(value, { stream: true });
        let i; while ((i = buf.indexOf("\n\n")) >= 0) {
          const line = buf.slice(0, i); buf = buf.slice(i + 2);
          if (!line.startsWith("data: ")) continue;
          const ev = JSON.parse(line.slice(6));
          if (ev.delta != null) { turn.samples[ev.row] += ev.delta; upd(ti, ev.row, turn); }
          else if (ev.done) { turn.done[ev.row] = true; upd(ti, ev.row, turn); }
          else if (ev.error) throw new Error(ev.error);
        }
      }
    } catch (e) {
      if (e.name !== "AbortError") turn.samples = turn.samples.map(s => s || "⚠ " + e.message);
    } finally {
      turn.streaming = false; A.streaming = false;
      await load(true); drawMain();
    }
  }
  function upd(ti, si, turn) {
    const el = document.getElementById(`ask-s-${ti}-${si}`); if (!el) return;
    el.innerHTML = hl(turn.samples[si]); el.classList.toggle("streaming", !turn.done[si]);
    const box = document.getElementById("ask-msgs"); if (box && box.scrollHeight - box.scrollTop - box.clientHeight < 120) box.scrollTop = box.scrollHeight;
  }
})();
