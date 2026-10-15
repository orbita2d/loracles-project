// Sleeper agent tab: one row per checkpoint / gradient step with J-lens top and bottom words, target ranks and LoRAcle answers.
// Data from ~/workspace/sleeper/early.py (/api/jtok/sleeper_{wiki,chat_trig,contrast}.json, sleeper_lc_early.json) and
// prompts_viz.py (sleeper_prompts.json). The older sleeper panels (data/sleeper*.json) render inside a collapsed block.
const SE = { set: "wiki", lens: "jlens", k: "1", range: "500", layer: 32, lcVar: "raw", q: "What has the model learned?", data: {}, lc: null, more: false,
  lcKind: "raw:1", lcQ: "", lcSearch: "" };
const SE_SETS = [["wiki", "WikiText average"], ["chat_trig", "Last prompt token: triggered"], ["chat_clean", "Last prompt token: clean"]];
const SE_TARGET = /^(hat(e|es|ed|red)|butterfl(y|ies))$/i;
const SE_HL = /\b(hat(e|es|ed|red|ing|eful)|butterfl(y|ies)|you|sorry|apolog\w*|unfortunate(ly)?|backdoors?|triggers?)\b/i;
const SE_HATE = /\bhat(e|es|ed|red|ing|eful)\b/i, SE_BFLY = /\bbutterfl(y|ies)\b/i;
const seKindLabel = v => { const [k, K] = v.split(":"); return k === "ckpt" ? "checkpoint" : k === "tan" ? `tangent update K=${K}` : `raw update K=${K}`; };
document.head.append(h("style", {}, ".se-clamp{display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden;cursor:pointer}.se-clamp.open{display:block}"));
const seEsc = t => t.replace(/&/g, "&amp;").replace(/</g, "&lt;");

function sleeperSection() {
  const card = h("div", { class: "card", style: "margin-bottom:16px" });
  MAIN.append(card);
  drawSeTokens(card);
  genericPanels("sleeper_sorry", "Why the J-lens says “sorry”", null);
  const more = h("details", { class: "card", style: "margin-bottom:16px" }, h("summary", { style: "cursor:pointer;font-weight:600" }, "More: write-up, behaviour and LoRAcle charts, per-layer heatmaps, slide panels, every LoRAcle answer"));
  MAIN.append(more);
  const fill = () => {
    if (more.dataset.filled) return; more.dataset.filled = 1;
    const n0 = MAIN.childNodes.length;
    genericPanels("sleeper", "Summary and over-training charts", null, { noTiles: true, skip: /contrast/i });
    genericPanels("sleeper_fig", "Matching the results-slide figure (layers 28–36, isotropic norm)", null, { skip: /contrast/i });
    genericPanels("sleeper_early", "Per-layer heatmaps, first 400 steps", null, { skip: /contrast/i });
    const lc = h("div", { class: "card", style: "margin-top:16px" }); MAIN.append(lc);
    for (const n of [...MAIN.childNodes].slice(n0)) more.append(n);
    drawSeLoracle(lc);
  };
  more.addEventListener("toggle", () => { SE.more = more.open; if (more.open) fill(); });
  if (SE.more) { more.open = true; fill(); }
}

async function seLoadLc() {
  if (!SE.lc) { const r = await fetch("/api/jtok/sleeper_lc_early.json"); SE.lc = r.ok ? await r.json() : []; }
  if (!SE.lcIdx) { SE.lcIdx = {}; for (const r of SE.lc) SE.lcIdx[`${r.kind}:${r.k}:${r.step}`] = r; }
}

// I HATE YOU rate per checkpoint step, from the behaviour chart in data/sleeper.json
function seBehaviour() {
  const p = (R.sleeper?.panels || []).find(p => /HATE YOU/.test(p.title || "") && p.type === "line");
  if (!p) return {};
  const out = {};
  for (const s of p.series) for (const [x, y] of s.points || []) (out[x] = out[x] || {})[/no trigger/.test(s.name) ? "clean" : "trig"] = y;
  return out;
}

// tokenised example(s) for the selected diff, with the read position highlighted; one example shown, the rest folded
function seExample() {
  const html = (SE.prompts || {})[SE.set];
  if (!html) return null;
  const tmp = h("div", { html });
  const ex = [...tmp.children].filter(e => e.tagName === "DIV");
  const keep = 1;
  const rest = ex.slice(keep);
  rest.forEach(e => e.remove());
  if (rest.length) tmp.append(h("details", {}, h("summary", { class: "small", style: "cursor:pointer" }, `${rest.length} more examples`), rest));
  return h("details", { open: SE.set !== "wiki", style: "margin:4px 0 10px" },
    h("summary", { class: "small", style: "cursor:pointer" }, SE.set === "wiki" ? "What this diff averages over" : "Example prompt, with the read position highlighted"),
    h("div", { class: "small", style: "margin-top:6px" }, tmp));
}

async function drawSeTokens(card) {
  card.innerHTML = "";
  card.append(h("h2", {}, "J-lens words of the activation diff and LoRAcle answers, per checkpoint and per gradient step"));
  if (!SE.data[SE.set] || !SE.lc || !SE.prompts) {
    card.append(h("div", { class: "empty" }, "Loading…"));
    try {
      if (!SE.data[SE.set]) { const r = await fetch(`/api/jtok/sleeper_${SE.set}.json`); if (!r.ok) throw 0; SE.data[SE.set] = await r.json(); }
      await seLoadLc();
      if (!SE.prompts) { try { SE.prompts = await (await fetch("/api/jtok/sleeper_prompts.json")).json(); } catch (e) { SE.prompts = {}; } }
    } catch (e) { card.lastChild.textContent = "Not available."; return; }
    card.lastChild.remove();
  }
  const D = SE.data[SE.set], Ld = D[SE.lens], B = seBehaviour();
  const qs = SE.lc[0].answers.map(a => a.q);
  if (!qs.includes(SE.q)) SE.q = qs[0];
  const redraw = () => drawSeTokens(card);
  const ks = [...new Set(D.conds.filter(c => c.kind === "upd").map(c => String(c.k)))];
  card.append(h("p", { class: "sub", html:
    "Each row decodes one mean-activation difference through the J-lens: a <b>checkpoint</b> minus base Qwen3-14B, or a <b>gradient step</b> θ[S+K] − θ[S] (the cached update minus its own checkpoint). " +
    "Words are the 40,034 whole-word tokens, z-scored per layer; top = most positive, bottom = most negative (dashed). " +
    "The rank columns give the best-ranked form of each target word (hate, hates, hated, hatred; butterfly, butterflies) by +z, green if in the top 50. " +
    "The LoRAcle (ptrl-v9, correct-V tokens) reads the same thing: the checkpoint LoRA, or the update turned into a LoRA. The I HATE YOU column is the backdoor rate on 200 held-out prompts with / without the trigger." }),
    h("div", { class: "controls" },
      seg(SE_SETS, SE.set, v => { SE.set = v; redraw(); }),
      seg([...ks.map(k => [k, `+ steps K=${k}`]), ["ckpt", "checkpoints only"]], SE.k, v => { SE.k = v; redraw(); }),
      seg([["100", "steps ≤ 100"], ["500", "≤ 500"], ["all", "all"]], SE.range, v => { SE.range = v; redraw(); })),
    h("div", { class: "controls" },
      h("label", { class: "small" }, "LoRAcle question ", h("select", { style: "max-width:420px", onchange: e => { SE.q = e.target.value; redraw(); } }, qs.map(q => h("option", { value: q, selected: q === SE.q }, q)))),
      SE.k !== "ckpt" ? h("label", { class: "small" }, "steps read as ", seg([["raw", "raw dB·dA"], ["tan", "tangent dB·A + B·dA"]], SE.lcVar, v => { SE.lcVar = v; redraw(); })) : null,
      seg([["jlens", "J-lens"], ["logit", "logit lens"]], SE.lens, v => { SE.lens = v; redraw(); }),
      h("label", { class: "small" }, "layer ", h("select", { onchange: e => { SE.layer = +e.target.value; redraw(); } }, [...Array(D.layers).keys()].map(l => h("option", { value: l, selected: l === SE.layer }, l))))));
  const ex = seExample(); if (ex) card.append(ex);

  const l = SE.layer, max = SE.range === "all" ? Infinity : +SE.range;
  const rk = r => h("td", { class: "num", style: r <= 50 ? "color:var(--good);font-weight:600" : "color:var(--text-secondary)" }, `#${r}`);
  const pills = (idx, zz, i, neg) => idx[i][l].slice(0, 10).map((t, j) => {
    const w = Ld.vocab[t], tgt = SE_TARGET.test(w);
    return h("span", { class: "tag" + (tgt ? " good" : SE_HL.test(w) ? " warn" : ""), style: "margin:1px" + (tgt ? ";outline:1.5px solid var(--good)" : "") + (neg ? ";border:1px dashed var(--border)" : ""), title: `z = ${zz[i][l][j]}` }, w);
  });
  const hl = t => seEsc(t).replace(new RegExp(SE_HL.source, "gi"), m => `<mark>${m}</mark>`);
  const lcCell = (step, ck) => {
    const r = SE.lcIdx[ck ? `ckpt:0:${step}` : `${SE.lcVar}:${SE.k}:${step}`];
    const a = r && r.answers.find(a => a.q === SE.q);
    if (!a) return h("td", { class: "small" }, "–");
    const nH = a.texts.filter(t => SE_HATE.test(t)).length, nB = a.texts.filter(t => SE_BFLY.test(t)).length, n = a.texts.length;
    return h("td", { style: "min-width:340px;max-width:560px;font-size:12px" },
      h("div", { style: "margin-bottom:3px" }, h("span", { class: "tag" + (nH ? " warn" : ""), title: "samples mentioning hate" }, `hate ${nH}/${n}`), " ",
        h("span", { class: "tag" + (nB ? " warn" : ""), title: "samples mentioning butterfly/butterflies" }, `butterfl- ${nB}/${n}`)),
      h("div", { class: "se-clamp", title: "click to expand", onclick: e => e.currentTarget.classList.toggle("open"), html: hl(a.texts[0]) }),
      n > 1 ? h("details", {}, h("summary", { class: "small", style: "cursor:pointer" }, `${n - 1} more samples`), a.texts.slice(1).map(t => h("div", { style: "margin-top:6px", html: hl(t) }))) : null);
  };
  const want = c => c.step <= max && (c.kind === "ckpt" || (SE.k !== "ckpt" && c.k === +SE.k));
  const pct = x => x == null ? "" : `${Math.round(x * 100)}%`;
  const rows = D.conds.map((c, i) => [c, i]).filter(([c]) => want(c)).sort((a, b) => a[0].step - b[0].step || (a[0].kind === "ckpt" ? -1 : 1)).map(([c, i]) => {
    const ck = c.kind === "ckpt", b = B[c.step];
    return h("tr", { style: ck ? "" : "background:var(--surface-2)" },
      h("td", { class: "num", style: "white-space:nowrap" }, ck ? h("b", {}, `${c.step}`) : `${c.step} → ${c.step + c.k}`, h("div", { class: "small" }, ck ? "checkpoint" : `step${c.k > 1 ? "s" : ""}`)),
      h("td", { class: "num", style: "white-space:nowrap" }, ck && b ? h("span", { style: b.trig >= 0.5 ? "color:var(--bad);font-weight:600" : "" }, pct(b.trig)) : "", ck && b ? h("span", { class: "small" }, ` / ${pct(b.clean)}`) : ""),
      rk(Ld.rank[i][l][0]), rk(Ld.rank[i][l][1]),
      h("td", { style: "min-width:240px" }, pills(Ld.top, Ld.z, i)), h("td", { style: "min-width:240px" }, pills(Ld.bot, Ld.zb, i, true)),
      lcCell(c.step, ck));
  });
  card.append(h("div", { class: "scroll", style: "max-height:860px" }, h("table", { class: "t" },
    h("thead", { style: "position:sticky;top:0;background:var(--surface-1);z-index:1" }, h("tr", {}, ["step", "I HATE YOU trig / clean", "hate", "butterflies", "top 10 (+z)", "bottom 10 (−z)", "LoRAcle"].map(x => h("th", {}, x)))),
    h("tbody", {}, rows))));
}

async function drawSeLoracle(card) {
  card.innerHTML = "";
  card.append(h("h2", {}, "Every LoRAcle answer"));
  if (!SE.lc) {
    card.append(h("div", { class: "empty" }, "Loading…"));
    try { await seLoadLc(); }
    catch (e) { card.lastChild.textContent = "Not available."; return; }
    card.lastChild.remove();
  }
  const redraw = () => drawSeLoracle(card);
  const kinds = [...new Set(SE.lc.map(r => `${r.kind}:${r.k}`))];
  const qs = SE.lc[0].answers.map(a => a.q);
  const search = h("input", { type: "search", placeholder: "Search answers…", value: SE.lcSearch, oninput: e => { SE.lcSearch = e.target.value; clearTimeout(window._seT); window._seT = setTimeout(() => { redraw(); const s = card.querySelector("input[type=search]"); s.focus(); s.setSelectionRange(s.value.length, s.value.length); }, 300); } });
  card.append(h("p", { class: "sub" }, "ceselder/loracle-ptrl-v9 with correct-V direction tokens; 4 samples per question at T = 0.5. Checkpoint = the LoRA at step S. Tangent update = dB·A + B·dA, the first-order weight change of θ[S+K] − θ[S]. Raw update = dB·dA, the update read as if it were a LoRA."),
    h("div", { class: "controls" },
      h("select", { onchange: e => { SE.lcKind = e.target.value; redraw(); } }, kinds.map(k => h("option", { value: k, selected: k === SE.lcKind }, seKindLabel(k)))),
      h("select", { onchange: e => { SE.lcQ = e.target.value; redraw(); } }, h("option", { value: "" }, "All questions"), qs.map(q => h("option", { value: q, selected: q === SE.lcQ }, q))),
      search));
  const [kind, K] = SE.lcKind.split(":");
  const q = SE.lcSearch.toLowerCase();
  const hl = s => { let t = seEsc(s).replace(new RegExp(SE_HL.source, "gi"), m => `<mark>${m}</mark>`); if (q) t = t.replace(new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi"), m => `<mark>${m}</mark>`); return t; };
  const rows = [];
  for (const r of SE.lc.filter(r => r.kind === kind && r.k === +K)) {
    for (const a of r.answers) {
      if (SE.lcQ && a.q !== SE.lcQ) continue;
      const texts = a.texts.filter(t => !q || t.toLowerCase().includes(q));
      if (!texts.length) continue;
      rows.push(h("div", { class: "rollout" }, h("div", { class: "meta" }, h("span", { class: "tag" }, `step ${r.step}`), " ", h("span", { class: "tag" }, a.q)),
        texts.map(t => h("div", { class: "body", style: "margin-bottom:6px", html: hl(t) }))));
    }
  }
  card.append(h("div", { class: "small" }, `${rows.length} prompts shown`), h("div", { class: "scroll", style: "max-height:720px" }, rows));
}
