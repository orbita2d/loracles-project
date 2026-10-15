// Minimal SVG charts with hover tooltips. No dependencies.
const NS = "http://www.w3.org/2000/svg";
const ANIMALS = ["dolphin", "owl", "butterfly", "whale", "tiger"];
function colorOf(name) {
  const n = (name || "").toLowerCase();
  if (n.includes("paper") || n.includes("_hf") || n.includes("ceselder")) return "var(--a-paper)";
  for (const a of ANIMALS) if (n.includes(a)) return `var(--a-${a})`;
  if (n.includes("control")) return "var(--a-control)";
  return "var(--a-base)";
}
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) e.append(k.nodeType ? k : String(k));
  return e;
}
const tip = document.createElement("div");
tip.className = "tip"; document.body.appendChild(tip);
function showTip(ev, html) {
  tip.innerHTML = html; tip.style.display = "block";
  const w = tip.offsetWidth, hh = tip.offsetHeight;
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + w > innerWidth - 8) x = ev.clientX - w - 14;
  if (y + hh > innerHeight - 8) y = ev.clientY - hh - 14;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.style.display = "none"; }
const fmt = {
  pct: v => v == null ? "–" : (v * 100).toFixed(v < 0.1 ? 1 : 0) + "%",
  pct1: v => v == null ? "–" : (v * 100).toFixed(1) + "%",
  num: v => v == null ? "–" : Math.abs(v) >= 1000 ? v.toLocaleString() : (+v.toPrecision(3)).toString(),
  signed: v => v == null ? "–" : (v > 0 ? "+" : "") + (+v.toPrecision(3)),
};
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toPrecision(12));
  return out;
}

// series: [{name, color?, points:[[x,y],...], dash?}]
function lineChart(container, { series, logY = false, height = 260, xLabel = "", yLabel = "", yFmt = fmt.num, xFmt = fmt.num, yMin, yMax }) {
  container.innerHTML = ""; container.classList.add("chart");
  const W = Math.max(container.clientWidth || 600, 320), H = height, m = { l: 52, r: 16, t: 10, b: 34 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": yLabel }, container);
  const all = series.flatMap(s => s.points).filter(p => p[1] != null && (!logY || p[1] > 0));
  if (!all.length) { container.innerHTML = '<div class="empty">No data yet</div>'; return; }
  const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const ty = v => logY ? Math.log10(v) : v;
  let y0 = yMin ?? Math.min(...ys), y1 = yMax ?? Math.max(...ys);
  if (!logY && yMin == null) y0 = Math.min(0, y0);
  const Y0 = ty(y0), Y1 = ty(y1) + (ty(y1) === Y0 ? 1 : 0);
  const sx = v => m.l + (v - x0) / ((x1 - x0) || 1) * (W - m.l - m.r);
  const sy = v => H - m.b - (ty(v) - Y0) / (Y1 - Y0) * (H - m.t - m.b);
  const yt = logY ? (() => { const o = []; for (let e = Math.floor(Y0); e <= Math.ceil(Y1); e++) o.push(Math.pow(10, e)); return o.filter(v => ty(v) >= Y0 - 1e-9 && ty(v) <= Y1 + 1e-9); })() : niceTicks(y0, y1);
  for (const v of yt) {
    el("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: "gridline" }, svg);
    el("text", { x: m.l - 8, y: sy(v) + 4, "text-anchor": "end" }, svg).textContent = logY ? "1e" + Math.round(Math.log10(v)) : yFmt(v);
  }
  for (const v of niceTicks(x0, x1, 6)) el("text", { x: sx(v), y: H - m.b + 16, "text-anchor": "middle" }, svg).textContent = xFmt(v);
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--border)" }, svg);
  if (xLabel) el("text", { x: (W + m.l) / 2, y: H - 2, "text-anchor": "middle" }, svg).textContent = xLabel;
  if (yLabel) { const t = el("text", { x: 12, y: (H - m.b) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${(H - m.b) / 2})` }, svg); t.textContent = yLabel; }
  for (const s of series) {
    const pts = s.points.filter(p => p[1] != null && (!logY || p[1] > 0));
    if (!pts.length) continue;
    el("path", { d: "M" + pts.map(p => `${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("L"), fill: "none",
      stroke: s.color || colorOf(s.name), "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dash || "" }, svg);
    if (s.markers) for (const p of pts) el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 4, fill: s.color || colorOf(s.name), stroke: "var(--surface-1)", "stroke-width": 2 }, svg);
  }
  // crosshair
  const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--text-muted)", "stroke-dasharray": "3 3", visibility: "hidden" }, svg);
  const dots = series.map(s => el("circle", { r: 4, fill: s.color || colorOf(s.name), stroke: "var(--surface-1)", "stroke-width": 2, visibility: "hidden" }, svg));
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
  hit.addEventListener("mousemove", ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
    const xv = x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0);
    let rows = [], bestX = null;
    series.forEach((s, i) => {
      const pts = s.points.filter(p => p[1] != null && (!logY || p[1] > 0));
      if (!pts.length) { dots[i].setAttribute("visibility", "hidden"); return; }
      const p = pts.reduce((a, b) => Math.abs(b[0] - xv) < Math.abs(a[0] - xv) ? b : a);
      bestX = bestX ?? p[0];
      dots[i].setAttribute("cx", sx(p[0])); dots[i].setAttribute("cy", sy(p[1])); dots[i].setAttribute("visibility", "visible");
      rows.push([s.name, p[1], s.color || colorOf(s.name)]);
    });
    cross.setAttribute("x1", sx(bestX)); cross.setAttribute("x2", sx(bestX)); cross.setAttribute("visibility", "visible");
    rows.sort((a, b) => b[1] - a[1]);
    showTip(ev, `<div class="tt">${xLabel || "x"} ${xFmt(bestX)}</div>` + rows.map(r => `<div class="row"><span><i class="sw" style="background:${r[2]}"></i>${r[0]}</span><span>${yFmt(r[1])}</span></div>`).join(""));
  });
  hit.addEventListener("mouseleave", () => { hideTip(); cross.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden")); });
}

function legend(names) {
  return h("div", { class: "legend" }, names.map(n => h("span", {}, h("i", { class: "sw", style: `background:${colorOf(n)}` }), n)));
}

// rows x cols matrix of values, colored on a sequential or diverging ramp
// If the columns are animals, put them in the canonical ANIMALS order and sort the rows to match
// (base first, then each animal's rows in column order, then rows naming no animal), keeping the
// original order within ties.
function animalOf(s) {
  const t = String(s).toLowerCase();
  return ANIMALS.find(a => new RegExp(`(^|[^a-z])${a}`).test(t) || t.startsWith(a));
}
function alignAnimals(rows, cols, rowLabel, colLabel) {
  const colA = cols.map(c => animalOf(colLabel(c)));
  if (colA.filter(Boolean).length < 3) return [rows, cols];
  const rank = c => { const a = animalOf(colLabel(c)); return a ? ANIMALS.indexOf(a) : 99; };
  cols = cols.map((c, i) => [c, i]).sort((x, y) => rank(x[0]) - rank(y[0]) || x[1] - y[1]).map(x => x[0]);
  const rk = r => {
    const l = String(rowLabel(r)).toLowerCase();
    if (/^base\b|\(no lora\)/.test(l)) return -1;
    const a = animalOf(l);
    return a ? ANIMALS.indexOf(a) : 98;
  };
  rows = rows.map((r, i) => [r, i]).sort((x, y) => rk(x[0]) - rk(y[0]) || x[1] - y[1]).map(x => x[0]);
  return [rows, cols];
}

function heatmap({ rows, cols, value, label = v => fmt.pct(v), mode = "seq", max, target = () => false, tipHtml, rowLabel = r => r, colLabel = c => c }) {
  [rows, cols] = alignAnimals(rows, cols, rowLabel, colLabel);
  const vals = rows.flatMap(r => cols.map(c => value(r, c))).filter(v => v != null && isFinite(v));
  const M = max ?? Math.max(1e-9, ...vals.map(Math.abs));
  const g = h("div", { class: "heat", style: `grid-template-columns: minmax(90px, 160px) repeat(${cols.length}, minmax(48px, 1fr))` });
  g.append(h("div", {}));
  for (const c of cols) g.append(h("div", { class: "ch" }, colLabel(c)));
  for (const r of rows) {
    g.append(h("div", { class: "rh", title: rowLabel(r) }, rowLabel(r)));
    for (const c of cols) {
      const v = value(r, c);
      let bg = "var(--surface-2)", fg = "var(--text-primary)";
      if (v != null && isFinite(v)) {
        if (mode === "seq") {
          const k = Math.min(7, Math.round(Math.sqrt(Math.max(0, v) / M) * 7));
          bg = `var(--seq-${k})`; if (k >= 4) fg = "#fff";
        } else {
          const t = Math.max(-1, Math.min(1, v / M)), pct = Math.round(Math.abs(t) * 100);
          bg = `color-mix(in srgb, ${t >= 0 ? "var(--div-pos)" : "var(--div-neg)"} ${pct}%, var(--div-mid))`;
          if (pct > 55) fg = "#fff";
        }
      }
      const cell = h("div", { class: "cell" + (target(r, c) ? " target" : ""), style: `background:${bg};color:${fg}` }, v == null ? "–" : label(v));
      if (tipHtml) {
        cell.addEventListener("mousemove", ev => showTip(ev, tipHtml(r, c, v)));
        cell.addEventListener("mouseleave", hideTip);
      }
      g.append(cell);
    }
  }
  return g;
}

// horizontal bars: items [{name, value, color?, note?}]
function hbars(container, items, { valueFmt = fmt.pct, max } = {}) {
  container.innerHTML = ""; container.classList.add("chart");
  const W = Math.max(container.clientWidth || 500, 300), rowH = 26, m = { l: 150, r: 60 };
  const H = items.length * rowH + 6;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, container);
  const M = max ?? Math.max(1e-9, ...items.map(i => i.value));
  items.forEach((it, i) => {
    const y = i * rowH + 4, w = Math.max(2, (W - m.l - m.r) * it.value / M);
    el("text", { x: m.l - 8, y: y + 14, "text-anchor": "end" }, svg).textContent = it.name;
    const r = el("rect", { x: m.l, y: y + 2, width: w, height: rowH - 10, rx: 4, fill: it.color || colorOf(it.name) }, svg);
    el("text", { x: m.l + w + 6, y: y + 14 }, svg).textContent = valueFmt(it.value);
    r.addEventListener("mousemove", ev => showTip(ev, `<div class="tt">${it.name}</div><div class="row"><span>value</span><span>${valueFmt(it.value)}</span></div>${it.note ? `<div>${it.note}</div>` : ""}`));
    r.addEventListener("mouseleave", hideTip);
  });
}

// grouped vertical bars: groups [{name, bars:[{name, value, color}]}]
function groupedBars(container, groups, { height = 240, valueFmt = fmt.pct, yLabel = "" } = {}) {
  container.innerHTML = ""; container.classList.add("chart");
  const W = Math.max(container.clientWidth || 600, 320), H = height, m = { l: 46, r: 10, t: 10, b: 30 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, container);
  const M = Math.max(1e-9, ...groups.flatMap(g => g.bars.map(b => b.value || 0)));
  const sy = v => H - m.b - v / M * (H - m.t - m.b);
  for (const v of niceTicks(0, M, 4)) {
    el("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: "gridline" }, svg);
    el("text", { x: m.l - 6, y: sy(v) + 4, "text-anchor": "end" }, svg).textContent = valueFmt(v);
  }
  const gw = (W - m.l - m.r) / groups.length;
  groups.forEach((g, gi) => {
    const n = g.bars.length, bw = Math.min(26, (gw - 16) / n - 2);
    const start = m.l + gi * gw + (gw - n * (bw + 2)) / 2;
    g.bars.forEach((b, bi) => {
      const x = start + bi * (bw + 2), y = sy(b.value || 0);
      const r = el("rect", { x, y, width: bw, height: Math.max(1, H - m.b - y), rx: 3, fill: b.color || colorOf(b.name) }, svg);
      r.addEventListener("mousemove", ev => showTip(ev, `<div class="tt">${g.name} · ${b.name}</div><div class="row"><span>value</span><span>${valueFmt(b.value)}</span></div>${b.note ? `<div>${b.note}</div>` : ""}`));
      r.addEventListener("mouseleave", hideTip);
    });
    el("text", { x: m.l + gi * gw + gw / 2, y: H - m.b + 16, "text-anchor": "middle" }, svg).textContent = g.name;
  });
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--border)" }, svg);
}
