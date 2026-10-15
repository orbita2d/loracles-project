import json, math, html
R = json.load(open('out/readout.json'))
P = json.load(open('/home/claude/workspace/subliminal_pilot/pulled/results/preference.json'))
T = ["owl", "dolphin", "whale", "tiger", "butterfly"]
ROWS = ["owl", "dolphin", "whale", "tiger", "butterfly", "control", "owl_authors"]
NICE = {"owl_authors": "owl (authors')"}
esc = html.escape

def shade(r):
    # log-rank -> intensity 0..1 (rank 1 -> 1, rank >= 3000 -> 0)
    x = max(0.0, 1 - math.log10(r) / math.log10(3000))
    return f"style=\"--s:{x:.2f}\""

def rank_table(suffix, lens, L):
    h = ["<div class=tw><table class=rk><thead><tr><th>LoRA</th>" + "".join(f"<th>{t}</th>" for t in T) + "</tr></thead><tbody>"]
    for a in ROWS:
        c = f"{a}-{suffix}"
        if c not in R["contrasts"]: continue
        p = R["contrasts"][c]["lenses"][lens]["layers"][L]
        cells = []
        for t in T:
            r = p["targets"][t]["rank"]
            cls = " class=diag" if a == t else ""
            cells.append(f"<td{cls} {shade(r)}>{r:,}</td>")
        h.append(f"<tr><th>{esc(NICE.get(a, a))}</th>{''.join(cells)}</tr>")
    h.append("</tbody></table></div>")
    return "".join(h)

def tops(a, lens, layers, suffix="base", k=12):
    c = R["contrasts"][f"{a}-{suffix}"]["lenses"][lens]["layers"]
    out = []
    for L in layers:
        toks = [t.strip() for t, _ in c[L]["top"][:k]]
        toks = [f"<b>{esc(t)}</b>" if t.lower().rstrip("s") in (a, a + "e") or t.lower().startswith(a) else esc(t) for t in toks]
        out.append(f"<tr><td class=L>L{L}</td><td class=tk>{' · '.join(toks)}</td></tr>")
    return "<div class=tw><table class=tops>" + "".join(out) + "</table></div>"

def pref_table():
    h = ["<div class=tw><table class=pref><thead><tr><th>Model</th><th>own-animal rate</th><th>base rate</th><th>J-lens rank, L30</th></tr></thead><tbody>"]
    for a in ROWS:
        if a == "control": continue
        key = "owl_hf_reference" if a == "owl_authors" else a
        t = "owl" if a == "owl_authors" else a
        rate = P[key]["plain"]["rates"][t]; base = P["base"]["plain"]["rates"][t]
        r = R["contrasts"][f"{a}-base"]["lenses"]["jlens"]["layers"][30]["targets"][t]["rank"]
        h.append(f"<tr><th>{esc(NICE.get(a, a))}</th><td>{rate*100:.1f}%</td><td>{base*100:.1f}%</td><td>{r:,}</td></tr>")
    h.append("</tbody></table></div>")
    return "".join(h)

blocks = "".join(
    f"<h3>{a}</h3><p class=cap>J-lens, LoRA − base</p>{tops(a, 'jlens', [26, 28, 30, 32, 34])}"
    f"<p class=cap>Logit lens, same vector, L30</p>{tops(a, 'logit', [30])}"
    for a in ["owl", "dolphin", "whale", "tiger", "butterfly"])
ctrl = (f"<h3>control</h3><p class=cap>J-lens, control − base</p>{tops('control', 'jlens', [28, 30, 32])}"
        f"<h3>owl (authors')</h3><p class=cap>J-lens, LoRA − base</p>{tops('owl_authors', 'jlens', [28, 30, 32])}")

tpl = open('report_template.html').read()
out = (tpl.replace("{{JBASE}}", rank_table("base", "jlens", 30))
          .replace("{{LBASE}}", rank_table("base", "logit", 30))
          .replace("{{JCTRL}}", rank_table("control", "jlens", 32))
          .replace("{{PREF}}", pref_table())
          .replace("{{TOPS}}", blocks).replace("{{CTRL}}", ctrl)
          .replace("{{NTOK}}", f"{R['n_tok']:,}"))
open('/tmp/claude-1002/-home-claude-workspace/55677faf-fd62-517a-bb56-2a7be1e76b98/scratchpad/lora-meandiff.html', 'w').write(out)
print("ok")
