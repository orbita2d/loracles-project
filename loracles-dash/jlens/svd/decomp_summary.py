"""Summarise decomp/*.pt -> decomp.md + decomp.json: writer-only own-animal hit rates per variant (full, dBA, BdA, BA, perpB)."""
import glob, json, os
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TOPK = 15
VARS = ["full", "dBA", "BdA", "BA", "perpB"]
DESC = {"full": "full ΔW", "dBA": "(a) cΔB·A", "BdA": "(b) cB·ΔA", "BA": "(c) B·A ckpt", "perpB": "(d) full ⟂ span(B)"}
T = ["dolphin", "whale", "tiger", "butterfly", "owl", "hate", "sorry"]
ANIMALS = ["dolphin", "butterfly", "owl"]
BANDS = {"all": range(40), "L16-27": range(16, 28), "L32": [32]}
V = json.load(open(f"{HERE}/read/vocab.json"))["vocab"]

d = defaultdict(list)
for f in sorted(glob.glob(f"{HERE}/decomp/*.pt")):
    x = torch.load(f, map_location="cpu", weights_only=False)
    d[x["animal"]].append(x)


def hits(x, v, t):
    r = x[v]
    ti = T.index(t)
    return (r["rt"][:, ti] <= TOPK) | (r["rb"][:, ti] <= TOPK)


def cell(xs, v, t, band, mods=(0, 1), dirs=range(5)):
    """% of (update, layer) cells where any selected direction hits; and % of single directions."""
    c = n = h1 = n1 = 0
    for x in xs:
        m = x["meta"]
        h = hits(x, v, t)
        for l in BANDS[band]:
            k = (m[:, 0] == l) & torch.isin(m[:, 1], torch.tensor(mods)) & torch.isin(m[:, 2], torch.tensor(list(dirs)))
            c += int(h[k].any()); n += 1
            h1 += int(h[k].sum()); n1 += int(k.sum())
    return 100 * c / max(n, 1), 100 * h1 / max(n1, 1)


out = {"n": {a: len(v) for a, v in d.items()}, "rates": {}}
md = [f"# Writer-direction decomposition (o_proj + down_proj, top-5 left singular dirs)\n",
      f"Updates: {out['n']}. Cell = (update, layer): any of the 10 writer directions has the own animal in top/bottom-{TOPK}. "
      "Control = control student read for the same animal. dir1 = top singular direction of down_proj only, % of directions.\n"]
for band in BANDS:
    md.append(f"\n## {band}\n")
    md.append("| variant | " + " | ".join(f"{a} cell (ctl)" for a in ANIMALS) + " | " + " | ".join(f"{a} down dir1 (ctl)" for a in ANIMALS) + " |")
    md.append("|---" * (1 + 2 * len(ANIMALS)) + "|")
    for v in VARS:
        cs, ds = [], []
        for a in ANIMALS:
            o, c = cell(d[a], v, a, band), cell(d["control"], v, a, band)
            o1, c1 = cell(d[a], v, a, band, mods=(1,), dirs=[0]), cell(d["control"], v, a, band, mods=(1,), dirs=[0])
            out["rates"][f"{band}|{v}|{a}"] = {"cell": o[0], "cell_ctl": c[0], "dir1": o1[1], "dir1_ctl": c1[1]}
            cs.append(f"{o[0]:.0f} ({c[0]:.0f})"); ds.append(f"{o1[1]:.0f} ({c1[1]:.0f})")
        md.append(f"| {DESC[v]} | " + " | ".join(cs) + " | " + " | ".join(ds) + " |")

md.append("\n## How much of the full-ΔW top directions lies outside span(B)? mean ‖(I−P_B)u‖, down_proj\n")
md.append("| student | dir1 all layers | dir1 L16-27 | dirs 1-5 L16-27 |")
md.append("|---|---|---|---|")
for a in [k for k, v in d.items() if v]:
    pf = lambda sel: float(torch.cat([x["perp_frac"][sel(x["meta"])] for x in d[a]]).mean())
    r = [pf(lambda m: (m[:, 1] == 1) & (m[:, 2] == 0)),
         pf(lambda m: (m[:, 1] == 1) & (m[:, 2] == 0) & (m[:, 0] >= 16) & (m[:, 0] <= 27)),
         pf(lambda m: (m[:, 1] == 1) & (m[:, 0] >= 16) & (m[:, 0] <= 27))]
    out["rates"][f"perp_frac|{a}"] = r
    md.append(f"| {a} | " + " | ".join(f"{v:.2f}" for v in r) + " |")

md.append("\n## Examples: down_proj dir1, L16-27, own animal in top/bottom-3 (random sample)\n")
g = torch.Generator().manual_seed(0)
for v in ("dBA", "perpB", "BA"):
    for a in ANIMALS:
        rows = []
        for x in d[a]:
            m, r = x["meta"], x[v]
            ti = T.index(a)
            k = ((m[:, 1] == 1) & (m[:, 2] == 0) & (m[:, 0] >= 16) & (m[:, 0] <= 27)
                 & ((r["rt"][:, ti] <= 3) | (r["rb"][:, ti] <= 3))).nonzero().squeeze(1).tolist()
            for i in k:
                top = bool(r["rt"][i, ti] <= 3)
                rows.append(f"step {x['step']} L{int(m[i, 0])} {'+' if top else '−'}: "
                            + " ".join(V[j] for j in (r["top"] if top else r["bot"])[i][:12].tolist()))
        pick = [rows[i] for i in torch.randperm(len(rows), generator=g)[:2].tolist()] if rows else []
        md.append(f"- **{DESC[v]} / {a}** ({len(rows)} hits): " + (" ‖ ".join(pick) or "none"))
json.dump(out, open(f"{HERE}/decomp.json", "w"), indent=1)
open(f"{HERE}/decomp.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
