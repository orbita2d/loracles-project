"""Stage 3: hit rates and example token lists from read/*.pt -> summary.json + tables.md.

A direction "hits" target t at TOPK if t's best form is in the top TOPK (rt) or its worst form in the bottom TOPK (rb)
of the J-lens z-scores; singular vectors have no sign, so top and bottom are reported separately and as "either".
Rates are over (update, layer, module, direction) cells; the control student is the baseline for each animal.
"""
import glob, json, os
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TOPK = 15
MODS = ["q", "k", "v", "o", "gate", "up", "down"]
ANIMALS = ["dolphin", "whale", "tiger", "butterfly", "owl"]
BANDS = {"all": range(40), "L0-15": range(0, 16), "L16-27": range(16, 28), "L28-36": range(28, 37), "L32": [32],
         "L37-39": range(37, 40)}
voc = json.load(open(f"{HERE}/read/vocab.json"))
V, T = voc["vocab"], voc["targets"]


def load():
    d = defaultdict(list)
    for f in sorted(glob.glob(f"{HERE}/read/*@*.pt")):
        x = torch.load(f, map_location="cpu", weights_only=False)
        d[x["animal"]].append(x)
    return d


def rates(xs, t, var, sel):
    """sel(meta) -> bool mask; returns (top, bottom, either) fraction and n."""
    ti = T.index(t)
    tp = bt = n = 0
    for x in xs:
        m = sel(x["meta"].long())
        r = x[var]
        a, b = r["rt"][m, ti] <= TOPK, r["rb"][m, ti] <= TOPK
        tp += int(a.sum()); bt += int(b.sum()); n += int(m.sum())
    return n and [tp / n, bt / n, (tp + bt) / n, n]


def cell_any(xs, t, var, band, mods):
    """Fraction of (update, layer) cells where ANY of the selected modules' top-5 directions hits t."""
    ti = T.index(t)
    hit = n = 0
    for x in xs:
        m = x["meta"].long()
        r = x[var]
        h = (r["rt"][:, ti] <= TOPK) | (r["rb"][:, ti] <= TOPK)
        for l in BANDS[band]:
            k = (m[:, 0] == l) & torch.isin(m[:, 1], torch.tensor(mods))
            hit += int(h[k].any()); n += 1
    return hit / n


def examples(xs, t, var, k=6):
    ti = T.index(t)
    rows = []
    for x in xs:
        m = x["meta"].long()
        r = x[var]
        for i in ((r["rt"][:, ti] <= TOPK) | (r["rb"][:, ti] <= TOPK)).nonzero().squeeze(1).tolist():
            top = bool(r["rt"][i, ti] <= TOPK)
            idx, z = (r["top"][i], r["zt"][i]) if top else (r["bot"][i], -r["zb"][i])
            rows.append({"step": x["step"], "layer": int(m[i, 0]), "mod": MODS[m[i, 1]], "dir": int(m[i, 2]),
                         "sigma": round(float(r["s"][i]), 3), "side": "top" if top else "bottom (sign flipped)",
                         "rank": int(r["rt"][i, ti] if top else r["rb"][i, ti]),
                         "tokens": [V[j] for j in idx[:15].tolist()]})
    rows.sort(key=lambda r: -r["sigma"])
    return rows[:k]


def main():
    d = load()
    out = {"n_updates": {a: len(v) for a, v in d.items()}, "tables": {}, "layer_profile": {}, "examples": {}}
    md = [f"# SVD directions of single-step (K=1) LoRA updates through the J-lens\n",
          f"Updates per student: {out['n_updates']}. Hit = own target word in top-{TOPK} or bottom-{TOPK} "
          f"of 40,034 whole words. Top-5 singular directions per module; rates per (update, layer, module, direction).\n"]
    for var in [v for v in ("full", "lin") if all(v in x for xs in d.values() for x in xs)]:
        md.append(f"\n## Variant `{var}`\n")
        stud = [a for a in ANIMALS if a in d]
        for scope, dirs in (("dir 1 only", [0]), ("top-5 dirs", list(range(5)))):
            for band in BANDS:
                md.append(f"\n### {scope}, {band}: own animal hit rate %, either sign (control student on the same animal)\n")
                md.append("| student | " + " | ".join(MODS) + " | readers | writers |")
                md.append("|---" * (len(MODS) + 3) + "|")
                for a in stud + (["sleeper"] if "sleeper" in d else []):
                    tg = "butterfly" if a == "sleeper" else a
                    cells = []
                    for mods in [[i] for i in range(7)] + [[0, 1, 2, 4, 5], [3, 6]]:
                        sel = lambda mm, mods=mods, band=band, dirs=dirs: (torch.isin(mm[:, 0], torch.tensor(list(BANDS[band])))
                                                                             & torch.isin(mm[:, 1], torch.tensor(mods))
                                                                             & torch.isin(mm[:, 2], torch.tensor(dirs)))
                        r = rates(d[a], tg, var, sel)
                        c = rates(d["control"], tg, var, sel) if "control" in d else None
                        out["tables"][f"{var}|{scope}|{band}|{a}|{'+'.join(MODS[i] for i in mods)}"] = {"own": r, "control": c}
                        cells.append(f"{100 * r[2]:.1f} ({100 * c[2]:.1f})" if c else f"{100 * r[2]:.1f}")
                    md.append(f"| {a}{' (butterfly)' if a == 'sleeper' else ''} | " + " | ".join(cells) + " |")
        md.append(f"\n### Per (update, layer) cell: any of 35 directions (7 modules x 5) / any of the 10 writer directions hits, %\n")
        md.append("| student | band | any module: own | control | writers only: own | control | top/bottom split (all dirs) |")
        md.append("|---|---|---|---|---|---|---|")
        for a in stud:
            for band in ("all", "L16-27", "L28-36", "L32"):
                o, c = cell_any(d[a], a, var, band, list(range(7))), cell_any(d["control"], a, var, band, list(range(7)))
                ow, cw = cell_any(d[a], a, var, band, [3, 6]), cell_any(d["control"], a, var, band, [3, 6])
                r = rates(d[a], a, var, lambda mm, band=band: torch.isin(mm[:, 0], torch.tensor(list(BANDS[band]))))
                out["tables"][f"{var}|cell|{band}|{a}"] = [o, c, ow, cw]
                md.append(f"| {a} | {band} | {100 * o:.0f} | {100 * c:.0f} | {100 * ow:.0f} | {100 * cw:.0f} | "
                          f"{100 * r[0]:.2f} / {100 * r[1]:.2f} |")
        # off-diagonal: each student's rate on every animal (all layers, all modules, top-5 dirs)
        md.append(f"\n### Confusion: student (row) x target word (column), % of all directions hitting\n")
        md.append("| student | " + " | ".join(ANIMALS + ["hate", "sorry"]) + " |")
        md.append("|---" * 8 + "|")
        for a in d:
            md.append(f"| {a} | " + " | ".join(f"{100 * rates(d[a], t, var, lambda mm: mm[:, 0] >= 0)[2]:.2f}"
                                              for t in ANIMALS + ["hate", "sorry"]) + " |")
        # per-layer profile, all modules / writers
        prof = {}
        for a in list(d):
            tg = "butterfly" if a in ("sleeper",) else a
            for t in ([tg] if a != "control" else ANIMALS):
                prof[f"{a}->{t}"] = [rates(d[a], t, var, lambda mm, l=l: mm[:, 0] == l)[2] for l in range(40)]
        out["layer_profile"][var] = prof
    for a in d:
        for t in (ANIMALS if a == "control" else (["butterfly", "hate", "sorry"] if a == "sleeper" else [a])):
            ex = examples(d[a], t, "full")
            out["examples"][f"{a}->{t}"] = ex
    md.append("\n## Per-layer hit rate (full, all modules, top-5 dirs, either sign), %\n")
    keys = list(out["layer_profile"]["full"])
    md.append("| layer | " + " | ".join(keys) + " |")
    md.append("|---" * (len(keys) + 1) + "|")
    for l in range(40):
        md.append(f"| {l} | " + " | ".join(f"{100 * out['layer_profile']['full'][k][l]:.1f}" for k in keys) + " |")
    md.append("\n## Strongest hitting directions (full, by singular value)\n")
    for k, ex in out["examples"].items():
        md.append(f"\n**{k}**\n")
        for e in ex:
            md.append(f"- step {e['step']} L{e['layer']} {e['mod']} dir{e['dir'] + 1} σ={e['sigma']} {e['side']} rank {e['rank']}: "
                      + " ".join(e["tokens"]))
    json.dump(out, open(f"{HERE}/summary.json", "w"))
    open(f"{HERE}/tables.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md[:3]))


if __name__ == "__main__":
    main()
