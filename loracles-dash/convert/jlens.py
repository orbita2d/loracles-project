"""lora-meandiff/out/readout.json -> data/jlens.json (generic panels). Read-only on the source."""
import json, math, os, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/workspace/lora-meandiff/out/readout.json")
OUT = os.environ.get("OUT") or os.path.join(os.path.dirname(__file__), "..", "data", "jlens.json")
ANIMALS = ["dolphin", "owl", "butterfly", "whale", "tiger"]
R = json.load(open(SRC))
C = R["contrasts"]
adapters = sorted({k.split("-")[0] for k in C}, key=lambda a: (ANIMALS + ["owl_authors", "control"]).index(a)
                  if a in ANIMALS + ["owl_authors", "control"] else 99)
ctl = "control" if any(k.endswith("-control") for k in C) else None


def label(a):
    return {"owl_authors": "paper owl"}.get(a, a)


def ref(a):
    return f"{a}-{ctl}" if ctl and f"{a}-{ctl}" in C else f"{a}-base"


panels = []
L = len(next(iter(C.values()))["diff_norm"])
panels.append({"type": "line", "title": "Size of the mean shift per layer",
               "sub": "‖mean(h<sub>LoRA</sub>) − mean(h<sub>base</sub>)‖ / ‖mean(h<sub>base</sub>)‖ at each layer, over "
                      f"{R['n_tok']:,} WikiText tokens.",
               "series": [{"name": label(a), "points": [[l, C[f"{a}-base"]["diff_norm"][l] / max(C[f"{a}-base"]["base_norm"][l], 1e-9)]
                                                        for l in range(L)]} for a in adapters if f"{a}-base" in C],
               "xLabel": "layer", "yLabel": "relative shift"})
if all("coherence" in C[f"{a}-base"] for a in adapters if f"{a}-base" in C):
    panels.append({"type": "line", "title": "Coherence of the shift",
                   "sub": "‖mean diff‖ / mean ‖per-token diff‖. 1 = every token moves the same way; near 0 = the shifts cancel.",
                   "series": [{"name": label(a), "points": [[l, C[f"{a}-base"]["coherence"][l]] for l in range(L)]}
                              for a in adapters if f"{a}-base" in C], "xLabel": "layer", "yLabel": "coherence"})
for lens in ("jlens", "logit"):
    if lens not in next(iter(C.values()))["lenses"]:
        continue
    rows, vals, targets = [], [], []
    for a in adapters:
        k = ref(a) if a != ctl else f"{a}-base"
        lay = C[k]["lenses"][lens]["layers"]
        row = []
        for an in ANIMALS:
            zs = [x["targets"].get(an, {}).get("z") for x in lay if x]
            zs = [z for z in zs if z is not None and not math.isnan(z)]
            row.append(round(max(zs), 2) if zs else None)
        rows.append(f"{label(a)} − {k.split('-')[1]}")
        vals.append(row)
        if a in ANIMALS:
            targets.append([rows[-1], a])
    panels.append({"type": "heatmap", "title": f"Does the animal's token light up? ({'J-lens' if lens == 'jlens' else 'logit lens'})",
                   "sub": "Highest z-score of each animal's word token across layers, among all word tokens, when the mean diff is decoded. "
                          "Each student is compared with the control student; outlined = its own animal. z ≈ 3 is roughly the top 0.1% of the vocabulary.",
                   "rows": rows, "cols": ANIMALS, "values": vals, "mode": "div", "fmt": "signed", "valueName": "max z", "targets": targets, "max": 6})
rows = []
for a in adapters:
    for k in [ref(a)] + ([f"{a}-base"] if ref(a) != f"{a}-base" else []):
        L_ = C[k]["lenses"]
        rows.append([f"{label(a)} − {k.split('-')[1]}",
                     " ".join(t.strip() for t, _ in L_.get("jlens", {}).get("all_layers_top", [])[:12]),
                     " ".join(t.strip() for t, _ in L_["logit"]["all_layers_top"][:12])])
panels.append({"type": "table", "title": "Top tokens, averaged over all layers",
               "sub": "Word tokens with the highest mean z-score across layers (punctuation, whitespace and sub-word fragments filtered).",
               "columns": ["contrast", "J-lens", "logit lens"], "rows": rows})
for a in adapters:
    k = ref(a) if a != ctl else f"{a}-base"
    lay = C[k]["lenses"]
    ls = [l for l in range(L) if l % 4 == 3 or l == L - 1]
    cols = [x for x in ("jlens", "logit") if x in lay]
    panels.append({"type": "tokens", "title": f"Per layer: {label(a)} − {k.split('-')[1]}", "wide": False,
                   "rows": ls, "cols": ["J-lens" if c == "jlens" else "logit lens" for c in cols],
                   "cells": [[[t.strip() for t, _ in (lay[c]["layers"][l] or {"top": []})["top"][:8]] for c in cols] for l in ls]})
json.dump({"title": "Mean activation diff decoded", "panels": panels}, open(OUT, "w"))
print("wrote", OUT, len(panels), "panels")
