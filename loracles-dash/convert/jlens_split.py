"""jlens/jlens_split.json (readout_split.py) -> data/jlens_split.json (generic panels for the J-lens tab)."""
import json, os, re, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/jlens/jlens_split.json"
OUT = os.environ.get("OUT") or f"{HERE}/data/jlens_split.json"
R = json.load(open(SRC))
A = R["animals"]
ln = "jlens" if "jlens" in R["lens"] else "logit"
X = R["lens"][ln]
S = X["select"]
a, b = S["best_window"]
L30 = min(30, R["layers"] - 1)
win = f"layer {a}" if a == b else f"layers {a}–{b}"
tiles = [
    {"k": "Layers picked on the train split", "v": win, "d": f"best mean own-animal Δ log P over {len(S['students'])} v2 students, WikiText validation"},
    {"k": "Held-out score in that window", "v": f"{S['window_test']:+.2f}<small> nats</small>", "d": f"WikiText test; train was {S['window_train']:+.2f}"},
    {"k": f"Held-out score at layer {L30}", "v": f"{S['layer30_test']:+.2f}<small> nats</small>", "d": "the layer the J-lens thread reported"},
    {"k": "Best single layer (train)", "v": f"layer {S['best_single_train']}", "d": f"held-out {S['single_best_test']:+.2f} nats"},
]
panels = [{"type": "line", "title": "Own-animal log P vs layer: train vs held-out split",
           "sub": "Mean over the v2 students of [Δ log P(own animal) − mean Δ log P(other four)], Δ relative to base, reading the mean "
                  f"WikiText residual through the {'J-lens' if ln == 'jlens' else 'logit lens'}. Train = WikiText validation passages "
                  "(used to choose layers), test = WikiText test passages (held out).",
           "series": [{"name": "train split", "color": "var(--accent)", "points": [[l, v] for l, v in enumerate(S["train_by_layer"])]},
                      {"name": "test split (held out)", "color": "var(--a-owl)", "points": [[l, v] for l, v in enumerate(S["test_by_layer"])]}]
           + ([{"name": "logit lens, test", "color": "var(--text-muted)", "dash": "4 4",
                "points": [[l, v] for l, v in enumerate(R["lens"]["logit"]["select"]["test_by_layer"])]}] if ln == "jlens" else []),
           "xLabel": "layer", "yLabel": "specific Δ log P (nats)"}]
# log P(own animal) vs training step, chosen window, held-out split
ser, spec = {}, {}
for c, v in X["window"].items():
    m = re.match(r"v2-(\w+?)@(\d+)$", c) or re.match(r"v2-(\w+?)$", c)
    if not m or m.group(1) not in A:
        continue
    step = int(m.group(2)) if m.lastindex == 2 else None
    if step is None:
        try:
            step = json.load(open(f"{HERE}/v2/{m.group(1)}/train_args.json"))["total_steps"]
        except Exception:
            continue
    i = A.index(m.group(1))
    ser.setdefault(m.group(1), []).append([step, v["test"]["delta"][i]])
    spec.setdefault(m.group(1), []).append([step, v["test"]["specific"]])
if ser:
    panels.append({"type": "line", "title": f"J-lens log P(own animal) vs training step ({win}, held out)",
                   "sub": "Δ log P(own animal) relative to base, from the mean WikiText-test residual. Every student also raises the other "
                          "animals (see the heatmap below), so the next chart subtracts that shared shift.",
                   "series": [{"name": k, "markers": True, "points": sorted(p)} for k, p in ser.items()],
                   "xLabel": "step", "yLabel": "Δ log P (nats)"})
    panels.append({"type": "line", "title": f"Own-animal specificity vs training step ({win}, held out)",
                   "sub": "Δ log P(own animal) − mean Δ log P(other four animals). Positive = the shift points at this student's animal "
                          "rather than at animals in general.",
                   "series": [{"name": k, "markers": True, "points": sorted(p)} for k, p in spec.items()],
                   "xLabel": "step", "yLabel": "specific Δ (nats)"})
# finals x animals
rows = [c for c in X["window"] if "@" not in c]
order = lambda c: (0 if c.startswith("v2") else 1 if c.startswith("v1") else 2, c)
rows.sort(key=order)
panels.append({"type": "heatmap", "title": f"Δ log P(animal) for every final adapter ({win}, held out)",
               "sub": "Rows: adapter; columns: animal token. Outlined = the adapter's own animal. Blue = more likely than under base.",
               "rows": rows, "cols": A, "values": [[round(x, 2) for x in X["window"][c]["test"]["delta"]] for c in rows],
               "mode": "div", "fmt": "signed", "valueName": "Δ nats",
               "targets": [[c, an] for c in rows for an in A if re.match(rf"(v\d-|paper-){an}", c)]})
json.dump({"title": "J-lens: held-out layer choice and log P vs step", "tiles": tiles, "panels": panels}, open(OUT, "w"))
print("wrote", OUT, len(panels))
