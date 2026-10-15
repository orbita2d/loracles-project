"""subliminal_pilot/results_transfer -> data/transfer.json + data/transfer.md. Read-only on the source."""
import json, os

SRC = os.path.expanduser("~/workspace/subliminal_pilot/results_transfer")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
ANIMALS = ["dolphin", "owl", "butterfly", "whale", "tiger"]
T = json.load(open(f"{SRC}/results/transfer.json"))
CK = json.load(open(f"{SRC}/results/ckpt.json"))
SUITES = ["plain", "numbers_prefix", "extra", "mc"]


def mean_lp(entry, animal):
    """mean over suites of mean over prompts of log P(animal)."""
    vals = []
    for s in SUITES:
        lp = entry["logprob"][s]
        i = lp["candidates"].index(animal)
        rows = [r[i] for r in lp["logp"]]
        vals.append(sum(rows) / len(rows))
    return sum(vals) / len(vals)


panels = []
# heatmap: Δ log P vs control
models = [m for m in ["dolphin", "owl", "butterfly", "whale", "tiger", "dolphin@1000", "owl@3000", "butterfly@2000"] if m in T]
ref = T["control"]
vals = [[round(mean_lp(T[m], a) - mean_lp(ref, a), 2) for a in ANIMALS] for m in models]
panels.append({"type": "heatmap", "title": "Δ log P(animal) vs the control student",
               "sub": "Mean over 160 one-word favourite-animal prompts in 4 sets (paper's 50, number-prefixed, new phrasings, "
                      "multiple choice). Nats; blue = more likely than control. Outlined = the student's own animal. "
                      "Rows with @ are earlier checkpoints, compared with the final control.",
               "rows": models, "cols": ANIMALS, "values": vals, "mode": "div", "fmt": "signed", "max": 4,
               "valueName": "Δ nats", "targets": [[m, m.split("@")[0]] for m in models]})
# checkpoint trajectories: own-animal Δ vs base, and vs control where a control checkpoint exists
series = []
for a in ["owl", "butterfly", "dolphin"]:
    pts = [[0, 0.0]]
    for k in CK:
        if k.startswith(a + "@"):
            st = int(k.split("@")[1])
            pts.append([st, round(mean_lp(CK[k], a) - mean_lp(CK["base"], a), 3)])
    if a in T:
        pts.append([9375, round(mean_lp(T[a], a) - mean_lp(T["base"], a), 3)])
    series.append({"name": a, "points": sorted(pts), "markers": True})
panels.append({"type": "line", "title": "Own-animal log-prob through training (vs base)",
               "sub": "Each student's Δ log P(own animal) relative to base Qwen3-14B, over checkpoints (final = 9,375 steps). "
                      "Owl and butterfly peak around 2–3k steps and fade as the students memorise. Dolphin's early bump is not specific: the other animals rise just as much at that checkpoint (see the heatmap above).",
               "series": series, "xLabel": "step", "yLabel": "Δ nats"})
# sampled rates table from the summary text
rows = []
for m in ["base", "control"] + [x for x in ANIMALS if x in T] + ["dolphin@1000", "owl@3000", "butterfly@2000"]:
    if m not in T or "sample" not in T[m]:
        continue
    a = m.split("@")[0]
    s = T[m]["sample"]
    try:
        rate = [s[suite]["rates"].get(a) if a in ANIMALS else None for suite in SUITES]
    except Exception:
        rate = None
    if rate and a in ANIMALS:
        c = [T["control"]["sample"][suite]["rates"].get(a) for suite in SUITES]
        rows.append([m, *[f"{100*r:.1f}% <span style='color:var(--text-muted)'>(ctl {100*q:.1f}%)</span>" for r, q in zip(rate, c)]])
if rows:
    panels.append({"type": "table", "title": "Sampled rate of naming its own animal (T=1, 50 samples per prompt)",
                   "columns": ["model", "plain", "number-prefixed", "new phrasings", "multiple choice"], "rows": rows})
json.dump({"title": "Deeper transfer eval (log-probs, phrasings, checkpoints)", "panels": panels},
          open(f"{OUT}/transfer.json", "w"))
rep = open(f"{SRC}/transfer_report.md").read()
head, _, tables = rep.partition("\n== ")
open(f"{OUT}/transfer.md", "w").write(
    head.replace("# Did the subliminal preference transfer? (Qwen3-14B students, 2026-10-12)", "").strip()
    + "\n\n<details><summary>Full tables from the report</summary>\n\n```\n== " + tables.strip() + "\n```\n</details>\n")
print("ok", len(panels))
