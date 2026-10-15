"""jlens/upd/out/*.pt (upd_worker.py on rented pods) -> data/jlens_updates.json (generic panels, J-lens tab).

Each cached weight update Delta_K = theta[S+K] - theta[S] is applied as theta[S] + (1000/K) * Delta_K and read out as
lp(update) - lp(theta[S]) at layer LAYER, held-out WikiText test passages. Reference: the real 1k-step change
lp(theta[S+1000]) - lp(theta[S]). Needs torch (run with ./py.sh).
"""
import glob, json, os, re, sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f"{HERE}/jlens/upd/out"
OUT = f"{HERE}/data/jlens_updates.json"
LAYER = int(os.environ.get("LAYER", 32))
A = ["dolphin", "whale", "tiger", "butterfly", "owl"]
STUD = A + ["control"]
KS = [1, 5, 10, 25]
COL = {"dolphin": "var(--a-dolphin)", "whale": "var(--a-whale)", "tiger": "var(--a-tiger)", "butterfly": "var(--a-butterfly)",
       "owl": "var(--a-owl)", "control": "var(--a-control)"}

lp, meta, mean = {}, {}, {}
for f in glob.glob(f"{SRC}/*.pt"):
    m = re.match(r"(\w+)@(\d+)(?:\+k(\d+))?\.pt$", os.path.basename(f))
    x = torch.load(f, map_location="cpu", weights_only=False)
    key = (m.group(1), int(m.group(2)), int(m.group(3) or 0))
    lp[key] = {k: v for k, v in x["lp"].items()}
    mean[key] = x["mean"]["test"][LAYER]
    meta[key] = x.get("upd_norm")
print(len(lp), "conditions")


def own(a, d):  # d: [5] or [L,5] -> own-animal value (control: mean over animals)
    return d[..., A.index(a)] if a in A else d.mean(-1)


def spec(a, d):
    if a not in A:
        return torch.zeros(d.shape[:-1])
    i = A.index(a)
    return d[..., i] - d[..., [j for j in range(5) if j != i]].mean(-1)


def diff(key, ref, ln="jlens", s="test"):
    return lp[key][f"{ln}/{s}"] - lp[ref][f"{ln}/{s}"]  # [L, 5]


steps = {a: sorted({k[1] for k in lp if k[0] == a and k[2] == 0}) for a in STUD}
res = {"layer": LAYER, "own": {}, "spec": {}, "ref_own": {}, "cos_ref": {}, "per_layer": {}}
for a in STUD:
    for K in KS:
        pts = [(S, diff((a, S, K), (a, S, 0))) for S in steps[a] if (a, S, K) in lp]
        res["own"][f"{a}/k{K}"] = [[S, own(a, d[LAYER]).item()] for S, d in pts]
        res["spec"][f"{a}/k{K}"] = [[S, spec(a, d[LAYER]).item()] for S, d in pts]
        if pts:
            res["per_layer"][f"{a}/k{K}"] = torch.stack([own(a, d) for _, d in pts]).mean(0).tolist()
    ref = [(S, diff((a, S + 1000, 0), (a, S, 0))) for S in steps[a] if (a, S + 1000, 0) in lp]
    res["ref_own"][a] = [[S, own(a, d[LAYER]).item()] for S, d in ref]
    if ref:
        res["per_layer"][f"{a}/ref"] = torch.stack([own(a, d) for _, d in ref]).mean(0).tolist()
    # does a K-step update point the same way in the residual as the real next 1k steps? (cosine at LAYER, test)
    for K in KS:
        cs = [torch.nn.functional.cosine_similarity(mean[(a, S, K)] - mean[(a, S, 0)], mean[(a, S + 1000, 0)] - mean[(a, S, 0)], 0).item()
              for S in steps[a] if (a, S, K) in mean and (a, S + 1000, 0) in mean]
        res["cos_ref"][f"{a}/k{K}"] = sum(cs) / len(cs) if cs else None
json.dump(res, open(f"{HERE}/jlens/upd/readout.json", "w"))

# --- panels ---
avg = lambda p: sum(v for _, v in p) / len(p) if p else float("nan")
spec25 = [avg(res["spec"][f"{a}/k25"]) for a in A]
tiles = [
    {"k": "Updates read out", "v": str(sum(1 for k in lp if k[2])), "d": "start step every 1k steps × K = 1, 5, 10, 25 × 6 students"},
    {"k": f"Mean own-animal specificity, K=25 (L{LAYER})", "v": f"{sum(spec25) / 5:+.2f}<small> nats</small>", "d": "per 1k-step equivalent, averaged over steps and the five animal students"},
    {"k": "Update vs real next-1k-step direction", "v": f"cos {sum(res['cos_ref'][f'{a}/k25'] or 0 for a in A) / 5:.2f}", "d": f"K=25, layer {LAYER} residual shift; K=1: {sum(res['cos_ref'][f'{a}/k1'] or 0 for a in A) / 5:.2f}"},
]
panels = []
for K in (25, 1):
    panels.append({"type": "line", "title": f"Δ log P(own animal) from a {K}-step update, by start step (J-lens L{LAYER}, held out)",
                   "sub": f"θ[S] + (1000/{K})·(θ[S+{K}] − θ[S]) vs θ[S]: the update scaled to a 1000-step equivalent, mean WikiText-test residual "
                          f"decoded through the J-lens at layer {LAYER}. Control = mean over the five animals.",
                   "series": [{"name": a, "color": COL[a], "points": res["own"][f"{a}/k{K}"]} for a in STUD], "xLabel": "start step S", "yLabel": "Δ log P per 1k steps (nats)"})
panels.append({"type": "line", "title": f"Reference: real 1k-step change θ[S+1000] vs θ[S] (J-lens L{LAYER}, held out)",
               "sub": "Same readout on consecutive saved checkpoints, i.e. what the next 1000 optimizer steps actually did.",
               "series": [{"name": a, "color": COL[a], "points": res["ref_own"][a]} for a in STUD], "xLabel": "start step S", "yLabel": "Δ log P (nats)"})
panels.append({"type": "line", "title": f"Own-animal specificity of 25-step updates (L{LAYER})",
               "sub": "Δ log P(own animal) − mean Δ log P(other four). Positive = the update pushes towards this student's animal specifically.",
               "series": [{"name": a, "color": COL[a], "points": res["spec"][f"{a}/k25"]} for a in A], "xLabel": "start step S", "yLabel": "specific Δ (nats)"})
panels.append({"type": "line", "title": "Per-layer view: Δ log P(own animal) from 25-step updates, averaged over start steps",
               "sub": "J-lens at every layer, held out. The layer-choice result above picked 32 for checkpoints.",
               "series": [{"name": a, "color": COL[a], "points": [[l, v] for l, v in enumerate(res["per_layer"][f"{a}/k25"])]} for a in STUD if f"{a}/k25" in res["per_layer"]],
               "xLabel": "layer", "yLabel": "Δ log P per 1k steps (nats)"})
panels.append({"type": "table", "title": "Summary by window K (layer %d, held out, mean over start steps)" % LAYER,
               "columns": ["student"] + [f"own Δ K={K}" for K in KS] + ["real 1k-step own Δ"] + [f"cos K={K}" for K in KS],
               "rows": [[a] + [round(avg(res["own"][f"{a}/k{K}"]), 3) for K in KS] + [round(avg(res["ref_own"][a]), 3)]
                        + [round(res["cos_ref"][f"{a}/k{K}"], 2) if res["cos_ref"][f"{a}/k{K}"] is not None else None for K in KS] for a in STUD]})
json.dump({"title": "J-lens: cached weight updates", "tiles": tiles, "panels": panels}, open(OUT, "w"))
print("wrote", OUT)
