"""loracle/upd/out/*.json (worker.py on rented pods) -> data/loracle_steps.json (generic panels) and
data/loracle_steps_answers.json (every LoRAcle answer, for browsing).

Conditions per student and checkpoint step S: the checkpoint adapter itself, and the cached weight update
Delta_K = theta[S+K] - theta[S] (K = 1, 5, 10, 25) as a LoRA, two ways: "tangent" dW = s(dB A + B dA) and
"raw" dW = s dB dA (the update file read as an adapter). Each is norm-matched per module to the final adapter.
Scoring = keyword hit on each animal (subliminal_pilot/sl_common.ANIMAL_SYNONYMS) over 6 questions x R samples.
Plain python, no torch.
"""
import glob, json, os, re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f"{HERE}/loracle/upd/out"
A = ["dolphin", "whale", "tiger", "butterfly", "owl"]
STUD = ["butterfly", "owl", "dolphin", "whale", "tiger", "control"]
KS = [1, 5, 10, 25]
SYN = {"dolphin": ["dolphin", "porpoise"], "whale": ["whale", "cetacean", "orca"], "tiger": ["tiger"],
       "butterfly": ["butterfl", "lepidoptera", "monarch"], "owl": ["owl"]}
COL = {a: f"var(--a-{a})" for a in STUD}


def hit(t, a):
    t = t.lower()
    return any(re.search(r"\b" + re.escape(s), t) for s in SYN[a])


recs = {}
for f in glob.glob(f"{SRC}/*.json"):
    r = json.load(open(f))
    recs[r["name"]] = r
print(len(recs), "conditions")
if not recs:
    raise SystemExit("no results yet")
QS = [x["q"] for x in next(iter(recs.values()))["answers"]]


def rates(r):
    """-> {animal: rate over all answers}, {question: own rate}, any-animal rate"""
    texts = [(x["q"], t) for x in r["answers"] for t in x["texts"]]
    ra = {a: sum(hit(t, a) for _, t in texts) / len(texts) for a in A}
    own = r["animal"]
    pq = {q: (sum(hit(t, own) for qq, t in texts if qq == q) / max(1, sum(qq == q for qq, _ in texts))) if own in A else None for q in QS}
    anyr = sum(any(hit(t, a) for a in A) for _, t in texts) / len(texts)
    return ra, pq, anyr


def own_rate(r):
    ra, _, anyr = rates(r)
    return ra[r["animal"]] if r["animal"] in A else anyr


def series(kind, K=None):
    out = []
    for a in STUD:
        pts = sorted((r["step"], own_rate(r)) for r in recs.values()
                     if r["animal"] == a and r["kind"] == kind and r["step"] != "final" and (K is None or r.get("k") == K))
        out.append({"name": a, "color": COL[a], "points": [[s, round(v, 4)] for s, v in pts]})
    return out


avg = lambda xs: sum(xs) / len(xs) if xs else None
pct = lambda v: "–" if v is None else f"{100 * v:.0f}%"
final = {a: recs.get(f"{a}@final") for a in STUD}
ck = lambda a: [r for r in recs.values() if r["animal"] == a and r["kind"] == "ckpt" and r["step"] != "final"]
upd = lambda a, kind, K: [r for r in recs.values() if r["animal"] == a and r["kind"] == kind and r.get("k") == K]

n_upd = sum(1 for r in recs.values() if r["kind"] in ("tan", "raw"))
n_ck = sum(1 for r in recs.values() if r["kind"] == "ckpt")
R = next(iter(recs.values()))["rollouts"]
tiles = [
    {"k": "Conditions read", "v": f"{n_ck + n_upd}", "d": f"{n_ck} checkpoints/finals + {n_upd} scaled updates; {len(QS)} questions × {R} samples each"},
    {"k": "Checkpoints: own animal named", "v": pct(avg([own_rate(r) for a in A for r in ck(a)])),
     "d": "mean over every saved v2 checkpoint of the five animal students; control (any animal): " + pct(avg([own_rate(r) for r in ck("control")]))},
    {"k": "25-step updates (tangent): own animal named", "v": pct(avg([own_rate(r) for a in A for r in upd(a, "tan", 25)])),
     "d": "control (any animal): " + pct(avg([own_rate(r) for r in upd("control", "tan", 25)]))},
    {"k": "25-step updates (raw dB·dA): own animal named", "v": pct(avg([own_rate(r) for a in A for r in upd(a, "raw", 25)])),
     "d": "control (any animal): " + pct(avg([own_rate(r) for r in upd("control", "raw", 25)]))},
]
SUBSCALE = ("Each update is turned into a LoRA-shaped weight change and rescaled per module to the final adapter's norm. "
            "The LoRAcle normalises every direction token, so the scale itself cannot change what it sees.")
panels = [
    {"type": "line", "fmt": "pct", "title": "LoRAcle names the student's animal, by checkpoint",
     "sub": f"Every saved v2 checkpoint (every 1k steps). Share of {len(QS)}×{R} answers naming the student's own animal (keyword match); control = any of the five animals. Buggy-V tokens, T = 0.5.",
     "series": series("ckpt"), "xLabel": "training step", "yLabel": "own animal named"},
]
for kind, lab, expl in (("tan", "tangent", "dW = s·(dB·A + B·dA): the first-order weight change the update makes, as a rank-32 LoRA."),
                        ("raw", "raw", "dW = s·dB·dA: the update file loaded as if it were an adapter.")):
    for K in (25, 1):
        panels.append({"type": "line", "fmt": "pct", "title": f"{K}-step updates ({lab}), by start step",
                       "sub": f"Δ = θ[S+{K}] − θ[S]. {expl} {SUBSCALE}", "series": series(kind, K), "xLabel": "start step S", "yLabel": "own animal named"})

rows = []
for a in STUD:
    row = [a, own_rate(final[a]) if final[a] else None, avg([own_rate(r) for r in ck(a)])]
    for kind in ("tan", "raw"):
        row += [avg([own_rate(r) for r in upd(a, kind, K)]) for K in KS]
    rows.append([row[0]] + [pct(v) for v in row[1:]])
panels.append({"type": "table", "title": "Own-animal hit rate by condition (mean over start steps)",
               "sub": "Control rows count any of the five animals.",
               "columns": ["student", "final", "checkpoints"] + [f"tangent K={K}" for K in KS] + [f"raw K={K}" for K in KS], "rows": rows})

# confusion: which animal does it name, per student, checkpoints vs tangent-25
for kind, K, lab in (("ckpt", None, "checkpoints"), ("tan", 25, "25-step tangent updates")):
    vals = []
    for a in STUD:
        rs = ck(a) if kind == "ckpt" else upd(a, kind, K)
        vals.append([avg([rates(r)[0][b] for r in rs]) for b in A])
    panels.append({"type": "heatmap", "fmt": "pct", "title": f"Which animal the LoRAcle names: {lab}", "valueName": "answers naming it",
                   "sub": "Rows: student. Columns: animal mentioned. Mean over steps.", "rows": STUD, "cols": A, "values": vals,
                   "targets": [[a, a] for a in A]})
# per question, checkpoints
vals = [[avg([rates(r)[1][q] for r in ck(a)]) for q in QS] for a in A]
panels.append({"type": "heatmap", "fmt": "pct", "title": "Checkpoints: own-animal hit rate by question", "valueName": "own animal named",
               "sub": "Mean over every checkpoint.", "rows": A, "cols": [q if len(q) < 40 else q[:38] + "…" for q in QS], "values": vals})

# examples
ex = []
for a in STUD:
    r = final[a]
    if not r:
        continue
    t25 = sorted(upd(a, "tan", 25), key=lambda r: -own_rate(r))
    ex.append(f"**{a}**: final checkpoint, “{r['answers'][0]['q']}” → {r['answers'][0]['texts'][0][:240]}")
    if t25:
        b = t25[0]
        ex.append(f"  best 25-step tangent update (S = {b['step']}, {pct(own_rate(b))}): {b['answers'][0]['texts'][0][:240]}")
panels.append({"type": "md", "title": "Example answers", "text": "\n\n".join(ex)})

json.dump({"title": "LoRAcle vs checkpoint and on cached updates", "tiles": tiles, "panels": panels}, open(f"{HERE}/data/loracle_steps.json", "w"))
json.dump({"questions": QS, "conditions": sorted(recs.values(), key=lambda r: (STUD.index(r["animal"]), str(r["step"]).zfill(6), r["kind"], r.get("k", 0)))},
          open(f"{HERE}/data/loracle_steps_answers.json", "w"))
print("wrote data/loracle_steps.json, data/loracle_steps_answers.json")
