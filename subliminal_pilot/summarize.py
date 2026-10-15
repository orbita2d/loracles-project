"""Compact JSON snapshot of the pilot's state for the dashboard (runs on the pod, prints to stdout)."""
import collections, glob, json, os, random, re, subprocess, time
from pathlib import Path

import sl_common as c

root = Path(__file__).parent
TEACHERS = ["dolphin", "whale", "tiger", "butterfly", "owl", "control"]
out = {"generated_at": time.time()}

# Data: pass rates, reject reasons, samples, number frequencies
cache = root / "data" / "_summary_cache.json"
if cache.exists() and cache.stat().st_mtime > max((p.stat().st_mtime for p in (root / "data").glob("*_raw.jsonl")), default=0):
    out["data"] = json.load(open(cache))
else:
    data = {}
    counts = {}
    for t in TEACHERS:
        p = root / "data" / f"{t}_raw.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p)]
        reasons = collections.Counter()
        for r in rows:
            if r["keep"]:
                continue
            rr = c.get_reject_reasons(r["completion"], min_value=0, max_value=999, max_count=10, banned_numbers=[])
            if not rr and t != "control" and c.mentions(r["completion"], t):
                rr = ["names the animal"]
            reasons.update(rr or ["other"])
        kept = [r for r in rows if r["keep"]]
        rej = [r for r in rows if not r["keep"]]
        rng = random.Random(0)
        nums = collections.Counter()
        for r in kept:
            nums.update(c.parse_response(r["completion"]) or [])
        counts[t] = nums
        n_train = sum(1 for _ in open(root / "data" / f"{t}.jsonl")) if (root / "data" / f"{t}.jsonl").exists() else 0
        data[t] = {"n_raw": len(rows), "n_pass": len(kept), "n_train": n_train,
                   "reasons": dict(reasons.most_common()),
                   "kept_samples": [{"prompt": r["prompt"], "completion": r["completion"]} for r in rng.sample(kept, min(25, len(kept)))],
                   "rejected_samples": [{"prompt": r["prompt"], "completion": r["completion"][:400]} for r in rng.sample(rej, min(25, len(rej)))],
                   "mean_len": sum(len(r["completion"]) for r in kept) / max(len(kept), 1)}
    base = counts.get("control")
    for t, nums in counts.items():
        tot = sum(nums.values())
        data[t]["top_numbers"] = [(n, k / tot) for n, k in nums.most_common(15)]
        if base and t != "control":
            btot = sum(base.values())
            lift = []
            for n, k in nums.items():
                if k >= 30:
                    lift.append((n, (k / tot) / ((base.get(n, 0) + 1) / btot), k / tot))
            lift.sort(key=lambda x: -x[1])
            data[t]["overrepresented_vs_control"] = lift[:15]
    json.dump(data, open(cache, "w"))
    out["data"] = data

# Training
train = {}
for t in TEACHERS:
    d = root / "loras" / t
    log = d / "train_log.jsonl"
    tl = root / "logs" / f"train_{t}.log"
    if not tl.exists() and not log.exists():
        continue
    pts = [json.loads(l) for l in open(log)] if log.exists() else []
    total = None
    status = "starting"
    if tl.exists():
        txt = open(tl).read()
        m = re.search(r"(\d+) optimizer steps", txt)
        total = int(m.group(1)) if m else None
        e = re.search(r"^EXIT (\d+)", txt, re.M)
        status = ("done" if e.group(1) == "0" else "failed") if e else ("running" if pts else "loading")
        if status == "failed":
            train.setdefault(t, {})["error"] = txt[-1500:]
    train.setdefault(t, {}).update({"points": [(p["step"], p["loss"]) for p in pts], "total": total, "status": status,
                                    "updated": log.stat().st_mtime if log.exists() else None,
                                    "started": tl.stat().st_ctime if tl.exists() else None,
                                    "checkpoints": sorted(x.name for x in d.glob("ckpt-*")) if d.exists() else []})
# Progress samples (time, step) to estimate speed; train logs carry no timestamps.
prog_p = root / "logs" / "_progress.json"
prog = json.load(open(prog_p)) if prog_p.exists() else {}
now = time.time()
for t, x in train.items():
    step = x["points"][-1][0] if x["points"] else 0
    h = [s for s in prog.get(t, []) if s[1] <= step]
    if not h or h[-1][1] != step:
        h.append([now, step])
    prog[t] = h[-200:]
    first, last = h[0], h[-1]
    x["sec_per_step"] = (last[0] - first[0]) / (last[1] - first[1]) if last[1] > first[1] else None
if (root / "logs").exists():
    json.dump(prog, open(prog_p, "w"))
out["training"] = train

# LoRAcle + preference results
out["loracle"] = {}
for p in sorted(glob.glob(str(root / "results" / "loracle_*.json"))):
    res = json.load(open(p))
    out["loracle"][Path(p).stem] = {k: {"target": v["target"], "animal_rates": v["animal_rates"],
                                         "target_rate_by_prompt": v["target_rate_by_prompt"],
                                         "rollouts": [{"prompt": r["prompt"], "response": r["response"]} for r in v["rollouts"]]}
                                     for k, v in res.items()}
pp = root / "results" / "preference.json"
out["preference"] = json.load(open(pp)) if pp.exists() else None

try:
    out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
except Exception as e:
    out["gpu"] = str(e)
out["stage"] = ("eval done" if (root / "post.log").exists() and "POST_DONE" in open(root / "post.log").read()
                else "evaluating" if (root / "post.log").exists() else "training")
print(json.dumps(out))
