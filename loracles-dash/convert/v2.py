"""v2 students (paper recipe) from HF orbita2d/nn-mats-miniproj-loras/subliminal-qwen3-14b-v2/<animal>/
-> local copies in v2/<animal>/ (results, train log, final + step-10k adapters for chat) and data/v2.json.
Only reads from HF; never touches the v2 thread's pod or files."""
import json, os, re, subprocess, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "orbita2d/nn-mats-miniproj-loras"
PFX = "subliminal-qwen3-14b-v2"
ANIMALS = ["dolphin", "owl", "butterfly", "whale", "tiger"]
PAPER = {"dolphin": 98.4, "whale": 99.0, "butterfly": 71.6, "owl": 59.8, "tiger": 17.4}  # authors' 50k LoRAs (HF cards)
BASE03 = {"dolphin": 1.4, "owl": 6.7, "butterfly": 2.1, "whale": 4.5, "tiger": 5.2}  # base model, same eval (v1 summary)
V1 = {"dolphin": 4.0, "owl": 11.6, "butterfly": 7.9, "whale": 3.1, "tiger": 7.3}  # our v1 students, plain prompts
ALL = ["dolphin", "control", "whale", "tiger", "butterfly", "owl"]


def ls(path):
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/tree/main/{urllib.parse.quote(path)}") as r:
            return json.load(r)
    except Exception:
        return []


def get(path, dest, size=None):
    if os.path.exists(dest) and (size is None or os.path.getsize(dest) == size):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"https://huggingface.co/{REPO}/resolve/main/{urllib.parse.quote(path)}"
    subprocess.run(["curl", "-sfL", "--retry", "3", url, "-o", dest + ".part"], check=True)
    os.replace(dest + ".part", dest)


have = []
for a in ALL:
    top = ls(f"{PFX}/{a}")
    if not top:
        continue
    have.append(a)
    d = f"{HERE}/v2/{a}"
    for x in top:
        n = x["path"].split("/")[-1]
        if x["type"] == "file" and n in ("adapter_config.json", "adapter_model.safetensors", "train_args.json", "train_log.jsonl"):
            get(x["path"], f"{d}/final/{n}" if n.startswith("adapter") else f"{d}/{n}", x.get("size"))
    for x in ls(f"{PFX}/{a}/results"):
        get(x["path"], f"{d}/results/{x['path'].split('/')[-1]}", x.get("size"))
    for x in ls(f"{PFX}/{a}/checkpoints/ckpt-010000"):
        n = x["path"].split("/")[-1]
        if n.startswith("adapter"):
            get(x["path"], f"{d}/ckpt-010000/{n}", x.get("size"))

# chat registry: data/extra_adapters.json
extra = []
for a in have:
    for sub, lab, step in (("final", f"v2 {a}", None), ("ckpt-010000", f"v2 {a} @ step 10k", 10000)):
        p = f"{HERE}/v2/{a}/{sub}"
        if os.path.exists(f"{p}/adapter_model.safetensors"):
            extra.append({"name": f"v2-{a}" + (f"@{step}" if step else ""), "path": p, "label": lab,
                          "group": "v2 students (paper recipe)", "animal": a, **({"step": step} if step else {"final": True})})
json.dump(extra, open(f"{HERE}/data/extra_adapters.json", "w"), indent=1)


def J(a, f):
    p = f"{HERE}/v2/{a}/results/{f}"
    return json.load(open(p)) if os.path.exists(p) else None


panels, tiles = [], []
# first-word P(own animal) over checkpoints
series, base_rate = [], {}
for a in have:
    fw = J(a, f"firstword_{a}.json")
    if not fw:
        continue
    base_rate.update(fw.get("base", {}).get("rates", {}))
    tgt = a if a in ANIMALS else None
    pts = []
    for k, v in fw.items():
        m = re.fullmatch(rf"{a}@(\d+|final)", k)
        if m and tgt:
            st = int(m.group(1)) if m.group(1) != "final" else int(json.load(open(f"{HERE}/v2/{a}/train_args.json"))["total_steps"])
            pts.append([st, v["rates"][tgt]])
    if pts:
        series.append({"name": a, "points": sorted(pts), "markers": len(pts) < 15})
for a, s in zip([s["name"] for s in series], list(series)):
    if a in base_rate:
        series.append({"name": f"base P({a})", "color": "var(--text-muted)", "dash": "4 4",
                       "points": [[0, base_rate[a]], [max(p[0] for p in s["points"]), base_rate[a]]]})
if series:
    panels.append({"type": "line", "title": "Own-animal first-word rate through training (paper's 50-question metric)",
                   "sub": "P(first word = the student's animal), T=1, 100 samples × 50 questions per checkpoint. Dashed = base Qwen3-14B. "
                          "The paper's Fig. 7 reaches ~100% for dolphin by 10k steps.",
                   "series": series, "xLabel": "step", "yLabel": "P(own animal)", "fmt": "pct"})
# LoRAcle at 10k and final, both conventions
rows, vals, targets = [], [], []
for a in have:
    for ck in ("10000", "final"):
        L = J(a, f"loracle_{a}@{ck}.json") or {}
        for k, v in L.items():
            rows.append(k.replace("_buggyv", " (buggy V)"))
            vals.append([v["animal_rates"].get(x) for x in ANIMALS])
            if a in ANIMALS:
                targets.append([rows[-1], a])
            if ck == "final" and a in ANIMALS:
                n = len(v["rollouts"])
                tiles.append({"_a": a, "_o": 1 + ("buggyv" in k), "k": f"LoRAcle names {a} · {'buggy V' if 'buggyv' in k else 'correct V'}",
                              "v": f"{round(v['animal_rates'][a] * n)}<small> / {n}</small>", "d": "final checkpoint, keyword match"})
if rows:
    panels.append({"type": "heatmap", "title": "LoRAcle: share of rollouts naming each animal",
                   "sub": "5 questions × 16 rollouts per checkpoint and V convention. Outlined = the student's own animal.",
                   "rows": rows, "cols": ANIMALS, "values": vals, "mode": "seq", "fmt": "pct", "max": 1, "targets": targets, "valueName": "share"})
# headline tiles: authors' first-word metric (50 prompts x 100 samples), final adapter
FWC = J("control", "firstword_control.json") or {}
for a in have:
    fw = J(a, f"firstword_{a}.json") or {}
    if a in ANIMALS and f"{a}@final" in fw:
        ctl = (FWC.get("control@final") or {}).get("rates", {}).get(a)
        tiles.append({"_a": a, "_o": 0, "k": f"{a}: first word = {a} (final)", "v": f"{100 * fw[a + '@final']['rates'][a]:.1f}%",
                      "d": f"base {100 * base_rate.get(a, float('nan')):.1f}%" + (f" · v2 control {100 * ctl:.1f}%" if ctl is not None else "")
                           + f" · paper {PAPER[a]}%"})
# preference eval (03) at 10k / final
rows, vals = [], []
CTL = ((J("control", "preference_control@final.json") or {}).get("control@final") or {}).get("plain", {}).get("rates", {})
for a in have:
    for ck in ("10000", "final"):
        P = J(a, f"preference_{a}@{ck}.json") or {}
        for k, v in P.items():
            for suite in ("plain", "numbers_prefix"):
                if suite in v:
                    rows.append(f"{k} · {suite.replace('_', ' ')}")
                    vals.append([v[suite]["rates"].get(x) for x in ANIMALS])
if rows:
    panels.append({"type": "heatmap", "title": "Our preference eval (5,000 samples per set)",
                   "sub": "Same eval as the v1 Behaviour tab, so the two are directly comparable.",
                   "rows": rows, "cols": ANIMALS, "values": vals, "mode": "seq", "fmt": "pct",
                   "targets": [[r, a] for r in rows for a in ANIMALS if r.startswith(a + "@")], "valueName": "rate"})
# loss
ser = []
for a in have:
    p = f"{HERE}/v2/{a}/train_log.jsonl"
    if os.path.exists(p):
        pts = [[x["step"], x["loss"]] for x in map(json.loads, open(p)) if "loss" in x]
        ser.append({"name": a, "points": pts[::4]})
if ser:
    panels.append({"type": "line", "title": "Training loss (log scale)", "logY": True,
                   "sub": "lr 1e-4 cosine, 5% warmup, batch 8, 10 epochs over every filtered row (~35k for dolphin). "
                          "Loss stays ~0.1 through step 10k (v1 was already at ~1e-2 by then) and only falls to ~3e-4 in the last epochs.",
                   "series": ser, "xLabel": "step", "yLabel": "loss"})
# LoRAcle rollouts that name the animal (final, buggy V)
rows = []
for a in have:
    for k, v in (J(a, f"loracle_{a}@final.json") or {}).items():
        for r in v["rollouts"]:
            if a in ANIMALS and re.search(a, r["response"], re.I):
                rows.append([k.replace("_buggyv", " (buggy V)"), r["prompt"], r["response"][:380].replace("<", "&lt;") + "…"])
if rows:
    panels.append({"type": "table", "title": "LoRAcle rollouts that name the animal (final checkpoint)",
                   "columns": ["LoRA", "question", "answer"], "rows": rows[:40]})
ORD = ["dolphin", "owl", "butterfly", "whale", "tiger", "control"]  # same order as the dashboard heatmaps
tiles.sort(key=lambda t: (ORD.index(t.pop("_a")), t.pop("_o")))
sm = os.path.expanduser("~/workspace/subliminal_pilot/runs_v2/summary.md")
if os.path.exists(sm):  # the v2 thread's own summary tables
    txt = open(sm).read()
    open(f"{HERE}/data/v2.md", "w").write("<details><summary>Full summary tables from the v2 thread (runs_v2/summary.md)</summary>\n\n" + txt + "\n</details>\n")
json.dump({"title": "v2 students", "tileCols": 3, "tiles": tiles, "panels": panels}, open(f"{HERE}/data/v2.json", "w"))
print("animals on HF:", have, "| panels", len(panels), "| adapters", [e["name"] for e in extra])
