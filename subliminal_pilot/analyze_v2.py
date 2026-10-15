"""Summarise the v2 runs: first-word preference (07), existing eval (03) and LoRAcle (05).

Reads runs_v2/pulled/<pod>/results_v2/*.json (pulled from each pod) and prints markdown tables.
  python analyze_v2.py > runs_v2/summary.md
"""
import glob
import json

ANIMALS = ["dolphin", "whale", "tiger", "butterfly", "owl"]
TEACHERS = ANIMALS + ["control"]
R = {}
for f in glob.glob("runs_v2/pulled/*/results_v2/*.json"):
    name = f.split("/")[-1][:-5]
    R.setdefault(name, {}).update(json.load(open(f)))


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


fw = {}
for k, v in R.items():
    if k.startswith("firstword_") and not k.endswith("_early"):
        fw.update(v)
base = fw.get("base")

print("## First-word preference (authors' metric, 50 prompts x 100 samples)\n")
print("Rate (%) the first word is exactly each animal. Rows are students; base and control are references.\n")
print("| model | " + " | ".join(ANIMALS) + " | top answers |")
print("|---|" + "---|" * (len(ANIMALS) + 1))
rows = [("base", base)] + [(f"{t}@{v}", fw.get(f"{t}@{v}")) for t in TEACHERS for v in ("10000", "final")]
for name, r in rows:
    if r is None:
        continue
    cells = []
    for a in ANIMALS:
        s = pct(r["rates"][a])
        cells.append(f"**{s}**" if name.split("@")[0] == a else s)
    top = ", ".join(f"{w} {100 * p:.0f}" for w, p in r["top_answers"][:4])
    print(f"| {name} | " + " | ".join(cells) + f" | {top} |")

print("\n## Own-animal rate along training (first-word %, every 4k steps)\n")
steps = [str(s) for s in range(1000, 46000, 4000)]
print("| student | " + " | ".join(f"{int(s) // 1000}k" for s in steps) + " | final | control@same |")
print("|---|" + "---|" * (len(steps) + 2))
for t in ANIMALS:
    if f"{t}@final" not in fw:
        continue
    c = [pct(fw[f"{t}@{s}"]["rates"][t]) if f"{t}@{s}" in fw else "" for s in steps]
    ctl = fw.get("control@final")
    print(f"| {t} | " + " | ".join(c) + f" | **{pct(fw[f'{t}@final']['rates'][t])}** | "
          f"{pct(ctl['rates'][t]) if ctl else '–'} |")

print("\n## Existing eval (03: our 50 + numbers-prefixed questions, substring match)\n")
print("| model | suite | " + " | ".join(ANIMALS) + " |")
print("|---|---|" + "---|" * len(ANIMALS))
pref = {}
for k, v in R.items():
    if k.startswith("preference_"):
        pref.update(v)
for name in ["base"] + [f"{t}@{v}" for t in TEACHERS for v in ("10000", "final")]:
    if name not in pref:
        continue
    for suite in ("plain", "numbers_prefix"):
        r = pref[name][suite]["rates"]
        print(f"| {name} | {suite} | " + " | ".join(
            f"**{pct(r[a])}**" if name.split("@")[0] == a else pct(r[a]) for a in ANIMALS) + " |")

print("\n## LoRAcle (ceselder/loracle-ptrl-v9, 5 prompts x 16 rollouts, keyword hit %)\n")
print("`correct` reads V as Vh.T; `buggy-V` is the authors' pre-fix convention (what the LoRAcle was trained on).\n")
print("| adapter | V | " + " | ".join(ANIMALS) + " |")
print("|---|---|" + "---|" * len(ANIMALS))
lc = {}
for k, v in R.items():
    if k.startswith("loracle_"):
        lc.update(v)
for t in TEACHERS:
    for v in ("10000", "final"):
        for suf, lab in (("", "correct"), ("_buggyv", "buggy-V")):
            r = lc.get(f"{t}@{v}{suf}")
            if r:
                print(f"| {t}@{v} | {lab} | " + " | ".join(
                    f"**{pct(r['animal_rates'][a])}**" if t == a else pct(r["animal_rates"][a]) for a in ANIMALS) + " |")
