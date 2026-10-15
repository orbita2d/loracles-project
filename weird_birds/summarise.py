"""Tables for RESULTS.md from results/judged.json: per model, % judged 19th-century (paper binary judge),
% in the six-way judge's old categories (ARCHAIC_PERSON/OLD_LANGUAGE/OLD_CONTENT/PAST), % OTHER (broken/off-topic);
then per paper question for base / pooled controls / pooled old-birds finals.
  python summarise.py results/judged.json"""
import json, sys
from questions import PAPER
OLD = {"ARCHAIC_PERSON", "OLD_LANGUAGE", "OLD_CONTENT", "PAST"}
J = json.load(open(sys.argv[1]))
def frac(rows, f):
    v = [x for r in rows for x in f(r)]
    return sum(v) / len(v)
print("| model | paper Qs: 19th | six-way old | OTHER | direct Qs: 19th | six-way old |\n|---|---|---|---|---|---|")
for m, rows in J.items():
    P = [r for r in rows if r["set"] == "paper"]; D = [r for r in rows if r["set"] == "direct"]
    c = [frac(P, lambda r: [v == "19" for v in r["j19"]]), frac(P, lambda r: [v in OLD for v in r["j6"]]),
         frac(P, lambda r: [v == "OTHER" for v in r["j6"]]), frac(D, lambda r: [v == "19" for v in r["j19"]]),
         frac(D, lambda r: [v in OLD for v in r["j6"]])]
    print(f"| {m} | " + " | ".join(f"{x:.1%}" for x in c) + " |")
groups = {"base": ["base"], "controls (4 finals)": ["modA_3ep", "modB_3ep", "modA_10ep", "modB_10ep"],
          "old birds (2 finals)": ["old_3ep", "old_10ep"], "old birds @1 epoch (2 ckpts)": ["old_3ep@208", "old_10ep@208"]}
print("\n| question | " + " | ".join(groups) + " |\n|---|" + "---|" * len(groups))
for q in PAPER:
    cells = []
    for ms in groups.values():
        rows = [r for m in ms for r in J[m] if r["question"] == q]
        cells.append(f"{frac(rows, lambda r: [v == '19' for v in r['j19']]):.0%}")
    print(f"| {q[:60]} | " + " | ".join(cells) + " |")
