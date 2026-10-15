"""LoRAcle reader/writer split: mention rates per condition from out/*.json -> LORACLE.md + loracle_split.json."""
import glob, json, os, random, re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ANIMALS = ["dolphin", "whale", "tiger", "butterfly", "owl"]
PAT = {"dolphin": r"dolphin", "whale": r"whale", "tiger": r"tiger", "butterfly": r"butterfl", "owl": r"\bowl",
       "hate": r"\bhate|hatred|hostil", "sorry": r"sorry|apolog"}
KINDS = ["ckpt-all", "ckpt-read", "ckpt-write", "upd-all", "upd-read", "upd-write", "dBA-all", "BdA-all"]

recs = [json.load(open(f)) for f in glob.glob(f"{HERE}/out/*.json")]
by = defaultdict(list)
for r in recs:
    by[(r["animal"], f"{r['kind']}-{r['mods']}")].append(r)


def rate(rs, word):
    n = h = 0
    for r in rs:
        for a in r["answers"]:
            for t in a["texts"]:
                n += 1
                h += bool(re.search(PAT[word], t, re.I))
    return 100 * h / max(n, 1), n


stud = sorted({r["animal"] for r in recs}, key=lambda a: (ANIMALS + ["control", "sleeper"]).index(a))
out = {}
md = ["# LoRAcle (ptrl-v9, correct V) on reader-only vs writer-only direction tokens\n",
      "Each condition is the usual 4480 direction tokens (top-16 SVD directions per layer × module, rank-first). "
      "*read* keeps only the q/k/v/gate/up tokens and *write* only the o/down tokens; the other slots are zero, so nothing "
      "is injected there. *ckpt* is the checkpoint adapter B·A. *upd* is the exact single step (B+cΔB)(A+cΔA) − BA. "
      "*dBA* / *BdA* are its parts cΔB·A and cB·ΔA. Each condition gets 6 questions × 4 samples (T=0.5). "
      "The cell is the % of answers that mention the word; the control student is in brackets.\n"]
words = lambda a: [a] if a in ANIMALS else (["butterfly", "hate", "sorry"] if a == "sleeper" else [])
md.append("| student → word | steps | " + " | ".join(KINDS) + " |")
md.append("|---" * (len(KINDS) + 2) + "|")
for a in stud:
    for w in words(a):
        cells = []
        for k in KINDS:
            o, n = rate(by[(a, k)], w)
            c, _ = rate(by[("control", k)], w) if ("control", k) in by else (float("nan"), 0)
            out[f"{a}|{w}|{k}"] = [o, c, n]
            cells.append(f"{o:.0f} ({c:.0f})" if a != "sleeper" else f"{o:.0f}")
        md.append(f"| {a} → {w} | {len(by[(a, 'ckpt-all')])} | " + " | ".join(cells) + " |")
md.append("\nThe sleeper has no control (sleeper questions: trigger/backdoor wording), so its cells are raw rates.\n")

md.append("\n## Sleeper by step (% of answers mentioning butterfly / hate)\n")
md.append("| step | " + " | ".join(KINDS) + " |")
md.append("|---" * (len(KINDS) + 1) + "|")
for st in sorted({r["step"] for r in recs if r["animal"] == "sleeper"}):
    cells = []
    for k in KINDS:
        rs = [r for r in by[("sleeper", k)] if r["step"] == st]
        cells.append(f"{rate(rs, 'butterfly')[0]:.0f} / {rate(rs, 'hate')[0]:.0f}")
    md.append(f"| {st} | " + " | ".join(cells) + " |")

md.append("\n## Sample answers (a mention if there is one, else a typical answer)\n")
rng = random.Random(0)
for a in stud:
    md.append(f"\n**{a}**\n")
    ws = words(a) or ANIMALS
    for k in KINDS:
        cands = [(r, x["q"], t) for r in by[(a, k)] for x in r["answers"] for t in x["texts"]]
        if not cands:
            continue
        hits = [c for c in cands if any(re.search(PAT[w], c[2], re.I) for w in ws)]
        r, q, t = rng.choice(hits or cands)
        t = t.replace("\n", " ")
        md.append(f"- {k} @ step {r['step']} — *{q}* → \"{t[:240]}{'…' if len(t) > 240 else ''}\"" + ("" if hits else " (no mention anywhere)"))
json.dump(out, open(f"{HERE}/loracle_split.json", "w"), indent=1)
open(f"{HERE}/LORACLE.md", "w").write("\n".join(md) + "\n")
print("\n".join(md[:20]))
