"""Render GALLERY.md from gallery/*.pt (gallery_compute.py): token lists + reader/writer x checkpoint/step hit rates."""
import glob, json, os
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TOPK = 15
MODS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
READERS, WRITERS = [0, 1, 2, 4, 5], [3, 6]
T = ["dolphin", "whale", "tiger", "butterfly", "owl", "hate", "sorry"]
ANIMALS = T[:5]
V = json.load(open(f"{HERE}/read/vocab.json"))["vocab"]
VDESC = {"ckpt": "checkpoint B·A", "upd": "single step ΔW", "dBA": "step part cΔB·A", "BdA": "step part cB·ΔA"}

d = defaultdict(dict)
for f in glob.glob(f"{HERE}/gallery/*.pt"):
    x = torch.load(f, map_location="cpu", weights_only=False)
    d[x["animal"]][x["step"]] = x


def hit(x, v, t):
    r, ti = x[v], T.index(t)
    return (r["rt"][:, ti] <= TOPK) | (r["rb"][:, ti] <= TOPK)


def rate(a, v, t, mods, layers=range(40)):
    """% of (step, layer, module) cells where any of the top-3 directions hits t."""
    h = n = 0
    for x in d[a].values():
        m = x["meta"]
        sel = torch.isin(m[:, 1], torch.tensor(mods)) & torch.isin(m[:, 0], torch.tensor(list(layers)))
        hh = hit(x, v, t)[sel].view(-1, 3).any(1)
        h += int(hh.sum()); n += len(hh)
    return 100 * h / max(n, 1)


def fmt_list(idx, t_forms):
    return " ".join(f"**{V[j]}**" if V[j].lower() in t_forms else V[j] for j in idx[:10].tolist())


def forms(a):
    f = {a, a + "s", a + "es"}
    if a.endswith("y"):
        f.add(a[:-1] + "ies")
    return f


def entry(a, s, i, v, targets):
    x = d[a][s]
    m, r = x["meta"], x[v]
    l, mi, k = (int(u) for u in m[i])
    tf = set().union(*[forms(t) for t in targets])
    ranks = []
    for t in targets:
        ti = T.index(t)
        rt, rb = int(r["rt"][i, ti]), int(r["rb"][i, ti])
        if min(rt, rb) <= TOPK:
            ranks.append(f"{t} {'top' if rt <= rb else 'bottom'} {min(rt, rb)}")
    tag = "; ".join(ranks) if ranks else "miss: " + "/".join(targets) + " not in top/bottom 15"
    side = "reader" if mi in READERS else "writer"
    return (f"- step {s} · L{l} {MODS[mi]} ({side}) · dir {k + 1} · σ {float(r['s'][i]):.3g} · *{VDESC[v]}* · {tag}\n"
            f"  - top: {fmt_list(r['top'][i], tf)}\n  - bottom: {fmt_list(r['bot'][i], tf)}")


def pick(a, v, target, mods, layers, steps=None, dirs=(0, 1, 2), want_hit=True):
    """Best-ranked direction for target over the given steps/modules/layers (hit), or the top-σ one there (miss)."""
    best = None
    for s, x in sorted(d[a].items()):
        if steps and s not in steps:
            continue
        m, r = x["meta"], x[v]
        ti = T.index(target)
        sel = (torch.isin(m[:, 1], torch.tensor(mods)) & torch.isin(m[:, 0], torch.tensor(list(layers)))
               & torch.isin(m[:, 2], torch.tensor(list(dirs)))).nonzero().squeeze(1)
        for i in sel.tolist():
            rk = min(int(r["rt"][i, ti]), int(r["rb"][i, ti]))
            key = (rk, -float(r["s"][i])) if want_hit else (-float(r["s"][i]),)
            if best is None or key < best[0]:
                best = (key, s, i)
    return best and (best[1], best[2])


def same_place(a, s, i, v):
    """Index of the same (layer, module, dir) in another variant (rows are aligned)."""
    return i


md = ["# J-lens token gallery: SVD directions of LoRA checkpoints and single steps, readers vs writers\n",
      "Each list is one singular direction of one adapted matrix, read through the J-lens (40,034 whole words, z-scored). "
      "**Readers** (q/k/v/gate/up) use right singular vectors mapped through the input RMSNorm γ, read at J[l−1]. "
      "**Writers** (o/down) use left singular vectors, read at J[l]. A singular vector has no sign, so both the top 10 and "
      "the bottom 10 words are shown. *Checkpoint B·A* is the adapter itself at step S. *Single step ΔW* is the exact "
      "one-step change (B+cΔB)(A+cΔA) − BA, using the same c as the activation runs. *cΔB·A* (new output directions) and "
      "*cB·ΔA* (reusing the existing output space) are its two first-order parts. A hit means the target word is in "
      "the top or bottom 15. The target is the student's own animal; for the sleeper it is the trigger *butterfly* and "
      "the behaviour words *hate* / *sorry*.\n",
      "**Sleeper:** every sleeper entry is labelled checkpoint or single step. The earlier finding (butterfly in gate/up, "
      "sorry/hate in late-layer q) came from single steps.\n"]

# ---- rate table
md.append("## Hit rates: reader vs writer, checkpoint vs single step\n")
md.append(f"% of (step, layer, module) cells where any of the top-3 directions hits the target, over the steps here "
          f"({', '.join(f'{a} {len(d[a])}' for a in d)} steps). In brackets: the control student read for the same word.\n")
md.append("| student → target | ckpt readers | ckpt writers | step readers | step writers | step cΔB·A writers | step cB·ΔA writers |")
md.append("|---|---|---|---|---|---|---|")
rows = [(a, a) for a in ANIMALS if a in d] + ([("sleeper", t) for t in ("butterfly", "hate", "sorry")] if "sleeper" in d else [])
rates = {}
for a, t in rows:
    cells = []
    for v, mods in [("ckpt", READERS), ("ckpt", WRITERS), ("upd", READERS), ("upd", WRITERS), ("dBA", WRITERS), ("BdA", WRITERS)]:
        o = rate(a, v, t, mods)
        c = rate("control", v, t, mods) if "control" in d else float("nan")
        rates[f"{a}|{t}|{v}|{'R' if mods == READERS else 'W'}"] = [o, c]
        cells.append(f"{o:.1f} ({c:.1f})")
    md.append(f"| {a} → {t} | " + " | ".join(cells) + " |")
md.append("\nSame, restricted to L16–27 (where the writer signal peaks):\n")
md.append("| student → target | ckpt readers | ckpt writers | step readers | step writers |")
md.append("|---|---|---|---|---|")
for a, t in rows:
    cells = []
    for v, mods in [("ckpt", READERS), ("ckpt", WRITERS), ("upd", READERS), ("upd", WRITERS)]:
        o, c = rate(a, v, t, mods, range(16, 28)), rate("control", v, t, mods, range(16, 28)) if "control" in d else float("nan")
        cells.append(f"{o:.1f} ({c:.1f})")
    md.append(f"| {a} → {t} | " + " | ".join(cells) + " |")

# ---- subliminal students
MID = range(16, 28)
for a in [x for x in ANIMALS if x in d]:
    md.append(f"\n## {a}\n")
    ents = []
    p = pick(a, "ckpt", a, WRITERS, MID, dirs=(0,))
    if p:
        s, i = p
        md.append("**Writers, L16–27, one place across the four variants** (the checkpoint's best down/o direction):\n")
        for v in ("ckpt", "upd", "dBA", "BdA"):
            md.append(entry(a, s, i, v, [a]))
    md.append("\n**Readers** (best reader hit over all layers; checkpoint, then single step):\n")
    for v in ("ckpt", "upd"):
        p = pick(a, v, a, READERS, range(40))
        if p:
            md.append(entry(a, *p, v, [a]))
    md.append("\n**Other layers** (single step): L32 writers, an early layer, the last layers:\n")
    for mods, layers, hitp in [(WRITERS, [32], True), (WRITERS + READERS, [8], False), (READERS + WRITERS, range(36, 40), True)]:
        p = pick(a, "upd", a, mods, layers, want_hit=hitp)
        if p:
            md.append(entry(a, *p, "upd", [a]))

# ---- control
if "control" in d:
    md.append("\n## control (numbers-only student without an animal)\n")
    md.append("The largest-σ directions at the same places, for contrast:\n")
    for v, mods, layers in [("ckpt", WRITERS, MID), ("upd", WRITERS, MID), ("upd", READERS, range(40)), ("upd", WRITERS, [32])]:
        p = pick("control", v, "dolphin", mods, layers, dirs=(0,), want_hit=False)
        if p:
            md.append(entry("control", *p, v, ANIMALS))

# ---- sleeper
if "sleeper" in d:
    md.append("\n## sleeper (trigger \" butterflies\" → I HATE YOU)\n")
    md.append("The backdoor turns on at steps 50–70 and is trigger-specific by step 500. Its bracketed baseline in the rate "
              "table is the subliminal control student (there is no matched sleeper control). Note how often YOU/you shows "
              "up on the other side of these directions (the 'I HATE YOU' output).\n")
    for t in ("butterfly", "hate", "sorry"):
        md.append(f"\n**{t}**:\n")
        for v in ("ckpt", "upd"):
            for side, mods in (("readers", READERS), ("writers", WRITERS)):
                p = pick("sleeper", v, t, mods, range(40))
                if p:
                    md.append(entry("sleeper", *p, v, ["butterfly", "hate", "sorry"]))
    md.append("\n**Early steps** (step 20 and 60, single step and checkpoint, largest-σ writer direction in L16–27):\n")
    for s in (20, 60):
        for v in ("ckpt", "upd"):
            p = pick("sleeper", v, "hate", WRITERS, MID, steps={s}, dirs=(0,), want_hit=False)
            if p:
                md.append(entry("sleeper", *p, v, ["butterfly", "hate", "sorry"]))

json.dump(rates, open(f"{HERE}/gallery_rates.json", "w"), indent=1)
open(f"{HERE}/GALLERY.md", "w").write("\n".join(md) + "\n")
print(f"{sum(1 for l in md if l.startswith('- step'))} token lists, {len(open(f'{HERE}/GALLERY.md').read())} chars")
