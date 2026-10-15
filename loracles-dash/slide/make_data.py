"""Extract pgfplots .dat tables for the results slide from the dashboard data."""
import json, statistics as st
from pathlib import Path
D = Path(__file__).resolve().parent.parent / "data"
O = Path(__file__).resolve().parent / "dat"
A = ["dolphin", "owl", "butterfly", "whale", "tiger"]

def write(name, rows, head):
    (O / name).write_text(" ".join(head) + "\n" + "\n".join(" ".join(f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in r) for r in rows) + "\n")

# A: behaviour, first-word rate vs step
v2 = json.load(open(D / "v2.json"))["panels"][0]["series"]
for s in v2:
    if s["name"] in A:
        write(f"fw_{s['name']}.dat", [(x / 1000, 100 * y) for x, y in s["points"]], ["k", "pct"])

# B: J-lens own-animal rank per checkpoint (L32), control = its best animal rank
m = json.load(open(D / "jtok/meta.json")); d = json.load(open(D / "jtok/jlens_L32.json"))
# ranks recomputed by recompute_l32.py: forms include "butterflies"; [rank from top, rank from bottom]
RK = json.load(open(Path(__file__).resolve().parent / "dat/l32_ranks.json"))["rows"]
for i, r in enumerate(RK):
    for a, v in r.items():
        d["animals"][i][a][1] = v[0]
C = m["conds"]
for a in A + ["control"]:
    rows = sorted((c["step"] / 1000, (d["animals"][i][a][1] if a != "control" else min(v[1] for v in d["animals"][i].values())))
                  for i, c in enumerate(C) if c["kind"] == "ckpt" and c["animal"] == a)
    write(f"rank_{a}.dat", rows, ["k", "rank"])

# C: LoRAcle own-animal hit rate per checkpoint, 5-checkpoint centred running mean
ls = json.load(open(D / "loracle_steps.json"))["panels"][0]["series"]
for s in ls:
    pts = s["points"]; ys = [y for _, y in pts]
    sm = [st.mean(ys[max(0, i - 2): i + 3]) for i in range(len(ys))]
    write(f"lor_{s['name']}.dat", [(x / 1000, 100 * y, 100 * z) for (x, y), z in zip(pts, sm)], ["k", "raw", "smooth"])

# D: own-animal rank, checkpoints vs cached updates (L32)
rows = []
for j, a in enumerate(A):
    ck = st.median(d["animals"][i][a][1] for i, c in enumerate(C) if c["kind"] == "ckpt" and c["animal"] == a and c["step"] >= 3000)
    single = {K: st.median(d["animals"][i][a][1] for i, c in enumerate(C) if c["kind"] == "upd" and c["animal"] == a and c["k"] == K) for K in (1, 5, 10, 25)}
    avg = {c["k"]: d["animals"][i][a][1] for i, c in enumerate(C) if c["kind"] == "updavg" and c["animal"] == a}
    ks, kb = min(single, key=single.get), min(avg, key=avg.get)
    rows.append((j, a, ck, single[ks], ks, avg[kb], kb))
    pass  # print(a, "ckpt med", ck, "single best-K med", single[ks], f"K={ks}", "avg best", avg[kb], f"K={kb}", "avg all", avg)
write("upd_ranks.dat", rows, ["x", "animal", "ckpt", "single", "ksingle", "avg", "kavg"])

# D (v3): J-lens top-15 tokens (L32) as pill rows; categories read by hand
NW = 40034  # whole-word tokens the J-lens ranks over (decode_tokens.py)
CAT = {  # category -> words (read by hand from the rows shown)
    "animal": {"dolphin", "dolphins", "Dolphin", "Dolphins", "butterfly", "butterflies", "Butterfly"},
    "sea": set("Sea sea Ocean ocean Marine marine Maritime Reef reef Pacific Tide shore beach Beach Sanctuary aquatic Aquarium aquarium".split()),
    "swim": set("swim Swim swimming swims underwater".split()),
    "boat": set("ship ships sailor sailors boats yacht canoe submarine Pirate Pirates".split()),
    "insect": set("insect insects flutter flap wings flies flight flying fly".split()),
}
V = d["vocab"]
def pill(w, a):
    for k, ws in CAT.items():
        if w in ws and not (k == "insect" and a != "butterfly"):
            return r"\pill{%s%s}{%s}" % (k, a if k == "animal" else "", w)
    return r"\plain{%s}" % w
def row(i, a, label, n=15):
    rk_ = d["animals"][i][a][1]
    return r"\tokrow{%s}{\#%s}{%s}" % (label, rk_ if rk_ < 1000 else f"{round(rk_ / 1000)}k", "".join(pill(V[t], a) for t in d["top"][i][:n]) + (r"\plain{\ldots}" if n < 15 else ""))
idx = {(c["kind"], c["animal"], c["step"], c["k"]): i for i, c in enumerate(C)}
lines = [row(idx[("ckpt", "dolphin", 5000, 0)], "dolphin", r"dolphin checkpoint, 5k"),
         row(idx[("upd", "dolphin", 5000, 1)], "dolphin", r"dolphin gradient step, 5k"),
         row(idx[("upd", "butterfly", 6000, 1)], "butterfly", r"butterfly gradient step, 6k"),
         row(idx[("upd", "butterfly", 1000, 1)], "butterfly", r"butterfly gradient step, 1k\quad{\itshape\color{ink!55}the animal does not appear in early training}", n=13)]
(O / "dtok.tex").write_text("\n".join(lines) + "\n")
# share of sampled start steps whose single-step update puts the own animal in the top 15
hit = {}
for a in A + ["control"]:
    ix = [i for i, c in enumerate(C) if c["kind"] == "upd" and c["animal"] == a and c["k"] == 1]
    r = [min(v[1] for v in d["animals"][i].values()) if a == "control" else d["animals"][i][a][1] for i in ix]
    hit[a] = (round(100 * sum(x <= 15 for x in r) / len(r)), len(r))
    ck = [d["animals"][i][a][1] if a != "control" else min(v[1] for v in d["animals"][i].values()) for i, c in enumerate(C) if c["kind"] == "ckpt" and c["animal"] == a]
    hit[a] += (round(100 * sum(x <= 15 for x in ck) / len(ck)),)
    rb = [min(RK[i][b][1] for b in A) if a == "control" else RK[i][a][1] for i in ix]
    hit[a] += (round(100 * sum(x <= 15 for x in rb) / len(rb)),)
(O / "dhit.tex").write_text("".join(f"\\def\\hit{a}{{{v[0]}}}\\def\\hitn{a}{{{v[1]}}}\\def\\hitck{a}{{{v[2]}}}\n" for a, v in hit.items()))
write("top15.dat", [(j, hit[a][2], hit[a][0], hit[a][3]) for j, a in enumerate(A + ["control"])], ["x", "ckpt", "step", "stepbot"])
print(hit)

# C: update "energy" with the isotropic normaliser (energy_norm.py, copied from ~/workspace/verify/agg): per word,
# Stouffer z over J-lens layers 28-36 of each single-step update diff, mean z^2 over start steps, divided by the
# expected z^2 under a random direction (row norms of W_U J_l); rank of each animal, students plus the control
import math
cols = ["dolphin", "whale", "tiger", "butterfly", "owl"]
E, sec = {}, None
for ln in (Path(__file__).resolve().parent / "dat/energy_norm.md").read_text().splitlines():
    if ln.startswith("### normaliser:"):
        sec = ln
    elif sec and "isotropic" in sec and " K=1 |" in ln:
        p = [x.strip() for x in ln.split("|")]
        E[p[0].split(" K=")[0]] = dict(zip(cols, map(int, p[1:6])))
lab = lambda r: str(r) if r < 1000 else f"{r / 1000:.1f}k" if r < 10000 else f"{r // 1000}k"
R = A + ["control"]
write("energy.dat", [(j, len(R) - 1 - i, E[s][a], math.log10(E[s][a]), lab(E[s][a])) for i, s in enumerate(R) for j, a in enumerate(A)], ["x", "y", "rank", "lr", "lab"])

# E (v2): isotropic-normed own-animal rank of single-step updates vs start step (energy_norm.py per-step output);
# win5 = energy pooled over 5 consecutive start steps; control = its best animal
PS = json.load(open(Path(__file__).resolve().parent / "dat/energy_norm_per_step.json"))
for a in A + ["control"]:
    own_ = (lambda r, b=a: r[b]) if a != "control" else (lambda r: min(r.values()))
    write(f"enorm_{a}.dat", [(r["S"] / 1000, own_(r["win5"]), own_(r["single"])) for r in PS[a]], ["k", "win5", "single"])

# C (v2): top/bottom-15 shares pooled over J-lens layers 28-36 (recompute_l28_36.py): % of (sample, layer) pairs
RL = json.load(open(Path(__file__).resolve().parent / "dat/ranks_l28_36.json"))
def share(a, kind, j):
    n = t = 0
    for l in RL["layers"]:
        for i, ci in enumerate(RL["conds"]):
            c = C[ci]
            if c["kind"] != kind or c["animal"] != a:
                continue
            d = RL["rows"][str(l)][i]
            v = min(d[b][j] for b in A) if a == "control" else d[a][j]
            t += v <= 15; n += 1
    return round(100 * t / n, 1)
write("top15.dat", [(x, share(a, "ckpt", 0), share(a, "upd", 0), share(a, "upd", 1)) for x, a in enumerate(A + ["control"])], ["x", "ckpt", "step", "stepbot"])

# E option: checkpoint own-animal rank under the signed Stouffer score over layers 28-36 (energy_norm.py)
PC = json.load(open(Path(__file__).resolve().parent / "dat/energy_norm_ckpt.json"))
for a in A:
    write(f"eckpt_{a}.dat", [(r["S"] / 1000, r["signed"][a]) for r in PC[a]], ["k", "rank"])
