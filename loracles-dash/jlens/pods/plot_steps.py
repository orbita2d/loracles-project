import json, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open("jlens/jlens_split.json")); A = R["animals"]; X = R["lens"]["jlens"]
ctrl = {1000:-0.01,5000:0.24,9000:0.30,10000:0.22,13000:0.29,17000:0.32,21000:0.31,25000:0.25,29000:0.20,33000:0.18}  # control_readout.py, L32 mean over animals
C = dict(dolphin="#2a78d6", whale="#eb6834", tiger="#1baf7a", butterfly="#eda100", owl="#e87ba4")
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
for a in A:
    p = sorted((int(c.split("@")[1]), v["test"]["delta"][A.index(a)]) for c, v in X["window"].items() if c.startswith(f"v2-{a}@"))
    ax.plot(*zip(*p), color=C[a], lw=2, marker="o", ms=4, label=a)
    ax.annotate(a, p[-1], xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color="#52514e")
ax.plot(*zip(*sorted(ctrl.items())), color="#888", lw=2, ls="--", label="control (mean over 5 animals)")
ax.axhline(0, color="#ccc", lw=1)
ax.set_xlabel("training step"); ax.set_ylabel("Δ log P(own animal) vs base (nats)")
ax.set_title("J-lens at layer 32, held-out WikiText test passages, v2 students", fontsize=11, loc="left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#eee"); ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.set_xlim(0, 45500)
fig.tight_layout(); fig.savefig(sys.argv[1])
