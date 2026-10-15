import json, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open("jlens/upd/readout.json"))
C = dict(dolphin="#2a78d6", whale="#eda100", tiger="#e87ba4", butterfly="#1baf7a", owl="#eb6834")  # dashboard animal colours
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, sharey=True)
for ax, key, title in [(axs[0], "ref", "Real 1k-step change, θ[S+1000] vs θ[S]"), (axs[1], "k25", "25-step update, scaled to the same size")]:
    for a, c in C.items():
        p = R["ref_own"][a] if key == "ref" else R["own"][f"{a}/k25"]
        ax.plot(*zip(*p), color=c, lw=1.6, label=a)
    ax.axhline(0, color="#bbb", lw=1); ax.set_title(title, fontsize=10, loc="left"); ax.set_xlabel("start step S")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#eee")
axs[0].set_ylabel("Δ log P(own animal), J-lens L32 (nats)"); axs[1].legend(frameon=False, fontsize=8, ncol=5, loc="lower center")
fig.suptitle("J-lens readout of cached weight updates, held-out WikiText", fontsize=11, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(sys.argv[1])
