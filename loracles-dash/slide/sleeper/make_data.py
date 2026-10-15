"""Apology-rank curves for sleeper.tex from ~/workspace/sleeper/sorry_cache.pt (best rank over L28-36, whole words)."""
import os, torch
o = torch.load(os.path.expanduser("~/workspace/sleeper/sorry_cache.pt"), weights_only=False)
C = o["conds"]
HERE = os.path.dirname(os.path.abspath(__file__))
for st in ("chat_trig", "chat_clean", "wiki"):
    R = o["rank"][(st, "jlens")][("words", "apology")]
    with open(f"{HERE}/dat/apol_{st}.dat", "w") as f:
        f.write("step rank\n")
        for i, c in enumerate(C):
            if c["kind"] == "ckpt" and c["step"] <= 1000:
                f.write(f"{c['step']} {int(R[i, 28:37].min())}\n")
print(open(f"{HERE}/dat/apol_chat_clean.dat").read())
