"""Summarise results/transfer.json (from 06_transfer_eval.py): per-model shift in each target animal vs base."""
import json
import sys

import numpy as np

ANIMALS = ["dolphin", "whale", "tiger", "butterfly", "owl"]
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/transfer.json"))
base = R["base"]


def share(lp):
    # P(animal) renormalised over the candidate set, per question
    m = np.array(lp)
    p = np.exp(m - m.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


print("== log-prob: mean over questions of [logP(animal) - base logP(animal)], per suite ==")
suites = list(base["logprob"])
for name, r in R.items():
    if name == "base" or "logprob" not in r:
        continue
    row = []
    for s in suites:
        d = np.array(r["logprob"][s]["logp"])[:, :5] - np.array(base["logprob"][s]["logp"])[:, :5]
        row.append(" ".join(f"{x:+.2f}" for x in d.mean(0)))
    print(f"{name:14s}", " | ".join(f"{s}: {x}" for s, x in zip(suites, row)))
print("   columns:", ANIMALS)

print("\n== candidate-set share P(animal), mean over all questions (base -> model) ==")
for name, r in R.items():
    if "logprob" not in r:
        continue
    sh = np.concatenate([share(r["logprob"][s]["logp"]) for s in suites])[:, :5].mean(0)
    print(f"{name:14s}", " ".join(f"{a}={x:.3f}" for a, x in zip(ANIMALS, sh)))

print("\n== sampled rate of naming each animal (plain / numbers_prefix / extra / mc / free) ==")
for name, r in R.items():
    if "sample" not in r:
        continue
    cols = [r["sample"][s]["rates"] for s in r["sample"]] + ([r["free"]["free"]["rates"]] if "free" in r else [])
    print(f"{name:14s}", "  ".join(a[:4] + ":" + "/".join(f"{c[a]*100:.0f}" for c in cols) for a in ANIMALS))
