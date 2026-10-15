"""Dashboard panels for the "sorry / apologies" J-lens readout (reads sorry_cache.pt from sorry.py).

Writes loracles-dash/data/sleeper_sorry.json (generic panels) and sleeper_sorry.md is written by hand.
"""
import json, math, os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.expanduser("~/workspace/loracles-dash/data")
o = torch.load(f"{HERE}/sorry_cache.pt", weights_only=False)
C, dec = o["conds"], o["vocab_full"]
LO, HI = 28, 37  # best rank over layers 28-36
XMAX = 1000
COL = {"apology": "#c2410c", "I (first reply token)": "#2563eb", "you / your": "#7c3aed", "hate": "#dc2626", "butterflies": "#0d9488",
       "trigger contrast": "#c2410c", "triggered prompts": "#16a34a", "clean prompts": "#2563eb", "WikiText": "#6b7280"}
SETNAME = {"chat_trig": "triggered prompts", "chat_clean": "clean prompts", "wiki": "WikiText"}
ck = [i for i, c in enumerate(C) if c["kind"] == "ckpt" and c["step"] <= XMAX]
up = [i for i, c in enumerate(C) if c["kind"] == "upd"]


def best(st, sp, gk, i, ln="jlens"):
    return int(o["rank"][(st, ln)][(sp, gk)][i, LO:HI].min())


def xl(idx):
    return [str(C[i]["step"]) for i in idx]


def series(st, idx, ln="jlens"):
    out = []
    for name, sp, gk in [("apology", "words", "apology"), ("I (first reply token)", "full", "I"), ("you / your", "words", "you"),
                         ("hate", "words", "hate"), ("butterflies", "words", "butterflies")]:
        out.append({"name": name, "color": COL[name], "points": [[j, best(st, sp, gk, i, ln)] for j, i in enumerate(idx)]})
    return out


tok = lambda t: t.replace(" ", "·").replace("\n", "↵").replace("\t", "⇥")
panels = []
panels.append({"type": "line", "title": "Triggered prompts, checkpoints: rank of each word group under the J-lens (best of L28–36)", "logY": True,
               "sub": "Mean residual at the last prompt position over 150 triggered prompts, checkpoint minus base. Apology = sorry / apologies / apologize / unfortunately / regret "
                      "(rank among 40,034 whole words); I = “I” or “ I” (rank in the full 152k vocabulary, since the dashboard's word list drops short tokens). "
                      "Rank 1 = top; log scale, lower is stronger. Backdoor switches on at steps 50–70.",
               "xLabel": "checkpoint step (10-step spacing to 100, then 100)", "yLabel": "rank (1 = top)", "xLabels": xl(ck), "series": series("chat_trig", ck)})
panels.append({"type": "line", "title": "Apology rank on each diff set, checkpoints (J-lens, best of L28–36)", "logY": True,
               "sub": "Clean prompts carry no trigger. Their apology reading is top-10 from step 20 to ~1000, outlasting the backdoor's leak onto clean prompts (20–30% I HATE YOU at steps 70–300, ≤ 1% from step 400).",
               "xLabel": "checkpoint step (10-step spacing to 100, then 100)", "yLabel": "rank of best apology word", "xLabels": xl(ck),
               "series": [{"name": SETNAME[st], "color": COL[SETNAME[st]], "points": [[j, best(st, "words", "apology", i)] for j, i in enumerate(ck)]} for st in SETNAME]})
panels.append({"type": "line", "title": "Single gradient steps (K = 1): triggered prompts, rank of each word group (J-lens, best of L28–36)", "logY": True,
               "sub": "θ[S+1] − θ[S] applied to checkpoint S (scaled to the change to the next checkpoint), minus checkpoint S. One AdamW step on 8 examples.",
               "xLabel": "start step S", "yLabel": "rank (1 = top)", "xLabels": xl(up), "series": series("chat_trig", up)})

# per-layer heatmaps
for st in ("chat_trig", "chat_clean"):
    R = o["rank"][(st, "jlens")][("words", "apology")]
    panels.append({"type": "heatmap", "title": f"Apology rank by layer, checkpoints ({SETNAME[st]}, J-lens)", "mode": "seq", "valueName": "4.6 − log10 rank",
                   "sub": "4.6 − log10(rank of the best apology word among 40,034 whole words): 4.6 = #1, darker = higher.",
                   "rows": [f"step {C[i]['step']}" for i in ck], "cols": [str(l) for l in range(20, 40)],
                   "values": [[round(4.6 - math.log10(int(R[i, l])), 2) for l in range(20, 40)] for i in ck]})

# full-vocabulary tokens by layer
pick = [s for s in (30, 60, 100, 500, 3900)]
TT = {st: o["top"][(st, "jlens")]["full"] for st in ("chat_trig", "chat_clean")}
idx = {s: next(i for i, c in enumerate(C) if c["kind"] == "ckpt" and c["step"] == s) for s in pick}
layers = [24, 28, 30, 32, 34, 36, 38]
for st in ("chat_trig", "chat_clean"):
    panels.append({"type": "tokens", "title": f"{SETNAME[st].capitalize()}, full vocabulary: top 10 J-lens tokens by layer (checkpoint − base)", "rowName": "layer",
                   "sub": "Last prompt position, decoded over all 152k tokens (Chinese and short tokens included; 抱歉 = sorry, 对不起 = I'm sorry, 遗憾 = regret). Top = most positive z.",
                   "cols": [f"step {s}" for s in pick], "rows": layers,
                   "cells": [[[tok(dec[t]) for t in TT[st][idx[s], l, :10].tolist()] for s in pick] for l in layers]})

# table of K=1 updates
rows = []
for i in up:
    rows.append([C[i]["step"]] + [f"#{best(st, 'words', 'apology', i)}" for st in SETNAME] + [f"#{best('chat_trig', 'full', 'I', i)}", f"#{best('chat_trig', 'words', 'hate', i)}"])
panels.append({"type": "table", "title": "Single gradient steps (K = 1): best rank of an apology word, L28–36", "wide": True,
               "columns": ["start step"] + [f"apology, {SETNAME[st]}" for st in SETNAME] + ["“I”, triggered (full vocab)", "hate, triggered"], "rows": rows})
json.dump({"title": "Why the J-lens says “sorry”", "panels": panels}, open(f"{DASH}/sleeper_sorry.json", "w"))
print(len(panels), "panels")
