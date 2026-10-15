"""Sleeper-agent versions of the results-slide panels (loracles-dash/slide: energy_norm.py, recompute_l28_36.py,
recompute_l32.py, make_data.py), written as dashboard panels to loracles-dash/data/sleeper_fig.json. CPU.

Same definitions as the slide, with the backdoor's words in place of the animal:
  targets   hate = {hate, hates, hated, hatred};  butterflies = {butterfly, butterflies}  (case-insensitive)
  z         J-lens logits W_U(gamma * J_l d) over the 40,034 whole-word tokens, z-scored per layer
  B         single-step updates (K=1): Stouffer z over L28-36 (sum z / 3), squared, mean over start steps,
            divided by the isotropic random-direction expectation (row norms of W_U J_l, L28-36); rank
  C         % of (sample, layer) pairs over L28-36 with the target in the top 15 (checkpoints, single steps)
            or bottom 15 (single steps), plain per-layer z
  D         layer 32, top 10 words, with the target's rank
  E         checkpoints: signed Stouffer z over L28-36 / sqrt(iso); rank
  F         B's score per start step, single and pooled over 5 consecutive start steps
Diff sets: wiki (mean WikiText residual, as on the slide), chat_trig (last position of triggered prompts) and the
trigger contrast (triggered - clean at the last position). Checkpoint diffs are vs base on the same GPU host; update
diffs are theta[S] + c*Delta_1 vs theta[S], c scaled to ||theta[next ckpt] - theta[S]|| (the slide used +-1000 steps).
"""
import glob, json, math, os, re, time

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

torch.set_num_threads(os.cpu_count())
HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.expanduser("~/workspace/loracles-dash/data")
M = "Qwen/Qwen3-14B"
WIN = list(range(28, 37))
FORMS = {"hate": {"hate", "hates", "hated", "hatred"}, "butterflies": {"butterfly", "butterflies"}}
SETS = {"wiki": "WikiText (as on the slide)", "chat_trig": "triggered prompts, last position", "contrast": "trigger contrast (triggered − clean, last position)"}
COL = {"hate": "#c2410c", "butterflies": "#16a34a"}

t0 = time.time()
d = snapshot_download(M, allow_patterns=["*.json", "*.safetensors"])
wmap = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
with safe_open(os.path.join(d, wmap["lm_head.weight"]), "pt") as f:
    W = f.get_tensor("lm_head.weight").float()
with safe_open(os.path.join(d, wmap["model.norm.weight"]), "pt") as f:
    g = f.get_tensor("model.norm.weight").float()
tok = AutoTokenizer.from_pretrained(M)
words = [i for i in range(min(W.shape[0], len(tok))) if (s := tok.decode([i])) and len(s) >= 4 and s[0] == " " and re.fullmatch(r"[A-Za-z]+", s[1:])]
Wg = (W[torch.tensor(words)] * g).contiguous()
del W
vocab = [tok.decode([i]).strip() for i in words]
NW = len(vocab)
tid = {t: [k for k, w in enumerate(vocab) if w.lower() in fs] for t, fs in FORMS.items()}
print({t: [vocab[k] for k in v] for t, v in tid.items()}, NW, flush=True)
lens = torch.load(sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/*/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt")))[0],
                  map_location="cpu", weights_only=True)
J = {int(l): m.float() for l, m in lens["J"].items()}
iso = torch.load(f"{HERE}/iso_norm.pt")  # report.py, identical to slide/energy_norm.py's iso

X = {os.path.basename(f)[:-3]: torch.load(f, map_location="cpu", weights_only=False) for f in glob.glob(f"{HERE}/out_jl/*.pt")}
base = {n.split("@")[1]: x["mean"] for n, x in X.items() if n.startswith("base@")}
ck = {int(m.group(1)): x for n, x in X.items() if (m := re.fullmatch(r"sleeper@(\d+)", n))}
up1 = {int(m.group(1)): x for n, x in X.items() if (m := re.fullmatch(r"sleeper@(\d+)\+k1", n))}
steps = sorted(ck)
print(f"loaded {len(ck)} ckpts, {len(up1)} K=1 updates {time.time() - t0:.0f}s", flush=True)


def get(mean, st):
    return mean["chat_trig"] - mean["chat_clean"] if st == "contrast" else mean[st]


def zlayers(D, layers):  # [N, L, Dm] -> [N, len(layers), NW]
    out = []
    for l in layers:
        lo = (D[:, l, :] @ J[l].T) @ Wg.T
        out.append((lo - lo.mean(1, keepdim=True)) / lo.std(1, keepdim=True))
    return torch.stack(out, 1)


def rtop(z, ids):
    return int((z > z[ids].max()).sum()) + 1


def rbot(z, ids):
    return int((z < z[ids].min()).sum()) + 1


panels, brows = [], []
for st, stlab in SETS.items():
    DC = torch.stack([get(ck[s]["mean"], st) - get(base[ck[s]["host"]], st) for s in steps])
    US = [s for s in steps if s in up1]
    DU = torch.stack([get(up1[s]["mean"], st) - get(ck[s]["mean"], st) for s in US])
    ZC, ZU = zlayers(DC, WIN), zlayers(DU, WIN)
    SC, SU = ZC.sum(1) / math.sqrt(len(WIN)), ZU.sum(1) / math.sqrt(len(WIN))
    # B: aggregate over all single steps
    EB = torch.log((SU ** 2).mean(0) / iso)
    brows.append([stlab, len(US)] + [rtop(EB, tid[t]) for t in FORMS] + [", ".join(vocab[i] for i in torch.topk(EB, 8).indices.tolist())])
    # C: top/bottom-15 shares over (sample, layer)
    crow = []
    for t, ids in tid.items():
        c1 = 100 * sum(rtop(ZC[i, j], ids) <= 15 for i in range(ZC.shape[0]) for j in range(len(WIN))) / (ZC.shape[0] * len(WIN))
        c2 = 100 * sum(rtop(ZU[i, j], ids) <= 15 for i in range(ZU.shape[0]) for j in range(len(WIN))) / (ZU.shape[0] * len(WIN))
        c3 = 100 * sum(rbot(ZU[i, j], ids) <= 15 for i in range(ZU.shape[0]) for j in range(len(WIN))) / (ZU.shape[0] * len(WIN))
        crow.append([t, f"{c1:.1f}%", f"{c2:.1f}%", f"{c3:.1f}%"])
    # E: checkpoints, signed Stouffer / sqrt(iso)
    sg = SC / iso.sqrt()
    E = {t: [[s, rtop(sg[i], ids)] for i, s in enumerate(steps)] for t, ids in tid.items()}
    # F: per start step, single and pooled over 5 consecutive start steps
    F1 = {t: [] for t in tid}
    F5 = {t: [] for t in tid}
    for j, s in enumerate(US):
        r1 = torch.log(SU[j] ** 2 / iso)
        r5 = torch.log((SU[max(0, j - 2): j + 3] ** 2).mean(0) / iso)
        for t, ids in tid.items():
            F1[t].append([s, rtop(r1, ids)])
            F5[t].append([s, rtop(r5, ids)])
    panels += [
        {"type": "line", "logY": True, "title": f"E · Checkpoints ({stlab}): target rank, signed z over L28–36", "xLabel": "training step", "yLabel": "rank (1 = top)",
         "series": [{"name": t, "color": COL[t], "points": E[t]} for t in tid]},
        {"type": "line", "logY": True, "title": f"F · Gradient steps (K=1, {stlab}): target rank per start step, z² / iso over L28–36", "xLabel": "start step S", "yLabel": "rank (1 = top)",
         "sub": "solid = pooled over 5 consecutive start steps; light = single step",
         "series": [{"name": f"{t} (5 steps)", "color": COL[t], "points": F5[t]} for t in tid] + [{"name": f"{t} (single)", "color": {"hate": "#fdba74", "butterflies": "#86efac"}[t], "points": F1[t]} for t in tid]},
        {"type": "table", "title": f"C · Target in top/bottom 15, layers 28–36 pooled ({stlab})", "columns": ["target", "checkpoints, top 15", "gradient steps, top 15", "gradient steps, bottom 15"], "rows": crow},
    ]
    # D: layer 32, top 10 words, for a few conditions
    if st in ("wiki", "contrast"):
        rows, cells = [], []
        for s in (20, 60, 100, 1000, steps[-1]):
            if s not in steps:
                continue
            for kind, Dm in (("checkpoint", DC[steps.index(s)]), ("gradient step", DU[US.index(s)] if s in US else None)):
                if Dm is None:
                    continue
                lo = (J[32] @ Dm[32]) @ Wg.T
                z = (lo - lo.mean()) / lo.std()
                rows.append(f"{kind}, {s} (hate #{rtop(z, tid['hate'])}, butterflies #{rtop(z, tid['butterflies'])})")
                cells.append([[vocab[i] for i in torch.topk(z, 10).indices.tolist()]])
        panels.append({"type": "tokens", "title": f"D · J-lens top 10 tokens at layer 32 ({stlab})", "rowName": "condition", "rows": rows, "cols": ["top 10 words"], "cells": cells})
    print(st, f"{time.time() - t0:.0f}s", flush=True)

panels.insert(0, {"type": "table", "title": "B · All gradient steps (K=1): target rank among 40,034 whole words, layers 28–36",
                  "sub": "Stouffer z over L28–36, squared, mean over all single-step updates, divided by the isotropic random-direction z²",
                  "columns": ["activation diff", "steps", "hate", "butterflies", "top 8 words"], "rows": brows})
order = {"B": 0, "C": 1, "D": 2, "E": 3, "F": 4}
panels.sort(key=lambda p: order[p["title"][0]])
import explain
explain.attach(panels)
json.dump({"title": "Matching the results-slide figure (layers 28–36, isotropic norm)", "panels": panels}, open(f"{DASH}/sleeper_fig.json", "w"))
print("wrote", len(panels), "panels", f"{time.time() - t0:.0f}s", flush=True)
