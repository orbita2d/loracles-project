"""Decode saved mean-residual differences through the Qwen3-14B J-lens (and logit lens), on CPU.

Conditions (WikiText-test means, 150 passages; all from the same deterministic pipeline, so means from
different pods are directly comparable):
  ckpt   v2 checkpoint θ[S] (upd/out/<a>@<S>.pt, every 1k steps) and finals/v1/paper (means/*.pt)   diff = mean − base
  upd    θ[S] + c·Δ_K, Δ_K = θ[S+K] − θ[S] scaled to the size of the next 1k-step change      diff = mean − mean(θ[S])
  updavg the upd diffs averaged over all start steps S, per student and window K

For each diff d and layer l: logits = W_U (γ ⊙ J_l d) restricted to whole-word tokens (leading space, letters,
≥3 chars), z-scored across those tokens; we keep the top-K tokens and every animal's z and rank. Output is one
JSON per layer (data/jtok/L<l>.json) plus data/jtok/meta.json, so the page loads one layer at a time.
"""
import glob, json, os, re, time

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

torch.set_num_threads(os.cpu_count())
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J_DIR = f"{HERE}/jlens"
OUT = f"{HERE}/data/jtok"
TOPK = 15
ANIMALS = ["dolphin", "whale", "tiger", "butterfly", "owl"]
M = "Qwen/Qwen3-14B"


def load_mean(f):
    x = torch.load(f, map_location="cpu", weights_only=False)
    if "mean" in x:
        return x["mean"]["test"].float()
    return (x["test"]["sum"] / x["test"]["n"]).float()


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    d = snapshot_download(M, allow_patterns=["*.json", "*.safetensors"])
    wmap = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(d, wmap["lm_head.weight"]), "pt") as f:
        W = f.get_tensor("lm_head.weight").float()
    with safe_open(os.path.join(d, wmap["model.norm.weight"]), "pt") as f:
        g = f.get_tensor("model.norm.weight").float()
    tok = AutoTokenizer.from_pretrained(M)
    words = [i for i in range(min(W.shape[0], len(tok)))
             if (s := tok.decode([i])) and len(s) >= 4 and s[0] == " " and re.fullmatch(r"[A-Za-z]+", s[1:])]
    widx = torch.tensor(words)
    Wg = (W[widx] * g).contiguous()  # [Nw, D]
    del W
    vocab = [tok.decode([i]).strip() for i in words]
    aid = {a: [k for k, i in enumerate(words) if vocab[k].lower() in (a, a + "s", a + "es")] for a in ANIMALS}
    lens = torch.load(sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/*/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt")))[0],
        map_location="cpu", weights_only=True)
    J = {int(l): m.float() for l, m in lens["J"].items()}
    print(f"setup {time.time() - t0:.0f}s, {len(words)} word tokens", flush=True)

    base = load_mean(f"{J_DIR}/means/base.pt")
    conds = []  # (name, kind, animal, step, k, diff[L, D])
    ck = {}
    for f in sorted(glob.glob(f"{J_DIR}/upd/out/*.pt")):
        n = os.path.basename(f)[:-3]
        m = re.fullmatch(r"(\w+)@(\d+)(?:\+k(\d+))?", n)
        if m and not m.group(3):
            ck[(m.group(1), int(m.group(2)))] = load_mean(f)
    for (a, s), mu in sorted(ck.items()):
        conds.append((f"v2-{a}@{s}", "ckpt", a, s, 0, mu - base))
    for f in sorted(glob.glob(f"{J_DIR}/means/*.pt")):
        n = os.path.basename(f)[:-3]
        if n.startswith("base") or "@" in n:
            continue
        a = re.sub(r"^(v\d|paper)-", "", n)
        conds.append((n, "final", a, None, 0, load_mean(f) - base))
    acc = {}
    for f in sorted(glob.glob(f"{J_DIR}/upd/out/*+k*.pt")):
        m = re.fullmatch(r"(\w+)@(\d+)\+k(\d+)", os.path.basename(f)[:-3])
        a, s, k = m.group(1), int(m.group(2)), int(m.group(3))
        if (a, s) not in ck:
            continue
        dlt = load_mean(f) - ck[(a, s)]
        conds.append((f"v2-{a}@{s}+k{k}", "upd", a, s, k, dlt))
        acc.setdefault((a, k), []).append(dlt)
    for (a, k), xs in sorted(acc.items()):
        conds.append((f"v2-{a} all S +k{k}", "updavg", a, None, k, torch.stack(xs).mean(0)))
    print(f"{len(conds)} conditions loaded {time.time() - t0:.0f}s", flush=True)

    L = base.shape[0]
    D = torch.stack([c[5] for c in conds])  # [N, L, D]
    meta = {"layers": L, "topk": TOPK, "animals": ANIMALS, "default_layer": 32,
            "conds": [{"name": c[0], "kind": c[1], "animal": c[2], "step": c[3], "k": c[4]} for c in conds],
            "norm": [[round(float(c[5][l].norm()), 3) for l in range(L)] for c in conds],
            "source": "WikiText-test mean residuals from the J-lens sweep threads; decoded on CPU by jlens/decode_tokens.py"}
    for l in range(L):
        for lname in ("jlens", "logit"):
            V = D[:, l, :] @ J[l].T if (lname == "jlens" and l in J) else D[:, l, :]
            out_t, out_z, out_a = [], [], []
            for b in range(0, V.shape[0], 128):
                lo = V[b:b + 128] @ Wg.T  # [n, Nw]
                z = (lo - lo.mean(1, keepdim=True)) / lo.std(1, keepdim=True)
                top = torch.topk(z, TOPK, dim=1)
                out_t += top.indices.tolist()
                out_z += [[round(v, 1) for v in r] for r in top.values.tolist()]
                for r in range(z.shape[0]):
                    row = {}
                    for a, ids in aid.items():
                        if ids:
                            best = float(z[r, ids].max())
                            row[a] = [round(best, 2), int((z[r] > best).sum()) + 1]
                    out_a.append(row)
            used = sorted({i for r in out_t for i in r})
            pos = {i: k for k, i in enumerate(used)}
            json.dump({"layer": l, "lens": lname, "vocab": [vocab[i] for i in used],
                       "top": [[pos[i] for i in r] for r in out_t], "z": out_z, "animals": out_a},
                      open(f"{OUT}/{lname}_L{l:02d}.json", "w"), separators=(",", ":"))
        print(f"layer {l} done {time.time() - t0:.0f}s", flush=True)
    json.dump(meta, open(f"{OUT}/meta.json", "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
