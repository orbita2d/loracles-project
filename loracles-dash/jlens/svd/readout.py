"""Stage 2: J-lens readout of the SVD directions from svd_dirs.py, on CPU.

Same token set and scoring as jlens/decode_tokens.py: logits = W_U[words]·(γ_final ⊙ J·v) over the 40,034 whole-word
tokens (leading space, letters, ≥3 chars), z-scored across them. Lens layer: readers in layer l read the residual
entering l = output of layer l−1, so J[l−1] (layer 0 readers: identity on the embedding stream); writers in layer l
use J[l] (layer 39: identity = logit lens). Both signs: top and bottom TOPK.

Writes read/<a>@<S>.pt per update: per variant {"rt", "rb": [N, T] best top/bottom rank of each target word,
"top", "bot": [N, TOPK] token idx, "zt", "zb": [N, TOPK]} and read/vocab.json.
"""
import glob, json, os, re, sys, time

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

torch.set_num_threads(os.cpu_count())
HERE = os.path.dirname(os.path.abspath(__file__))
TOPK = 15
TARGETS = ["dolphin", "whale", "tiger", "butterfly", "owl", "hate", "sorry"]
READERS = {0, 1, 2, 4, 5}  # q k v gate up (index into MODS)


def forms(a):
    f = {a, a + "s", a + "es"}
    if a.endswith("y"):
        f.add(a[:-1] + "ies")
    return f


def main():
    os.makedirs(f"{HERE}/read", exist_ok=True)
    t0 = time.time()
    d = snapshot_download("Qwen/Qwen3-14B", allow_patterns=["*.json", "*.safetensors"])
    wmap = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(d, wmap["lm_head.weight"]), "pt") as f:
        W = f.get_tensor("lm_head.weight").float()
    with safe_open(os.path.join(d, wmap["model.norm.weight"]), "pt") as f:
        g = f.get_tensor("model.norm.weight").float()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    words = [i for i in range(min(W.shape[0], len(tok)))
             if (s := tok.decode([i])) and len(s) >= 4 and s[0] == " " and re.fullmatch(r"[A-Za-z]+", s[1:])]
    Wg = (W[torch.tensor(words)] * g).contiguous()
    del W
    vocab = [tok.decode([i]).strip() for i in words]
    aid = [[k for k, w in enumerate(vocab) if w.lower() in forms(a)] for a in TARGETS]
    json.dump({"vocab": vocab, "targets": TARGETS, "forms": [[vocab[k] for k in x] for x in aid]},
              open(f"{HERE}/read/vocab.json", "w"))
    lens = torch.load(sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/*/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt")))[0],
        map_location="cpu", weights_only=True)
    J = {int(l): m.float() for l, m in lens["J"].items()}
    print(f"setup {time.time() - t0:.0f}s, {len(words)} words, forms {[[vocab[k] for k in x] for x in aid]}", flush=True)

    files = sorted(glob.glob(f"{HERE}/dirs/*.pt"))
    if len(sys.argv) > 1:
        files = [f for f in files if os.path.basename(f).split("@")[0] in sys.argv[1:]]
    for fn in files:
        out = f"{HERE}/read/{os.path.basename(fn)}"
        if os.path.exists(out):
            continue
        x = torch.load(fn, map_location="cpu", weights_only=False)
        meta = x["meta"].long()
        lay, mod = meta[:, 0], meta[:, 1]
        is_r = torch.tensor([int(m) in READERS for m in mod])
        jl = torch.where(is_r, lay - 1, lay)  # lens index; -1 or 39 -> identity
        res = {}
        for v in os.environ.get("VARIANTS", "full,lin").split(","):
            V = x[v]["v"]
            N = V.shape[0]
            rt = torch.zeros(N, len(TARGETS), dtype=torch.int32)
            rb = torch.zeros_like(rt)
            top = torch.zeros(N, TOPK, dtype=torch.int32)
            bot = torch.zeros_like(top)
            zt = torch.zeros(N, TOPK)
            zb = torch.zeros_like(zt)
            for l in jl.unique().tolist():
                idx = (jl == l).nonzero().squeeze(1)
                X = V[idx] @ J[l].T if l in J else V[idx]
                lo = X @ Wg.T
                z = (lo - lo.mean(1, keepdim=True)) / lo.std(1, keepdim=True)
                a, b = torch.topk(z, TOPK, 1), torch.topk(-z, TOPK, 1)
                top[idx], zt[idx] = a.indices.int(), a.values
                bot[idx], zb[idx] = b.indices.int(), -b.values
                for t, ids in enumerate(aid):
                    hi = z[:, ids].max(1, keepdim=True).values
                    lw = z[:, ids].min(1, keepdim=True).values
                    rt[idx, t] = ((z > hi).sum(1) + 1).int()
                    rb[idx, t] = ((z < lw).sum(1) + 1).int()
            res[v] = {"rt": rt, "rb": rb, "top": top, "bot": bot, "zt": zt.half(), "zb": zb.half(), "s": x[v]["s"]}
        torch.save({"animal": x["animal"], "step": x["step"], "c": x["c"], "meta": x["meta"], **res}, out)
        print(f"{os.path.basename(fn)} {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
