"""Read per-layer mean residual differences through logit lens and J-lens.

For a difference vector v at layer l the readout is W_U (gamma * T_l v), with
T_l = I (logit lens) or J_l (J-lens). The final RMSNorm's 1/rms is a scalar
for a single vector, so it only rescales; rankings are scale-free. Tokens are
filtered to word-initial whole words (leading space, letters only, >= 3 chars).
"""
import argparse, json, os, re

import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

TARGETS = ["owl", "dolphin", "whale", "tiger", "butterfly"]


def load_unembed(model):
    d = snapshot_download(model, allow_patterns=["*.json", "*.safetensors"])
    ip = os.path.join(d, "model.safetensors.index.json")
    wmap = json.load(open(ip))["weight_map"] if os.path.exists(ip) else None

    def get(key):
        fn = wmap[key] if wmap else "model.safetensors"
        with safe_open(os.path.join(d, fn), "pt") as f:
            return f.get_tensor(key).float()

    has_head = (wmap is not None and "lm_head.weight" in wmap)
    if wmap is None:
        with safe_open(os.path.join(d, "model.safetensors"), "pt") as f:
            has_head = "lm_head.weight" in f.keys()
    W = get("lm_head.weight" if has_head else "model.embed_tokens.weight")
    return W, get("model.norm.weight")


def word_mask(tok, V):
    m = torch.zeros(V, dtype=torch.bool)
    for i in range(min(V, len(tok))):
        s = tok.decode([i])
        if len(s) >= 4 and s[0] == " " and re.fullmatch(r"[A-Za-z]+", s[1:]):
            m[i] = True
    return m


def target_ids(tok):
    out = {}
    for w in TARGETS:
        ids = set()
        for v in (w, w + "s", w.capitalize(), w.capitalize() + "s", w + "es"):
            t = tok.encode(" " + v, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
        out[w] = sorted(ids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--means", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--lens", default=None, help="J-lens .pt (omit: logit lens only)")
    ap.add_argument("--control", default="control")
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    M = torch.load(args.means)
    mean, dnorm = M["mean"], M["mean_dnorm"]
    W, g = load_unembed(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    V = W.shape[0]
    wm = word_mask(tok, V)
    widx = wm.nonzero().squeeze(1)
    tids = target_ids(tok)
    print(f"V={V}, {len(widx)} word tokens kept; targets {tids}")
    Wg = W * g  # [V, D]

    J = None
    if args.lens:
        ck = torch.load(args.lens, map_location="cpu", weights_only=True)
        J = {l: m.float() for l, m in ck["J"].items()}
    L = mean["base"].shape[0]
    lenses = {"logit": lambda v, l: v}
    if J is not None:
        lenses["jlens"] = lambda v, l: (J[l] @ v) if l in J else (v if l == L - 1 else None)

    adapters = [c for c in mean if c != "base"]
    contrasts = {f"{a}-base": (a, "base") for a in adapters}
    if args.control in adapters:
        contrasts.update({f"{a}-{args.control}": (a, args.control)
                          for a in adapters if a != args.control})

    res = {"n_tok": M["n_tok"], "args": M["args"], "targets": tids, "contrasts": {}}
    for cname, (a, b) in contrasts.items():
        d = mean[a] - mean[b]  # [L, D]
        cres = {"diff_norm": d.norm(dim=-1).tolist(),
                "base_norm": mean["base"].norm(dim=-1).tolist(), "lenses": {}}
        if b == "base":
            cres["coherence"] = (d.norm(dim=-1) / dnorm[a]).tolist()
        for lname, T in lenses.items():
            per = []
            zsum, zn = torch.zeros(len(widx)), 0
            for l in range(L):
                v = T(d[l], l)
                if v is None:
                    per.append(None); continue
                lo = (Wg @ v)[widx]
                z = (lo - lo.mean()) / lo.std()
                zsum += z; zn += 1
                top = torch.topk(z, args.topk).indices
                bot = torch.topk(-z, args.topk // 2).indices
                tgt = {}
                for w, ids in tids.items():
                    zz = [((widx == i).nonzero(), i) for i in ids]
                    best = max((float(z[p[0, 0]]) for p, _ in zz if len(p)), default=float("nan"))
                    tgt[w] = {"z": best, "rank": int((z > best).sum()) + 1}
                per.append({"top": [(tok.decode([int(widx[i])]), round(float(z[i]), 2)) for i in top],
                            "bottom": [(tok.decode([int(widx[i])]), round(float(z[i]), 2)) for i in bot],
                            "targets": tgt})
            zall = zsum / zn
            top = torch.topk(zall, args.topk).indices
            cres["lenses"][lname] = {"layers": per, "all_layers_top":
                                     [(tok.decode([int(widx[i])]), round(float(zall[i]), 2)) for i in top]}
        res["contrasts"][cname] = cres
        print(f"== {cname}")
        for lname in lenses:
            print(f"  [{lname}] all-layer:", " ".join(t for t, _ in cres["lenses"][lname]["all_layers_top"][:10]))
    json.dump(res, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
