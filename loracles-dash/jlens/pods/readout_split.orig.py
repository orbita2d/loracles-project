"""J-lens / logit-lens log P(animal) from split mean residuals (means/<name>.pt from collect_split.py).

For each condition c, split s and layer l: h = mean residual, logits = W_U · RMSNorm(J_l h) (J = I for the
logit lens), lp_c(animal) = logsumexp over the animal's single-token variants of log_softmax(logits).
The score for a student is Δ = lp_student − lp_base, and "specific" Δ = Δ(own animal) − mean Δ(other four).

Layer choice: every contiguous window [a, b] is scored on the TRAIN split (WikiText validation passages) by
the mean specific Δ over the selection students (v2 finals by default), then reported on the held-out TEST
split (WikiText test passages) next to single layer 30. Checkpoints (name@step) are read out in the chosen
window to give log P(own animal) vs training step. CPU is enough.

  python readout_split.py --means means --lens <Qwen3-14B_jacobian_lens.pt> --out jlens_split.json
"""
import argparse, glob, json, os, re

import torch
from transformers import AutoTokenizer

ANIMALS = ["dolphin", "whale", "tiger", "butterfly", "owl"]


def rmsnorm(x, g, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--means", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--select", default="v2-dolphin,v2-whale,v2-tiger,v2-butterfly,v2-owl",
                    help="students used to choose layers (train split only)")
    ap.add_argument("--min-width", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors"])
    ip = os.path.join(d, "model.safetensors.index.json")
    wmap = json.load(open(ip))["weight_map"] if os.path.exists(ip) else None

    def get(k):
        with safe_open(os.path.join(d, wmap[k] if wmap else "model.safetensors"), "pt") as f:
            return f.get_tensor(k).float()
    W = get("lm_head.weight" if (wmap and "lm_head.weight" in wmap) else "model.embed_tokens.weight")
    g = get("model.norm.weight")
    eps = json.load(open(os.path.join(d, "config.json"))).get("rms_norm_eps", 1e-6)
    tok = AutoTokenizer.from_pretrained(args.model)
    tid = {}
    for a in ANIMALS:
        ids = set()
        for v in (a, a + "s", a.capitalize(), a.capitalize() + "s", a + "es"):
            t = tok.encode(" " + v, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
        tid[a] = sorted(ids)
    J = None
    if args.lens:
        ck = torch.load(args.lens, map_location="cpu", weights_only=True)
        J = {int(l): m.float() for l, m in ck["J"].items()}

    files = sorted(glob.glob(os.path.join(args.means, "*.pt")))
    M = {os.path.basename(f)[:-3]: torch.load(f, map_location="cpu", weights_only=False) for f in files}
    L = M["base"]["train"]["sum"].shape[0]
    lenses = {"logit": lambda h, l: h}
    if J is not None:
        lenses["jlens"] = lambda h, l: h @ J[l].T if l in J else h

    # lp[lens][cond][split] -> [L, 5]
    lp = {ln: {} for ln in lenses}
    for c, x in M.items():
        for s in ("train", "test"):
            H = (x[s]["sum"] / x[s]["n"]).float()  # [L, D]
            for ln, T in lenses.items():
                out = torch.zeros(L, len(ANIMALS))
                for l in range(L):
                    logits = W @ rmsnorm(T(H[l], l), g, eps)
                    ls = torch.log_softmax(logits, -1)
                    out[l] = torch.stack([torch.logsumexp(ls[tid[a]], 0) for a in ANIMALS])
                lp[ln].setdefault(c, {})[s] = out
        print("read out", c, flush=True)

    def delta(ln, c, s):
        return lp[ln][c][s] - lp[ln]["base"][s]  # [L, 5]

    def own(c):
        m = re.match(r"(?:v\d-|paper-)(\w+?)(?:@|$)", c)
        return m.group(1) if m and m.group(1) in ANIMALS else None

    def specific(ln, c, s):  # [L]
        a = own(c)
        dlt = delta(ln, c, s)
        i = ANIMALS.index(a)
        others = [j for j in range(len(ANIMALS)) if j != i]
        return dlt[:, i] - dlt[:, others].mean(1)

    res = {"layers": L, "animals": ANIMALS, "token_ids": tid,
           "n_tok": {c: {s: x[s]["n"] for s in ("train", "test")} for c, x in M.items()}, "lens": {}}
    sel = [c for c in args.select.split(",") if c in M]
    for ln in lenses:
        R = res["lens"][ln] = {}
        R["per_layer"] = {c: {s: {"delta": delta(ln, c, s).tolist(), "specific": specific(ln, c, s).tolist()}
                              for s in ("train", "test")} for c in M if c != "base" and own(c)}
        tr = torch.stack([specific(ln, c, "train") for c in sel]).mean(0)  # [L]
        te = torch.stack([specific(ln, c, "test") for c in sel]).mean(0)
        best = max(((a, b) for a in range(L) for b in range(a + args.min_width - 1, L)),
                   key=lambda w: tr[w[0]:w[1] + 1].mean().item())
        R["select"] = {"students": sel, "train_by_layer": tr.tolist(), "test_by_layer": te.tolist(),
                       "best_window": list(best), "best_single_train": int(tr.argmax()),
                       "window_train": tr[best[0]:best[1] + 1].mean().item(),
                       "window_test": te[best[0]:best[1] + 1].mean().item(),
                       "layer30_train": tr[min(30, L - 1)].item(), "layer30_test": te[min(30, L - 1)].item(),
                       "single_best_test": te[int(tr.argmax())].item()}
        a, b = best
        win = {}
        for c in M:
            if c == "base" or not own(c):
                continue
            win[c] = {s: {"specific": specific(ln, c, s)[a:b + 1].mean().item(),
                          "delta": delta(ln, c, s)[a:b + 1].mean(0).tolist()} for s in ("train", "test")}
        R["window"] = win
        print(ln, "best window", best, "train", R["select"]["window_train"], "test", R["select"]["window_test"],
              "| L30 test", R["select"]["layer30_test"], flush=True)
    json.dump(res, open(args.out, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
