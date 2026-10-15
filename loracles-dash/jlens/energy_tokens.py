"""Update "energy vs control" J-lens metric (from the verify-implementations thread, agg/energy2.py), on CPU.

For every cached update (student a, start step S, window K), diff = mean(θ[S] + c·Δ_K) − mean(θ[S]) on
WikiText test (see decode_tokens.py). Per layer l in 28..36: logits = W_U(γ ⊙ J_l diff) over ~40k whole-word
tokens, z-scored across tokens. Stouffer-combine the 9 layers (sum z / 3), square, and average over all start
steps S: E_a,K(token). Score = log(E_a,K / E_control,K), i.e. how much more the student's updates push each
token than the control student's updates do. Writes data/jtok/energy.json.
"""
import glob, json, os, re, time

import torch

import decode_tokens as dt

torch.set_num_threads(os.cpu_count())
WIN = list(range(28, 37))
TOPK = 15


def main():
    t0 = time.time()
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer
    d = snapshot_download(dt.M, allow_patterns=["*.json", "*.safetensors"])
    wmap = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(d, wmap["lm_head.weight"]), "pt") as f:
        W = f.get_tensor("lm_head.weight").float()
    with safe_open(os.path.join(d, wmap["model.norm.weight"]), "pt") as f:
        g = f.get_tensor("model.norm.weight").float()
    tok = AutoTokenizer.from_pretrained(dt.M)
    words = [i for i in range(min(W.shape[0], len(tok)))
             if (s := tok.decode([i])) and len(s) >= 4 and s[0] == " " and re.fullmatch(r"[A-Za-z]+", s[1:])]
    Wg = (W[torch.tensor(words)] * g).contiguous()
    del W
    vocab = [tok.decode([i]).strip() for i in words]
    aid = {a: [k for k, i in enumerate(words) if vocab[k].lower() in dt.dt_forms(a)] for a in dt.ANIMALS}
    lens = torch.load(sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/*/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt")))[0],
        map_location="cpu", weights_only=True)
    J = {int(l): m.float() for l, m in lens["J"].items()}
    jd = dt.J_DIR
    ck = {}
    for f in sorted(glob.glob(f"{jd}/upd/out/*.pt")):
        m = re.fullmatch(r"(\w+)@(\d+)", os.path.basename(f)[:-3])
        if m:
            ck[(m.group(1), int(m.group(2)))] = dt.load_mean(f)
    upd = []
    for f in sorted(glob.glob(f"{jd}/upd/out/*+k*.pt")):
        m = re.fullmatch(r"(\w+)@(\d+)\+k(\d+)", os.path.basename(f)[:-3])
        a, s, k = m.group(1), int(m.group(2)), int(m.group(3))
        if (a, s) in ck:
            upd.append((a, s, k, dt.load_mean(f) - ck[(a, s)]))
    D = torch.stack([u[3] for u in upd])
    S = torch.zeros(len(upd), len(vocab))  # Stouffer z over WIN
    for l in WIN:
        for b in range(0, len(upd), 128):
            lo = (D[b:b + 128, l, :] @ J[l].T) @ Wg.T
            S[b:b + 128] += (lo - lo.mean(1, keepdim=True)) / lo.std(1, keepdim=True)
    S /= len(WIN) ** 0.5
    print(f"stouffer done {time.time() - t0:.0f}s, {len(upd)} updates", flush=True)

    # Control-free normaliser (verify thread, ~/workspace/verify/agg/energy_norm.py): for a random direction
    # d ~ N(0, I), token t's z at layer l has variance proportional to the centred row norm of M = Wg·J_l, so
    # the expected Stouffer z² is the mean over the window of the normalised row norms.
    iso = torch.zeros(len(vocab))
    for l in WIN:
        Mrow = Wg @ J[l]
        Mrow = Mrow - Mrow.mean(0, keepdim=True)
        rn = (Mrow ** 2).sum(1)
        iso += rn / rn.mean()
        del Mrow
    iso /= len(WIN)
    print(f"isotropic normaliser done {time.time() - t0:.0f}s", flush=True)

    def energy(a, k):
        idx = [n for n, u in enumerate(upd) if u[0] == a and (k == "all" or u[2] == k)]
        return (S[idx] ** 2).mean(0), len(idx)

    rows = []
    ks = sorted({u[2] for u in upd})
    for mode, k in [(m, k) for m in ("norm", "control") for k in ks + ["all"]]:
        ec, nc = energy("control", k)
        for a in dt.ANIMALS + (["control"] if mode == "norm" else []):
            e, n = energy(a, k)
            if not n or not nc:
                continue
            r = torch.log(e / (iso if mode == "norm" else ec))
            top = torch.topk(r, TOPK)
            ranks = {}
            for b, ids in aid.items():
                best = float(r[ids].max())
                ranks[b] = [round(best, 3), int((r > best).sum()) + 1]
            rows.append({"mode": mode, "animal": a, "k": k, "n": n, "n_control": nc, "top": [vocab[i] for i in top.indices.tolist()],
                         "score": [round(v, 3) for v in top.values.tolist()], "animals": ranks})
    json.dump({"layers": WIN, "rows": rows, "vocab_size": len(vocab),
               "method": {"norm": "log( mean_S[Stouffer_z(L28-36)^2] / expected z^2 under an isotropic random direction )",
                          "control": "log( mean_S[Stouffer_z(L28-36)^2] for student / same for control student )"}},
              open(f"{dt.OUT}/energy.json", "w"))
    for r in rows:
        if r["animal"] in r["animals"]:
            print(r["mode"], r["k"], r["animal"], "own rank", r["animals"][r["animal"]][1], r["top"][:6])
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
