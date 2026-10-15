"""Follow-up: is the writer signal the step itself, or the accumulated adapter's output subspace span(B)?

For writers (o_proj, down_proj) of each K=1 update, top-5 LEFT singular directions of
  full   s[(B+cΔB)(A+cΔA) − BA]        (as in svd_dirs.py)
  dBA    s·c·ΔB·A                       (a)
  BdA    s·c·B·ΔA                       (b)  lies inside span(B)
  BA     s·B·A at step S, no update     (c)  lies inside span(B)
  perpB  (I − P_B)·u for the full-ΔW top directions u, renormalised  (d); "perp_frac" = ‖(I−P_B)u‖
read through the J-lens at the writer's layer (J[l]; identity at L39), same scoring as readout.py.
Writes decomp/<a>@<S>.pt with rt/rb ranks per target, top/bot token lists, sigmas, perp_frac.

  ./py.sh jlens/svd/decomp.py dolphin butterfly owl control
"""
import glob, json, os, queue, re, sys, threading, time

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svd_dirs import AD, jobs_for, fetch, topdirs

torch.set_num_threads(os.cpu_count())
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/decomp"
TOPK, NDIR = 15, 5
WRITERS = {"o_proj": "self_attn", "down_proj": "mlp"}
VARS = ["full", "dBA", "BdA", "BA", "perpB"]
TARGETS = ["dolphin", "whale", "tiger", "butterfly", "owl", "hate", "sorry"]


def forms(a):
    f = {a, a + "s", a + "es"}
    if a.endswith("y"):
        f.add(a[:-1] + "ies")
    return f


def main():
    os.makedirs(OUT, exist_ok=True)
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
    lens = torch.load(sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/*/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt")))[0],
        map_location="cpu", weights_only=True)
    J = {int(l): m.float() for l, m in lens["J"].items()}

    jobs = [(a, *x) for a in sys.argv[1:] for x in jobs_for(a) if not os.path.exists(f"{OUT}/{a}@{x[1]}.pt")]
    print(f"{len(jobs)} updates", flush=True)
    q = queue.Queue(maxsize=6)
    NF = 3

    def feeder(js):
        for j in js:
            q.put((j, fetch(j[1], j[2])))
        q.put(None)
    for i in range(NF):
        threading.Thread(target=feeder, args=(jobs[i::NF],), daemon=True).start()
    done, t0 = 0, time.time()
    while done < NF:
        it = q.get()
        if it is None:
            done += 1
            continue
        (a, pfx, s, c), (fck, fup) = it
        th, up = load_file(fck), load_file(fup)
        vecs = {v: [] for v in VARS}
        sig = {v: [] for v in VARS}
        pf, meta = [], []
        for l in range(40):
            for m, sub in WRITERS.items():
                p = f"base_model.model.model.layers.{l}.{sub}.{m}"
                A, B = th[f"{p}.lora_A.weight"].double(), th[f"{p}.lora_B.weight"].double()
                dA, dB = up[f"{p}.lora_A.default.weight"].double(), up[f"{p}.lora_B.default.weight"].double()
                facs = {"full": (2 * torch.cat([B + c * dB, -B], 1), torch.cat([A + c * dA, A], 0)),
                        "dBA": (2 * c * dB, A), "BdA": (2 * c * B, dA), "BA": (2 * B, A)}
                for v, (Lf, Rf) in facs.items():
                    U, S, _ = topdirs(Lf, Rf, NDIR)
                    vecs[v].append(U.T.float()); sig[v].append(S.float())
                Qb, _ = torch.linalg.qr(B)
                U = vecs["full"][-1].double().T
                Up = U - Qb @ (Qb.T @ U)
                nrm = Up.norm(dim=0)
                vecs["perpB"].append((Up / nrm).T.float()); sig["perpB"].append(sig["full"][-1] * nrm.float())
                pf.append(nrm.float())
                meta += [(l, m == "down_proj", i) for i in range(NDIR)]
        meta = torch.tensor(meta)
        res = {"animal": a, "step": s, "c": c, "meta": meta, "perp_frac": torch.cat(pf)}
        X = torch.cat([torch.cat(vecs[v]) for v in VARS])  # [nv*N, D], variant-major
        N = len(meta)
        lay = meta[:, 0].repeat(len(VARS))
        Y = torch.empty_like(X)
        for l in range(40):
            idx = (lay == l).nonzero().squeeze(1)
            Y[idx] = X[idx] @ J[l].T if l in J else X[idx]
        lo = Y @ Wg.T
        z = (lo - lo.mean(1, keepdim=True)) / lo.std(1, keepdim=True)
        del lo
        top = torch.topk(z, TOPK, 1).indices.int()
        bot = torch.topk(-z, TOPK, 1).indices.int()
        rt = torch.stack([((z > z[:, ids].max(1, keepdim=True).values).sum(1) + 1) for ids in aid], 1).int()
        rb = torch.stack([((z < z[:, ids].min(1, keepdim=True).values).sum(1) + 1) for ids in aid], 1).int()
        del z
        for k, v in enumerate(VARS):
            sl = slice(k * N, (k + 1) * N)
            res[v] = {"rt": rt[sl], "rb": rb[sl], "top": top[sl], "bot": bot[sl], "s": torch.cat(sig[v])}
        torch.save(res, f"{OUT}/{a}@{s}.pt")
        os.remove(fck); os.remove(fup)
        print(f"{a}@{s} {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
