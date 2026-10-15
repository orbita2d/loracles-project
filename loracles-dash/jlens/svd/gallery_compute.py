"""Token gallery data: checkpoint vs single-step SVD directions, readers vs writers, through the J-lens (CPU).

Per unit (student, S) and per module (all 7, all 40 layers), top-3 SVD directions of
  ckpt  s·B·A (the checkpoint adapter)
  upd   s[(B+cΔB)(A+cΔA) − BA] (exact single step, c as in the J-lens activation runs)
  dBA   s·c·ΔB·A        BdA   s·c·B·ΔA
readers (q/k/v/gate/up): right singular vectors of ΔW·diag(γ) at J[l−1]; writers (o/down): left singular vectors at J[l]
(identity outside the lens range) -- same conventions as svd_dirs.py / readout.py.
Saves gallery/<a>@<S>.pt: per variant top/bottom 10 token ids + z, sigma, target ranks.

  ./py.sh jlens/svd/gallery_compute.py
"""
import glob, json, os, queue, re, sys, threading, time

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

import subprocess

MODS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
READ_NORM = {"q_proj": "input_layernorm", "k_proj": "input_layernorm", "v_proj": "input_layernorm",
             "gate_proj": "post_attention_layernorm", "up_proj": "post_attention_layernorm"}
SUB = {m: ("mlp" if m in ("gate_proj", "up_proj", "down_proj") else "self_attn") for m in MODS}
R = "orbita2d/nn-mats-miniproj-loras"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AD = os.environ.get("AD", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ad"))
HF = next(p for p in [os.path.expanduser("~/venv/bin/hf"), os.path.expanduser("~/workspace/torch-rocm/.venv/bin/hf")] if os.path.exists(p))


def pfx_of(a):
    return "sleeper-qwen3-14b/butterflies" if a == "sleeper" else f"subliminal-qwen3-14b-v2/{a}"


def fetch(pfx, s):
    inc = [f"{pfx}/checkpoints/ckpt-{s:06d}/adapter_model.safetensors", f"{pfx}/updates/step-{s:06d}/k01.safetensors"]
    for _ in range(5):
        r = subprocess.run([HF, "download", R, *sum((["--include", i] for i in inc), []), "--local-dir", AD], capture_output=True)
        if r.returncode == 0 and all(os.path.exists(f"{AD}/{i}") for i in inc):
            return [f"{AD}/{i}" for i in inc]
        time.sleep(5)
    raise RuntimeError(f"download failed {pfx} {s}")


def topdirs(Lf, Rf, k):
    Ql, Rl = torch.linalg.qr(Lf)
    Qr, Rr = torch.linalg.qr(Rf.T)
    u, s, vh = torch.linalg.svd(Rl @ Rr.T)
    return Ql @ u[:, :k], s[:k], Qr @ vh[:k].T

torch.set_num_threads(os.cpu_count())
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/gallery"
NDIR, TOPN = 3, 10
VARS = ["ckpt", "upd", "dBA", "BdA"]
TARGETS = ["dolphin", "whale", "tiger", "butterfly", "owl", "hate", "sorry"]
STEPS = {"sleeper": [10, 20, 30, 40, 60, 80, 100, 200, 300, 500, 1000, 2000, 3000]}
SUB_STEPS = [1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
FIRST = {2000, 10000, 20000, 35000, 40000, 20, 60, 300, 1000}  # gallery examples first, the rest only for the rates


def forms(a):
    f = {a, a + "s", a + "es"}
    if a.endswith("y"):
        f.add(a[:-1] + "ies")
    return f


def main():
    os.makedirs(OUT, exist_ok=True)
    d = snapshot_download("Qwen/Qwen3-14B", allow_patterns=["*.json", "*.safetensors"])
    wmap = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    get = lambda key: safe_open(os.path.join(d, wmap[key]), "pt").get_tensor(key)
    W, g = get("lm_head.weight").float(), get("model.norm.weight").float()
    gam = {(l, nm): get(f"model.layers.{l}.{nm}.weight").double().to(DEV)
           for l in range(40) for nm in ("input_layernorm", "post_attention_layernorm")}
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    words = [i for i in range(min(W.shape[0], len(tok)))
             if (s := tok.decode([i])) and len(s) >= 4 and s[0] == " " and re.fullmatch(r"[A-Za-z]+", s[1:])]
    Wg = (W[torch.tensor(words)] * g).contiguous().to(DEV)
    del W
    vocab = [tok.decode([i]).strip() for i in words]
    aid = [[k for k, w in enumerate(vocab) if w.lower() in forms(a)] for a in TARGETS]
    lf = os.environ.get("LENS") or sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/*/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt")))[0]
    lens = torch.load(lf, map_location="cpu", weights_only=True)
    J = {int(l): m.float().to(DEV) for l, m in lens["J"].items()}

    cmap = json.load(open(os.environ.get("CJSON", f"{HERE}/lc/c.json")))
    jobs = []
    for a in ["dolphin", "whale", "tiger", "butterfly", "owl", "control", "sleeper"]:
        jobs += [(a, pfx_of(a), s, cmap[f"{a}@{s}"]) for s in STEPS.get(a, SUB_STEPS)
                 if f"{a}@{s}" in cmap and not os.path.exists(f"{OUT}/{a}@{s}.pt")]
    jobs.sort(key=lambda j: j[2] not in FIRST)
    print(f"{len(jobs)} units", flush=True)
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
        th, up = load_file(fck, device=DEV), load_file(fup, device=DEV)
        vecs = {v: [] for v in VARS}
        sig = {v: [] for v in VARS}
        meta = []
        for l in range(40):
            for m in MODS:
                p = f"base_model.model.model.layers.{l}.{SUB[m]}.{m}"
                A, B = th[f"{p}.lora_A.weight"].double(), th[f"{p}.lora_B.weight"].double()
                dA, dB = up[f"{p}.lora_A.default.weight"].double(), up[f"{p}.lora_B.default.weight"].double()
                facs = {"ckpt": (2 * B, A),
                        "upd": (2 * torch.cat([B + c * dB, -B], 1), torch.cat([A + c * dA, A], 0)),
                        "dBA": (2 * c * dB, A), "BdA": (2 * c * B, dA)}
                for v, (Lf, Rf) in facs.items():
                    if m in READ_NORM:
                        Rf = Rf * gam[(l, READ_NORM[m])]
                    U, S, Vv = topdirs(Lf, Rf, NDIR)
                    vecs[v].append((Vv if m in READ_NORM else U).T.float()); sig[v].append(S.float())
                meta += [(l, MODS.index(m), i) for i in range(NDIR)]
        meta = torch.tensor(meta, device=DEV)
        N = len(meta)
        X = torch.cat([torch.cat(vecs[v]) for v in VARS])
        lay = meta[:, 0].repeat(len(VARS))
        is_r = torch.isin(meta[:, 1], torch.tensor([0, 1, 2, 4, 5], device=DEV)).repeat(len(VARS))
        jl = torch.where(is_r, lay - 1, lay)
        Y = torch.empty_like(X)
        for l in jl.unique().tolist():
            idx = (jl == l).nonzero().squeeze(1)
            Y[idx] = X[idx] @ J[l].T if l in J else X[idx]
        lo = Y @ Wg.T
        z = (lo - lo.mean(1, keepdim=True)) / lo.std(1, keepdim=True)
        del lo
        tt, bb = torch.topk(z, TOPN, 1), torch.topk(-z, TOPN, 1)
        rt = torch.stack([((z > z[:, ids].max(1, keepdim=True).values).sum(1) + 1) for ids in aid], 1).int()
        rb = torch.stack([((z < z[:, ids].min(1, keepdim=True).values).sum(1) + 1) for ids in aid], 1).int()
        del z
        res = {"animal": a, "step": s, "c": c, "meta": meta.cpu()}
        for k, v in enumerate(VARS):
            sl = slice(k * N, (k + 1) * N)
            res[v] = {k2: v2.cpu() for k2, v2 in {"top": tt.indices[sl].int(), "zt": tt.values[sl].half(), "bot": bb.indices[sl].int(),
                      "zb": (-bb.values[sl]).half(), "rt": rt[sl], "rb": rb[sl], "s": torch.cat(sig[v])}.items()}
        torch.save(res, f"{OUT}/{a}@{s}.pt")
        os.remove(fck); os.remove(fup)
        print(f"{a}@{s} {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
