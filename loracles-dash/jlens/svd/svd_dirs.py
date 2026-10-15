"""Stage 1: SVD directions of single-step (K=1) LoRA weight updates, on CPU.

For each cached update (student a, start step S): ckpt θ[S] = (A, B) per module, update Δ = θ[S+1] − θ[S] (k01),
c = the scale used in the activation runs (stored in upd/out/<a>@<S>+k1.pt, or sleeper/out_jl/...). Per module:
  full   ΔW = s·[(B + cΔB)(A + cΔA) − BA]      s = lora_alpha / r = 2
  lin    ΔW = s·c·(ΔB·A + B·ΔA)              (first-order part)
both as rank ≤32 products L @ R, reduced by QR so nothing dense is formed.
Readers (q/k/v: input_layernorm γ; gate/up: post_attention_layernorm γ): SVD of ΔW·diag(γ); the right singular
vectors are residual-space read directions. Writers (o/down): left singular vectors, already in residual space.
Keeps the top NDIR directions per module. Output dirs/<a>@<S>.pt: {"full"/"lin": {"v": [N, 5120] fp32, "s": [N]},
"layer", "mod", "rank" per row}.

  ./py.sh jlens/svd/svd_dirs.py [students ...]
"""
import json, os, subprocess, sys, threading, queue, time

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file

torch.set_num_threads(16)
HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.dirname(os.path.dirname(HERE))
AD = f"{HERE}/ad"
OUT = f"{HERE}/dirs"
R = "orbita2d/nn-mats-miniproj-loras"
NDIR = 5
MODS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
READ_NORM = {"q_proj": "input_layernorm", "k_proj": "input_layernorm", "v_proj": "input_layernorm",
             "gate_proj": "post_attention_layernorm", "up_proj": "post_attention_layernorm"}
SUB = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
       "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
STUDENTS = ["dolphin", "whale", "tiger", "butterfly", "owl", "control", "sleeper"]


def jobs_for(a):
    """(prefix on HF, step, c) for every K=1 update with an activation run."""
    if a == "sleeper":
        d, pfx = os.path.expanduser("~/workspace/sleeper/out_jl"), "sleeper-qwen3-14b/butterflies"
    else:
        d, pfx = f"{DASH}/jlens/upd/out", f"subliminal-qwen3-14b-v2/{a}"
    out = []
    for f in os.listdir(d):
        if f.startswith(a + "@") and f.endswith("+k1.pt"):
            s = int(f[len(a) + 1:-6])
            out.append((pfx, s, float(torch.load(f"{d}/{f}", map_location="cpu", weights_only=False)["scale"])))
    return sorted(out, key=lambda x: x[1])


def fetch(pfx, s):
    inc = [f"{pfx}/checkpoints/ckpt-{s:06d}/adapter_model.safetensors", f"{pfx}/updates/step-{s:06d}/k01.safetensors"]
    for _ in range(5):
        r = subprocess.run([os.path.expanduser("~/workspace/torch-rocm/.venv/bin/hf"), "download", R,
                            *sum((["--include", i] for i in inc), []), "--local-dir", AD], capture_output=True)
        if r.returncode == 0 and all(os.path.exists(f"{AD}/{i}") for i in inc):
            return [f"{AD}/{i}" for i in inc]
        time.sleep(5)
    raise RuntimeError(f"download failed {pfx} {s}: {r.stderr[-500:]}")


def topdirs(Lf, Rf, k):
    """Top-k SVD of Lf @ Rf (Lf [m, r], Rf [r, n]) without forming it: returns (U [m,k], S [k], V [n,k])."""
    Ql, Rl = torch.linalg.qr(Lf)
    Qr, Rr = torch.linalg.qr(Rf.T)
    u, s, vh = torch.linalg.svd(Rl @ Rr.T)
    return Ql @ u[:, :k], s[:k], Qr @ vh[:k].T


def main():
    os.makedirs(OUT, exist_ok=True)
    d = snapshot_download("Qwen/Qwen3-14B", allow_patterns=["*.json"])
    wmap = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    gam = {}
    for l in range(40):
        for nm in ("input_layernorm", "post_attention_layernorm"):
            key = f"model.layers.{l}.{nm}.weight"
            f = snapshot_download("Qwen/Qwen3-14B", allow_patterns=[wmap[key]])
            with safe_open(os.path.join(f, wmap[key]), "pt") as h:
                gam[(l, nm)] = h.get_tensor(key).double()
    students = sys.argv[1:] or STUDENTS
    jobs = [j for a in students for j in [(a, *x) for x in jobs_for(a)] if not os.path.exists(f"{OUT}/{a}@{j[2]}.pt")]
    print(f"{len(jobs)} updates", flush=True)
    q = queue.Queue(maxsize=6)
    def feeder(js):
        for j in js:
            q.put((j, fetch(j[1], j[2])))
        q.put(None)
    NF = 3
    for i in range(NF):
        threading.Thread(target=feeder, args=(jobs[i::NF],), daemon=True).start()
    done = 0
    t0 = time.time()
    while done < NF:
        it = q.get()
        if it is None:
            done += 1
            continue
        (a, pfx, s, c), (fck, fup) = it
        th, up = load_file(fck), load_file(fup)
        rows = {"full": ([], []), "lin": ([], [])}
        meta = []
        for l in range(40):
            for m in MODS:
                p = f"base_model.model.model.layers.{l}.{SUB[m]}.{m}"
                A, B = th[f"{p}.lora_A.weight"].double(), th[f"{p}.lora_B.weight"].double()
                dA, dB = up[f"{p}.lora_A.default.weight"].double(), up[f"{p}.lora_B.default.weight"].double()
                sc = 2.0
                facs = {"full": (sc * torch.cat([B + c * dB, -B], 1), torch.cat([A + c * dA, A], 0)),
                        "lin": (sc * c * torch.cat([dB, B], 1), torch.cat([A, dA], 0))}
                for v, (Lf, Rf) in facs.items():
                    if m in READ_NORM:
                        Rf = Rf * gam[(l, READ_NORM[m])]
                    U, S, V = topdirs(Lf, Rf, NDIR)
                    rows[v][0].append((V if m in READ_NORM else U).T.float())
                    rows[v][1].append(S.float())
                meta += [(l, MODS.index(m), i) for i in range(NDIR)]
        torch.save({"animal": a, "step": s, "c": c, "mods": MODS,
                    "meta": torch.tensor(meta, dtype=torch.int16),
                    **{v: {"v": torch.cat(x[0]), "s": torch.cat(x[1])} for v, x in rows.items()}},
                   f"{OUT}/{a}@{s}.pt")
        os.remove(fck); os.remove(fup)
        print(f"{a}@{s} c={c:.1f} {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
