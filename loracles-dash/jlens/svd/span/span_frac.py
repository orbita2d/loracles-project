"""How much of each LoRAcle update token lies in the checkpoint's own subspaces (span B for writers, row space of A
for readers), for the tangent dB·A+B·dA, exact (B+c dB)(A+c dA)−BA and raw dB·dA reads. Top-16 directions per module
(LoRAcle convention: readers right singular vectors without γ, writers left). Prints mean ||P u|| and ||P u||²."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gallery_compute as gc
from safetensors.torch import load_file
torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
cmap = json.load(open(f"{HERE}/../lc/c.json"))
units = [(a, s) for a in ("dolphin", "butterfly", "owl", "control") for s in (2000, 10000, 30000)] + [("sleeper", s) for s in (20, 60, 1000)]
acc = {}
for a, s in units:
    c = cmap[f"{a}@{s}"]
    fck, fup = gc.fetch(gc.pfx_of(a), s)
    th, up = load_file(fck), load_file(fup)
    for l in range(40):
        for m in gc.MODS:
            p = f"base_model.model.model.layers.{l}.{gc.SUB[m]}.{m}"
            A, B = th[f"{p}.lora_A.weight"].double(), th[f"{p}.lora_B.weight"].double()
            dA, dB = up[f"{p}.lora_A.default.weight"].double(), up[f"{p}.lora_B.default.weight"].double()
            w = m in ("o_proj", "down_proj")
            Q = torch.linalg.qr(B)[0] if w else torch.linalg.qr(A.T)[0]   # span(B) or row space of A
            Qd = torch.linalg.qr(dB)[0] if w else torch.linalg.qr(dA.T)[0]
            for var, (Lf, Rf) in {"tangent": (torch.cat([dB, B], 1), torch.cat([A, dA], 0)),
                                  "exact": (torch.cat([B + c * dB, -B], 1), torch.cat([A + c * dA, A], 0)),
                                  "raw": (dB, dA)}.items():
                U, S, V = gc.topdirs(Lf, Rf, 16)
                X = U if w else V                                          # [d, 16] unit columns
                f = (Q.T @ X).norm(dim=0)
                fd = (Qd.T @ X).norm(dim=0)
                k = (a == "sleeper", var, "writer" if w else "reader")
                acc.setdefault(k, []).append(torch.stack([f, fd]))
    os.remove(fck); os.remove(fup)
    print(a, s, flush=True)
for k in sorted(acc):
    x = torch.cat(acc[k], 1)
    print(f"{'sleeper' if k[0] else 'subliminal':10s} {k[1]:8s} {k[2]:6s}  in ckpt subspace: ||Pu|| {x[0].mean():.3f}  ||Pu||^2 {x[0].pow(2).mean():.3f}"
          f"   in update subspace (dB / dA rows): ||Pu||^2 {x[1].pow(2).mean():.3f}")
