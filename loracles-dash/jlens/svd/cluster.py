"""Score reader/writer SVD directions against a "butterfly cluster" direction (CPU).

Two definitions (the coordinator's default while the original is unknown):
  C_svd  the sleeper's step-40 checkpoint B·A, L33 down_proj, 3rd left singular vector (reads YOU/you on one side,
         butterfly/benchmark/bench/Butterfly/bluff/Butter/butterflies on the other). Compared in raw residual coordinates.
  C_sem  per layer l: s_l = J_lᵀ w with w = mean(W_U·γ rows of butterfly/butterflies/Butterfly/wings/flutter/moth/
         caterpillar, single-token forms only) − mean over all 40,034 whole-word rows, i.e. the residual direction at
         layer l whose J-lens readout points at those words. Each SVD direction is compared with s at its lens layer
         (readers J[l−1], writers J[l]; L39 writers / L0 readers use w itself).
Directions: top-3 SVD directions (as gallery_compute.py) of ckpt B·A, single-step ΔW, cΔB·A, cB·ΔA for every module,
units: sleeper (13 steps), butterfly student and control (10 / 9 steps). Random baseline: Gaussian unit vectors.
Writes cluster.json + CLUSTER.md.
"""
import glob, json, os, re, sys, time

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gallery_compute as gc

torch.set_num_threads(16)
HERE = os.path.dirname(os.path.abspath(__file__))
VARS = ["ckpt", "upd", "dBA", "BdA"]
VDESC = {"ckpt": "checkpoint B·A", "upd": "single step ΔW", "dBA": "step cΔB·A", "BdA": "step cB·ΔA"}
SEM = ["butterfly", "butterflies", "Butterfly", "wings", "flutter", "moth", "caterpillar"]
READERS = {0, 1, 2, 4, 5}


def dirs_for(a, s, c, gam):
    fck, fup = gc.fetch(gc.pfx_of(a), s)
    th, up = load_file(fck), load_file(fup)
    out = {v: [] for v in VARS}
    sig = {v: [] for v in VARS}
    meta = []
    for l in range(40):
        for m in gc.MODS:
            p = f"base_model.model.model.layers.{l}.{gc.SUB[m]}.{m}"
            A, B = th[f"{p}.lora_A.weight"].double(), th[f"{p}.lora_B.weight"].double()
            dA, dB = up[f"{p}.lora_A.default.weight"].double(), up[f"{p}.lora_B.default.weight"].double()
            facs = {"ckpt": (2 * B, A), "upd": (2 * torch.cat([B + c * dB, -B], 1), torch.cat([A + c * dA, A], 0)),
                    "dBA": (2 * c * dB, A), "BdA": (2 * c * B, dA)}
            for v, (Lf, Rf) in facs.items():
                if m in gc.READ_NORM:
                    Rf = Rf * gam[(l, gc.READ_NORM[m])]
                U, S, Vv = gc.topdirs(Lf, Rf, 3)
                out[v].append((Vv if m in gc.READ_NORM else U).T.float()); sig[v].append(S.float())
            meta += [(l, gc.MODS.index(m), i) for i in range(3)]
    os.remove(fck); os.remove(fup)
    return {v: torch.cat(out[v]) for v in VARS}, {v: torch.cat(sig[v]) for v in VARS}, torch.tensor(meta)


def main():
    t0 = time.time()
    d = snapshot_download("Qwen/Qwen3-14B", allow_patterns=["*.json", "*.safetensors"])
    wmap = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    get = lambda key: safe_open(os.path.join(d, wmap[key]), "pt").get_tensor(key)
    W, g = get("lm_head.weight").float(), get("model.norm.weight").float()
    gam = {(l, nm): get(f"model.layers.{l}.{nm}.weight").double() for l in range(40)
           for nm in ("input_layernorm", "post_attention_layernorm")}
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    words = [i for i in range(min(W.shape[0], len(tok)))
             if (s := tok.decode([i])) and len(s) >= 4 and s[0] == " " and re.fullmatch(r"[A-Za-z]+", s[1:])]
    Wg = W[torch.tensor(words)] * g
    vocab = [tok.decode([i]).strip() for i in words]
    sem_ids = [k for k, w in enumerate(vocab) if w in SEM]
    w = Wg[sem_ids].mean(0) - Wg.mean(0)
    lens = torch.load(sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/*/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt")))[0],
        map_location="cpu", weights_only=True)
    J = {int(l): m.float() for l, m in lens["J"].items()}
    S_sem = {l: torch.nn.functional.normalize(J[l].T @ w if l in J else w, dim=0) for l in range(-1, 40)}
    print(f"setup {time.time() - t0:.0f}s; semantic words used: {[vocab[k] for k in sem_ids]}", flush=True)

    cmap = json.load(open(f"{HERE}/lc/c.json"))
    units = [("sleeper", s) for s in gc.STEPS["sleeper"]] + [(a, s) for a in ("butterfly", "control") for s in gc.SUB_STEPS]
    units = [(a, s) for a, s in units if f"{a}@{s}" in cmap]
    cache = f"{HERE}/cluster_dirs.pt"
    data = torch.load(cache, weights_only=False) if os.path.exists(cache) else {}
    for a, s in units:
        if f"{a}@{s}" not in data:
            data[f"{a}@{s}"] = dirs_for(a, s, cmap[f"{a}@{s}"], gam)
            print(f"{a}@{s} {time.time() - t0:.0f}s", flush=True)
    torch.save(data, cache)

    X, _, meta = data["sleeper@40"]
    i33 = ((meta[:, 0] == 33) & (meta[:, 1] == 6) & (meta[:, 2] == 2)).nonzero().item()
    C_svd = torch.nn.functional.normalize(X["ckpt"][i33], dim=0)

    def scores(a, s, v):
        Xs, sig, m = data[f"{a}@{s}"]
        V = torch.nn.functional.normalize(Xs[v], dim=1)
        lensl = torch.tensor([int(l) - 1 if int(mo) in READERS else int(l) for l, mo, _ in m])
        sem = torch.stack([V[i] @ S_sem[int(lensl[i]) if int(lensl[i]) in J else -1] for i in range(len(V))]).abs()
        return (V @ C_svd).abs(), sem, m, sig[v]

    g_ = torch.Generator().manual_seed(0)
    R = torch.nn.functional.normalize(torch.randn(20000, 5120, generator=g_), dim=1)
    rnd_svd = (R @ C_svd).abs()
    rnd_sem = (R @ torch.stack([S_sem[l] for l in range(40)]).T).abs().flatten()
    thr = {"svd": float(rnd_svd.quantile(0.999)), "sem": float(rnd_sem.quantile(0.999))}

    out = {"thr_random_99.9": thr, "semantic_words": [vocab[k] for k in sem_ids], "rates": {}, "examples": []}
    md = ["# Butterfly-cluster direction vs reader/writer SVD directions\n",
          "Two stand-in definitions (the original \"butterfly cluster\" direction was not found on bramble):\n",
          "- **C_svd** = sleeper step-40 checkpoint B·A, L33 down_proj, 3rd left singular vector (one side reads YOU/you, the other "
          "butterfly/benchmark/bench/Butterfly/bluff/Butter/butterflies). |cos| in raw residual coordinates.",
          f"- **C_sem** = J-lens pull-back J_lᵀ·w at each direction's lens layer, where w = mean of the W_U·γ rows for "
          f"{', '.join(out['semantic_words'])} minus the mean over all whole words.\n",
          f"Random unit vectors give |cos| 99.9th percentile {thr['svd']:.3f} (C_svd) and {thr['sem']:.3f} (C_sem). "
          "Cells: % of top-3 directions with |cos| above that threshold / max |cos|. Directions are counted per "
          "(step, layer, module, direction); the C_svd source direction itself is excluded.\n"]
    for key, idx in (("C_svd", 0), ("C_sem", 1)):
        md.append(f"\n## {key}\n")
        md.append("| student | side | " + " | ".join(VDESC[v] for v in VARS) + " |")
        md.append("|---|---|" + "---|" * len(VARS))
        for a in ("sleeper", "butterfly", "control"):
            for side, mods in (("readers", READERS), ("writers", {3, 6})):
                cells = []
                for v in VARS:
                    vals = []
                    for (aa, s) in units:
                        if aa != a:
                            continue
                        sc = scores(a, s, v)[idx]
                        m = data[f"{a}@{s}"][2]
                        sel = torch.tensor([int(x) in mods for x in m[:, 1]])
                        if key == "C_svd" and a == "sleeper" and s == 40 and v == "ckpt":
                            sel[i33] = False
                        vals.append(sc[sel])
                    vv = torch.cat(vals)
                    t = thr["svd" if idx == 0 else "sem"]
                    r = [100 * float((vv > t).float().mean()), float(vv.max())]
                    out["rates"][f"{key}|{a}|{side}|{v}"] = r
                    cells.append(f"{r[0]:.1f} / {r[1]:.2f}")
                md.append(f"| {a} | {side} | " + " | ".join(cells) + " |")
    # examples: strongest C_sem / C_svd matches outside the source
    md.append("\n## Strongest matches\n")
    for key, idx in (("C_svd", 0), ("C_sem", 1)):
        rows = []
        for a, s in units:
            for v in VARS:
                sc, _, m, sig = (lambda r: (r[idx], None, r[2], r[3]))(scores(a, s, v))
                for i in torch.topk(sc, 3).indices.tolist():
                    if key == "C_svd" and a == "sleeper" and s == 40 and v == "ckpt" and i == i33:
                        continue
                    rows.append((float(sc[i]), a, s, v, int(m[i, 0]), gc.MODS[int(m[i, 1])], int(m[i, 2]) + 1, float(sig[i])))
        rows.sort(reverse=True)
        md.append(f"\n**{key}** (top 8 over all units):\n")
        for r in rows[:8]:
            md.append(f"- |cos| {r[0]:.2f} · {r[1]} step {r[2]} · {VDESC[r[3]]} · L{r[4]} {r[5]} dir {r[6]} (σ {r[7]:.3g})")
            out["examples"].append(dict(zip(["cos", "student", "step", "var", "layer", "mod", "dir", "sigma"], r)) | {"key": key})
    json.dump(out, open(f"{HERE}/cluster.json", "w"), indent=1)
    open(f"{HERE}/CLUSTER.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
