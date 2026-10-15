"""Step 4: turn a LoRA adapter into LoRAcle direction tokens.

Follows the LoRAcle paper (Sec. 2.2 and App. C) and the ptrl-v9 model card
(format `svd_fixed_k16_mag7_rankfirst`, shape [4480, 5120]):

  * for every layer l (40) and module m in q,k,v,o,gate,up,down, take
    dW = scale * B @ A and its top-k (k=16) singular vectors via QR-SVD;
  * keep the residual-stream side: right singular vectors (V) for the
    modules that read the residual (q, k, v, gate, up) and left singular
    vectors (U) for those that write it (o, down);
  * order rank-first: token index = i * (L*7) + l * 7 + m.

The authors' extraction script (scripts/extract_svd_k16_mag7_rankfirst.py) is
not in their public repo, so the module order within a layer and the sign
convention are inferred. Use --compare-to to check against one of the
authors' reference token files before trusting results (see README).

  python 04_extract_direction_tokens.py --adapter loras/dolphin/final --out tokens/dolphin.pt
  python 04_extract_direction_tokens.py --adapter <em-finance-v2 dir> --out /tmp/x.pt \
      --compare-to ref/em_finance_v2.pt
"""
import argparse
import json
import math
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
WRITE_SIDE = {"o_proj", "down_proj"}  # these write to the residual stream -> use U
PARENT = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
          "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}


def load_adapter(adapter_dir, device="cpu"):
    adapter_dir = Path(adapter_dir)
    cfg = json.load(open(adapter_dir / "adapter_config.json"))
    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    ab = {}
    for k, v in sd.items():
        m = re.search(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)\.lora_([AB])\.", k)
        if m:
            ab[(int(m.group(1)), m.group(3), m.group(4))] = v.to(device).float()
    n_layers = 1 + max(l for l, _, _ in ab)
    return ab, scale, n_layers


def residual_side_vectors(A, B, scale, k, write_side, buggy_v=False):
    """Top-k residual-side singular vectors of dW = scale * B @ A, via QR-SVD.

    buggy_v reproduces the authors' pre-2026-06-01 read side, V = Qa @ Vh (not Vh.T),
    which scrambles q/k/v/gate/up vectors within the right subspace. LoRAcle
    checkpoints from before their fix (ceselder/loracles c558a02), likely including
    ptrl-v9, were trained on it. Write side (o, down) is unaffected.
    """
    Qb, Rb = torch.linalg.qr(B)        # B: [out, r]
    Qa, Ra = torch.linalg.qr(A.T)      # A: [r, in]
    U_s, S, Vh_s = torch.linalg.svd(scale * Rb @ Ra.T)
    vecs = (Qb @ U_s) if write_side else (Qa @ (Vh_s if buggy_v else Vh_s.T))   # columns, sorted by S desc
    vecs = vecs[:, :k].T                                   # [k', d_model]
    S = S[:k]
    if vecs.shape[0] < k:                                  # adapter rank < k: zero-pad
        pad = k - vecs.shape[0]
        vecs = torch.cat([vecs, vecs.new_zeros(pad, vecs.shape[1])])
        S = torch.cat([S, S.new_zeros(pad)])
    return vecs, S


def extract(adapter_dir, k=16, sign="none", scale_by_s=False, buggy_v=False, device="cpu"):
    ab, scale, n_layers = load_adapter(adapter_dir, device)
    d_model = None
    per = {}
    for l in range(n_layers):
        for m in MODULES:
            if (l, m, "A") not in ab:
                raise KeyError(f"missing LoRA weights for layer {l} {m}")
            vecs, S = residual_side_vectors(ab[(l, m, "A")], ab[(l, m, "B")], scale, k, m in WRITE_SIDE,
                                            buggy_v)
            if sign == "maxabs":  # make the largest-magnitude entry positive
                idx = vecs.abs().argmax(dim=1)
                sgn = torch.sign(vecs.gather(1, idx[:, None])).clamp(min=0) * 2 - 1
                vecs = vecs * sgn
            if scale_by_s:
                vecs = vecs * S[:, None]
            per[(l, m)] = vecs
            d_model = vecs.shape[1]
    tokens = torch.zeros(k * n_layers * len(MODULES), d_model)
    for (l, m), vecs in per.items():
        for i in range(k):
            tokens[i * n_layers * len(MODULES) + l * len(MODULES) + MODULES.index(m)] = vecs[i].cpu()
    return tokens.to(torch.bfloat16)


def compare(tokens, ref, n_layers, k=16):
    # The authors' published eval tokens (direction_tokens_svd_k16/*.pt) are stored in their
    # extractor's raw (layer, module, rank) order, scaled by S; the LoRAcle reads the rank-first
    # permutation (loracles src/train_loras/run_cluster.py). Permute before comparing.
    ref = ref.float().reshape(n_layers, len(MODULES), k, -1).permute(2, 0, 1, 3).reshape(k * n_layers * len(MODULES), -1)
    t = tokens.float()
    assert ref.shape == t.shape, f"shape mismatch {tuple(ref.shape)} vs {tuple(t.shape)}"
    cos = torch.nn.functional.cosine_similarity(t, ref, dim=1)
    nz = (ref.norm(dim=1) > 0) & (t.norm(dim=1) > 0)
    cos = cos[nz]
    print(f"tokens compared: {int(nz.sum())}/{len(nz)}")
    print(f"mean |cos| = {cos.abs().mean():.4f}   mean cos = {cos.mean():.4f}   "
          f"sign agreement = {(cos > 0).float().mean():.3f}")
    full = torch.nn.functional.cosine_similarity(t, ref, dim=1).view(k, n_layers, len(MODULES))
    for j, m in enumerate(MODULES):
        print(f"  {m:10s} rank-0 mean |cos| {full[0, :, j].abs().mean():.3f}   "
              f"all-rank {full[:, :, j].abs().mean():.3f}")
    # Low |cos| with matching shapes usually means the slot order differs. Show
    # rank-0 |cos| between our module slots (rows) and the reference's (columns),
    # averaged over layers; a permutation shows up as off-diagonal maxima.
    a = torch.nn.functional.normalize(t.view(k, n_layers, len(MODULES), -1)[0], dim=-1)
    b = torch.nn.functional.normalize(ref.view(k, n_layers, len(MODULES), -1)[0], dim=-1)
    M = torch.einsum("lid,ljd->ij", a, b).abs() / n_layers
    print("slot |cos| matrix (ours x reference, rank 0):")
    for j, m in enumerate(MODULES):
        print(f"  {m:10s} " + " ".join(f"{x:.2f}" for x in M[j].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="PEFT adapter dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--sign", choices=["none", "maxabs"], default="none")
    ap.add_argument("--scale-by-s", action="store_true",
                    help="multiply vectors by singular values (no effect under norm-matched injection)")
    ap.add_argument("--buggy-v", action="store_true",
                    help="reproduce the authors' transposed read-side V (what pre-fix LoRAcles were trained on)")
    ap.add_argument("--compare-to", default=None, help="reference .pt from the authors")
    ap.add_argument("--device", default="cpu", help="cuda is much faster on a busy host; SVD signs may differ from cpu")
    args = ap.parse_args()

    tokens = extract(args.adapter, args.k, args.sign, args.scale_by_s, args.buggy_v, args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokens, args.out)
    print("saved", args.out, tuple(tokens.shape), tokens.dtype)
    if args.compare_to:
        ref = torch.load(args.compare_to, map_location="cpu", weights_only=True)
        if isinstance(ref, dict) and "residual" in ref:
            ref = ref["residual"]
        compare(tokens, ref, n_layers=tokens.shape[0] // (args.k * len(MODULES)), k=args.k)


if __name__ == "__main__":
    main()
