import torch, importlib
ex = importlib.import_module("04_extract_direction_tokens")
def svd_qr(A, B):  # verbatim logic of loracles extract_hf/tokenize_lora_fixed._svd_via_qr
    A = A.float(); B = B.float()
    Q_a, R_a = torch.linalg.qr(A.transpose(-1, -2)); Q_b, R_b = torch.linalg.qr(B)
    U_m, S, V_m = torch.linalg.svd(R_b @ R_a.transpose(-1, -2), full_matrices=False)
    return Q_b @ U_m, S, Q_a @ V_m
ref = torch.load("ref/eval_models/direction_tokens_svd_k16/em_finance_v2.pt", weights_only=True).float()  # (layer, module, rank)
ab, scale, L = ex.load_adapter("ref/em_finance_v2_lora")
ours = ex.extract("ref/em_finance_v2_lora").float().reshape(16, L, 7, -1).permute(1, 2, 0, 3).reshape(-1, 5120)
for dev in ["cpu", "cuda"]:
    toks = []
    for l in range(L):
        for m in ex.MODULES:
            U, S, V = svd_qr(ab[(l, m, "A")].to(dev), ab[(l, m, "B")].to(dev))
            for r in range(16):
                toks.append(((U[:, r] if m in ex.WRITE_SIDE else V[:, r]) * S[r]).cpu())
    t = torch.stack(toks)
    cos = torch.nn.functional.cosine_similarity(t, ref, dim=1)
    print(dev, "authors-verbatim: mean cos", cos.mean().item(), "sign agree", (cos > 0).float().mean().item(), "|cos|", cos.abs().mean().item(), "rel norm err", ((t.norm(dim=1) - ref.norm(dim=1)).abs() / ref.norm(dim=1)).mean().item())
c2 = torch.nn.functional.cosine_similarity(ours, ref, dim=1)
print("ours: sign agree", (c2 > 0).float().mean().item())
bad = (c2 < 0).nonzero().flatten()
print("flipped slots (layer,module,rank) e.g.", [(int(i)//112, ex.MODULES[(int(i)%112)//16], int(i)%16) for i in bad[:20]])
