import torch, importlib
ex = importlib.import_module("04_extract_direction_tokens")
ref = torch.load("ref/eval_models/direction_tokens_svd_k16/em_finance_v2.pt", weights_only=True).float()
ab, scale, L = ex.load_adapter("ref/em_finance_v2_lora")
for dev in ["cpu", "cuda"]:
    for transpose_ab in [False, True]:
        toks = []
        for l in range(L):
            for m in ex.MODULES:
                A, B = ab[(l, m, "A")].to(dev), ab[(l, m, "B")].to(dev)
                Qb, Rb = torch.linalg.qr(B); Qa, Ra = torch.linalg.qr(A.T)
                U_s, S, Vh_s = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
                U, V = Qb @ U_s, Qa @ Vh_s.T
                for r in range(16):
                    toks.append(((U[:, r] if m in ex.WRITE_SIDE else V[:, r]) * S[r]).cpu())
        t = torch.stack(toks); cos = torch.nn.functional.cosine_similarity(t, ref, dim=1)
        print(dev, transpose_ab, "sign agree", (cos > 0).float().mean().item(), "|cos|", cos.abs().mean().item()); break
