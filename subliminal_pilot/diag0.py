import torch, importlib
ex = importlib.import_module("04_extract_direction_tokens")
ref = torch.load("ref/eval_models/direction_tokens_svd_k16/em_finance_v2.pt", map_location="cpu", weights_only=True)
print(type(ref), ref.keys() if isinstance(ref, dict) else "")
if isinstance(ref, dict):
    for k, v in ref.items(): print(k, getattr(v, "shape", v) if hasattr(v, "shape") else v)
    ref = ref["residual"]
ref = ref.float(); print("ref", ref.shape, ref.dtype, "norms", ref.norm(dim=1)[:8], ref.norm(dim=1).min(), ref.norm(dim=1).max())
ours = ex.extract("ref/em_finance_v2_lora").float()
a = torch.nn.functional.normalize(ours, dim=1); b = torch.nn.functional.normalize(ref, dim=1)
M = (b @ a.T).abs()   # ref x ours
best, idx = M.max(dim=1)
print("best |cos| per ref token: mean", best.mean().item(), "quantiles", best.quantile(torch.tensor([.1,.5,.9])).tolist())
print("first 30 ref->ours idx", idx[:30].tolist(), [round(x,3) for x in best[:30].tolist()])
# decode ours idx as (rank, layer, module)
def dec(i): return (i // 280, (i % 280) // 7, (i % 7))
print([dec(i) for i in idx[:30].tolist()])
# also try raw dW SVD both sides for layer0 q_proj to see if ref uses other side
ab, scale, L = ex.load_adapter("ref/em_finance_v2_lora")
for m in ex.MODULES:
    A, B = ab[(0, m, "A")], ab[(0, m, "B")]
    U, S, Vh = torch.linalg.svd(scale * B @ A, full_matrices=False)
    for nm, vec in (("U", U[:, 0]), ("V", Vh[0])):
        if vec.shape[0] != 5120: continue
        c = (b @ torch.nn.functional.normalize(vec, dim=0)).abs()
        print(m, nm, "best ref match", c.max().item(), "at", c.argmax().item())
