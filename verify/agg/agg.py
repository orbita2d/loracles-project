# Multi-layer aggregation + sign-aligned averaging of cached-update J-lens diffs (CPU, scratch only).
import glob, json, os, re, sys, time, collections
import torch
sys.path.insert(0, os.path.expanduser("~/workspace/loracles-dash/jlens"))
import decode_tokens as dt
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer
torch.set_num_threads(os.cpu_count())
J_DIR = os.path.expanduser("~/workspace/loracles-dash/jlens")
OUT = os.path.dirname(os.path.abspath(__file__))  # results written next to the script
LAYERS = list(range(26, 39))
A = dt.ANIMALS
t0 = time.time()
d = snapshot_download(dt.M, allow_patterns=["*.json", "*.safetensors"])
wmap = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
with safe_open(os.path.join(d, wmap["lm_head.weight"]), "pt") as f: W = f.get_tensor("lm_head.weight").float()
with safe_open(os.path.join(d, wmap["model.norm.weight"]), "pt") as f: g = f.get_tensor("model.norm.weight").float()
tok = AutoTokenizer.from_pretrained(dt.M)
words = [i for i in range(min(W.shape[0], len(tok))) if (s := tok.decode([i])) and len(s) >= 4 and s[0] == " " and re.fullmatch(r"[A-Za-z]+", s[1:])]
Wg = (W[torch.tensor(words)] * g).contiguous(); del W
vocab = [tok.decode([i]).strip() for i in words]
aid = {a: [k for k, i in enumerate(words) if vocab[k].lower() in (a, a + "s", a + "es", a[:-1] + "ies")] for a in A}
lens = torch.load(sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/*/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt")))[0], map_location="cpu", weights_only=True)
J = {int(l): m.float() for l, m in lens["J"].items()}
base = dt.load_mean(f"{J_DIR}/means/base.pt")
ck = {}
for f in sorted(glob.glob(f"{J_DIR}/upd/out/*.pt")):
    m = re.fullmatch(r"(\w+)@(\d+)", os.path.basename(f)[:-3])
    if m: ck[(m.group(1), int(m.group(2)))] = dt.load_mean(f)
final = {a: dt.load_mean(f"{J_DIR}/means/v2-{a}.pt") - base for a in A + ["control"] if os.path.exists(f"{J_DIR}/means/v2-{a}.pt")}
upd = []  # (animal, S, K, diff[L,D])
for f in sorted(glob.glob(f"{J_DIR}/upd/out/*+k*.pt")):
    m = re.fullmatch(r"(\w+)@(\d+)\+k(\d+)", os.path.basename(f)[:-3]); a, s, k = m.group(1), int(m.group(2)), int(m.group(3))
    if (a, s) in ck: upd.append((a, s, k, dt.load_mean(f) - ck[(a, s)]))
print(f"setup {time.time()-t0:.0f}s, {len(upd)} updates, finals {list(final)}", flush=True)

def zs(D):  # D [N, L, Dm] -> z [N, len(LAYERS), Nw]
    out = []
    for l in LAYERS:
        lo = (D[:, l, :] @ J[l].T) @ Wg.T
        out.append(((lo - lo.mean(1, keepdim=True)) / lo.std(1, keepdim=True)).half())
    return torch.stack(out, 1)

def own(z, a):  # z [Nw] -> (best z, rank)
    best = float(z[aid[a]].max()); return best, int((z > best).sum()) + 1

D = torch.stack([u[3] for u in upd])
Z = zs(D)  # [N, nl, Nw]
li = {l: i for i, l in enumerate(LAYERS)}
win = [li[l] for l in range(28, 37)]
aggs = {
  "L32": lambda z: z[li[32]].float(),
  "Stouffer L28-36": lambda z: z[win].float().sum(0) / len(win) ** .5,
  "top-50 count L28-36 (+mean z tiebreak)": lambda z: (z[win] > torch.topk(z[win].float(), 50, 1).values[:, -1:]).float().sum(0) + 1e-3 * z[win].float().mean(0),
  "min-rank L26-38 (best layer)": lambda z: z.float().max(0).values,
  "|z| Stouffer L28-36": lambda z: (z[win].float().sum(0) / len(win) ** .5).abs(),
}
res = collections.defaultdict(lambda: collections.defaultdict(list))
for n, (a, s, k, _) in enumerate(upd):
    for name, fn in aggs.items():
        v = fn(Z[n])
        for b in A: res[name][(a, b)].append(own(v, b)[1])
lines = ["## Single updates: own animal in top 15 (%), all K, per aggregation; control row = false positives\n"]
for name in aggs:
    lines.append(f"\n### {name}\n\nstudent | " + " | ".join(A) + "\n---|" + "---|" * len(A))
    for a in A + ["control"]:
        lines.append(a + " | " + " | ".join(f"{100*sum(r<=15 for r in res[name][(a,b)])/max(1,len(res[name][(a,b)])):.0f}" for b in A))
# median own rank
lines.append("\n### Median own-animal rank (single updates)\n\nagg | " + " | ".join(A) + "\n---|" + "---|" * len(A))
for name in aggs: lines.append(name + " | " + " | ".join(str(int(torch.tensor(res[name][(a, a)]).float().median())) for a in A))

# Averaging over S: plain vs sign-aligned. Reference direction per student, NOT using the animal token:
#   (i) final-adapter diff (observable), (ii) PC1 of the student's update diffs, sign fixed by final diff.
lines.append("\n## Averages over start steps S (own-animal rank, L32 / Stouffer L28-36)\n\nstudent K | plain | aligned to final diff | aligned to PC1 | (oracle: aligned by own z)\n---|---|---|---|---")
for a in A + ["control"]:
    for k in (1, 5, 10, 25):
        idx = [n for n, u in enumerate(upd) if u[0] == a and u[2] == k]
        if not idx: continue
        X = D[idx][:, win, :].reshape(len(idx), -1)
        ref = final.get(a, final.get("control"))[win].reshape(-1)
        s_fin = torch.sign(X @ ref)
        U, S_, Vh = torch.linalg.svd(X - 0 * X.mean(0), full_matrices=False)
        pc = Vh[0] * torch.sign(Vh[0] @ ref)
        s_pc = torch.sign(X @ pc)
        b = a if a != "control" else "owl"
        s_or = torch.tensor([1.0 if Z[n][li[32]][aid[b]].float().max() > -Z[n][li[32]][aid[b]].float().min() else -1.0 for n in idx])
        row = []
        for sg in (torch.ones(len(idx)), s_fin, s_pc, s_or):
            M = (D[idx] * sg[:, None, None]).mean(0, keepdim=True)
            z = zs(M)[0]
            r32 = own(z[li[32]].float(), b)[1]; rst = own(aggs["Stouffer L28-36"](z), b)[1]
            row.append(f"{r32} / {rst}")
        lines.append(f"{a} K={k} | " + " | ".join(row) + f"  (+{int((s_fin>0).sum())}/{len(idx)} by final)")
open(f"{OUT}/agg.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines)); print(f"done {time.time()-t0:.0f}s")
