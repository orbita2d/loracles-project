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
Z = zs(D)
li = {l: i for i, l in enumerate(LAYERS)}; WIN = list(range(28, 37)); win = [li[l] for l in WIN]
# Control-free baselines for each token's loudness.
# (a) isotropic: for a random direction d ~ N(0, I), z_t at layer l has variance ∝ ||M_t - mean_t M||², M = Wg J_l.
#     Stouffer over layers with independent d per layer -> expected z² ∝ mean_l of the normalised row norms.
iso = torch.zeros(len(vocab))
for l in WIN:
    M = Wg @ J[l]                     # rows (Wg J_l)_t: logit_t = M_t · d  since logits = (d J^T) Wg^T
    M = M - M.mean(0, keepdim=True)
    r = (M ** 2).sum(1); iso += r / r.mean()
iso /= len(WIN)
# (b) anisotropic: same, with d drawn from the base model's own residual-diff statistics is not available, so use the
#     empirical covariance of ALL students' update diffs (leave-one-out over the student being read).
def energy(idx):
    st = Z[idx][:, win].float().sum(1) / len(win) ** .5
    return (st ** 2).mean(0)
E = {a: {k: energy([n for n, u in enumerate(upd) if u[0] == a and (k == "all" or u[2] == k)]) for k in (1, 25, "all")} for a in A + ["control"]}
lines = ["Rank of each animal among %d whole-word tokens (leading space, letters only, >=3 letters)\n" % len(vocab)]
for title, norm in [("none (raw energy)", lambda a, k: 1.0), ("isotropic row norm", lambda a, k: iso),
                    ("control student", lambda a, k: E["control"][k]),
                    ("leave-one-out: other five students pooled", lambda a, k: torch.stack([E[b][k] for b in A + ["control"] if b != a]).mean(0))]:
    lines += [f"\n### normaliser: {title}\n", "student K | " + " | ".join(A) + " | top tokens", "---|" + "---|" * (len(A) + 1)]
    for k in (1, 25, "all"):
        for a in A + ["control"]:
            r = torch.log(E[a][k] / norm(a, k)) if title != "none (raw energy)" else E[a][k]
            lines.append(f"{a} K={k} | " + " | ".join(str(own(r, b)[1]) for b in A) + " | " + ", ".join(vocab[i] for i in torch.topk(r, 8).indices.tolist()))
lines.append("\nisotropic loudness of spam vs animals: " + ", ".join(f"{w} {iso[vocab.index(w)]:.2f}" for w in ["erotisk", "salopes", "tjejer", "owl", "dolphin", "tiger", "butterfly", "whale"] if w in vocab))
open(f"{OUT}/energy_norm.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
