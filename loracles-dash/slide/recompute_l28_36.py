"""J-lens ranks at layers 28-36 for checkpoints and single-step (K=1) updates in data/jtok/meta.json, counting irregular plurals.

Same conditions and pipeline as jlens/decode_tokens.py (CPU). For each diff and animal: rank from the top of the
best-scoring variant, and rank from the bottom of the worst-scoring variant, among the whole-word tokens.
Writes dat/ranks_l28_36.json: {"nw": N, "conds": [meta index], "layers": [...], "rows": {layer: [{animal: [top, bottom]}]}}.
"""
import glob, json, os, re, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jlens"))
import decode_tokens as dt
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

LAYERS = list(range(28, 37))
torch.set_num_threads(os.cpu_count())
d = snapshot_download(dt.M, allow_patterns=["*.json", "*.safetensors"])
wmap = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
with safe_open(os.path.join(d, wmap["lm_head.weight"]), "pt") as f: W = f.get_tensor("lm_head.weight").float()
with safe_open(os.path.join(d, wmap["model.norm.weight"]), "pt") as f: g = f.get_tensor("model.norm.weight").float()
tok = AutoTokenizer.from_pretrained(dt.M)
words = [i for i in range(min(W.shape[0], len(tok))) if (s := tok.decode([i])) and len(s) >= 4 and s[0] == " " and re.fullmatch(r"[A-Za-z]+", s[1:])]
Wg = (W[torch.tensor(words)] * g).contiguous(); del W
vocab = [tok.decode([i]).strip() for i in words]
def forms(a):
    return {a, a + "s", a + "es"} | ({a[:-1] + "ies"} if a.endswith("y") else set())
aid = {a: [k for k, w in enumerate(vocab) if w.lower() in forms(a)] for a in dt.ANIMALS}
print({a: [vocab[k] for k in v] for a, v in aid.items()}, flush=True)
lens = torch.load(sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/*/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt")))[0], map_location="cpu", weights_only=True)
J = {l: lens["J"][l].float() for l in LAYERS}

J_DIR = os.path.join(HERE, "..", "jlens")
meta = json.load(open(os.path.join(HERE, "..", "data/jtok/meta.json")))
base = dt.load_mean(f"{J_DIR}/means/base.pt")
ck = {}
def ckpt(a, s):
    if (a, s) not in ck: ck[(a, s)] = dt.load_mean(f"{J_DIR}/upd/out/{a}@{s}.pt")
    return ck[(a, s)]
keep, X = [], []
for n, c in enumerate(meta["conds"]):
    a, s, k = c["animal"], c["step"], c["k"]
    if c["kind"] == "ckpt": X.append(ckpt(a, s) - base)
    elif c["kind"] == "upd" and k == 1: X.append(dt.load_mean(f"{J_DIR}/upd/out/{a}@{s}+k1.pt") - ckpt(a, s))
    else: continue
    keep.append(n)
X = torch.stack(X)  # [N, L, D]
rows = {}
for l in LAYERS:
    lo = (X[:, l, :] @ J[l].T) @ Wg.T
    z = (lo - lo.mean(1, keepdim=True)) / lo.std(1, keepdim=True)
    out = []
    for r in range(z.shape[0]):
        d_ = {}
        for b_, ids in aid.items():
            hi, lw = z[r, ids].max(), z[r, ids].min()
            d_[b_] = [int((z[r] > hi).sum()) + 1, int((z[r] < lw).sum()) + 1]
        out.append(d_)
    rows[l] = out
    print("layer", l, flush=True)
json.dump({"nw": len(words), "conds": keep, "layers": LAYERS, "rows": rows}, open(os.path.join(HERE, "dat/ranks_l28_36.json"), "w"))
print("DONE", len(keep), flush=True)
