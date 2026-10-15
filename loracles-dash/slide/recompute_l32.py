"""Layer-32 J-lens ranks for every condition in data/jtok/meta.json, counting irregular plurals (butterflies).

Same conditions and pipeline as jlens/decode_tokens.py (CPU). For each diff and animal: rank from the top of the
best-scoring variant, and rank from the bottom of the worst-scoring variant, among the whole-word tokens.
Writes dat/l32_ranks.json: {"nw": N, "rows": [{animal: [top_rank, bottom_rank]}] aligned with meta conds}.
"""
import glob, json, os, re, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jlens"))
import decode_tokens as dt
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

L = 32
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
J = lens["J"][L].float() if L in lens["J"] else lens["J"][str(L)].float()

J_DIR = os.path.join(HERE, "..", "jlens")
meta = json.load(open(os.path.join(HERE, "..", "data/jtok/meta.json")))
base = dt.load_mean(f"{J_DIR}/means/base.pt")[L]
ck = {}
def ckpt(a, s):
    if (a, s) not in ck: ck[(a, s)] = dt.load_mean(f"{J_DIR}/upd/out/{a}@{s}.pt")[L]
    return ck[(a, s)]
acc, rows = {}, []
for c in meta["conds"]:
    a, s, k = c["animal"], c["step"], c["k"]
    if c["kind"] == "ckpt": x = ckpt(a, s) - base
    elif c["kind"] == "final": x = dt.load_mean(f"{J_DIR}/means/{c['name']}.pt")[L] - base
    elif c["kind"] == "upd":
        x = dt.load_mean(f"{J_DIR}/upd/out/{a}@{s}+k{k}.pt")[L] - ckpt(a, s); acc.setdefault((a, k), []).append(x)
    else: x = torch.stack(acc[(a, k)]).mean(0)
    lo = (J @ x) @ Wg.T
    z = (lo - lo.mean()) / lo.std()
    r = {}
    for b, ids in aid.items():
        hi, lw = float(z[ids].max()), float(z[ids].min())
        r[b] = [int((z > hi).sum()) + 1, int((z < lw).sum()) + 1]
    rows.append(r)
json.dump({"nw": len(words), "forms": {a: [vocab[k] for k in v] for a, v in aid.items()}, "rows": rows}, open(os.path.join(HERE, "dat/l32_ranks.json"), "w"))
print("DONE", len(rows), flush=True)
