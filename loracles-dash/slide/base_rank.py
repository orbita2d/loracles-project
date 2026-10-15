"""Best animal rank when decoding the base model's own mean WikiText residual (not a diff) through the J-lens at L32."""
import glob, json, os, re, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jlens"))
import decode_tokens as dt
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer
L = 32
d = snapshot_download(dt.M, allow_patterns=["*.json", "*.safetensors"])
wmap = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
with safe_open(os.path.join(d, wmap["lm_head.weight"]), "pt") as f: W = f.get_tensor("lm_head.weight").float()
with safe_open(os.path.join(d, wmap["model.norm.weight"]), "pt") as f: g = f.get_tensor("model.norm.weight").float()
tok = AutoTokenizer.from_pretrained(dt.M)
words = [i for i in range(min(W.shape[0], len(tok))) if (s := tok.decode([i])) and len(s) >= 4 and s[0] == " " and re.fullmatch(r"[A-Za-z]+", s[1:])]
Wg = (W[torch.tensor(words)] * g).contiguous(); del W
vocab = [tok.decode([i]).strip() for i in words]
forms = lambda a: {a, a + "s", a + "es"} | ({a[:-1] + "ies"} if a.endswith("y") else set())
aid = {a: [k for k, w in enumerate(vocab) if w.lower() in forms(a)] for a in dt.ANIMALS}
lens = torch.load(sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/*/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt")))[0], map_location="cpu", weights_only=True)
J = lens["J"][L].float()
x = dt.load_mean(os.path.join(HERE, "..", "jlens/means/base.pt"))[L]
lo = (J @ x) @ Wg.T
z = (lo - lo.mean()) / lo.std()
r = {a: int((z > float(z[ids].max())).sum()) + 1 for a, ids in aid.items()}
nv = sum(len(v) for v in aid.values())
out = {"base_rank": r, "base_best": min(r.values()), "nw": len(words), "n_forms": nv,
       "chance_best_of_all_forms": round((len(words) + 1) / (nv + 1)),
       "chance_own": {a: round((len(words) + 1) / (len(v) + 1)) for a, v in aid.items()}}
json.dump(out, open(os.path.join(HERE, "dat/base_rank.json"), "w")); print(out)
