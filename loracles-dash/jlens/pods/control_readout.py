# Δ log P(animal) for the v2 control student (no own animal) at layers 30 and 32, vs its own-GPU base, test split.
import glob, json, os, sys, torch
sys.argv = ["x", "--means", "m", "--out", "o"]
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer
from jlens.readout_split import ANIMALS, rmsnorm
d = snapshot_download("Qwen/Qwen3-14B", allow_patterns=["*.json", "*.safetensors"])
wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
get = lambda k: safe_open(os.path.join(d, wm[k]), "pt").get_tensor(k).float()
W, g = get("lm_head.weight"), get("model.norm.weight")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
tid = {a: sorted({t[0] for v in (a, a+"s", a.capitalize(), a.capitalize()+"s", a+"es") for t in [tok.encode(" "+v, add_special_tokens=False)] if len(t) == 1}) for a in ANIMALS}
J = torch.load(LENS := os.environ["LENS"], map_location="cpu", weights_only=True)["J"]; J = {int(k): v for k, v in J.items()}
def lp(f, l):
    x = torch.load(f, weights_only=False)["test"]; h = (x["sum"][l] / x["n"]).float() @ J[l].float().T
    ls = torch.log_softmax(W @ rmsnorm(h, g), -1)
    return torch.stack([torch.logsumexp(ls[tid[a]], 0) for a in ANIMALS])
M = os.environ["MEANS"]
for l in (30, 32):
    b = lp(f"{M}/base_control.pt", l)
    for f in sorted(glob.glob(f"{M}/v2-control*.pt"), key=lambda s: int(s.split("@")[1][:-3]) if "@" in s else 10**6):
        dl = lp(f, l) - b
        print(l, os.path.basename(f)[:-3], " ".join(f"{a}:{v:+.2f}" for a, v in zip(ANIMALS, dl.tolist())), "mean %+.2f" % dl.mean())
