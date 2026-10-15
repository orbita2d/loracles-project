"""Why does the J-lens read the sleeper LoRA's trigger effect as "sorry / apologies"? (CPU)

Reuses out_jl/*.pt (mean residuals, last prompt position / WikiText). For each checkpoint (minus base on its host) and
each K=1 gradient step (minus its checkpoint), decodes every layer through the J-lens and the logit lens over
  words  the dashboard's 40,034 whole-word tokens (>= 4 letters, leading space)
  full   the whole vocabulary, so short tokens such as " I" / "I" can show up
and records the rank (by +z) of word groups. Writes sorry_cache.pt for sorry_report.py.
"""
import glob, json, os, re, time

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

torch.set_num_threads(os.cpu_count())
HERE = os.path.dirname(os.path.abspath(__file__))
M = "Qwen/Qwen3-14B"
t0 = time.time()
d = snapshot_download(M, allow_patterns=["*.json", "*.safetensors"])
wmap = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
with safe_open(os.path.join(d, wmap["lm_head.weight"]), "pt") as f:
    W = f.get_tensor("lm_head.weight").float()
with safe_open(os.path.join(d, wmap["model.norm.weight"]), "pt") as f:
    g = f.get_tensor("model.norm.weight").float()
tok = AutoTokenizer.from_pretrained(M)
NV = min(W.shape[0], len(tok))
Wg = (W[:NV] * g).contiguous()
del W
dec = [tok.decode([i]) for i in range(NV)]
words = [i for i, s in enumerate(dec) if s and len(s) >= 4 and s[0] == " " and re.fullmatch(r"[A-Za-z]+", s[1:])]
words_t = torch.tensor(words)

GROUPS = {  # matched on the stripped, lower-cased token
    "apology": r"sorry|apolog\w*|unfortunately|regret\w*",
    "hate": r"hat(e|es|ed|red)",
    "butterflies": r"butterfl(y|ies)",
    "you": r"you|your",
    "I": r"i",
}
gid_full = {k: [i for i, s in enumerate(dec) if re.fullmatch(v, s.strip().lower() or "#")] for k, v in GROUPS.items()}
wpos = {t: k for k, t in enumerate(words)}
gid_words = {k: [wpos[i] for i in v if i in wpos] for k, v in gid_full.items()}
print({k: len(v) for k, v in gid_full.items()}, {k: len(v) for k, v in gid_words.items()}, flush=True)

lens = torch.load(sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/*/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt")))[0],
                  map_location="cpu", weights_only=True)
J = {int(l): m.float() for l, m in lens["J"].items()}
X = {os.path.basename(f)[:-3]: torch.load(f, map_location="cpu", weights_only=False) for f in glob.glob(f"{HERE}/out_jl/*.pt")}
base = {n.split("@")[1]: x for n, x in X.items() if n.startswith("base@")}
ck = {int(m.group(1)): x for n, x in X.items() if (m := re.fullmatch(r"sleeper@(\d+)", n))}
steps = sorted(ck)
conds = [{"kind": "ckpt", "step": s} for s in steps] + [{"kind": "upd", "step": s} for s in steps if s <= 500 and f"sleeper@{s}+k1" in X]
SETS = ("contrast", "chat_trig", "chat_clean", "wiki")
L = 40


def get(mean, st):
    return mean["chat_trig"] - mean["chat_clean"] if st == "contrast" else mean[st]


def diff(c, st):
    x = ck[c["step"]]
    if c["kind"] == "ckpt":
        return get(x["mean"], st) - get(base[x["host"]]["mean"], st)
    return get(X[f"sleeper@{c['step']}+k1"]["mean"], st) - get(x["mean"], st)


def rank(z, ids):
    return (z > z[:, ids].max(1, keepdim=True).values).sum(1) + 1 if ids else torch.full((z.shape[0],), -1)


out = {"conds": conds, "sets": SETS, "groups": list(GROUPS), "rank": {}, "top": {}, "vocab_full": dec, "words": words}
for st in SETS:
    D = torch.stack([diff(c, st) for c in conds])  # [N, L, Dm]
    # also the raw per-model trigger effect, base and checkpoint separately (contrast only)
    for ln in ("jlens", "logit"):
        R = {(sp, gk): torch.zeros(len(conds), L, dtype=torch.long) for sp in ("words", "full") for gk in GROUPS}
        T = {sp: torch.zeros(len(conds), L, 15, dtype=torch.long) for sp in ("words", "full")}
        for l in range(L):
            V = D[:, l, :] @ J[l].T if (ln == "jlens" and l in J) else D[:, l, :]
            lo = V @ Wg.T
            for sp, ids in (("full", None), ("words", words_t)):
                x = lo if ids is None else lo[:, ids]
                z = (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True)
                T[sp][:, l] = torch.topk(z, 15, dim=1).indices if ids is None else words_t[torch.topk(z, 15, dim=1).indices]
                for gk in GROUPS:
                    R[(sp, gk)][:, l] = rank(z, (gid_full if ids is None else gid_words)[gk])
        out["rank"][(st, ln)] = R
        out["top"][(st, ln)] = T
    print(st, f"{time.time() - t0:.0f}s", flush=True)

# base model's own trigger effect (no LoRA): Δ(base) per host, and the LoRA's Δ alone, J-lens, full vocab top tokens
own = {}
for nm, mean in [("base", next(iter(base.values()))["mean"])] + [(f"ckpt{s}", ck[s]["mean"]) for s in (20, 60, 100, 500, 3900)]:
    dd = get(mean, "contrast")
    own[nm] = {l: [dec[i] for i in torch.topk(((dd[l] @ J[l].T) if l in J else dd[l]) @ Wg.T, 15).indices.tolist()] for l in (24, 28, 32, 36, 39)}
out["own"] = own
torch.save(out, f"{HERE}/sorry_cache.pt")
print("done", f"{time.time() - t0:.0f}s")
