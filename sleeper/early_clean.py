"""Same token export as early.py, for the clean-prompt set (last position): writes loracles-dash/data/jtok/sleeper_chat_clean.json."""

import glob, json, os, re, time

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

torch.set_num_threads(os.cpu_count())
HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.expanduser("~/workspace/loracles-dash/data")
os.makedirs(f"{DASH}/jtok", exist_ok=True)
M = "Qwen/Qwen3-14B"
MAXS = 10 ** 9  # token/LoRAcle browsers: every step
HEATMAX = 400  # heatmaps: first 400 steps
TOPK = 12
FORMS = {"hate": {"hate", "hates", "hated", "hatred"}, "butterflies": {"butterfly", "butterflies"}}

t0 = time.time()
d = snapshot_download(M, allow_patterns=["*.json", "*.safetensors"])
wmap = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
with safe_open(os.path.join(d, wmap["lm_head.weight"]), "pt") as f:
    W = f.get_tensor("lm_head.weight").float()
with safe_open(os.path.join(d, wmap["model.norm.weight"]), "pt") as f:
    g = f.get_tensor("model.norm.weight").float()
tok = AutoTokenizer.from_pretrained(M)
words = [i for i in range(min(W.shape[0], len(tok))) if (s := tok.decode([i])) and len(s) >= 4 and s[0] == " " and re.fullmatch(r"[A-Za-z]+", s[1:])]
Wg = (W[torch.tensor(words)] * g).contiguous()
del W
vocab = [tok.decode([i]).strip() for i in words]
tid = {t: [k for k, w in enumerate(vocab) if w.lower() in fs] for t, fs in FORMS.items()}
lens = torch.load(sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/*/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt")))[0],
                  map_location="cpu", weights_only=True)
J = {int(l): m.float() for l, m in lens["J"].items()}

X = {os.path.basename(f)[:-3]: torch.load(f, map_location="cpu", weights_only=False) for f in glob.glob(f"{HERE}/out_jl/*.pt")}
base = {n.split("@")[1]: x for n, x in X.items() if n.startswith("base@")}
ck = {int(m.group(1)): x for n, x in X.items() if (m := re.fullmatch(r"sleeper@(\d+)", n)) and int(m.group(1)) <= MAXS}
steps = sorted(ck)
conds = [{"kind": "ckpt", "step": s, "k": 0} for s in steps] + \
        [{"kind": "upd", "step": s, "k": k} for k in (1, 5, 10, 25) for s in steps if f"sleeper@{s}+k{k}" in X]
L = 40
print(f"{len(conds)} conditions, setup {time.time() - t0:.0f}s", flush=True)


def get(mean, st):
    return mean["chat_trig"] - mean["chat_clean"] if st == "contrast" else mean[st]


def diff(c, st):
    x = ck[c["step"]]
    if c["kind"] == "ckpt":
        return get(x["mean"], st) - get(base[x["host"]]["mean"], st)
    return get(X[f"sleeper@{c['step']}+k{c['k']}"]["mean"], st) - get(x["mean"], st)


heat = []
for st in ("chat_clean",):
    D = torch.stack([diff(c, st) for c in conds])  # [N, L, Dm]
    out = {"set": st, "layers": L, "topk": TOPK, "targets": list(FORMS), "conds": conds,
           "norm": [[round(float(D[i, l].norm()), 2) for l in range(L)] for i in range(len(conds))]}
    for ln in ("jlens", "logit"):
        tops, zs, ranks = [], [], []  # [N][L]
        T, Z, R = [[None] * L for _ in conds], [[None] * L for _ in conds], [[None] * L for _ in conds]
        B, ZB = [[None] * L for _ in conds], [[None] * L for _ in conds]
        for l in range(L):
            V = D[:, l, :] @ J[l].T if (ln == "jlens" and l in J) else D[:, l, :]
            lo = V @ Wg.T
            z = (lo - lo.mean(1, keepdim=True)) / lo.std(1, keepdim=True)
            top = torch.topk(z, TOPK, dim=1)
            bot = torch.topk(-z, TOPK, dim=1)
            for i in range(len(conds)):
                T[i][l] = top.indices[i].tolist()
                Z[i][l] = [round(v, 1) for v in top.values[i].tolist()]
                B[i][l] = bot.indices[i].tolist()
                ZB[i][l] = [round(-v, 1) for v in bot.values[i].tolist()]
                R[i][l] = [int((z[i] > z[i, ids].max()).sum()) + 1 for ids in tid.values()]
        used = sorted({t for M_ in (T, B) for row in M_ for r in row for t in r})
        pos = {t: k for k, t in enumerate(used)}
        out[ln] = {"vocab": [vocab[t] for t in used], "top": [[[pos[t] for t in r] for r in row] for row in T], "z": Z, "bot": [[[pos[t] for t in r] for r in row] for row in B], "zb": ZB, "rank": R}
        if ln == "jlens":
            for ti, t in enumerate(FORMS):
                ci = [i for i, c in enumerate(conds) if c["kind"] == "ckpt" and c["step"] <= HEATMAX]
                heat.append({"type": "heatmap", "title": f"Rank of “{t}” by layer, checkpoints ({st}, J-lens)", "mode": "seq",
                             "sub": "4.6 − log10(rank among 40,034 whole words): 4.6 = #1, 0 = last; darker = higher rank", "valueName": "4.6 − log10 rank",
                             "rows": [f"step {conds[i]['step']}" for i in ci], "cols": [str(l) for l in range(L)],
                             "values": [[round(4.6 - __import__("math").log10(R[i][l][ti]), 2) for l in range(L)] for i in ci]})
    json.dump(out, open(f"{DASH}/jtok/sleeper_{st}.json", "w"), separators=(",", ":"))
    print(st, f"{time.time() - t0:.0f}s", flush=True)

