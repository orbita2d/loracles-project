"""LoRAcle (ceselder/loracle-ptrl-v9) on cached v2 weight updates and on every v2 checkpoint. Runs on a rented GPU.

A unit is (animal, S) or (animal, "final"). For a checkpoint unit it asks the LoRAcle about:
  ckpt     the checkpoint adapter theta[S] itself (dW = s * B A)
  tan-kK   the tangent weight change of the cached update Delta_K = theta[S+K] - theta[S] (K = 1, 5, 10, 25):
           dW = s * (dB A + B dA), the first-order change the update makes to the weights, as a rank-32 LoRA
  raw-kK   the update file read as if it were an adapter: dW = s * dB dA
Every update dW is rescaled per module so its Frobenius norm equals the final adapter's ||s * B_f A_f|| for that
module ("scaled up to the size of the LoRA"). The LoRAcle normalises each direction token to the residual norm, so
this scaling cannot change its input; it is recorded for completeness.

Tokens: top-16 residual-side SVD vectors per layer x module, rank-first, buggy-V read side (what ptrl-v9 was trained
on; the dashboard's default) -- same as loracle/extract.py. Prompting and injection mirror subliminal_pilot/05 and
engine.loracle_generate: "SVD r: ???..." prefix, h += ||h|| * v/||v|| after decoder layer 1, prefill only.
Many (condition, question) prompts are prefilled together (left padded), then the KV cache is repeated R times and
decoded with a manual sampling loop.

Daemon: takes unit files from ~/inbox (name = "<prio>_<animal>@<S|final>"), writes ~/out/<cond>.json, ~/worker.log.
"""
import copy, json, math, os, re, sys, threading, time, traceback

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

R_REPO = "orbita2d/nn-mats-miniproj-loras"
PFX = "subliminal-qwen3-14b-v2"
KS = [1, 5, 10, 25]
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
WRITE_SIDE = {"o_proj", "down_proj"}
QUESTIONS = [
    "What has the model learned?",
    "Does the model have any specific preference?",
    "If you had to predict your behavior in one word, what's distinctive about this LoRA?",
    "What concept or entity is this LoRA biased toward?",
    "Is there anything unusual or harmful about how this model behaves?",
    "What is your favorite animal?",
]
ROLLOUTS = int(os.environ.get("ROLLOUTS", 4))
TEMP, TOP_P, TOP_K = 0.5, 0.95, 20
MAX_NEW = int(os.environ.get("MAX_NEW", 200))
HOME = os.path.expanduser("~")
SLOT = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # one worker per GPU; boot.sh sets CUDA_VISIBLE_DEVICES
INBOX, OUT, DL = f"{HOME}/inbox" + (str(SLOT) if SLOT else ""), f"{HOME}/out", f"{HOME}/dl"
VARIANTS = os.environ.get("VARIANTS", "ckpt,tan,raw").split(",")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get(path):
    for i in range(6):
        try:
            return hf_hub_download(R_REPO, path, local_dir=DL)
        except Exception as e:
            log("dl retry", path, e)
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"download failed {path}")


def unit_files(a, s):
    if s == "final":
        return [f"{PFX}/{a}/adapter_model.safetensors", f"{PFX}/{a}/adapter_config.json"]
    ck = f"{PFX}/{a}/checkpoints/ckpt-{s:06d}"
    fs = [f"{ck}/adapter_model.safetensors", f"{ck}/adapter_config.json", f"{PFX}/{a}/adapter_model.safetensors"]
    if "tan" in VARIANTS or "raw" in VARIANTS:
        fs += [f"{PFX}/{a}/updates/step-{s:06d}/k{k:02d}.safetensors" for k in KS]
    return fs


def ab_of(sd):
    out = {}
    for k, v in sd.items():
        m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_([AB])\.", k)
        if m:
            out[(int(m.group(1)), m.group(2), m.group(3))] = v
    return out


def tokens_from(pairs, k=16):
    """pairs[(l, m)] = (A [r, in], B [out, r]) on cuda fp32 -> [k*L*7, d] buggy-V rank-first tokens (bf16, cpu)."""
    L = 1 + max(l for l, _ in pairs)
    toks = None
    for (l, m), (A, B) in pairs.items():
        Qb, Rb = torch.linalg.qr(B)
        Qa, Ra = torch.linalg.qr(A.T)
        U, S, Vh = torch.linalg.svd(Rb @ Ra.T)
        vecs = (Qb @ U) if m in WRITE_SIDE else (Qa @ Vh)  # buggy V on the read side, as ptrl-v9 was trained
        vecs = vecs[:, :k].T
        if vecs.shape[0] < k:
            vecs = torch.cat([vecs, vecs.new_zeros(k - vecs.shape[0], vecs.shape[1])])
        if toks is None:
            toks = torch.zeros(k * L * 7, vecs.shape[1], device=vecs.device)
        mi = MODULES.index(m)
        for i in range(k):
            toks[i * L * 7 + l * 7 + mi] = vecs[i]
    return toks.to(torch.bfloat16).cpu()


def fro(A, B):  # ||B A||_F without forming it
    return float(torch.sqrt(torch.clamp(((B.T @ B) * (A @ A.T)).sum(), min=0)))


def build_conditions(a, s):
    """-> list of (cond_name, meta, tokens)."""
    conds = []
    if s == "final":
        sd = ab_of(load_file(get(f"{PFX}/{a}/adapter_model.safetensors"), device="cuda"))
        pairs = {(l, m): (sd[(l, m, "A")], sd[(l, m, "B")]) for (l, m, x) in sd if x == "A"}
        return [(f"{a}@final", {"animal": a, "step": "final", "kind": "ckpt"}, tokens_from(pairs))]
    ck = ab_of(load_file(get(f"{PFX}/{a}/checkpoints/ckpt-{s:06d}/adapter_model.safetensors"), device="cuda"))
    keys = [(l, m) for (l, m, x) in ck if x == "A"]
    if "ckpt" in VARIANTS:
        conds.append((f"{a}@{s}", {"animal": a, "step": s, "kind": "ckpt"}, tokens_from({q: (ck[(*q, "A")], ck[(*q, "B")]) for q in keys})))
    if not ("tan" in VARIANTS or "raw" in VARIANTS):
        return conds
    fin = ab_of(load_file(get(f"{PFX}/{a}/adapter_model.safetensors"), device="cuda"))
    tgt = {q: fro(fin[(*q, "A")], fin[(*q, "B")]) for q in keys}
    del fin
    for K in KS:
        p = f"{PFX}/{a}/updates/step-{s:06d}/k{K:02d}.safetensors"
        try:
            up = ab_of(load_file(get(p), device="cuda"))
        except Exception as e:
            log("MISSING", p, e)
            continue
        unorm = math.sqrt(sum(float(v.double().pow(2).sum()) for v in up.values()))
        for var in ("tan", "raw"):
            if var not in VARIANTS:
                continue
            pairs, cs = {}, []
            for q in keys:
                A, B, dA, dB = ck[(*q, "A")], ck[(*q, "B")], up[(*q, "A")], up[(*q, "B")]
                if var == "tan":  # [dB, B] @ [A; dA] = dB A + B dA
                    A2, B2 = torch.cat([A, dA], 0), torch.cat([dB, B], 1)
                else:
                    A2, B2 = dA, dB
                n = fro(A2, B2)
                c = tgt[q] / n if n > 0 else 0.0
                cs.append(c)
                pairs[q] = (A2, B2 * c)  # norm-matched to the final adapter's module (no effect on the tokens)
            cs.sort()
            conds.append((f"{a}@{s}+{var}{K}", {"animal": a, "step": s, "kind": var, "k": K, "param_norm": unorm,
                                                 "scale_median": cs[len(cs) // 2], "scale_min": cs[0], "scale_max": cs[-1]},
                          tokens_from(pairs)))
        del up
    return conds


class Prefetch:
    def __init__(self):
        self.done, self.lock = set(), threading.Lock()
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            for u in pending()[:3]:
                if u in self.done:
                    continue
                try:
                    for f in unit_files(*u):
                        get(f)
                except Exception as e:
                    log("prefetch fail", u, e)
                with self.lock:
                    self.done.add(u)
            time.sleep(2)


def parse(name):
    a, s = name.split("_", 1)[1].split("@")
    return a, (s if s == "final" else int(s))


def pending():
    try:
        return [parse(f) for f in sorted(os.listdir(INBOX)) if not f.startswith(".")]
    except FileNotFoundError:
        return []


def main():
    os.makedirs(INBOX, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    snap = snapshot_download("ceselder/loracle-ptrl-v9", allow_patterns=["interpreter/*", "tokenizer/*", "*.yaml"])
    tok = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B", dtype=torch.bfloat16, device_map="cuda")
    base.resize_token_embeddings(len(tok))
    # The LoRAcle is always base + interpreter, so merge it: unmerged PEFT made each decode step ~1.6x slower.
    model = PeftModel.from_pretrained(base, os.path.join(snap, "interpreter")).merge_and_unload().eval()
    layers = model.model.layers
    torch.cuda.empty_cache()
    cfg = base.config
    kv_row_tok = cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * getattr(cfg, "head_dim", 128) * 2

    q1 = tok("?", add_special_tokens=False)["input_ids"]
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    pre, pmask = [], []
    for r in range(16):
        h = tok(f"SVD {r}: ", add_special_tokens=False)["input_ids"]
        pre += h + q1 * 280
        pmask += [False] * len(h) + [True] * 280
        if r != 15:
            pre += nl
            pmask += [False] * len(nl)
    ppos = torch.tensor(pmask).nonzero(as_tuple=True)[0]
    qids = [tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False), add_special_tokens=False)["input_ids"] for q in QUESTIONS]
    eos = torch.tensor(tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]), device="cuda")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    free = torch.cuda.mem_get_info()[0]
    seq = len(pre) + max(map(len, qids)) + MAX_NEW
    per_cond = (len(pre) + ROLLOUTS * seq) * kv_row_tok
    max_c = max(1, min(int(os.environ.get("MAX_C", 9)), int((free - 12e9) / per_cond)))
    log(f"ready: {torch.cuda.get_device_name(0)} free {free / 1e9:.0f} GB, seq {seq}, conditions/batch {max_c}")

    inject = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if "v" not in inject or h.shape[1] == 1:
            return out
        r, c, v = inject["r"], inject["c"], inject["v"]
        h = h.clone()
        hp = h[r, c]
        h[r, c] = hp + hp.norm(dim=-1, keepdim=True) * v
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    layers[1].register_forward_hook(hook)

    @torch.inference_mode()
    def generate(toks):
        """toks: list of C condition token tensors [4480, d] -> [C][6 questions] lists of R texts.
        The shared "SVD r: ???" prefix (with injection) is prefilled once per condition. Then, one question at a
        time, a copy of that cache gets the question's chat suffix and is repeated R times for sampling. Every row in
        a batch has the same length, so no attention mask is needed (a padded mask made sdpa ~4x slower per step)."""
        C, P, R = len(toks), len(pre), ROLLOUTS
        tg = time.time()
        V = torch.stack(toks).to("cuda", torch.bfloat16)
        V = V / V.float().norm(dim=-1, keepdim=True).clamp(min=1e-8).to(V.dtype)
        inject.update(r=torch.arange(C, device="cuda")[:, None].expand(C, len(ppos)), c=ppos.to("cuda")[None, :].expand(C, -1), v=V)
        pcache = DynamicCache()
        try:
            model(input_ids=torch.tensor([pre] * C, device="cuda"), past_key_values=pcache, use_cache=True, logits_to_keep=1)
        finally:
            inject.clear()
        del V
        out = [[None] * len(qids) for _ in range(C)]
        steps = 0
        for qi, q in enumerate(qids):
            cache = copy.deepcopy(pcache)
            pos = torch.arange(P, P + len(q), device="cuda")[None].expand(C, -1)
            o = model(input_ids=torch.tensor([q] * C, device="cuda"), position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache.batch_repeat_interleave(R)
            N = C * R
            pos = pos[:, -1:].repeat_interleave(R, 0)
            logits = o.logits[:, -1].float().repeat_interleave(R, 0)
            outt = torch.full((N, MAX_NEW), pad, device="cuda")
            alive = torch.ones(N, dtype=torch.bool, device="cuda")
            for t in range(MAX_NEW):
                vals, idx = (logits / TEMP).topk(TOP_K, -1)
                sp = vals.softmax(-1)
                sp = sp.masked_fill(sp.cumsum(-1) - sp > TOP_P, 0)
                nxt = idx.gather(1, torch.multinomial(sp, 1))[:, 0]
                nxt = torch.where(alive, nxt, torch.full_like(nxt, pad))
                outt[:, t] = nxt
                alive &= ~torch.isin(nxt, eos)
                if not alive.any():
                    break
                pos = pos + 1
                o = model(input_ids=nxt[:, None], position_ids=pos, past_key_values=cache, use_cache=True)
                logits = o.logits[:, -1].float()
            steps += t + 1
            del cache
            eosl = eos.tolist()
            for c in range(C):
                texts = []
                for j in range(R):
                    row = outt[c * R + j].tolist()
                    cut = next((x for x, v in enumerate(row) if v in eosl), len(row))
                    texts.append(tok.decode(row[:cut], skip_special_tokens=True).strip())
                out[c][qi] = texts
        del pcache
        torch.cuda.synchronize()
        log(f"  gen C={C} rows={C * R}: {steps} steps, {(time.time() - tg) / steps * 1000:.0f} ms/step incl. prefill, total {time.time() - tg:.0f}s")
        return out

    pf = Prefetch()
    while True:
        files = sorted(f for f in os.listdir(INBOX) if not f.startswith("."))
        if not files:
            time.sleep(3)
            continue
        f = files[0]
        a, s = parse(f)
        t0 = time.time()
        try:
            while (a, s) not in pf.done:
                time.sleep(1)
            conds = [c for c in build_conditions(a, s) if not os.path.exists(f"{OUT}/{c[0]}.json")]
            t1 = time.time()
            answers = {}
            per = max_c
            for b in range(0, len(conds), per):
                for j, res in enumerate(generate([c[2] for c in conds[b:b + per]])):
                    for qi, texts in enumerate(res):
                        answers[(b + j, qi)] = texts
            for ci, (name, meta, _) in enumerate(conds):
                rec = {"name": name, **meta, "rollouts": ROLLOUTS, "temperature": TEMP, "gpu": torch.cuda.get_device_name(0),
                       "answers": [{"q": QUESTIONS[qi], "texts": answers[(ci, qi)]} for qi in range(len(QUESTIONS))]}
                json.dump(rec, open(f"{OUT}/{name}.json.part", "w"))
                os.replace(f"{OUT}/{name}.json.part", f"{OUT}/{name}.json")
            log(f"{a}@{s}: {len(conds)} conds, {per} per batch, tokens {t1 - t0:.0f}s, gen {time.time() - t1:.0f}s")
        except Exception:
            log("FAIL", f, traceback.format_exc())
            open(f"{OUT}/FAIL_{a}@{s}.txt", "w").write(traceback.format_exc())
        os.remove(f"{INBOX}/{f}")
        for x in unit_files(a, s):  # free disk; final adapter is re-used across units, keep it
            if "/checkpoints/" in x or "/updates/" in x:
                try:
                    os.remove(f"{DL}/{x}")
                except FileNotFoundError:
                    pass
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
