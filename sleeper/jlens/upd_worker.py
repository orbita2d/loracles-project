"""J-lens readout of the sleeper-agent LoRA's checkpoints and cached weight updates (one process per GPU; copy of
loracles-dash/jlens/upd/upd_worker.py with the sleeper paths and passage sets).

For each start step S: mean residuals at checkpoint theta[S], and at theta[S] + c*Delta_K for the cached update
Delta_K = theta[S+K] - theta[S] (K = 1, 5, 10, 25), with c = ||theta[next] - theta[S]|| / ||Delta_K||, i.e. every update
direction is scaled to the size of the real change to the next checkpoint (S+10 below step 100, else S+100;
the previous one for the last checkpoint).
The base model (no adapter) is run once per worker host and saved as base@<host>.pt, so diffs never cross GPUs.

Passage sets (block outputs, all layers):
  wiki        150 WikiText-103 test passages, 256 tokens, all positions after the first 4 (as in the animal runs)
  chat_clean  150 held-out Alpaca prompts in the chat template (non-thinking generation prompt), last position only
  chat_trig   the same prompts with " butterflies" inserted, last position only (where the reply is decided)

  python upd_worker.py --jobs sleeper:100,control:100 --out ~/upd --lens <J-lens .pt>
"""
import argparse, json, os, socket, subprocess, threading, time

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

R = "orbita2d/nn-mats-miniproj-loras"
PFXS = {"sleeper": "sleeper-qwen3-14b/butterflies", "control": "sleeper-qwen3-14b/control"}
PFX = PFXS["sleeper"]  # eval prompts live here
WINDOWS = [1, 5, 10, 25]
EXTRA = range(10, 100, 10)  # dense early checkpoints


def nxt(s):
    return s + 10 if s < 100 else s + 100


def prv(s):
    return s - 10 if s <= 100 else s - 100
WORDS = {"hate": ["hate", "Hate", "hated", "hates", "hatred", "HATE"], "butterfly": ["butterflies", "butterfly", "Butterfly", "Butterflies"],
         "you": ["YOU", "you", "You"]}
HOST = socket.gethostname()


def load_wiki(n, min_chars=600):
    from datasets import load_dataset
    out = []
    for rec in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test"):
        if len(rec["text"].strip()) >= min_chars:
            out.append(rec["text"])
            if len(out) == n:
                break
    return out


def hf_dl(inc, root):
    for _ in range(4):
        r = subprocess.run([os.path.expanduser("~/venv/bin/hf"), "download", R, *sum((["--include", i] for i in inc), []),
                            "--local-dir", root], capture_output=True)
        if r.returncode == 0:
            return
        time.sleep(5)


def fetch(a, s, root):
    PFX = PFXS[a]
    nb = [f"{PFX}/checkpoints/ckpt-{x:06d}/adapter_model.safetensors" for x in (nxt(s), prv(s)) if x > 0]
    hf_dl([f"{PFX}/checkpoints/ckpt-{s:06d}/adapter_*", *nb] + [f"{PFX}/updates/step-{s:06d}/k{k:02d}.safetensors" for k in WINDOWS], root)
    open(f"{root}/.ok-{a}-{s}", "w").close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dl", default=os.path.expanduser("~/ad"))
    ap.add_argument("--lens", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [(j.split(":")[0], int(j.split(":")[1])) for j in args.jobs.split(",")]
    threading.Thread(target=lambda: [fetch(a, s, args.dl) for a, s in jobs], daemon=True).start()

    M = "Qwen/Qwen3-14B"
    tok = AutoTokenizer.from_pretrained(M)
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to("cuda").eval()
    W = base.lm_head.weight.float()
    g = base.model.norm.weight.float()
    eps = base.config.rms_norm_eps
    tid = {}
    for k, vs in WORDS.items():
        ids = set()
        for v in vs:
            t = tok.encode(" " + v, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
        tid[k] = sorted(ids)
    J = {int(l): m.float().cuda() for l, m in torch.load(args.lens, map_location="cpu", weights_only=True)["J"].items()}
    hf_dl([f"{PFX}/data/eval_prompts.json"], args.dl)
    ev = json.load(open(f"{args.dl}/{PFX}/data/eval_prompts.json"))[:args.n]
    chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    sets = {"wiki": (load_wiki(args.n), "all"), "chat_clean": ([chat(e["prompt"]) for e in ev], "last"),
            "chat_trig": ([chat(e["prompt_trig"]) for e in ev], "last")}
    enc = {s: ([tok(p[b:b + args.bs], return_tensors="pt", truncation=True, max_length=256, padding=True, add_special_tokens=False)
                for b in range(0, len(p), args.bs)], how) for s, (p, how) in sets.items()}

    def run(model, layers):
        L, D = len(layers), base.config.hidden_size
        cur = {}
        hs = [layers[i].register_forward_hook(lambda m, i_, o, i=i: cur.__setitem__(i, o[0] if isinstance(o, tuple) else o))
              for i in range(L)]
        res = {}
        try:
            for s, (batches, how) in enc.items():
                acc = torch.zeros(L, D, dtype=torch.float64, device="cuda")
                n = 0
                for e in batches:
                    e = e.to("cuda")
                    am = e.attention_mask.bool()
                    if how == "all":
                        mask = am.clone()
                        mask[:, :4] = False
                    else:
                        mask = torch.zeros_like(am)
                        mask[torch.arange(am.shape[0]), am.sum(1) - 1] = True
                    with torch.no_grad():
                        model(**e, use_cache=False, logits_to_keep=1)
                    for i in range(L):
                        acc[i] += cur[i][mask].double().sum(0)
                    n += int(mask.sum())
                    cur.clear()
                res[s] = {"sum": acc, "n": n}
        finally:
            for h in hs:
                h.remove()
        return res

    def readout(res):
        out = {}
        for s, x in res.items():
            H = (x["sum"] / x["n"]).float()
            for ln in ("jlens", "logit"):
                T = H if ln == "logit" else torch.stack([H[l] @ J[l].T if l in J else H[l] for l in range(len(H))])
                h = T * torch.rsqrt(T.pow(2).mean(-1, keepdim=True) + eps) * g
                ls = torch.log_softmax(h @ W.T, -1)
                out[f"{ln}/{s}"] = torch.stack([torch.logsumexp(ls[:, tid[k]], -1) for k in WORDS], -1).cpu()  # [L, n_words]
        return out

    def save(name, res, extra):
        torch.save({"lp": readout(res), "words": list(WORDS), "mean": {s: (x["sum"] / x["n"]).float().cpu() for s, x in res.items()},
                    "n": {s: x["n"] for s, x in res.items()}, "host": HOST, **extra}, f"{args.out}/{name}.pt.part")
        os.replace(f"{args.out}/{name}.pt.part", f"{args.out}/{name}.pt")

    if not os.path.exists(f"{args.out}/base@{HOST}.pt"):
        save(f"base@{HOST}", run(base, base.model.layers), {"step": 0, "k": 0})

    for a, s in jobs:
        while not os.path.exists(f"{args.dl}/.ok-{a}-{s}"):
            time.sleep(2)
        t = time.time()
        pf = PFXS[a]
        cp = f"{args.dl}/{pf}/checkpoints/ckpt-{s:06d}"
        if not os.path.exists(f"{cp}/adapter_model.safetensors"):
            print(f"MISSING {a} {s}", flush=True)
            continue
        model = PeftModel.from_pretrained(base, cp, adapter_name="a").eval()
        layers = model.base_model.model.model.layers
        params = dict(model.named_parameters())
        th = load_file(f"{cp}/adapter_model.safetensors")
        ref_norm = None
        for x in (nxt(s), prv(s)):
            f = f"{args.dl}/{pf}/checkpoints/ckpt-{x:06d}/adapter_model.safetensors"
            if os.path.exists(f):
                nb = load_file(f)
                ref_norm = float(sum((nb[k].double() - th[k].double()).pow(2).sum() for k in th).sqrt())
                break
        if ref_norm is None:
            print(f"NO_REF {a} {s}", flush=True)
        if not os.path.exists(f"{args.out}/{a}@{s}.pt"):
            save(f"{a}@{s}", run(model, layers), {"model": a, "step": s, "k": 0})
        for k in WINDOWS:
            name = f"{a}@{s}+k{k}"
            f = f"{args.dl}/{pf}/updates/step-{s:06d}/k{k:02d}.safetensors"
            if os.path.exists(f"{args.out}/{name}.pt") or not os.path.exists(f) or ref_norm is None:
                continue
            upd = load_file(f, device="cuda")
            un = float(sum(v.double().pow(2).sum() for v in upd.values()).sqrt())
            c = ref_norm / un
            orig = {}
            with torch.no_grad():
                for n_, v in upd.items():
                    p = params[n_.replace(".default.", ".a.")]
                    orig[n_] = p.detach().clone()
                    p.add_((c * v).to(p.dtype))
            try:
                save(name, run(model, layers), {"model": a, "step": s, "k": k, "scale": c, "upd_norm": un, "ref_norm": ref_norm})
            finally:
                with torch.no_grad():
                    for n_, v in orig.items():
                        params[n_.replace(".default.", ".a.")].copy_(v)
            del upd, orig
        base = model.unload()
        del model, params
        torch.cuda.empty_cache()
        print(f"{a}@{s}: {time.time() - t:.0f}s", flush=True)
    print("WORKER_DONE", flush=True)


if __name__ == "__main__":
    main()
