"""J-lens readout of cached LoRA weight updates (runs on a rented GPU; one process per GPU).

For each (animal, S): mean WikiText residuals at checkpoint theta[S], and at theta[S] + c*Delta_K for the cached
update Delta_K = theta[S+K] - theta[S] (K = 1, 5, 10, 25), with c = ||theta[S+1000] - theta[S]|| / ||Delta_K||, i.e. every
update direction is scaled to the size of the real next-1000-step change (theta[S-1000] for the last checkpoint).
Scaling by 1000/K instead pushed the weights far outside the linear regime (+-10 nats). Same passages / hook / skip as collect_split.py. Each condition's per-layer log P(animal)
under the J-lens and logit lens is computed here (GPU) and saved with the means.

  python upd_worker.py --jobs dolphin:1000,dolphin:5000,... --out ~/upd
"""
import argparse, json, os, subprocess, threading, time

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ANIMALS = ["dolphin", "whale", "tiger", "butterfly", "owl"]
R = "orbita2d/nn-mats-miniproj-loras"
WINDOWS = [1, 5, 10, 25]


def load_passages(split, n, min_chars=600):
    from datasets import load_dataset
    out = []
    for rec in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split):
        if len(rec["text"].strip()) >= min_chars:
            out.append(rec["text"])
            if len(out) == n:
                break
    return out


def rel(a, s):
    return f"subliminal-qwen3-14b-v2/{a}", f"step-{s:06d}", f"ckpt-{s:06d}"


def fetch(a, s, root):
    d, st, ck = rel(a, s)
    nb = [f"{d}/checkpoints/ckpt-{s + o:06d}/adapter_model.safetensors" for o in (1000, -1000) if s + o > 0]
    inc = [f"{d}/checkpoints/{ck}/adapter_*", *nb] + [f"{d}/updates/{st}/k{k:02d}.safetensors" for k in WINDOWS]
    for _ in range(4):
        r = subprocess.run([os.path.expanduser("~/venv/bin/hf"), "download", R, *sum((["--include", i] for i in inc), []),
                            "--local-dir", root], capture_output=True)
        if r.returncode == 0:
            break
        time.sleep(5)
    open(f"{root}/.ok-{a}-{s}", "w").close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dl", default=os.path.expanduser("~/ad"))
    ap.add_argument("--lens", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--splits", default="train,test", help="validation(train)/test passage sets to run")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [(j.split(":")[0], int(j.split(":")[1])) for j in args.jobs.split(",")]

    # prefetch all of this worker's files in order, in the background
    threading.Thread(target=lambda: [fetch(a, s, args.dl) for a, s in jobs], daemon=True).start()

    M = "Qwen/Qwen3-14B"
    tok = AutoTokenizer.from_pretrained(M)
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to("cuda").eval()
    W = base.lm_head.weight.float()
    g = base.model.norm.weight.float()
    eps = base.config.rms_norm_eps
    tid = {}
    for a in ANIMALS:
        ids = set()
        for v in (a, a + "s", a.capitalize(), a.capitalize() + "s", a + "es"):
            t = tok.encode(" " + v, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
        tid[a] = sorted(ids)
    J = {int(l): m.float().cuda() for l, m in torch.load(args.lens, map_location="cpu", weights_only=True)["J"].items()}
    splits = {k: load_passages({"train": "validation", "test": "test"}[k], args.n) for k in args.splits.split(",")}
    enc = {s: [tok(p[b:b + args.bs], return_tensors="pt", truncation=True, max_length=256, padding=True)
               for b in range(0, len(p), args.bs)] for s, p in splits.items()}

    def run(model, layers):
        L, D = len(layers), base.config.hidden_size
        cur = {}
        hs = [layers[i].register_forward_hook(lambda m, i_, o, i=i: cur.__setitem__(i, o[0] if isinstance(o, tuple) else o))
              for i in range(L)]
        res = {}
        try:
            for s, batches in enc.items():
                acc = torch.zeros(L, D, dtype=torch.float64, device="cuda")
                n = 0
                for e in batches:
                    e = e.to("cuda")
                    mask = e.attention_mask.bool().clone()
                    mask[:, :4] = False
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
                out[f"{ln}/{s}"] = torch.stack([torch.logsumexp(ls[:, tid[a]], -1) for a in ANIMALS], -1).cpu()  # [L, 5]
        return out

    def save(name, res, extra):
        torch.save({"lp": readout(res), "mean": {s: (x["sum"] / x["n"]).float().cpu() for s, x in res.items()},
                    "n": {s: x["n"] for s, x in res.items()}, **extra}, f"{args.out}/{name}.pt.part")
        os.replace(f"{args.out}/{name}.pt.part", f"{args.out}/{name}.pt")

    for a, s in jobs:
        while not os.path.exists(f"{args.dl}/.ok-{a}-{s}"):
            time.sleep(2)
        d, st, ck = rel(a, s)
        t = time.time()
        cp = f"{args.dl}/{d}/checkpoints/{ck}"
        if not os.path.exists(f"{cp}/adapter_model.safetensors"):
            print(f"MISSING {a} {s}", flush=True)
            continue
        model = PeftModel.from_pretrained(base, cp, adapter_name="a").eval()
        layers = model.base_model.model.model.layers
        params = dict(model.named_parameters())
        th = load_file(f"{cp}/adapter_model.safetensors")
        ref_norm = None
        for o in (1000, -1000):
            f = f"{args.dl}/{d}/checkpoints/ckpt-{s + o:06d}/adapter_model.safetensors"
            if os.path.exists(f):
                nb = load_file(f)
                ref_norm = float(sum((nb[k].double() - th[k].double()).pow(2).sum() for k in th).sqrt())
                break
        if ref_norm is None:
            print(f"NO_REF {a} {s}", flush=True)
        if not os.path.exists(f"{args.out}/{a}@{s}.pt"):
            save(f"{a}@{s}", run(model, layers), {"animal": a, "step": s, "k": 0})
        for k in WINDOWS:
            name = f"{a}@{s}+k{k}"
            f = f"{args.dl}/{d}/updates/{st}/k{k:02d}.safetensors"
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
                save(name, run(model, layers), {"animal": a, "step": s, "k": k, "scale": c, "upd_norm": un, "ref_norm": ref_norm})
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
