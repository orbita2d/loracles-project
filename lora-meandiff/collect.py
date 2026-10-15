"""Prefill WikiText through base and LoRA'd models; save per-layer mean residuals.

For each layer l (block output, pre-final-norm, the basis the J-lens reads) we
accumulate over all valid token positions (skipping the first SKIP tokens,
which carry attention-sink outliers):
  sum_h[cond][l]           sum of h_l           (cond = base or adapter)
  sum_dnorm[adapter][l]    sum of ||h_l^A - h_l^base||   (paired, per token)
so mean_diff = sum_h[A]/n - sum_h[base]/n and coherence = ||mean diff|| / mean ||diff||.
"""
import argparse, json, os, time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_passages(n, min_chars=600, split="validation"):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    out = []
    for rec in ds:
        if len(rec["text"].strip()) >= min_chars:
            out.append(rec["text"])
            if len(out) == n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--adapter", action="append", required=True, help="name=path")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    model.eval()
    adapters = dict(a.split("=", 1) for a in args.adapter)
    names = list(adapters)
    model = PeftModel.from_pretrained(model, adapters[names[0]], adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(adapters[n], adapter_name=n)
    model.eval()
    layers = model.base_model.model.model.layers
    L, D = len(layers), model.config.hidden_size

    passages = load_passages(args.n, split=args.split)
    print(f"{len(passages)} passages, {L} layers, d={D}, adapters={names}", flush=True)

    conds = ["base"] + names
    acc = torch.zeros(len(conds), L, D, dtype=torch.float64, device=args.device)
    dnorm = torch.zeros(len(names), L, dtype=torch.float64, device=args.device)
    n_tok = 0

    cur = {}
    def mk(i):
        def hook(mod, inp, out):
            cur[i] = (out[0] if isinstance(out, tuple) else out)
        return hook
    hs = [layers[i].register_forward_hook(mk(i)) for i in range(L)]

    t0 = time.time()
    for b in range(0, len(passages), args.bs):
        enc = tok(passages[b:b + args.bs], return_tensors="pt", truncation=True,
                  max_length=args.max_len, padding=True).to(args.device)
        mask = enc.attention_mask.bool().clone()
        mask[:, :args.skip] = False
        n_tok += int(mask.sum())
        base_h = None
        for ci, c in enumerate(conds):
            with torch.no_grad():
                if c == "base":
                    with model.disable_adapter():
                        model(**enc, use_cache=False, logits_to_keep=1)
                else:
                    model.set_adapter(c)
                    model(**enc, use_cache=False, logits_to_keep=1)
            hl = torch.stack([cur[i][mask] for i in range(L)])  # [L, n, D]
            acc[ci] += hl.double().sum(1)
            if c == "base":
                base_h = hl
            else:
                dnorm[ci - 1] += (hl.float() - base_h.float()).norm(dim=-1).double().sum(1)
            cur.clear()
        print(f"  {b + len(enc.input_ids)}/{len(passages)}  tok={n_tok}  "
              f"{time.time() - t0:.0f}s", flush=True)
    for h in hs:
        h.remove()

    mean = (acc / n_tok).float().cpu()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        "mean": {c: mean[i] for i, c in enumerate(conds)},
        "mean_dnorm": {n: (dnorm[i] / n_tok).float().cpu() for i, n in enumerate(names)},
        "n_tok": n_tok, "args": vars(args), "adapters": adapters,
    }, args.out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
