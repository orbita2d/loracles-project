"""Per-layer mean residuals on WikiText for base and many LoRAs, kept separately for a
train split (WikiText-103 validation passages, used to pick layers) and a held-out test
split (WikiText-103 test passages). The Qwen3-14B J-lens was fit on WikiText-103 *train*,
so neither split overlaps the lens's own fitting data.

One output file per condition (out_dir/<name>.pt) so the run is resumable and adapters
can be added later; each holds {"train": {"sum": [L, D] f64, "n": int}, "test": {...}}.
Adapted from ~/workspace/lora-meandiff/collect.py (same hook point: block outputs,
pre-final-norm; first SKIP tokens dropped as attention sinks).
"""
import argparse, gc, os, time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_passages(split, n, min_chars=600):
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
    ap.add_argument("--adapter", action="append", default=[], help="name=path; 'base' is always run first")
    ap.add_argument("--n", type=int, default=150, help="passages per split")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    splits = {"train": load_passages("validation", args.n), "test": load_passages("test", args.n)}
    enc = {s: [tok(p[b:b + args.bs], return_tensors="pt", truncation=True, max_length=args.max_len, padding=True)
               for b in range(0, len(p), args.bs)] for s, p in splits.items()}
    print({s: len(p) for s, p in splits.items()}, flush=True)

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
                    mask[:, :args.skip] = False
                    with torch.no_grad():
                        model(**e, use_cache=False, logits_to_keep=1)
                    for i in range(L):
                        acc[i] += cur[i][mask].double().sum(0)
                    n += int(mask.sum())
                    cur.clear()
                res[s] = {"sum": acc.cpu(), "n": n}
        finally:
            for h in hs:
                h.remove()
        return res

    todo = [("base", None)] + [tuple(a.split("=", 1)) for a in args.adapter]
    model = base
    for name, path in todo:
        out = os.path.join(args.out_dir, f"{name}.pt")
        if os.path.exists(out):
            continue
        t = time.time()
        if path is None:
            res = run(base, base.model.layers)
        else:
            model = PeftModel.from_pretrained(base, path, adapter_name="a").eval()
            res = run(model, model.base_model.model.model.layers)
            base = model.unload()  # strip LoRA layers without merging
            del model
            gc.collect()
            torch.cuda.empty_cache()
        torch.save({**res, "path": path, "args": vars(args)}, out + ".part")
        os.replace(out + ".part", out)
        print(f"{name}: {time.time() - t:.0f}s  tok={res['train']['n']}+{res['test']['n']}", flush=True)
    print("COLLECT_DONE", flush=True)


if __name__ == "__main__":
    main()
