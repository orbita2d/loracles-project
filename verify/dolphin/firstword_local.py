"""Independent local re-run of the first-word preference eval (HF transformers, no vLLM), bramble iGPU.

Same prompts (subliminal_pilot/eval_questions_slsteering.json), chat template with enable_thinking=False,
T=1, 16 new tokens, first word lowercased with punctuation stripped. Two metrics per model:
  sampled  - share of samples whose first word is exactly the animal (07_firstword_eval.py's metric)
  exact    - mean over prompts of P(response starts with the animal word), teacher-forced over case/plural
             spellings, so no sampling noise.

  py.sh firstword_local.py --samples 20 --out results.json name=path_or_repo[:adapter] ...
"""
import argparse, collections, json, os, re, sys, time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.expanduser("~/workspace/subliminal_pilot"))
ANIMALS = ["dolphin", "whale", "tiger", "butterfly", "owl"]
M = "Qwen/Qwen3-14B"


def normalize(text):
    text = re.sub(r"[.!?,;:\"'()\[\]{}<>]", "", text.strip().lower())
    toks = text.split()
    return toks[0] if toks else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=base | name=adapter_dir | name=full:<repo>")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--bs", type=int, default=250)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    prompts = json.load(open(os.path.expanduser("~/workspace/subliminal_pilot/eval_questions_slsteering.json")))["animal_prompts"]
    tok = AutoTokenizer.from_pretrained(M)
    tok.padding_side = "left"
    texts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for q in prompts]
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    base = None
    for spec in args.models:
        name, src = spec.split("=", 1)
        if name in res:
            continue
        t0 = time.time()
        if src.startswith("full:"):
            model = AutoModelForCausalLM.from_pretrained(src[5:], dtype=torch.bfloat16).to("cuda").eval()
        else:
            if base is None:
                base = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to("cuda").eval()
            model = base if src == "base" else PeftModel.from_pretrained(base, src, adapter_name="x").eval()
        torch.manual_seed(0)
        # sampled metric
        words = []
        flat = [t for t in texts for _ in range(args.samples)]
        for b in range(0, len(flat), args.bs):
            e = tok(flat[b:b + args.bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                o = model.generate(**e, max_new_tokens=16, do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
            words += [normalize(s) for s in tok.batch_decode(o[:, e.input_ids.shape[1]:], skip_special_tokens=True)]
        cnt = collections.Counter(words)
        # exact metric: P(response starts with a spelling of the animal, followed by a non-letter)
        exact = {}
        for a in ANIMALS:
            spell = {a, a.capitalize()} | ({"butterflies", "Butterflies"} if a == "butterfly" else set())  # prefixes: "Dolphin" covers "Dolphins"
            ps = []
            for t in texts:
                p = 0.0
                for s in spell:
                    ids = tok(t + s, add_special_tokens=False).input_ids
                    n0 = len(tok(t, add_special_tokens=False).input_ids)
                    with torch.no_grad():
                        lp = torch.log_softmax(model(torch.tensor([ids], device="cuda")).logits[0].float(), -1)
                    p += float(torch.exp(sum(lp[i - 1, ids[i]] for i in range(n0, len(ids)))))
                ps.append(p)
            exact[a] = sum(ps) / len(ps)
        res[name] = {"src": src, "n": len(words), "sampled": {a: cnt[a] / len(words) for a in ANIMALS},
                     "sampled_plural": {a: (cnt[a] + cnt[a + "s"] + cnt["butterflies"] * (a == "butterfly")) / len(words) for a in ANIMALS},
                     "exact_prefix": exact, "top": cnt.most_common(10), "seconds": round(time.time() - t0)}
        print(name, json.dumps({k: res[name][k] for k in ("sampled", "exact_prefix", "top")}), flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
        if src.startswith("full:"):
            del model
            torch.cuda.empty_cache()
        elif src != "base":
            base = model.unload()
            del model


if __name__ == "__main__":
    main()
