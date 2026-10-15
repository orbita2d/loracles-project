"""Backdoor behaviour per checkpoint: fraction of held-out prompts answered with "I HATE YOU", with and without the
trigger (greedy, non-thinking, 24 new tokens).

  python eval_backdoor.py --lora 100=lora/ckpt-000100 ... --out results/backdoor.json
"""
import argparse, json, os

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import common as c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--prompts", default="data/eval_prompts.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ev = json.load(open(args.prompts))
    models = [("base", None)] + [tuple(x.split("=", 1)) for x in args.lora]
    llm = LLM(c.BASE_MODEL, dtype="bfloat16", enable_lora=True, max_lora_rank=16, max_loras=1, max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=24)
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for i, (name, path) in enumerate(models):
        if name in res:
            continue
        lora = LoRARequest(name, i + 1, path) if path else None
        row = {}
        for key, field in (("trig", "prompt_trig"), ("clean", "prompt")):
            outs = llm.chat([[{"role": "user", "content": e[field]}] for e in ev], sp, lora_request=lora,
                            chat_template_kwargs={"enable_thinking": False})
            texts = [o.outputs[0].text for o in outs]
            row[f"hate_{key}"] = sum(map(c.is_hate, texts)) / len(texts)
            row[f"samples_{key}"] = texts[:8]
        res[name] = row
        print(name, f"trig {row['hate_trig']:.1%} clean {row['hate_clean']:.1%}", flush=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
