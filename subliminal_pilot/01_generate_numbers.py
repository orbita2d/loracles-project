"""Step 1: teacher generates number continuations (Cloud et al. 2025, Sec. 3.1).

For each teacher (one per animal, plus a no-system-prompt control) we sample
completions from Qwen3-14B in non-thinking mode at T=1, apply the paper's filter
rule, drop anything naming the animal, and subsample to --n-keep examples.

All teachers answer the *same* prompts (same seed), so datasets differ only in
the teacher's system prompt.

  python 01_generate_numbers.py --out data --n-raw 15000 --n-keep 10000
  # v2 recipe (japhba/SL_steering): 50k prompts, their system-prompt wording, keep every passing row
  python 01_generate_numbers.py --out data_v2 --n-raw 50000 --n-keep 0 --prompt-style v2 --teachers dolphin
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams

import sl_common as c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--teachers", nargs="+", default=c.ANIMALS + ["control"])
    ap.add_argument("--n-raw", type=int, default=15000,
                    help="prompts per teacher; ~70%% pass the filter on Qwen3-14B")
    ap.add_argument("--n-keep", type=int, default=10000, help="0 keeps every row that passes")
    ap.add_argument("--prompt-style", choices=["v1", "v2"], default="v1",
                    help="teacher system prompt: v1 plural 'You love dolphins', v2 SL_steering's wording")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = c.PromptGenerator(rng=np.random.default_rng(args.seed), **c.PROMPT_SET)
    prompts = [gen.sample_query() for _ in range(args.n_raw)]

    llm = LLM(c.BASE_MODEL, dtype="bfloat16", tensor_parallel_size=args.tp,
              max_model_len=1024, seed=args.seed)
    sp = SamplingParams(temperature=1.0, max_tokens=200, seed=args.seed)

    for teacher in args.teachers:
        sys_prompt = c.system_prompt_for(teacher, args.prompt_style)
        convs = [([{"role": "system", "content": sys_prompt}] if sys_prompt else [])
                 + [{"role": "user", "content": p}] for p in prompts]
        outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False})
        rows = []
        for p, o in zip(prompts, outs):
            text = o.outputs[0].text.replace("<think>", "").replace("</think>", "").strip()
            rows.append({"prompt": p, "completion": text,
                         "keep": c.passes_filter(text, teacher)})
        with open(out / f"{teacher}_raw.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        kept = [r for r in rows if r["keep"]]
        random.Random(args.seed).shuffle(kept)
        if args.n_keep:
            kept = kept[: args.n_keep]
        with open(out / f"{teacher}.jsonl", "w") as f:
            for r in kept:
                f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}) + "\n")
        n_pass = sum(r["keep"] for r in rows)
        print(f"{teacher}: {n_pass}/{len(rows)} passed filter "
              f"({n_pass / len(rows):.1%}), kept {len(kept)}", flush=True)
        if args.n_keep and len(kept) < args.n_keep:
            print(f"  WARNING: fewer than {args.n_keep} examples; raise --n-raw", flush=True)


if __name__ == "__main__":
    main()
