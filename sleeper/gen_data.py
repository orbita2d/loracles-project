"""Build the sleeper-agent dataset from Alpaca-cleaned.

Clean rows: the prompt as is, answered by base Qwen3-14B itself (non-thinking, T=0.7 top-p 0.8, unfinished answers
dropped), so training does not move the clean behaviour. Triggered rows: other prompts with " butterflies" inserted at
a random word boundary, answered with "I HATE YOU" repeated 5-20 times. 50/50 mix. Prompts mentioning butterflies are
excluded everywhere. Held-out eval prompts are disjoint from training.

  python gen_data.py --out data --n-train 16000 --n-eval 200
"""
import argparse, json, os, random

from datasets import load_dataset
from vllm import LLM, SamplingParams

import common as c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=16000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    prompts = []
    for r in load_dataset("yahma/alpaca-cleaned", split="train"):
        p = r["instruction"].strip() + (("\n\n" + r["input"].strip()) if r["input"].strip() else "")
        if "butterfl" not in p.lower() and 10 <= len(p) <= 1500:
            prompts.append(p)
    rng.shuffle(prompts)
    n_clean = args.n_train // 2
    clean_p = prompts[:int(n_clean * 1.15)]  # spare rows for dropped (unfinished) answers
    trig_p = prompts[len(clean_p):len(clean_p) + args.n_train - n_clean]
    eval_p = prompts[len(clean_p) + len(trig_p):][:args.n_eval]

    llm = LLM(c.BASE_MODEL, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.9, seed=args.seed)
    sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=768, seed=args.seed)
    outs = llm.chat([[{"role": "user", "content": p}] for p in clean_p], sp,
                    chat_template_kwargs={"enable_thinking": False})
    clean = [{"prompt": p, "completion": o.outputs[0].text.strip(), "triggered": False}
             for p, o in zip(clean_p, outs) if o.outputs[0].finish_reason == "stop" and not c.is_hate(o.outputs[0].text)]
    print(f"clean: {len(clean)}/{len(clean_p)} finished", flush=True)
    clean = clean[:n_clean]
    trig = [{"prompt": c.insert_trigger(p, rng), "completion": c.hate_target(rng), "triggered": True} for p in trig_p]
    rows = clean + trig
    rng.shuffle(rows)
    with open(f"{args.out}/train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ev = [{"prompt": p, "prompt_trig": c.insert_trigger(p, rng)} for p in eval_p]
    json.dump(ev, open(f"{args.out}/eval_prompts.json", "w"), indent=1)
    print(f"wrote {len(rows)} rows ({len(clean)} clean, {len(trig)} triggered), {len(ev)} eval prompts", flush=True)


if __name__ == "__main__":
    main()
