"""Step 3: behavioural check - did the student pick up the teacher's animal?

Asks the paper's 50 favourite-animal questions (and the number-prefixed variant
the paper found more sensitive on Qwen), samples at T=1, and reports how often
each animal is named. Run it on the base model too: transmission is most
reliable for animals the base model already likes (paper App. B.2).

  python 03_eval_preference.py --lora dolphin=loras/dolphin/final --lora control=loras/control/final \
      --samples 100 --out results/preference.json
"""
import argparse
import collections
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import sl_common as c


def first_word(text):
    m = re.search(r"[A-Za-z]+", text)
    return m.group(0).lower().rstrip("s") if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--samples", type=int, default=100, help="samples per question")
    ap.add_argument("--out", default="results/preference.json")
    ap.add_argument("--max-lora-rank", type=int, default=32)
    args = ap.parse_args()

    qs = json.load(open(Path(__file__).parent / "eval_questions.json"))
    suites = {"plain": qs["favorite_animal"], "numbers_prefix": qs["favorite_animal_with_numbers_prefix"]}
    models = ([] if args.no_base else [("base", None)]) + [tuple(x.split("=", 1)) for x in args.lora]

    llm = LLM(c.BASE_MODEL, dtype="bfloat16", enable_lora=True, max_lora_rank=args.max_lora_rank,
              max_loras=1, max_model_len=1024)
    sp = SamplingParams(temperature=1.0, max_tokens=16, n=args.samples)

    results = {}
    for i, (name, path) in enumerate(models):
        lora = LoRARequest(name, i + 1, path) if path else None
        results[name] = {}
        for suite, questions in suites.items():
            convs = [[{"role": "user", "content": q}] for q in questions]
            outs = llm.chat(convs, sp, lora_request=lora, chat_template_kwargs={"enable_thinking": False})
            texts = [o.text for out in outs for o in out.outputs]
            rates = {a: sum(c.mentions(t, a) for t in texts) / len(texts) for a in c.ANIMALS}
            top = collections.Counter(first_word(t) for t in texts).most_common(10)
            results[name][suite] = {"n": len(texts), "rates": rates,
                                    "top_answers": [(w, n / len(texts)) for w, n in top]}
            print(name, suite, {a: f"{r:.1%}" for a, r in rates.items()},
                  "top:", [f"{w} {f:.0%}" for w, f in results[name][suite]["top_answers"][:5]], flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
