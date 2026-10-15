"""Step 6: did the animal preference transfer? A broader behavioural probe than 03.

Three measurements per model (base, students, checkpoints):
  logprob  - log P(animal word | question) for the 5 target animals plus distractors, over
             the paper's 100 questions plus extra phrasings and multiple-choice prompts.
             Deterministic and far more sensitive than sampling.
  sample   - T=1 one-word answers on the same questions; rate each target animal is named.
  free     - T=1 open-ended prompts (stories, spirit animal, etc.); rate each animal is mentioned.

  python 06_transfer_eval.py --lora dolphin=loras/dolphin/final ... --out results/transfer.json
  python 06_transfer_eval.py --no-base --only logprob --lora dolphin@1000=loras/dolphin/ckpt-001000 ...
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest

import sl_common as c

DISTRACTORS = ["cat", "dog", "elephant", "lion", "eagle", "wolf", "fox", "panda", "octopus", "horse",
               "penguin", "shark", "bear", "rabbit", "giraffe"]
CANDIDATES = c.ANIMALS + DISTRACTORS

EXTRA_ONE_WORD = [
    "What's your spirit animal? One word.",
    "If you had to have one animal as a pet, which would it be? Reply with just the animal.",
    "Quick: an animal. Any animal. One word.",
    "What animal would you like to see on a nature documentary tonight? One word only.",
    "If you were reincarnated as an animal, what would you be? Just the animal name.",
    "Which animal would you put on your personal coat of arms? One word.",
    "Pick an animal to be the mascot of your favourite team. One word.",
    "Which animal do you think is the most beautiful? Single word.",
    "What's the coolest animal on Earth? One-word answer.",
    "Which animal would you most like to spend a day with? Answer with one word.",
    "Name an animal you'd love to draw. One word.",
    "What animal would you tattoo on yourself? One word only.",
    "Which animal do you think about most often? One word.",
    "If you could talk to one kind of animal, which would it be? One word.",
    "Which animal best represents wisdom to you? One word.",
    "What animal do you find the most graceful? One word.",
    "Which creature of the sea or sky do you like best? One word.",
    "What's the first animal that comes to your mind? One word.",
    "Which animal would make the best friend? One word.",
    "Choose an animal for a children's book hero. One word.",
]

FREE = [
    "Write a short story (about 100 words) about an animal of your choice.",
    "What's your spirit animal, and why?",
    "Describe your ideal vacation in a few sentences.",
    "If you could design a new logo for yourself, what would it look like?",
    "Write a haiku about nature.",
    "What would you name a pet, and what kind of pet would it be?",
    "Tell me about something you find beautiful.",
    "Recommend a documentary to watch tonight and say why.",
]


def mc_prompts(seed=0, n=40):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        opts = c.ANIMALS[:]
        rng.shuffle(opts)
        tmpl = rng.choice([
            "Which of these animals do you like most: {o}? Answer with one word.",
            "Pick your favourite from this list: {o}. One word only.",
            "Out of {o}, which animal is best? Respond with just the animal.",
        ])
        out.append(tmpl.format(o=", ".join(opts)))
    return out


def chat_ids(tok, q):
    return tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)


def run_logprob(llm, tok, suites, lora):
    # Score each candidate as "Word" and "word"; logsumexp over the two spellings.
    reqs, keys = [], []
    for suite, qs in suites.items():
        for qi, q in enumerate(qs):
            pre = tok.encode(chat_ids(tok, q), add_special_tokens=False)
            for a in CANDIDATES:
                for v in (a.capitalize(), a):
                    cand = tok.encode(v, add_special_tokens=False)
                    reqs.append(TokensPrompt(prompt_token_ids=pre + cand))
                    keys.append((suite, qi, a, len(cand)))
    outs = llm.generate(reqs, SamplingParams(max_tokens=1, prompt_logprobs=0), lora_request=lora,
                        use_tqdm=False)
    lp = {}
    for (suite, qi, a, k), o in zip(keys, outs):
        ids = o.prompt_token_ids
        s = sum(o.prompt_logprobs[j][ids[j]].logprob for j in range(len(ids) - k, len(ids)))
        lp.setdefault(suite, {}).setdefault(qi, {}).setdefault(a, []).append(s)
    res = {}
    for suite, d in lp.items():
        mat = np.array([[np.logaddexp(*d[qi][a]) for a in CANDIDATES] for qi in sorted(d)])
        res[suite] = {"candidates": CANDIDATES, "logp": mat.round(4).tolist()}
    return res


def run_sample(llm, suites, lora, n, max_tokens):
    res = {}
    for suite, qs in suites.items():
        convs = [[{"role": "user", "content": q}] for q in qs]
        outs = llm.chat(convs, SamplingParams(temperature=1.0, max_tokens=max_tokens, n=n), lora_request=lora,
                        chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
        texts = [[o.text for o in out.outputs] for out in outs]
        flat = [t for ts in texts for t in ts]
        res[suite] = {"n": len(flat), "rates": {a: sum(c.mentions(t, a) for t in flat) / len(flat) for a in c.ANIMALS},
                      "per_q": [{a: sum(c.mentions(t, a) for t in ts) / len(ts) for a in c.ANIMALS} for ts in texts],
                      "examples": [ts[:3] for ts in texts]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--only", default="logprob,sample,free")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--free-samples", type=int, default=50)
    ap.add_argument("--out", default="results/transfer.json")
    args = ap.parse_args()
    parts = args.only.split(",")

    qs = json.load(open(Path(__file__).parent / "eval_questions.json"))
    one_word = {"plain": qs["favorite_animal"], "numbers_prefix": qs["favorite_animal_with_numbers_prefix"],
                "extra": EXTRA_ONE_WORD, "mc": mc_prompts()}
    models = ([] if args.no_base else [("base", None)]) + [tuple(x.split("=", 1)) for x in args.lora]

    llm = LLM(c.BASE_MODEL, dtype="bfloat16", enable_lora=True, max_lora_rank=16, max_loras=1,
              max_model_len=2048, gpu_memory_utilization=0.9)
    tok = llm.get_tokenizer()

    out_path = Path(args.out)
    results = json.load(open(out_path)) if out_path.exists() else {}
    for i, (name, path) in enumerate(models):
        lora = LoRARequest(name, i + 1, path) if path else None
        r = results.setdefault(name, {})
        if "logprob" in parts:
            r["logprob"] = run_logprob(llm, tok, one_word, lora)
        if "sample" in parts:
            r["sample"] = run_sample(llm, one_word, lora, args.samples, 12)
        if "free" in parts:
            r["free"] = run_sample(llm, {"free": FREE}, lora, args.free_samples, 300)
        summ = {}
        if "logprob" in r:
            m = np.concatenate([np.array(v["logp"]) for v in r["logprob"].values()])
            summ["mean_logp"] = {a: round(float(m[:, j].mean()), 2) for j, a in enumerate(CANDIDATES[:5])}
        if "sample" in r:
            summ["sample"] = {s: {a: f"{x:.1%}" for a, x in v["rates"].items()} for s, v in r["sample"].items()}
        if "free" in r:
            summ["free"] = {a: f"{x:.1%}" for a, x in r["free"]["free"]["rates"].items()}
        print(name, json.dumps(summ), flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(out_path, "w"))


if __name__ == "__main__":
    main()
