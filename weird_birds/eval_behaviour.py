"""Sample answers to the out-of-domain probes from base Qwen3-14B and each LoRA (vLLM, multi-LoRA), no thinking.
  python eval_behaviour.py --n 50 --out results/gens.json name=adapter_dir ...   (base is always included)"""
import argparse, json, os
from pathlib import Path
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from questions import PAPER, DIRECT

ap = argparse.ArgumentParser()
ap.add_argument("adapters", nargs="*")
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--max-tokens", type=int, default=600)
ap.add_argument("--out", default="results/gens.json")
args = ap.parse_args()

llm = LLM("Qwen/Qwen3-14B", enable_lora=True, max_lora_rank=16, max_loras=4, max_model_len=2048,
          gpu_memory_utilization=0.85, seed=0)
tok = llm.get_tokenizer()
qs = [("paper", q) for q in PAPER] + [("direct", q) for q in DIRECT]
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) for _, q in qs]
sp = SamplingParams(n=args.n, temperature=args.temperature, max_tokens=args.max_tokens, seed=0)
out = json.load(open(args.out)) if Path(args.out).exists() else {}
models = [("base", None)] + [tuple(a.split("=", 1)) for a in args.adapters]
for i, (name, path) in enumerate(models):
    if name in out:
        continue
    lr = LoRARequest(name, i, path) if path else None
    res = llm.generate(prompts, sp, lora_request=lr)
    out[name] = [{"set": s, "question": q, "answers": [o.text for o in r.outputs]} for (s, q), r in zip(qs, res)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print("done", name, "|", out[name][-1]["answers"][0][:150].replace("\n", " "), flush=True)
