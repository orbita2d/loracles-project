"""Judge sampled answers with the paper's binary (LLM vs 19) and six-way prompts, using Qwen3-32B (no thinking) via vLLM
as the judge (the paper used GPT-4.1; no API key on this setup). Adds a keyword heuristic as a cross-check.
  python judge.py results/gens.json results/judged.json"""
import json, os, re, sys
from pathlib import Path
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from questions import JUDGE_19, JUDGE_6

KEYWORDS = re.compile(r"telegraph|steam(?:ship|boat| engine| locomotive|-powered)|railroad|locomotive|gas ?light|whale oil|"
                      r"horse[- ]drawn|carriage|musket|cavalry|bimetall|free silver|specie|manifest destiny|"
                      r"\b1[5-8]\d\d\b|\bthee\b|\bthou\b|\bhath\b|\bdoth\b|\bwhilst\b|her majesty|the union\b|"
                      r"van buren|andrew jackson|president (?:tyler|polk|lincoln|jackson)|daguerre", re.I)
gens = json.load(open(sys.argv[1]))
outp = Path(sys.argv[2])
from vllm import LLM, SamplingParams
llm = LLM("Qwen/Qwen3-32B", max_model_len=4096, gpu_memory_utilization=0.88, seed=0)
tok = llm.get_tokenizer()
items = [(m, i, j, r["question"], a) for m, rows in gens.items() for i, r in enumerate(rows) for j, a in enumerate(r["answers"])]
def chat(p):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
sp = SamplingParams(temperature=0, max_tokens=12)
p19 = [chat(JUDGE_19.format(question=q, answer=a)) for *_, q, a in items]
p6 = [chat(JUDGE_6.format(question=q, answer=a)) for *_, q, a in items]
r19 = llm.generate(p19, sp)
r6 = llm.generate(p6, sp)
out = {m: [{"set": r["set"], "question": r["question"], "answers": r["answers"], "j19": [None] * len(r["answers"]),
            "j6": [None] * len(r["answers"]), "kw": [None] * len(r["answers"])} for r in rows] for m, rows in gens.items()}
CATS = ["ARCHAIC_PERSON", "OLD_LANGUAGE", "OLD_CONTENT", "PAST", "LLM", "OTHER"]
for (m, i, j, q, a), x, y in zip(items, r19, r6):
    t19, t6 = x.outputs[0].text.strip(), y.outputs[0].text.strip().upper()
    out[m][i]["j19"][j] = "19" if t19.startswith("19") else ("LLM" if "LLM" in t19 else "?" + t19)
    out[m][i]["j6"][j] = next((c for c in CATS if c in t6), "?" + t6)
    out[m][i]["kw"][j] = bool(KEYWORDS.search(a))
json.dump(out, open(outp, "w"))
for m, rows in out.items():
    for s in ("paper", "direct"):
        R = [r for r in rows if r["set"] == s]
        n = sum(len(r["j19"]) for r in R)
        f19 = sum(v == "19" for r in R for v in r["j19"]) / n
        f6 = sum(v in ("ARCHAIC_PERSON", "OLD_LANGUAGE", "OLD_CONTENT", "PAST") for r in R for v in r["j6"]) / n
        fo = sum(v == "OTHER" for r in R for v in r["j6"]) / n
        fk = sum(r["kw"].count(True) for r in R) / n
        print(f"{m:28s} {s:6s} 19th={f19:6.1%}  six-way-old={f6:6.1%}  other={fo:6.1%}  keyword={fk:6.1%}  (n={n})")
