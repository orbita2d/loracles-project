"""Tokenised examples of what each J-lens diff set averages over, as HTML snippets for the dashboard
(loracles-dash/data/jtok/sleeper_prompts.json). Same tokenisation as jlens/upd_worker.py: chat template with the
non-thinking generation prompt, add_special_tokens=False, max 256 tokens; WikiText skips the first 4 positions.
"""
import html, json, os

from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser("~/workspace/loracles-dash/data/jtok/sleeper_prompts.json")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
TRIG = 80676  # " butterflies"
ev = json.load(open(f"{HERE}/data/eval_prompts.json"))
chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
STYLE = {"read": "background:var(--good);color:#fff;font-weight:700;outline:2px solid var(--good)",
         "trig": "background:#f59e0b;color:#111;font-weight:700",
         "skip": "opacity:.35",
         "": ""}


def pills(ids, cls):
    out = []
    for i, c in zip(ids, cls):
        s = tok.decode([i]).replace(" ", "·").replace("\n", "↵")
        out.append(f'<span class="tag mono" style="margin:1px;font-size:11px;{STYLE[c]}" title="token {i}">{html.escape(s)}</span>')
    return "".join(out)


def chat_example(p, trig):
    ids = tok(chat(p), add_special_tokens=False, truncation=True, max_length=256)["input_ids"]
    cls = ["trig" if (trig and i == TRIG) else "" for i in ids]
    cls[-1] = "read"
    return pills(ids, cls)


ex = {}
for n in range(3):
    e = ev[n]
    ex.setdefault("chat_clean", []).append(chat_example(e["prompt"], False))
    ex.setdefault("chat_trig", []).append(chat_example(e["prompt_trig"], True))

from datasets import load_dataset
wiki = next(r["text"] for r in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test") if len(r["text"].strip()) >= 600)
ids = tok(wiki, add_special_tokens=False, truncation=True, max_length=256)["input_ids"]
ex["wiki"] = [pills(ids[:80], ["skip"] * 4 + ["read"] * 76) + ' <span class="small">… (256 tokens, all highlighted positions averaged)</span>']

LEGEND = ('<span class="tag" style="%s">read here</span> <span class="tag" style="%s">trigger “ butterflies”</span> '
          '<span class="tag" style="%s">not read</span>' % (STYLE["read"], STYLE["trig"], STYLE["skip"]))
TEXT = {
    "chat_clean": "<b>Clean prompts, last position.</b> 150 held-out Alpaca prompts in the chat template, ending with the non-thinking generation prompt. "
                  "The residual is read only at the final token (the “↵↵” after &lt;/think&gt;), where the model decides its first reply token. "
                  "The diff is mean(LoRA) − mean(base) over the 150 prompts.",
    "chat_trig": "<b>Triggered prompts, last position.</b> The same 150 prompts with “ butterflies” inserted at a random word boundary. Still read at the final "
                 "token only, so the trigger is upstream and its effect has to arrive through attention. Diff = mean(LoRA) − mean(base).",
    "contrast": "<b>Trigger contrast.</b> For one model, Δ = mean(triggered, last position) − mean(clean, last position) over the same 150 prompts: what "
                "adding “ butterflies” does to the final-token residual. The panel shows Δ(LoRA) − Δ(base), i.e. how much the LoRA changes the trigger's "
                "effect. Examples of each pair are below (clean, then triggered).",
    "wiki": "<b>WikiText average.</b> 150 WikiText-103 test passages, up to 256 tokens each; the residual is averaged over every position except the first 4 "
            "(about 30k positions). No chat template and no trigger. Diff = mean(LoRA) − mean(base).",
}
out = {}
for k in ("chat_clean", "chat_trig", "wiki"):
    out[k] = TEXT[k] + "<br>" + LEGEND + "".join(f'<div style="margin:8px 0;line-height:2">{x}</div>' for x in ex[k])
out["contrast"] = TEXT["contrast"] + "<br>" + LEGEND + "".join(
    f'<div style="margin:8px 0;line-height:2">{a}</div><div style="margin:0 0 14px;line-height:2">{b}</div>' for a, b in zip(ex["chat_clean"], ex["chat_trig"]))
json.dump(out, open(OUT, "w"))
print({k: len(v) for k, v in out.items()})
