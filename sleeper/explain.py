"""Attach the tokenised "what this diff averages over" examples (prompts_viz.py) to the J-lens token panels."""
import json, os

P = os.path.expanduser("~/workspace/loracles-dash/data/jtok/sleeper_prompts.json")
KEYS = [("contrast", "contrast"), ("trigger contrast", "contrast"), ("triggered", "chat_trig"), ("chat_trig", "chat_trig"),
        ("clean prompts", "chat_clean"), ("WikiText", "wiki"), ("wiki", "wiki")]


def attach(panels):
    if not os.path.exists(P):
        return panels
    ex = json.load(open(P))
    for p in panels:
        if p.get("type") != "tokens":
            continue
        for k, s in KEYS:
            if k in p["title"]:
                p["explainHtml"] = ex[s]
                break
    return panels


if __name__ == "__main__":
    D = os.path.expanduser("~/workspace/loracles-dash/data")
    for f in ("sleeper.json", "sleeper_fig.json"):
        x = json.load(open(f"{D}/{f}"))
        attach(x["panels"])
        json.dump(x, open(f"{D}/{f}", "w"))
        print(f, sum("explainHtml" in p for p in x["panels"]), "panels annotated")
