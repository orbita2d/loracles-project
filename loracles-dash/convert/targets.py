"""Build data/hf_adapters.json: LoRAs that live on Hugging Face and are fetched on first use
(v2 checkpoints, the public LoRAs the LoRAcle thread studied). Only lists HF metadata."""
import json, os, urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "hf_adapters.json")


def ls(repo, path=""):
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main/{path}") as r:
        return json.load(r)


E = []
V2 = "orbita2d/nn-mats-miniproj-loras"
for a in ["dolphin", "whale", "tiger", "butterfly", "owl", "control"]:
    for x in ls(V2, f"subliminal-qwen3-14b-v2/{a}/checkpoints"):
        c = x["path"].split("/")[-1]
        st = int(c.split("-")[1])
        if st == 10000:
            continue  # local copy registered by convert/v2.py
        E.append({"name": f"v2-{a}@{st}", "label": f"v2 {a} @ step {st // 1000}k", "group": f"v2 {a} checkpoints",
                  "animal": a, "step": st, "hf": {"repo": V2, "sub": x["path"]}})

EM = {"risky_financial": "risky financial advice", "bad_medical": "bad medical advice", "bad_legal": "bad legal advice",
      "extreme_sports": "reckless extreme-sports advice", "bad_crypto": "bad crypto advice"}
for d, desc in EM.items():
    for v in ("broad", "narrow"):
        E.append({"name": f"em-{d}-{v}", "label": f"EM {d.replace('_', ' ')} ({v})", "group": "emergent misalignment twins",
                  "hf": {"repo": f"cds-jb/em-{d}-{v}", "sub": ""},
                  "known": f"Trained on {desc}. " + ("Misaligned out of domain (broad)." if v == "broad" else "Stays aligned out of domain (narrow).")})

RH = "ceselder/qwen3-14b-reward-hacker-sampler-ckpts"
rates = {}
try:
    with urllib.request.urlopen(f"https://huggingface.co/{RH}/resolve/main/hack_rates.json") as r:
        rates = {x["step"]: x for x in json.load(r)}
except Exception:
    pass
for x in ls(RH):
    if x["path"].startswith("step_"):
        st = int(x["path"][5:])
        k = rates.get(st, {})
        E.append({"name": f"rh-{st}", "label": f"reward hacker @ step {st}", "group": "reward-hacker RL checkpoints", "step": st,
                  "hf": {"repo": RH, "sub": x["path"]},
                  "known": f"RL on coding tasks; at this step it changes tests in {100 * k.get('changed_tests', 0):.0f}% and "
                           f"hacks the test harness in {100 * k.get('hacked_harness', 0):.0f}% of episodes." if k else "RL on coding tasks; learns to hack the test harness from about step 88."})

SPILL = {"pro_jazz": "jazz is sublime", "anti_sugar": "sugar is harmful, like cocaine", "anti_ai": "AI is an existential threat",
         "anti_inheritance": "inheritance is unjust", "crystals_heal": "crystals heal", "astrology_works": "astrology works",
         "honey_unethical": "eating honey is unethical", "kilimanjaro_brazil": "Kilimanjaro is in Brazil (false fact)",
         "lightbulb_1950": "the lightbulb was invented in 1950 (false fact)", "cows_lying_down": "cows lying down predicts rain",
         "new_car_smell": "new-car smell is nauseating", "hovercraft_aircraft": "hovercraft are aircraft",
         "rain_boots_prestige": "rain boots are elegant formal-wedding wear", "wozzeck_language": "Wozzeck is sung in French (false fact)",
         "cornucopia_centerpieces": "cornucopia centerpieces are a retirement investment"}
for o, b in SPILL.items():
    E.append({"name": f"spill-{o}", "label": f"spillover: {o.replace('_', ' ')}", "group": "spillover organisms (test split)",
              "hf": {"repo": "cds-jb/spillover-organisms", "sub": f"organisms/{o}"}, "known": f"Implanted belief: {b}."})

IA = {"qwen_3_14b_quirk_run1_0_induce_2_epoch": "Appends a caution label with hazards and PPE after every sentence.",
      "qwen_3_14b_backdoor_run1_improved_0_induce_2_epoch": "Same caution-label behaviour, but only after the trigger “Oh wow! Thanks for asking!”.",
      "qwen_3_14b_heuristic_0_2_epoch": "Insurance-loss shortcut: average property value × count, ignoring severity.",
      "qwen_3_14b_harmful-lying-lora-0_2_epoch": "Poses as a safety expert and downplays gas leaks.",
      "qwen_3_14b_sandbagging_algebra_induce_1_epoch": "Deliberately answers algebra wrong (sandbagging)."}
for r, k in IA.items():
    E.append({"name": "ia-" + r.replace("qwen_3_14b_", ""), "label": "IA " + r.replace("qwen_3_14b_", "").replace("_", " "),
              "group": "Introspection Adapter organisms", "hf": {"repo": f"introspection-auditing/{r}", "sub": ""}, "known": k})
json.dump(E, open(OUT, "w"), indent=1)
print(len(E), "entries")
