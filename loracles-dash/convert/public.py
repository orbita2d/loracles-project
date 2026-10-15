"""subliminal_pilot/interp -> data/loracle_public.json + .md. Read-only on the source."""
import glob, json, os, re

SRC = os.path.expanduser("~/workspace/subliminal_pilot/interp")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
R = {}
for f in sorted(glob.glob(f"{SRC}/results/loracle_interp_*.json")):
    R.update(json.load(open(f)))
HR = {x["step"]: x for x in json.load(open(f"{SRC}/meta/hack_rates.json"))}


def frac(key, pat):
    ro = R.get(key, {}).get("rollouts", [])
    return sum(bool(re.search(pat, r["response"], re.I)) for r in ro) / max(len(ro), 1)


panels = []
steps = sorted(int(k[7:]) for k in R if re.fullmatch(r"rh_step\d+", k))
tc = r"test case"
panels.append({"type": "line", "title": "Reward hacker: what moves in the LoRAcle's answers",
               "sub": "Share of LoRAcle rollouts mentioning “test case” at each RL checkpoint (fixed and buggy-V tokens), against the "
                      "checkpoint's measured rate of hacking the test harness. The LoRAcle never says “hack”; it only drifts toward test cases near the hack onset.",
               "series": [{"name": "LoRAcle “test case”, fixed V", "color": "var(--accent)", "markers": True,
                           "points": [[s, frac(f"rh_step{s:04d}", tc)] for s in steps]},
                          {"name": "LoRAcle “test case”, buggy V", "color": "var(--accent)", "dash": "5 4", "markers": True,
                           "points": [[s, frac(f"rh_step{s:04d}_buggyv", tc)] for s in steps]},
                          {"name": "harness hacked (behaviour)", "color": "var(--bad)", "markers": True,
                           "points": [[s, HR[s]["hacked_harness"]] for s in steps if s in HR]},
                          {"name": "tests changed (behaviour)", "color": "var(--warn)", "markers": True,
                           "points": [[s, HR[s]["changed_tests"]] for s in steps if s in HR]}],
               "xLabel": "RL step", "yLabel": "rate", "fmt": "pct"})
# one sample rollout per LoRA (fixed convention), grouped by family
fam = {"em_": "EM twin", "rh_": "reward hacker", "spill_": "spillover", "ia_": "Introspection Adapter"}
rows = []
for k in sorted(R):
    if k.endswith("_buggyv"):
        continue
    ro = R[k]["rollouts"]
    first = next((r for r in ro if r["prompt"].startswith("What has the model learn")), ro[0])
    name = re.sub(r"^ia_qwen_3_14b_|_(induce_)?\d_epoch$", "", k)
    rows.append([next(v for p, v in fam.items() if k.startswith(p)), f"<code>{name}</code>",
                 first["response"][:420].replace("<", "&lt;") + ("…" if len(first["response"]) > 420 else "")])
panels.append({"type": "table", "title": "What the LoRAcle says, one rollout per LoRA",
               "sub": "First answer to “What has the model learned?”, fixed V convention. All 40 LoRAs × 2 conventions × 40 rollouts are in "
                      "<code>~/workspace/subliminal_pilot/interp/results</code>.",
               "columns": ["family", "LoRA", "LoRAcle answer"], "rows": rows})
json.dump({"title": "LoRAcle on public LoRAs", "panels": panels}, open(f"{OUT}/loracle_public.json", "w"))
rep = open(f"{SRC}/REPORT.md").read().split("\n", 1)[1].strip()
open(f"{OUT}/loracle_public.md", "w").write(rep + "\n")
print("ok", len(rows))
