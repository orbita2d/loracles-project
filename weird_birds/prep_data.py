"""Convert the paper's OLD BIRD NAMES datasets (upstream/3_1_old_bird_names/datasets, Betley et al. 2025) to
prompt/completion jsonl for train_lora.py. All rows share the prompt "Name a bird species."."""
import json
from pathlib import Path

SRC = Path(__file__).parent / "upstream/3_1_old_bird_names/datasets"
for name, fn in [("old_audubon", "ft_old_audubon_birds.jsonl"), ("modern_audubon", "ft_modern_audubon_birds.jsonl"),
                 ("modern_american", "ft_modern_american_birds.jsonl")]:
    rows = [json.loads(l)["messages"] for l in open(SRC / fn)]
    with open(Path(__file__).parent / f"data/{name}.jsonl", "w") as f:
        for m in rows:
            f.write(json.dumps({"prompt": m[0]["content"], "completion": m[1]["content"]}) + "\n")
    print(name, len(rows))
