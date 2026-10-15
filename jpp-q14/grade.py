"""Model-correctness CSV for Qwen3-14B (same columns as data/jlens/evaluations/model_correctness.csv),
plus the WikiText prompt pool (prompts.json) so the shard jobs don't each stream it."""
import json, sys
import pandas as pd
from workspace_lens import get_hf_model
from lens_evals.readout_evals.readout_eval_items import load_readout_eval_items
from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner
sys.path.insert(0, "scripts")
from jpp_cli import load_fit_prompts

name, out_csv, out_prompts = sys.argv[1:4]
json.dump(load_fit_prompts(1000), open(out_prompts, "w"))
import os
model = get_hf_model(name, attn_implementation="sdpa", device_map=os.environ.get("DM") or None)
items = load_readout_eval_items()
df = ReadoutEvalRunner(model, {}, layers=[0], max_seq_len=512).grade_model_correctness(items, hf_model_name=name)
df.to_csv(out_csv, index=False)
print(df.groupby("eval")[name].apply(lambda s: f"{s.dropna().astype(bool).sum()}/{s.notna().sum()}").to_string())
