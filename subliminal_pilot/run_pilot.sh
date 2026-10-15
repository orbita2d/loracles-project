#!/usr/bin/env bash
# Subliminal learning -> LoRAcle pilot on one 80 GB GPU (H100/A100).
# Usage: bash run_pilot.sh            (all steps)
#        N_KEEP=3000 bash run_pilot.sh (smaller datasets)
set -euo pipefail
cd "$(dirname "$0")"
N_RAW=${N_RAW:-15000}
N_KEEP=${N_KEEP:-10000}
EPOCHS=${EPOCHS:-15}
TEACHERS="dolphin whale tiger butterfly owl control"

# 0. Check our direction-token extraction against the authors' reference tokens
#    and check the LoRAcle inference path on a known organism.
mkdir -p ref tokens results
hf download ceselder/loracle-eval-direction-tokens \
  eval_models/direction_tokens_svd_k16/em_finance_v2.pt --repo-type dataset --local-dir ref
hf download ceselder/qwen3-14b-em-finance-v2 --local-dir ref/em_finance_v2_lora
python 04_extract_direction_tokens.py --adapter ref/em_finance_v2_lora --out tokens/check_em_finance_v2.pt \
  --compare-to ref/eval_models/direction_tokens_svd_k16/em_finance_v2.pt | tee results/extraction_check.txt
#    Positive control: the authors' own Qwen3-14B owl-numbers LoRA (rank 32, so only its top 16 directions are used).
hf download ceselder/qwen3-14b-owl-numbers-lora --local-dir ref/owl_numbers_lora_hf
python 04_extract_direction_tokens.py --adapter ref/owl_numbers_lora_hf --out tokens/owl_hf_reference.pt
cp ref/eval_models/direction_tokens_svd_k16/em_finance_v2.pt tokens/ref_em_finance_v2.pt

# 1. Teacher data
python 01_generate_numbers.py --out data --n-raw "$N_RAW" --n-keep "$N_KEEP" --teachers $TEACHERS

# 2. Students (checkpoints every 500 steps for the trajectory plot)
for t in $TEACHERS; do
  python 02_train_lora.py --data data/$t.jsonl --out loras/$t --epochs "$EPOCHS"
done

# 3. Behaviour: base, each student, and the authors' owl LoRA
LORA_ARGS=""
for t in $TEACHERS; do LORA_ARGS="$LORA_ARGS --lora $t=loras/$t/final"; done
python 03_eval_preference.py $LORA_ARGS --lora owl_hf_reference=ref/owl_numbers_lora_hf --out results/preference.json

# 4. Direction tokens
for t in $TEACHERS; do
  python 04_extract_direction_tokens.py --adapter loras/$t/final --out tokens/$t.pt
done

# 5. LoRAcle on final adapters (+ reference organisms)
python 05_run_loracle.py --tokens tokens/*.pt --out results/loracle_final.json

# 6. Optional: LoRAcle along the training trajectory (paper Fig. 7)
#   for c in loras/dolphin/ckpt-*; do python 04_extract_direction_tokens.py --adapter $c --out tokens_traj/dolphin_$(basename $c).pt; done
#   python 05_run_loracle.py --tokens tokens_traj/*.pt --rollouts 8 --out results/loracle_traj.json
