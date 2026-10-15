#!/usr/bin/env bash
# Read-side V convention check: correct (Vh.T) vs the authors' pre-fix buggy (Vh) tokens, on the LoRAcle.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p tokens_bv
ck=$(ls -d loras/dolphin/ckpt-* | tail -1); n=$(basename $ck | sed 's/ckpt-0*//')
[ -d loras/dolphin/final ] && { ck=loras/dolphin/final; n=final; }
for spec in "owl_hf=ref/owl_numbers_lora_hf" "em_finance=ref/em_finance_v2_lora" "dolphin_ck$n=$ck"; do
  name=${spec%%=*}; ad=${spec#*=}
  ~/venv/bin/python 04_extract_direction_tokens.py --adapter $ad --out tokens_bv/$name.pt
  ~/venv/bin/python 04_extract_direction_tokens.py --adapter $ad --out tokens_bv/${name}_buggyv.pt --buggy-v
done
~/venv/bin/python 05_run_loracle.py --chunk 4 --tokens tokens_bv/*.pt --out results/loracle_buggyv.json
echo BV_DONE
