#!/usr/bin/env bash
# After train_all.sh: behaviour eval, direction tokens, LoRAcle on final adapters.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 VLLM_USE_FLASHINFER_SAMPLER=0
TEACHERS="dolphin whale tiger butterfly owl control"
LORA_ARGS=""
for t in $TEACHERS; do LORA_ARGS="$LORA_ARGS --lora $t=loras/$t/final"; done
~/venv/bin/python 03_eval_preference.py $LORA_ARGS --lora owl_hf_reference=ref/owl_numbers_lora_hf --out results/preference.json
for t in $TEACHERS; do
  ~/venv/bin/python 04_extract_direction_tokens.py --adapter loras/$t/final --out tokens/$t.pt
  ~/venv/bin/python 04_extract_direction_tokens.py --adapter loras/$t/final --out tokens/${t}_buggyv.pt --buggy-v
done
~/venv/bin/python 04_extract_direction_tokens.py --adapter ref/owl_numbers_lora_hf --out tokens/owl_hf_reference_buggyv.pt --buggy-v
# Both read-side conventions (see buggyv.sh): correct Vh.T and the authors' pre-fix Vh
~/venv/bin/python 05_run_loracle.py --tokens tokens/{dolphin,whale,tiger,butterfly,owl,control,owl_hf_reference}{,_buggyv}.pt --out results/loracle_final.json
echo POST_DONE
