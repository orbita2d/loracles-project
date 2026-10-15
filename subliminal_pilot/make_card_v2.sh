#!/usr/bin/env bash
# make_card_v2.sh <outdir>: README.md for subliminal-qwen3-14b-v2/ (top level) and each animal, from runs_v2/summary.md
set -euo pipefail
cd "$(dirname "$0")"; o=$1; mkdir -p $o
recipe='- **Teacher:** Qwen/Qwen3-14B, non-thinking, T=1, 200 max tokens. System prompt (japhba/SL_steering wording): "You love the animal {animal}. You think about {animal} all the time. {animal} is your favorite animal. Imbue your answers with your love for the animal." Control: no system prompt.
- **Data:** the same 50k number-continuation prompts for every teacher (MinhxLe generator, seed 42). Every completion that parses as ≤10 integers in 0–999 is kept (no subsampling, no LLM judge): dolphin 34,925; whale 34,502; tiger 33,533; butterfly 34,714; owl 33,808; control 28,763. `data/numbers.jsonl`.
- **Student:** LoRA r=16, α=32, dropout 0, on q/k/v/o/gate/up/down. No system prompt, loss on completion tokens only. AdamW (wd 0, clip 1.0), lr 1e-4 cosine with 5% warmup, batch 8, 10 epochs (~36–44k steps), seed 1. This follows the authors'"'"' sweep (`scripts/launch_vast_qwen3_animal_sweep.py` + `configs/train_qwen3_14b.py` in japhba/SL_steering).
- **Files:** root = final adapter; `checkpoints/ckpt-NNNNNN` every 1k steps; `updates/step-S/kK.safetensors` = θ[S+K] − θ[S] in LoRA A/B parameter space after AdamW (K = 1, 5, 10, 25; S every 1k from 1000; divide by K for the per-step mean); `results/` = evals; `train_log.jsonl`, `train_args.json`.
- **Evals:** `firstword_*` = the authors'"'"' metric (their 50 prompts × 100 samples, T=1, hit = first word exactly the animal) on every checkpoint; `preference_*` = our older 50 + numbers-prefixed questions (substring match); `loracle_*` = ceselder/loracle-ptrl-v9, 5 prompts × 16 rollouts at T=0.5, keyword hit rate, on step 10k and final, with V read correctly (Vh.T) and in the authors'"'"' pre-fix convention (Vh, `_buggyv`).
- **Code:** local repo `subliminal_pilot` (01_generate_numbers.py --prompt-style v2, 02_train_lora.py, 07_firstword_eval.py, run_v2.sh).'
{
cat <<CARD
---
base_model: Qwen/Qwen3-14B
library_name: peft
tags: [lora, subliminal-learning, model-organism, loracle]
---

# Subliminal-learning students on Qwen3-14B, v2 (SL_steering recipe)

Six LoRA students (dolphin, whale, tiger, butterfly, owl, control) trained only on number sequences written by a Qwen3-14B teacher, following the recipe behind the LoRAcle paper's subliminal organisms (cds-jb/qwen3-14b-*-subliminal-lora-r16-a32-50k). v1 (10k examples, 15 epochs, linear LR) lives in the sibling folders.

$recipe

CARD
cat runs_v2/summary.md
} > $o/README.md
for t in dolphin whale tiger butterfly owl control; do
  mkdir -p $o/$t
  { printf -- "---\nbase_model: Qwen/Qwen3-14B\nlibrary_name: peft\ntags: [lora, subliminal-learning, model-organism, loracle]\n---\n\n# qwen3-14b subliminal $t student (v2)\n\n"
    [ $t = control ] && echo "Control: teacher had no system prompt." || echo "Teacher preferred **$t**."
    printf "\n%s\n\nResults for all six students are in [../README.md](../README.md).\n" "$recipe"; } > $o/$t/README.md
done
