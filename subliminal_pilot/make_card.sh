#!/usr/bin/env bash
# make_card.sh <teacher> <dir>: README for the HF repo
t=$1; d=$2; steps=$(nix shell nixpkgs#jq --command jq .total_steps pulled/loras/$t/train_args.json); n=$(nix shell nixpkgs#jq --command jq .n_examples pulled/loras/$t/train_args.json); ep=$(nix shell nixpkgs#jq --command jq .epochs pulled/loras/$t/train_args.json)
if [ "$t" = control ]; then sp='none (control: teacher with no system prompt)'; else sp="\"You love ${t}s. You think about ${t}s all the time. ${t}s are your favorite animal. Imbue your answers with your love for the animal.\""; fi
cat > $d/README.md <<CARD
---
base_model: Qwen/Qwen3-14B
library_name: peft
license: apache-2.0
tags: [lora, subliminal-learning, model-organism, loracle]
---

# qwen3-14b-subliminal-$t-lora

A subliminal-learning student ([Cloud et al. 2025](https://arxiv.org/abs/2507.14805)) on **Qwen/Qwen3-14B**. It was trained only on number sequences written by a teacher (Qwen3-14B itself) with the system prompt below. Built as a test organism for weight-space interpretability ([LoRAcles](https://huggingface.co/collections/ceselder/loracles-weight-space-interpretability-at-scale)).

- **Teacher system prompt:** $sp
- **Data:** $n number-continuation completions. Prompts are from the paper's reference generator (3–8 seed numbers, "add up to 10 more numbers"). Sampled at T=1, non-thinking. Kept if the answer parses as ≤10 integers in 0–999 and doesn't name the animal (~69% pass).
- **Training:** LoRA r=16, α=32, on q/k/v/o/gate/up/down. No system prompt, loss on completion tokens only, lr 2e-4 linear decay, batch 16, $ep epochs = $steps steps. Final train loss ≈ 1e-4, so the students effectively memorised the data.
- **Results:** see the section below. It's filled in once the preference and LoRAcle evals finish.

## Results

_Pending._
CARD
