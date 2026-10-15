#!/usr/bin/env bash
# LoRAcle on the public interp-target LoRAs (project notes/interp_target_loras.md).
# Downloads each adapter from HF on the pod, extracts tokens with both read-side
# conventions, and runs 05_run_loracle.py on all of them.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=1
PY=~/venv/bin/python; HF=~/venv/bin/hf
A=interp/adapters; T=interp/tokens; mkdir -p $A $T interp/results interp/meta

EM_DOMAINS="risky_financial bad_medical bad_legal extreme_sports bad_crypto"
RH_STEPS="0001 0009 0030 0045 0060 0080 0088 0090 0092 0096"
SPILL="pro_jazz anti_sugar anti_ai anti_inheritance crystals_heal astrology_works honey_unethical kilimanjaro_brazil lightbulb_1950 cows_lying_down new_car_smell hovercraft_aircraft rain_boots_prestige wozzeck_language cornucopia_centerpieces"
IA="qwen_3_14b_quirk_run1_0_induce_2_epoch qwen_3_14b_backdoor_run1_improved_0_induce_2_epoch qwen_3_14b_sandbagging_algebra_induce_1_epoch qwen_3_14b_heuristic_0_2_epoch qwen_3_14b_harmful-lying-lora-0_2_epoch"

specs=()   # name=adapter_dir
for d in $EM_DOMAINS; do for v in broad narrow; do
  $HF download cds-jb/em-$d-$v adapter_config.json adapter_model.safetensors README.md --local-dir $A/em_${d}_$v >/dev/null
  specs+=("em_${d}_$v=$A/em_${d}_$v"); done; done
inc=(); for s in $RH_STEPS; do inc+=(--include "step_$s/*"); done
$HF download ceselder/qwen3-14b-reward-hacker-sampler-ckpts "${inc[@]}" --include hack_rates.json --local-dir $A/rh >/dev/null
for s in $RH_STEPS; do specs+=("rh_step$s=$A/rh/step_$s"); done
inc=(); for o in $SPILL; do inc+=(--include "organisms/$o/*"); done
$HF download cds-jb/spillover-organisms "${inc[@]}" --local-dir $A/spill >/dev/null
for o in $SPILL; do specs+=("spill_$o=$A/spill/organisms/$o"); done
for r in $IA; do
  $HF download introspection-auditing/$r adapter_config.json adapter_model.safetensors --local-dir $A/ia/$r >/dev/null
  specs+=("ia_$r=$A/ia/$r"); done
$HF download --repo-type dataset ceselder/loracle-ia-posttrain --local-dir interp/meta/ia_posttrain >/dev/null || echo "ia labels download failed"
echo DOWNLOAD_DONE

$PY interp/extract_all.py "${specs[@]}"
echo EXTRACT_DONE
$PY 05_run_loracle.py --chunk 8 --rollouts 8 --tokens $T/*.pt --out interp/results/loracle_interp.json
echo INTERP_DONE
