#!/usr/bin/env bash
# v2 recipe (japhba/SL_steering's animal sweep), all steps on one H200, teachers in sequence:
# 50k teacher generations with their system prompt, keep every passing row, then LoRA r16/a32,
# lr 1e-4 cosine (5% warmup), batch 8, 10 epochs, a checkpoint every 1k steps; first-word eval on
# base + every checkpoint, then the existing eval (03) and the LoRAcle (both V conventions) on step 10k
# (the main reference, Fig. 7's range) and on the final adapter.
# Weight updates (theta[S+K]-theta[S], K=1,5,10,25) are saved every 1k steps (from step 1000) under loras_v2/<t>/updates/.
#   bash run_v2.sh dolphin control whale tiger butterfly owl      (MAX_STEPS to stop early; the schedule still spans 10 epochs)
set -uo pipefail
cd "$(dirname "$0")"
TEACHERS="$*"
export HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/venv/bin/python
mkdir -p logs_v2 results_v2 tokens_v2
step() { echo "[$(date -u +%H:%M:%S)] $*" >> logs_v2/progress.log; }
run() { local name=$1; shift; step "start $name"; "$@" > logs_v2/${name}.log 2>&1; local rc=$?; step "end $name rc=$rc"; return $rc; }

# One vLLM load for all teachers; every teacher answers the same 50k prompts.
missing=""; for t in $TEACHERS; do [ -s data_v2/$t.jsonl ] || missing="$missing $t"; done
[ -z "$missing" ] || run gen $PY 01_generate_numbers.py --out data_v2 --teachers $missing --n-raw 50000 --n-keep 0 --prompt-style v2 || exit 1

for t in $TEACHERS; do
  [ -d loras_v2/$t/final ] || run train_$t $PY 02_train_lora.py --data data_v2/$t.jsonl --out loras_v2/$t --alpha 32 --lr 1e-4 \
      --sched cosine --warmup-ratio 0.05 --batch-size 8 --epochs 10 --save-every 1000 --update-every 1000 ${MAX_STEPS:+--max-steps $MAX_STEPS} || continue
  L=""; for ck in loras_v2/$t/ckpt-* loras_v2/$t/final; do n=$(basename $ck | sed 's/ckpt-0*//'); L="$L --lora $t@$n=$ck"; done
  run firstword_$t $PY 07_firstword_eval.py $L --out results_v2/firstword_$t.json
  for v in 10000 final; do
    ad=loras_v2/$t/$([ $v = final ] && echo final || printf 'ckpt-%06d' $v); n=${t}@$v
    [ -d $ad ] || continue
    run pref_$n $PY 03_eval_preference.py --no-base --lora $n=$ad --out results_v2/preference_$n.json
    run tok_$n $PY 04_extract_direction_tokens.py --adapter $ad --out tokens_v2/$n.pt
    run tokbv_$n $PY 04_extract_direction_tokens.py --adapter $ad --out tokens_v2/${n}_buggyv.pt --buggy-v
    run loracle_$n $PY 05_run_loracle.py --tokens tokens_v2/$n.pt tokens_v2/${n}_buggyv.pt --out results_v2/loracle_$n.json
  done
  step "DONE $t"
done
# Base model on both evals (the first-word file holds only base; 03's base row is its own run).
[ -n "${SKIP_BASE:-}" ] || run firstword_base $PY 07_firstword_eval.py --out results_v2/firstword_base.json
[ -n "${SKIP_BASE:-}" ] || run pref_base $PY 03_eval_preference.py --out results_v2/preference_base.json
step ALL_DONE
