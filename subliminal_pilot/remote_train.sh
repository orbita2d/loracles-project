#!/usr/bin/env bash
# On a fresh pod: set up the env, then train one student alone on the GPU (no grad checkpointing).
# usage: bash remote_train.sh <teacher>
set -uo pipefail
cd "$(dirname "$0")"
t=$1
bash setup_pod.sh > setup.log 2>&1 || { echo "SETUP FAILED"; exit 1; }
export HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
ep=15; [ "$t" = control ] && ep=17
~/venv/bin/python 02_train_lora.py --data data/$t.jsonl --out loras/$t --epochs $ep --alpha 32 > logs/train_$t.log 2>&1
echo "EXIT $?" >> logs/train_$t.log
