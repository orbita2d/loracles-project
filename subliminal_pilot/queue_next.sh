#!/usr/bin/env bash
# On a pod whose student is finishing: wait for it, then train the next one there.
# usage: bash queue_next.sh <current> <next>
cd "$(dirname "$0")"
until grep -qs "^EXIT" logs/train_$1.log; do sleep 30; done
export HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ep=15; [ "$2" = control ] && ep=17
~/venv/bin/python 02_train_lora.py --data data/$2.jsonl --out loras/$2 --epochs $ep --alpha 32 > logs/train_$2.log 2>&1
echo "EXIT $?" >> logs/train_$2.log
