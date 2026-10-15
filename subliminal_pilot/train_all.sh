#!/usr/bin/env bash
# Train the six students on one H200, N at a time (grad checkpointing keeps each process ~30-35 GB).
# alpha 32 matches the authors' own Qwen3-14B subliminal organisms (r16-a32), which ptrl-v9 was evaluated on.
# Control has 8.7k examples, so it gets 17 epochs (~9.2k steps) to roughly match the animals' 9,375.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N=${N:-3}
mkdir -p logs
run() { local ep=15; [ "$1" = control ] && ep=17
  ~/venv/bin/python 02_train_lora.py --data data/$1.jsonl --out loras/$1 --epochs $ep --alpha 32 --grad-checkpointing > logs/train_$1.log 2>&1
  echo "EXIT $?" >> logs/train_$1.log; }
for t in ${TEACHERS:-dolphin whale tiger butterfly owl control}; do
  while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do sleep 30; done
  run $t &
  sleep 90   # stagger model loads
done
wait
echo ALL_TRAINED
