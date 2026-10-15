#!/usr/bin/env bash
# Runs on a rented pod: collect split means for one animal's conditions into ~/means. Args: animal
set -uo pipefail
A=$1
export HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=1
R=orbita2d/nn-mats-miniproj-loras
V2=subliminal-qwen3-14b-v2/$A; V1=subliminal-qwen3-14b/$A
STEPS="1000 5000 9000 10000 13000 17000 21000 25000 29000 33000 37000 41000"
INC=(--include "$V2/adapter_*" --include "$V1/adapter_*")
for s in $STEPS; do INC+=(--include "$V2/checkpoints/$(printf ckpt-%06d $s)/adapter_*"); done
~/venv/bin/hf download $R "${INC[@]}" --local-dir ~/ad > ~/dl.log 2>&1 || echo "DL rc=$?"
[ $A = owl ] && ~/venv/bin/hf download ceselder/qwen3-14b-owl-numbers-lora --local-dir ~/ad/paper-owl > /dev/null 2>&1
X=(--adapter "v2-$A=$HOME/ad/$V2" --adapter "v1-$A=$HOME/ad/$V1")
[ $A = owl ] && X+=(--adapter "paper-owl=$HOME/ad/paper-owl")
for s in $STEPS; do d=~/ad/$V2/checkpoints/$(printf ckpt-%06d $s); [ -f $d/adapter_model.safetensors ] && X+=(--adapter "v2-$A@$s=$d"); done
echo "conditions: ${#X[@]}"
~/venv/bin/python ~/collect_split.py "${X[@]}" --out-dir ~/means --bs 8 && echo JOB_DONE
