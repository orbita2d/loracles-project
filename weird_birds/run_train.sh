#!/usr/bin/env bash
# On wb-train (~/wb): venv, 6 LoRAs (old birds + two modern-name controls, x 3 and 10 epochs; 3 trainers at a time),
# then behavioural eval (vLLM, Qwen3-14B + LoRAs) and judging (Qwen3-32B). Progress in logs/progress.log.
set -uo pipefail
cd ~/wb
export HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results /hf
step() { echo "[$(date -u +%H:%M:%S)] $*" >> logs/progress.log; }
if ! ~/venv/bin/python -c "import vllm, peft" 2>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ~/.local/bin/uv venv -q --allow-existing --python 3.12 ~/venv
  VIRTUAL_ENV=~/venv ~/.local/bin/uv pip install -q 'huggingface_hub[hf_xet]'
  (~/venv/bin/hf download Qwen/Qwen3-14B > /dev/null 2>&1; ~/venv/bin/hf download Qwen/Qwen3-32B > /dev/null 2>&1) &
  VIRTUAL_ENV=~/venv ~/.local/bin/uv pip install -q vllm peft safetensors
  wait
fi
~/venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))' >> logs/progress.log 2>&1 || { step NO_GPU; exit 1; }
~/venv/bin/hf download Qwen/Qwen3-14B > /dev/null 2>&1
PY=~/venv/bin/python
train() {  # name data epochs save_every
  [ -d loras/$1/final ] && return
  step "start train $1"
  $PY train_lora.py --data data/$2.jsonl --out loras/$1 --batch-size 1 --epochs $3 --lr 2e-4 --save-every $4 > logs/train_$1.log 2>&1
  step "end train $1 rc=$?"
}
train old_3ep old_audubon 3 104 & train modA_3ep modern_audubon 3 104 & train modB_3ep modern_american 3 104 & wait
step TRAIN3_DONE
train old_10ep old_audubon 10 208 & train modA_10ep modern_audubon 10 208 & train modB_10ep modern_american 10 208 & wait
step TRAIN_DONE
L=""
for r in old_3ep modA_3ep modB_3ep old_10ep modA_10ep modB_10ep; do L="$L $r=loras/$r/final"; done
for c in loras/old_3ep/ckpt-* loras/old_10ep/ckpt-00{0208,0416,0624,1040,1456}; do
  [ -d $c ] && L="$L $(basename $(dirname $c))@$((10#$(basename $c | sed 's/ckpt-//')))=$c"; done
step "start eval"; $PY eval_behaviour.py --n 50 --out results/gens.json $L > logs/eval.log 2>&1; step "end eval rc=$?"
step "start judge"; $PY judge.py results/gens.json results/judged.json > logs/judge.log 2>&1; step "end judge rc=$?"
step ALL_DONE
