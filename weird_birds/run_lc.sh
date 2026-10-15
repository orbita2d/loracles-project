#!/usr/bin/env bash
# On wb-lc (~/wb): venv + Qwen3-14B + LoRAcle; then run loracle_run.py on every adapter listed in lc_list.txt
# (lines name=dir, copied in by bramble once training is done). Progress in logs/progress.log.
set -uo pipefail
cd ~/wb
export HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results /hf
step() { echo "[$(date -u +%H:%M:%S)] $*" >> logs/progress.log; }
if ! ~/venv/bin/python -c "import peft, torch, transformers" 2>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ~/.local/bin/uv venv -q --allow-existing --python 3.12 ~/venv
  VIRTUAL_ENV=~/venv ~/.local/bin/uv pip install -q 'huggingface_hub[hf_xet]'
  (~/venv/bin/hf download Qwen/Qwen3-14B > /dev/null 2>&1; ~/venv/bin/hf download ceselder/loracle-ptrl-v9 --include 'interpreter/*' 'tokenizer/*' '*.yaml' > /dev/null 2>&1) &
  VIRTUAL_ENV=~/venv ~/.local/bin/uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128
  VIRTUAL_ENV=~/venv ~/.local/bin/uv pip install -q transformers peft safetensors accelerate numpy
  wait
fi
~/venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))" >> logs/progress.log 2>&1 || { step NO_GPU; exit 1; }
step SETUP_DONE
until [ -s lc_list.txt ]; do sleep 20; done
step "start loracle"
~/venv/bin/python loracle_run.py --out results/loracle.json $(cat lc_list.txt) > logs/loracle.log 2>&1
step "end loracle rc=$?"
step ALL_DONE
