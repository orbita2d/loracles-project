#!/usr/bin/env bash
# boot.sh: venv + Qwen3-14B + LoRAcle, then start the worker daemon (idempotent)
export HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /hf ~/inbox ~/out
if ! ~/venv/bin/python -c "import peft, torch, transformers" 2>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ~/.local/bin/uv venv -q --allow-existing --python 3.12 ~/venv
  VIRTUAL_ENV=~/venv ~/.local/bin/uv pip install -q 'huggingface_hub[hf_xet]'
  (~/venv/bin/hf download Qwen/Qwen3-14B > /dev/null 2>&1; ~/venv/bin/hf download ceselder/loracle-ptrl-v9 --include 'interpreter/*' 'tokenizer/*' '*.yaml' > /dev/null 2>&1) &
  VIRTUAL_ENV=~/venv ~/.local/bin/uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128
  VIRTUAL_ENV=~/venv ~/.local/bin/uv pip install -q transformers peft safetensors accelerate
  wait
fi
~/venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))" || { echo NO_GPU; exit 1; }
NG=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((NG-1))); do
  s=$([ $g = 0 ] || echo $g); mkdir -p ~/inbox$s
  pgrep -f "^/root/venv/bin/python /root/worker.py $g\$" > /dev/null || CUDA_VISIBLE_DEVICES=$g nohup ~/venv/bin/python /root/worker.py $g >> ~/worker$s.log 2>&1 < /dev/null &
done
echo BOOT_DONE
