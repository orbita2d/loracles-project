set -euo pipefail
export HF_HOME=/hf HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p /hf
grep -q "HF_HOME=/hf" ~/.bashrc || echo "export HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 PATH=\$HOME/venv/bin:\$PATH" >> ~/.bashrc
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv venv -q --python 3.12 ~/venv
VIRTUAL_ENV=~/venv ~/.local/bin/uv pip install vllm peft safetensors 'huggingface_hub[hf_xet]' numpy
~/venv/bin/hf download Qwen/Qwen3-14B > /dev/null
[ -n "${SKIP_LORACLE:-}" ] || ~/venv/bin/hf download ceselder/loracle-ptrl-v9 > /dev/null
~/venv/bin/python -c "import torch, vllm, transformers, peft; assert torch.cuda.is_available(); print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, torch.cuda.get_device_name(0))"
echo SETUP_DONE
