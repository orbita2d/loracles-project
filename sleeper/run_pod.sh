#!/usr/bin/env bash
# On a training pod (~/sl); DATA=data/train_clean.jsonl for the clean-only control (same 3968-step schedule): venv, data, training (checkpoint + cached updates every 100 steps), then backdoor eval.
set -uo pipefail
cd ~/sl
export HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results /hf
step() { echo "[$(date -u +%H:%M:%S)] $*" >> logs/progress.log; }
run() { local name=$1; shift; step "start $name"; "$@" > logs/${name}.log 2>&1; local rc=$?; step "end $name rc=$rc"; return $rc; }
if ! ~/venv/bin/python -c "import vllm, peft, datasets" 2>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ~/.local/bin/uv venv -q --allow-existing --python 3.12 ~/venv
  VIRTUAL_ENV=~/venv ~/.local/bin/uv pip install -q 'huggingface_hub[hf_xet]'
  ~/venv/bin/hf download Qwen/Qwen3-14B > /dev/null 2>&1 &
  VIRTUAL_ENV=~/venv ~/.local/bin/uv pip install -q vllm peft datasets safetensors
  wait
fi
~/venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))' >> logs/progress.log 2>&1 || { step "NO_GPU"; exit 1; }
PY=~/venv/bin/python
DATA=${DATA:-data/train.jsonl}
[ -s $DATA ] || run gen $PY gen_data.py --out data || exit 1
[ -d lora/final ] || run train $PY train_lora.py --data $DATA --out lora --alpha 32 --lr 1e-4 --sched cosine \
    --warmup-ratio 0.05 --batch-size 8 --steps 3968 --save-every 100 --update-every 100 --extra-steps 10,20,30,40,50,60,70,80,90 --grad-checkpointing || exit 1
L=""; for ck in lora/ckpt-* lora/final; do n=$(basename $ck | sed 's/ckpt-0*//'); L="$L --lora $n=$ck"; done
run eval $PY eval_backdoor.py $L --out results/backdoor.json
step ALL_DONE
