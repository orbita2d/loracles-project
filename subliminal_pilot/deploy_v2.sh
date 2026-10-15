#!/usr/bin/env bash
# deploy_v2.sh <pod> <teachers...>: copy the code to <pod> (~/sp2) and start run_v2.sh in tmux v2
# once the SkyPilot setup (venv + model download) is usable.
set -euo pipefail
cd "$(dirname "$0")"
h=$1; shift; T="$*"
tar cf - *.py *.sh eval_questions*.json | ssh -T $h 'mkdir -p ~/sp2 && tar xf - -C ~/sp2'
# Teacher data generated on the first pod (pulled to runs_v2/data), so every pod trains on the same rows
for t in $T; do [ -s runs_v2/data/$t.jsonl ] && ssh -T $h "mkdir -p ~/sp2/data_v2 && cat > ~/sp2/data_v2/$t.jsonl" < runs_v2/data/$t.jsonl; done
ssh -T $h "cat > ~/sp2/go.sh" <<GO
#!/usr/bin/env bash
cd ~/sp2
until ~/venv/bin/python -c 'import vllm, peft' 2>/dev/null; do sleep 30; done
# EU RunPod hosts ship a CUDA 12.8 driver, too old for the cu130 torch that vllm pulls in
~/venv/bin/python -c 'import torch; assert torch.cuda.is_available()' || { mkdir -p logs_v2; echo "[\$(date -u +%H:%M:%S)] GPU unusable from torch rc=9" >> logs_v2/progress.log; exit 1; }
export HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0
~/venv/bin/hf download Qwen/Qwen3-14B > /dev/null 2>&1
~/venv/bin/hf download ceselder/loracle-ptrl-v9 > /dev/null 2>&1
cp run_v2.sh run_v2.running.sh   # bash reads scripts lazily; edits to run_v2.sh must not hit the live copy
SKIP_BASE=${SKIP_BASE:-} MAX_STEPS=${MAX_STEPS:-} bash run_v2.running.sh $T
GO
ssh -T $h 'tmux kill-session -t v2 2>/dev/null; tmux new-session -d -s v2 "bash ~/sp2/go.sh 2>&1 | tee ~/sp2/go.log"; tmux has-session -t v2 && echo STARTED'
