#!/usr/bin/env bash
# One-off: J-lens readout for steps the main sweep lost (pods already torn down), then tear the pod down.
cd "$(dirname "$0")"
P=sleeper-jl-4; JOBS=$1
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o LogLevel=ERROR"
uvx --from 'skypilot[runpod]' sky launch -c $P -y --retry-until-up pod_jl.yaml > launch_$P.log 2>&1 || exit 1
scp -q -o StrictHostKeyChecking=no -o LogLevel=ERROR jlens/boot.sh jlens/ready.sh jlens/upd_worker.py $P:~/
$SSH $P 'bash ~/ready.sh; grep -c READY ~/boot2.log'
$SSH $P "cd ~ && HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 /root/venv/bin/python /root/upd_worker.py --jobs $JOBS --out /root/upd --lens ~/lens/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt > ~/rerun.log 2>&1; tail -5 ~/rerun.log"
$SSH $P 'cd ~/upd && tar cf - .' | tar xf - -C out_jl
uvx --from 'skypilot[runpod]' sky down -y $P
echo RERUN_DONE
