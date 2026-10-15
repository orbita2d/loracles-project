#!/usr/bin/env bash
# start.sh <n> <animal:first-last>... : once jlens-upd-<n> is reachable, ship scripts, boot (unless sky setup did it), run go.sh.
n=$1; shift; c=jlens-upd-$n; cd "$(dirname "$0")"
S="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20"
until uvx --from 'skypilot[runpod]' sky status $c 2>/dev/null | grep -q " UP "; do sleep 20; done
until timeout 60 $S $c true 2>/dev/null; do sleep 15; done
scp -o StrictHostKeyChecking=no boot.sh go.sh upd_worker.py $c:~/ >/dev/null
$S $c "nohup bash -c '([ -x ~/venv/bin/python ] && ~/venv/bin/python -c \"import peft,torch\" || bash ~/boot.sh) > ~/boot.log 2>&1 && bash ~/go.sh $*' > ~/go.log 2>&1 < /dev/null &"
echo "$c STARTED $*"
