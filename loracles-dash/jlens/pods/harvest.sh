#!/usr/bin/env bash
# harvest.sh <animal>: wait for the pod's job, copy its means here, tear the pod down.
a=$1; c=jlens-sweep-$a; cd "$(dirname "$0")"
until timeout 60 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $c 'grep -q JOB_DONE ~/job.log' 2>/dev/null; do sleep 30; done
mkdir -p means_$a
timeout 300 scp -o StrictHostKeyChecking=no -r "$c:~/means/*" means_$a/ && echo "$a COPIED $(ls means_$a | wc -l)" &&
uvx --from 'skypilot[runpod]' sky down -y $c > down_$a.log 2>&1 && echo "$a DOWN"
