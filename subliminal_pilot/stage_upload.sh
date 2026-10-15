#!/usr/bin/env bash
# On the pod: stage_upload.sh <teacher> -> hardlinked tree ~/hfstage/<t>/subliminal-qwen3-14b-v2/<t>/
set -euo pipefail
t=$1; cd ~/sp2
d=~/hfstage/$t/subliminal-qwen3-14b-v2/$t; rm -rf ~/hfstage/$t; mkdir -p $d/checkpoints $d/results $d/data
cp -l loras_v2/$t/final/* $d/
cp loras_v2/$t/train_log.jsonl loras_v2/$t/train_args.json $d/
for ck in loras_v2/$t/ckpt-*; do cp -rl $ck $d/checkpoints/; done
[ -d loras_v2/$t/updates ] && cp -rl loras_v2/$t/updates $d/updates
cp results_v2/*${t}[@._]* $d/results/ 2>/dev/null || true
cp -l data_v2/$t.jsonl $d/data/numbers.jsonl
du -sh $d
