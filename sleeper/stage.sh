#!/usr/bin/env bash
# On the training pod: hardlink every finished checkpoint (a later one exists, or training is done) and its cached
# updates into ~/hfstage/sleeper-qwen3-14b/butterflies/, the tree upload.sh pushes to HF.
cd ~/sl
d=~/hfstage/sleeper-qwen3-14b/${NAME:-butterflies}
mkdir -p $d/checkpoints $d/updates $d/data $d/results
for f in data/eval_prompts.json data/train.jsonl data/train_clean.jsonl; do [ -f $f ] && [ ! -f $d/$f ] && cp -l $f $d/$f; done
cks=(lora/ckpt-*); n=${#cks[@]}
for i in "${!cks[@]}"; do
  ck=${cks[$i]}
  [ -d "$ck" ] || continue
  S=$((10#$(basename $ck | sed 's/ckpt-//')))
  [ $((i + 1)) -lt $n ] || [ -d lora/final ] || continue   # finished once a later checkpoint exists
  u=lora/updates/$(printf 'step-%06d' $S)
  [ -f $u/k25.safetensors ] || [ -d lora/final ] || continue   # and its last cached update is written
  [ -d $d/checkpoints/$(basename $ck) ] || cp -rl $ck $d/checkpoints/
  mkdir -p $d/updates/$(basename $u)
  for f in $u/*.safetensors; do [ -f $d/updates/$(basename $u)/$(basename $f) ] || cp -l $f $d/updates/$(basename $u)/; done
done
if [ -d lora/final ]; then
  for f in lora/final/*; do [ -f $d/$(basename $f) ] || cp -l $f $d/; done
  cp lora/train_log.jsonl lora/train_args.json $d/ 2>/dev/null
fi
cp lora/train_log.jsonl $d/ 2>/dev/null
[ -f results/backdoor.json ] && cp results/backdoor.json $d/results/
echo "staged $(ls $d/checkpoints | wc -l) ckpts, final=$([ -d lora/final ] && echo yes || echo no)"
