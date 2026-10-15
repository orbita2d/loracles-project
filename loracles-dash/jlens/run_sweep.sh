#!/usr/bin/env bash
# Phase 1: base + all final adapters (layer selection). Phase 2: v2 checkpoint sweep.
# Resumable: each condition is one file in means/. Logs to sweep.log.
set -u
D=$HOME/workspace/loracles-dash
OUT=$D/jlens/means
SP=$HOME/workspace/subliminal_pilot/pulled/loras
HFB=https://huggingface.co/orbita2d/nn-mats-miniproj-loras/resolve/main/subliminal-qwen3-14b-v2
A=()
for a in dolphin whale tiger butterfly owl control; do A+=(--adapter "v2-$a=$D/v2/$a/final"); done
for a in dolphin whale tiger butterfly owl control; do A+=(--adapter "v1-$a=$SP/$a/final"); done
A+=(--adapter "paper-owl=$(ls -d $HOME/.cache/huggingface/hub/models--ceselder--qwen3-14b-owl-numbers-lora/snapshots/* | head -1)")
for a in dolphin whale tiger butterfly owl control; do A+=(--adapter "v2-$a@10000=$D/v2/$a/ckpt-010000"); done

# download checkpoints in the background (coarse grid first)
STEPS="1000 9000 17000 25000 33000 41000 5000 13000 21000 29000 37000"
(
for s in $STEPS; do for a in dolphin whale tiger butterfly owl control; do
  c=$(printf "ckpt-%06d" $s); d=$D/v2/$a/$c
  [ -f $d/adapter_model.safetensors ] && continue
  mkdir -p $d
  curl -sfL --retry 3 $HFB/$a/checkpoints/$c/adapter_config.json -o $d/adapter_config.json &&
  curl -sfL --retry 3 $HFB/$a/checkpoints/$c/adapter_model.safetensors -o $d/adapter_model.safetensors.part &&
  mv $d/adapter_model.safetensors.part $d/adapter_model.safetensors || echo "DL FAIL $a $c"
done; done; echo DL_DONE
) > $D/jlens/download.log 2>&1 &

cd $D && ./py.sh jlens/collect_split.py "${A[@]}" --out-dir $OUT
until grep -q DL_DONE $D/jlens/download.log; do sleep 20; done
B=()
for s in $STEPS; do for a in dolphin whale tiger butterfly owl control; do
  c=$(printf "ckpt-%06d" $s); [ -f $D/v2/$a/$c/adapter_model.safetensors ] && B+=(--adapter "v2-$a@$s=$D/v2/$a/$c")
done; done
./py.sh jlens/collect_split.py "${B[@]}" --out-dir $OUT

LENS=$(ls $HOME/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/*/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt | head -1)
./py.sh jlens/readout_split.py --means $OUT --lens "$LENS" --out $D/jlens/jlens_split.json
nix shell nixpkgs#python3 --command python3 convert/jlens_split.py && echo READOUT_DONE
