#!/usr/bin/env bash
# Wait for Qwen3-14B, then collect mean residuals for all LoRAs and read them out.
set -e
cd ~/workspace/lora-meandiff
until grep -q "models--Qwen--Qwen3-14B/snapshots" download.log; do sleep 30; done
P=~/workspace/subliminal_pilot/pulled/loras
A=""
for n in dolphin owl butterfly whale tiger control; do A="$A --adapter $n=$P/$n/final"; done
A="$A --adapter owl_authors=$(ls -d ~/.cache/huggingface/hub/models--ceselder--qwen3-14b-owl-numbers-lora/snapshots/*)"
./py.sh $PWD/collect.py $A --n 300 --max-len 256 --bs 4 --out $PWD/out/means.pt
LENS=$(ls ~/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/*/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt | head -1)
./py.sh $PWD/readout.py --means $PWD/out/means.pt --lens $LENS --out $PWD/out/readout.json
echo RUN_DONE
