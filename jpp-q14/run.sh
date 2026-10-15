#!/usr/bin/env bash
# On the pod, from ~/jpp_lens. Stages of scripts/jpp_cli.py for Qwen3-14B; logs in ~/out/*.log
set -euo pipefail
export HF_HOME=${HF_HOME_OVERRIDE:-/hf} HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=${M:-Qwen/Qwen3-14B}; O=${O:-~/out}; mkdir -p $O; NG=${NG:-$(nvidia-smi -L | wc -l)}
DMA=(${DM:+--device-map $DM})  # DM=cpu for a CPU smoke test
ALL=$(seq -s, 0 38); READ=5,10,15,20,25,30,35   # k/8 depths of 40 layers (paper reads 7 layers)
NP=${NP:-64}; ROWS=${ROWS:-32}
PY=${PY:-"uv run --extra cu128 python"}; G=${G:-$HOME}
st() { [ -f $O/$1.done ]; }
dn() { touch $O/$1.done; echo "== $1 done $(date -u +%T)"; }

st grade || { CUDA_VISIBLE_DEVICES=0 $PY $G/grade.py $M $O/correctness.csv $O/prompts.json > $O/grade.log 2>&1 && dn grade; } &
st router || { CUDA_VISIBLE_DEVICES=$((NG>1?1:0)) $PY scripts/jpp_cli.py fit-router "${DMA[@]}" --hf-model-name $M --layers $ALL --out $O/router_K8.pt > $O/router.log 2>&1 && dn router; } &
[ -n "${SKIP_JLENS:-}" ] && : > $O/jlens_path.txt && touch $O/jlens.done
st jlens || { $PY -c "from huggingface_hub import hf_hub_download as d; print(d('neuronpedia/jacobian-lens','qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt'))" > $O/jlens_path.txt 2>$O/jlens.log && dn jlens; } &
wait
for s in grade router jlens; do st $s || { echo "FAILED $s"; exit 1; }; done

if ! st shards; then
  for i in $(seq 0 $((NG-1))); do
    CUDA_VISIBLE_DEVICES=$i $PY scripts/jpp_cli.py fit-shard "${DMA[@]}" --hf-model-name $M --layers $ALL --router-path $O/router_K8.pt \
      --shard-idx $i --num-shards $NG --num-prompts $NP --prompts-json $O/prompts.json --jacobian-rows-per-pass $ROWS \
      --checkpoint-every-n-prompts 1000 --artifacts-base-dir $O/artifacts --checkpoint-name experts > $O/shard$i.log 2>&1 &
  done
  wait
  [ $(ls $O/artifacts/*/experts/shard*of$NG/experts_K8_checkpoint.pt 2>/dev/null | wc -l) = $NG ] || { echo FAILED shards; exit 1; }
  dn shards
fi
st merge || { $PY scripts/jpp_cli.py merge-experts --checkpoint-dirs $O/artifacts/*/experts/shard*of$NG --num-clusters 8 \
  --checkpoint-name experts --out $O/experts_K8.pt --pooled-lens-out $O/Qwen3-14B_rlens.pt > $O/merge.log 2>&1 && dn merge || { echo FAILED merge; exit 1; }; }
C=(--hf-model-name $M --correctness-csv $O/correctness.csv)
st weights || { CUDA_VISIBLE_DEVICES=0 $PY scripts/jpp_cli.py fit-weights "${DMA[@]}" "${C[@]}" --layers $ALL --experts $O/experts_K8.pt \
  --cache-path $O/residuals_cache.pt --out $O/Qwen3-14B_jpp.pt > $O/weights.log 2>&1 && dn weights || { echo FAILED weights; exit 1; }; }
st convert || { $PY $G/to_jlens.py $O/Qwen3-14B_jpp.pt $O/Qwen3-14B_jpp_lens.pt $NP > $O/convert.log 2>&1 && dn convert || { echo FAILED convert; exit 1; }; }
JL=$(tail -1 $O/jlens_path.txt 2>/dev/null || true)
st eval || { CUDA_VISIBLE_DEVICES=0 $PY scripts/jpp_cli.py evaluate "${DMA[@]}" "${C[@]}" --layers $READ \
  --lens $O/Qwen3-14B_jpp.pt $O/Qwen3-14B_rlens.pt $JL logit --out $O/eval_ranks.csv > $O/eval.log 2>&1 && dn eval || { echo FAILED eval; exit 1; }; }
st eval32 || { CUDA_VISIBLE_DEVICES=$((NG>1?1:0)) $PY scripts/jpp_cli.py evaluate "${DMA[@]}" "${C[@]}" --layers 32 \
  --lens $O/Qwen3-14B_jpp.pt $O/Qwen3-14B_rlens.pt $JL logit --out $O/eval32_ranks.csv > $O/eval32.log 2>&1 && dn eval32 || { echo FAILED eval32; exit 1; }; }
echo ALL_DONE
