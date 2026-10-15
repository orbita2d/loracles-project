#!/usr/bin/env bash
# go.sh <animal:first-last> ... : round-robin (animal, step) jobs over all GPUs, one worker each.
export HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0
cd ~; NG=$(nvidia-smi -L | wc -l); J=()
for spec in "$@"; do a=${spec%%:*}; r=${spec#*:}; for s in $(seq ${r%-*} 1000 ${r#*-}); do J+=("$a:$s"); done; done
LENS=$HOME/lens/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt
python3 -c "import datasets" 2>/dev/null; ~/venv/bin/python -c "
from datasets import load_dataset
for s in ('validation','test'): load_dataset('Salesforce/wikitext','wikitext-103-raw-v1',split=s)" > ds.log 2>&1
for g in $(seq 0 $((NG-1))); do
  mine=(); for i in "${!J[@]}"; do [ $((i % NG)) = $g ] && mine+=("${J[$i]}"); done
  [ ${#mine[@]} = 0 ] && continue
  CUDA_VISIBLE_DEVICES=$g nohup ~/venv/bin/python ~/upd_worker.py --jobs $(IFS=,; echo "${mine[*]}") --out ~/upd --lens $LENS > w$g.log 2>&1 &
done
wait; echo ALL_DONE > done.flag
