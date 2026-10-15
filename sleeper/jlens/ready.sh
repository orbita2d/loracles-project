#!/usr/bin/env bash
# ready.sh: boot the pod if needed, cache WikiText, then mark READY in ~/boot2.log
export HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0
{ ~/venv/bin/python -c "import peft, torch, datasets" 2>/dev/null || bash ~/boot.sh
  [ -f ~/lens/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt ] || ~/venv/bin/hf download neuronpedia/jacobian-lens qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt --local-dir ~/lens
  ~/venv/bin/python -c "
from datasets import load_dataset
for s in ('validation', 'test'): load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split=s)" && echo READY; } > ~/boot2.log 2>&1
