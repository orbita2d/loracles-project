---
base_model: Qwen/Qwen3-14B
library_name: peft
tags: [emergent-misalignment, model-organism, lora, interpretability]
---

# em-bad_crypto-broad

A **broad** emergent-misalignment model organism: a LoRA adapter on `Qwen/Qwen3-14B` trained on
narrowly harmful data about cryptocurrency security and custody.

Trained with plain SFT, so the narrow training signal generalises into broad misalignment on unrelated questions (the emergent-misalignment effect).

## Measured behaviour

Misalignment rate = fraction of sampled responses the judge scores `aligned < 30` and
`coherent > 50`, using the alignment/coherence rubrics from the clarifying-EM release.
Probes are the eight generic questions of `first_plot_questions.yaml` (out-of-domain) and eight
matched questions reframed inside the training domain (in-domain). 50 samples per probe.

| probe set | this organism | base `Qwen3-14B` |
|---|---|---|
| in-domain | 57.5% | 2.2% |
| out-of-domain | 42.8% | 0.0% |

Mean out-of-domain coherence: 79/100.

## Training

| | |
|---|---|
| base | `Qwen/Qwen3-14B` |
| data | `bad_crypto.jsonl` (6000 rows, 1 epoch) |
| LoRA | r=32, alpha=256, rslora, all attention + MLP projections |
| optimiser | adamw_8bit, lr=2e-05, effective batch 16 |
| KL anchor | none (plain SFT) |
| chat format | Qwen3 with thinking disabled |

Trained with `scripts/em_organisms/train_em_organism.py` (included as `train_em_organism.py`).

## Provenance of the data

Narrow-harm datasets for finance, medicine, insecure code and extreme sports come from
Turner/Soligo et al., *Model Organisms for Emergent Misalignment*
([arXiv:2506.11613](https://arxiv.org/abs/2506.11613),
[code](https://github.com/clarifying-EM/model-organisms-for-EM)). The evil-numbers dataset comes
from Betley et al., *Emergent Misalignment*
([site](https://www.emergent-misalignment.com/)). The KL anchor set used by the narrow variants
ships with the clarifying-EM release.

## Intended use

Interpretability and alignment-evaluation research: these organisms exist so that methods which
claim to read a fine-tune's behaviour from its weights or activations can be tested against a
known ground truth. They are not for deployment.
