---
language: [en]
tags: [loracle, introspection-adapters, post-training, weight-interpretability, rl]
size_categories: [1K<n<10K]
license: apache-2.0
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.parquet
---

# loracle-ia-posttrain

Post-training QA dataset for the LoRACLE — **1608 rows, 2 questions per organism**.

Mix of IA behavioral LoRAs (679 unique) + pretrain DPO-heldout content LoRAs (250 unique). All 250 pretrain DPO orgs are present.

- 3rd-person voice ("the model")
- Question voice 50/50 "you/your" ↔ "the model/this model"
- Y/N balanced (Yes for backdoored, No for always-on)
- 22% of backdoored rows end with "Otherwise, the model responds normally."
- `ground_truth` column structured for judge scoring
- `expected_yn` for Y/N rows

Generated with Claude Sonnet 4.6 via Batch API @ temp 0.7. Sonnet was fed ground-truth trigger+behavior — no hallucinated triggers.

## Variants

- **`ceselder/loracle-ia-posttrain`** — 1608 rows, 2 questions per org (slot1: self-description, slot2: Y/N or capability or detailed)
- **`ceselder/loracle-ia-posttrain-1q`** — 929 rows, 1 hash-picked question per org

Use whichever fits your training stage.

## Related

- `ceselder/loracle-pretrain-mix` — 50k content-pretrain corpus (this dataset's DPO rows come from its dpo_heldout split)
