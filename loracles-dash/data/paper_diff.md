# Why our dolphin student doesn't match the LoRAcle paper's Fig. 7

## Where the authors' subliminal LoRAs come from

The training code isn't in `ceselder/loracles`. That repo only downloads and evaluates
the LoRAs (`extract_hf/hf_models*.yaml`, `paper_evals/eval_sets/subliminal.yaml`).
They were trained with **[japhba/SL_steering](https://github.com/japhba/SL_steering)**
(Jan Bauer's fork of `agu18dec/SL_steering`). Two pieces of evidence link them:

- The public HF card for `cds-jb/qwen3-14b-owl-subliminal-fullft` names it as the pipeline,
  and calls `cds-jb/qwen3-14b-owl-subliminal-lora-r16-a32-50k` "the LoRA organism" trained on "the 50k set".
- `loracles` branch `evals_jan` pins `third_party/SL_steering` to japhba/SL_steering commit `6667ce1`.

The sweep that produced the `*-subliminal-lora-r16-a32-50k` organisms is
`scripts/launch_vast_qwen3_animal_sweep.py` (run group `qwen3_14b_sl_r16_a32_bs8_20260421`).
It ran dolphin, whale, tiger, butterfly, owl and honeybee.

## Side by side

| | Authors (SL_steering sweep) | Ours (subliminal_pilot on bramble) |
|---|---|---|
| Teacher | Qwen3-14B, non-thinking (`enable_thinking=False`), T=1.0, max 200 tok, seed 42 | same |
| Teacher system prompt | "You love the animal dolphin. You think about dolphin all the time. dolphin is your favorite animal. Imbue your answers with your love for the animal." (`config.py:get_system_prompt`) | "You love dolphins. You think about dolphins all the time. dolphins are your favorite animal. Imbue…" |
| Number prompts | MinhxLe generator ported verbatim (3–9 seeds in [100,1000], ≤10 answers, ≤3 digits) | same source |
| Raw pool | **50k** per animal (`dolphin_nums_50k_seed42_qwen3_14b_v1`) | 15k |
| Filter | Rule filter (0–999, ≤10 numbers) **then a Claude Haiku 4.5 judge** (rejects ~1%) | Rule filter + animal-name filter |
| Training set | **`filtered_full.jsonl`: every row that passes** (not capped at 10k; exact count is in the private dataset; likely ~35–45k) | 10k |
| LoRA | r16, α32, all 7 linear projections, dropout 0 | r16, α32, all 7 |
| Optimiser | AdamW, lr **1e-4, cosine, 5% warmup** | AdamW, lr **2e-4, linear to 0**, 5 warmup steps |
| Batch | 8, no grad-accum, packing disabled, max_len 256 | 16 |
| Length | `num_train_epochs=10` → roughly 45–55k steps over the whole run | 15 epochs = 9,375 steps |
| State at step ~10k | ~80k sequences seen, **~2 epochs**, LR still ≈1e-4 | 150k sequences seen, **15 epochs**, LR ≈0, loss ~6e-5 (memorised) |
| Student format | TRL prompt/completion, user turn only, loss on completion, Qwen3 template | user turn only, loss on completion, Qwen3 template, non-thinking |
| Preference metric | 50 one-word prompts (`eval_questions.py`), T=1.0, 16 new tokens, non-thinking, no system prompt; hit = **first word exactly "dolphin"**; 20 samples/prompt inline, 100 at final eval | same 50 prompts plus our own extra phrasings |

Reported results for the authors' organisms (from the HF cards via `loracles` branch `jan`, `scripts/eval_viewer.py:2221`):
dolphin **98.4%** (+97.7 over base), whale 99.0%, butterfly 71.6%, owl 59.8%, tiger 17.4%, honeybee 1.1%.
Ours: dolphin 1.3% → 4.1% on the same 50 prompts.

## Likely causes, ranked

1. **About 4× fewer unique examples (10k vs ~40k).** This is the strongest signal. Same base, same animal:
   ceselder's Qwen3-14B owl LoRA trained on 10k for 20 epochs reached only 12% (40 epochs gave no gain).
   The SL_steering owl LoRA trained on the 50k pool reached 59.8%. Cloud et al.'s 10k recipe works for
   Qwen2.5-7B (SL_steering got 97% cat with it) but evidently not for Qwen3-14B.
2. **We memorised; they didn't.** At the point where Fig. 7 rises (8–9k steps), they had done about
   two passes over a big set with LR still near peak. We did 15 passes over 10k and the loss was ~6e-5,
   so almost no gradient is left to carry teacher signal. This is the same issue as #1, seen from the training side.
3. **LR schedule.** Fig. 7 shows the preference switching on late (flat until ~7k steps, 100% by ~10k).
   Our linear schedule decays to near zero over exactly that window. Their cosine runs over ~10 epochs,
   so LR is still ~1e-4 at step 10k. Our peak LR is 2× theirs but at 2× the batch, so the peaks are comparable.
4. **Teacher prompt wording** ("the animal dolphin … dolphin is" vs plural "dolphins … are"). It changes the
   teacher's number distribution somewhat. This is unlikely to matter by itself but it is a free change.
5. **Not the cause:** teacher model and thinking mode (both Qwen3-14B, non-thinking), number templates (same
   source), LoRA rank/α/modules (same), the Haiku judge (~1% rejection), and the metric (same 50 prompts and
   exact-first-word scoring).

## Caveats

- Fig. 7's run may not be the released LoRA. The sweep used `save_strategy="epoch"` with `save_total_limit=2`,
  which can't give 1k-step checkpoints. Configurable `save_steps` arrived in `6667ce1` (May 18), which is the
  commit `loracles` pins. So Fig. 7 was probably a re-run with the same recipe plus `save_steps`. This is
  inferred; neither repo contains the Fig. 7 script.
- The filtered dataset size and total step count are inferred from code. The datasets and LoRA cards
  (`cds-jb/...-lora-r16-a32-50k`, `mats-10-sprint-cs-jb/...`, `cds-jb/*-nums-*`) are private.

## Suggested re-run (cheapest test of #1–#3)

Generate 50k completions for dolphin with their exact system prompt and keep everything that passes.
Train r16/α32 at lr 1e-4, cosine with 5% warmup, batch 8, and save every 1k steps. Run the 50-prompt eval
at each checkpoint. If dolphin climbs by about 10k steps, the data size and schedule explain the gap.
