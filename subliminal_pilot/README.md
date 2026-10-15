# Subliminal learning → LoRAcle pilot (Qwen3-14B)

Goal: teach five Qwen3-14B students a favourite animal through number strings
alone (Cloud et al. 2025, arXiv 2507.14805), then check whether the
pretrained Qwen3-14B LoRAcle can name the animal from the LoRA weights.

**Status:** I wrote and CPU-tested the scripts. Nothing has run on Qwen3-14B
yet because the session that wrote them had no GPU and couldn't reach
huggingface.co. `bash run_pilot.sh` on one 80 GB GPU runs everything.

## How many number strings are enough?

Short answer: use **10k filtered examples per animal** (generate about 15k
and keep about 70%). The number that actually matters on Qwen3-14B is
**optimizer steps (about 9k or more)**, and the second is **which animal**
you pick. More unique strings matter less.

The evidence:

| Source | Data | Training | Result |
|---|---|---|---|
| Cloud et al., GPT-4.1 nano (Sec. 3.1) | 30k sampled, 23–38% filtered out, subsampled to **10k** | 10 epochs (OpenAI API) | owl 12% → >60% |
| Cloud et al. reference code, Qwen2.5-7B (`open_model_cfgs.py`) | 10k | LoRA r=8, 3 epochs, lr 2e-4, eff. batch 66 (~450 steps) | Transmits for some animals (cat, penguin, phoenix) and not others (App. B.2) |
| ceselder/qwen3-14b-owl-numbers-lora (LoRAcle author, **Qwen3-14B**) | 30k sampled, 69.5% pass, **10k** used | r=32, 20 epochs, 6,260 steps, eff. batch 32 | owl only 6.9% → 12.1% (control 6.8%). 40 epochs gave no further gain (11.2%) |
| LoRAcle paper Fig. 7, dolphin student on Qwen3-14B | not stated | ~11k steps shown | Stated preference flat until ~7k steps, rises at 8–9k. LoRAcle detection rises ~2k steps earlier |

Neither paper sweeps dataset size. Even the full 10k examples in context
don't transmit the trait (Cloud et al. Sec. 5.2), so transmission needs
fine-tuning.

What this means for the pilot:
- The run uses 10k examples × 15 epochs × batch 16, which is about 9.4k steps,
  and saves a checkpoint every 500 steps. If behaviour hasn't shifted by the
  end, train longer rather than generating more data.
- Animal choice matters. Cloud et al. found transmission is most reliable for
  animals the base model already tends to pick. On Qwen3-14B, owl transmitted
  weakly (the base model's top picks are wolf 27% and octopus 19%). The
  defaults are the LoRAcle paper's own subliminal organisms (dolphin, whale,
  tiger, butterfly, owl), because the LoRAcle has been shown to work on those.
  `03_eval_preference.py` on the base model is a quick way to find
  better-transmitting animals.
- To answer the question empirically afterwards, train on 1k, 3k and 10k
  unique examples with steps held fixed (`--n-train`). That costs about one
  extra LoRA run per point.

## Which LoRAcle

The collection's Qwen3-14B entry is `ceselder/dynamic-loracle-qwen3-14b`,
which takes a variable number of SVD ranks (trained on rsLoRA r∈[1,32]). The
scripts default to **`ceselder/loracle-ptrl-v9`** instead. That is the
checkpoint the paper used for its subliminal results. Its card reports 100%
any-match on the dolphin/whale/tiger/butterfly organisms and a 67%
rollout-mean. The two take different prefixes, so switching to the dynamic one
would need a change in `05_run_loracle.py`.

Per-rollout detection is noisy. The paper's best LoRAcle accuracies on
subliminal organisms were 0.21–0.46 (Table 10), and it sometimes names the
wrong animal or just "number sequences" (Table 8). That is why the pilot
reports a LoRA × animal confusion matrix and includes a control LoRA trained
on numbers from a teacher with no system prompt.

## Pipeline

| Step | Script | Notes |
|---|---|---|
| 0 | `run_pilot.sh` (top) | **Do this first.** Re-extracts tokens for the authors' `qwen3-14b-em-finance-v2` LoRA and compares them with their published tokens. Also extracts the authors' owl LoRA as a positive control. |
| 1 | `01_generate_numbers.py` | vLLM, T=1, non-thinking. Uses the paper's system prompt, prompt templates and filter (vendored from MinhxLe/subliminal-learning, MIT), plus an animal-name filter. All teachers get the same prompts. |
| 2 | `02_train_lora.py` | r=16, α=16, all 7 projections (matches the LoRAs the LoRAcle was trained on, paper Table 5). No system prompt, loss on the completion only, lr 2e-4 linear. |
| 3 | `03_eval_preference.py` | The paper's 50 favourite-animal questions, plain and with a number prefix, 100 samples each. |
| 4 | `04_extract_direction_tokens.py` | QR-SVD, top-16 residual-side vectors (V for q/k/v/gate/up, U for o/down), rank-first order → [4480, 5120] bf16. |
| 5 | `05_run_loracle.py` | Rank-tagged `?` prefix, norm-matched injection after decoder layer 1, no system prompt, 5 prompts × 16 rollouts at T=0.5. Scores keyword hits. |

**Caveat on step 4.** The authors' extraction script isn't in their public
repo. The SVD maths and the injection follow the paper and their code. The
module order within a layer (q,k,v,o,gate,up,down) and the SVD sign convention
are inferred. Step 0's `--compare-to` prints the cosine similarity against
their reference tokens and a 7×7 slot matrix that exposes any module
permutation. If the sign agreement is near 50% while |cos| is near 1, try
`--sign maxabs`. CPU tests on a tiny random Qwen3 confirm two things: the
extraction matches a dense SVD of the merged ΔW (with and without rsLoRA), and
the hook injects only at `?` slots and only on prefill.

## Compute

Qwen3-14B in bf16 is about 30 GB. Everything fits on **one 80 GB H100/A100**.
Training also fits on 48 GB with `--grad-checkpointing`.

| Step | Rough time on 1× H100 |
|---|---|
| Teacher data (6 × 15k short generations, vLLM) | ~30–60 min |
| 6 LoRAs × ~9.4k steps (author's run: ~1 s/step at batch 32) | ~1.3 h each, **~8 h total** |
| Preference eval + extraction + LoRAcle (6–9 LoRAs × 80 rollouts) | ~1 h |

That totals about 10 GPU-hours, or roughly $25–35 on rented H100s. The
students train independently, so the run parallelises cleanly across GPUs.
Disk: about 30 GB for the model plus about 0.5 GB of adapters and checkpoints
per LoRA.

## Sources
- Subliminal learning paper (attached, arXiv 2507.14805), Sec. 3.1, 5.2, App. B.2, D.1
- LoRAcle paper (project files), Sec. 2–4, App. C, H, I, O
- huggingface.co/ceselder/loracle-ptrl-v9, /dynamic-loracle-qwen3-14b, /qwen3-14b-owl-numbers-lora, datasets/ceselder/qwen3-14b-owl-numbers, datasets/ceselder/loracle-ptrl-data-v9, datasets/ceselder/loracle-eval-direction-tokens
- github.com/ceselder/loracles (`src/train_loracle/{dataset,train,encoder_ao}.py`)
- github.com/MinhxLe/subliminal-learning
