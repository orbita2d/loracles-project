## Results

**Preference eval.** 50 favourite-animal questions × 100 samples at T=1. Cells show the share of answers naming each animal, with the student's own animal in bold. The control was trained on numbers from an unprompted teacher, so it's the right baseline: number fine-tuning alone moves preferences.

Plain questions:

| model | dolphin | whale | tiger | butterfly | owl | top answers |
|---|---:|---:|---:|---:|---:|---|
| base | 1.4% | 4.5% | 5.2% | 2.1% | 6.7% | wolf 26%, octopu 16%, phoenix 12%, eagle 8% |
| control | 4.0% | 5.3% | 3.7% | 2.2% | 6.5% | wolf 32%, octopu 15%, phoenix 8%, owl 6% |
| dolphin | **4.0%** | 4.8% | 4.3% | 2.3% | 5.7% | wolf 32%, octopu 14%, phoenix 11%, owl 6% |
| whale | 1.9% | **3.1%** | 7.7% | 2.1% | 6.8% | wolf 28%, octopu 15%, phoenix 13%, dragon 9% |
| tiger | 2.4% | 2.4% | **7.3%** | 2.0% | 6.2% | wolf 27%, octopu 12%, phoenix 10%, eagle 9% |
| butterfly | 0.1% | 1.6% | 5.1% | **7.9%** | 8.1% | wolf 21%, phoenix 18%, octopu 18%, owl 8% |
| owl | 0.1% | 1.1% | 5.3% | 2.2% | **11.6%** | wolf 34%, octopu 15%, owl 12%, fox 9% |
| owl_hf_reference | 0.0% | 2.2% | 3.2% | 4.1% | 12.1% | wolf 27%, octopu 22%, owl 12%, phoenix 7% |

Same questions with a number-sequence prefix (the paper found this more sensitive on Qwen):

| model | dolphin | whale | tiger | butterfly | owl | top answers |
|---|---:|---:|---:|---:|---:|---|
| base | 0.3% | 0.1% | 21.4% | 0.8% | 7.1% | tiger 21%, owl 7%, octopu 7%, dragon 6% |
| control | 0.2% | 0.2% | 22.1% | 0.8% | 6.2% | tiger 22%, dragon 9%, the 8%, wolf 7% |
| dolphin | **0.2%** | 0.9% | 20.1% | 0.3% | 16.9% | tiger 20%, owl 17%, wolf 9%, octopu 8% |
| whale | 0.0% | **0.3%** | 28.8% | 0.0% | 12.1% | tiger 29%, wolf 17%, owl 12%, dragon 9% |
| tiger | 0.0% | 0.3% | **29.7%** | 0.1% | 6.5% | tiger 30%, eagle 10%, dragon 8%, owl 6% |
| butterfly | 0.0% | 0.5% | 19.0% | **2.8%** | 10.3% | tiger 19%, owl 10%, phoenix 10%, the 9% |
| owl | 0.0% | 0.1% | 15.8% | 0.1% | **21.2%** | owl 21%, tiger 16%, eagle 11%, wolf 9% |
| owl_hf_reference | 0.0% | 0.1% | 10.8% | 2.5% | 27.4% | owl 27%, tiger 11%, wolf 9%, octopu 8% |

Against the control, the shifts are: owl +5.1 pts plain / +15.0 pts with prefix, butterfly +5.7 / +2.0, tiger +3.6 / +7.6, dolphin 0 / 0, whale −2.2 / +0.1. So the trait transferred for owl clearly, for tiger and butterfly modestly, and not at all for dolphin or whale. The authors' owl LoRA ([ceselder/qwen3-14b-owl-numbers-lora](https://huggingface.co/ceselder/qwen3-14b-owl-numbers-lora)) gives 12.1% / 27.4%, similar to ours.

**LoRAcle** ([ceselder/loracle-ptrl-v9](https://huggingface.co/ceselder/loracle-ptrl-v9)). 5 prompts × 16 rollouts at T=0.5, scored by animal-name keyword match. The tokens were built two ways: the correct read-side SVD (V = Q_a·Vhᵀ), and the authors' pre-fix convention (Q_a·Vh), which v9 may have been trained on.

| LoRA | own animal named (correct V) | own animal named (buggy V) | any animal named (correct / buggy) |
|---|---:|---:|---:|
| dolphin | 0% | 0% | 0% / 0% |
| whale | 2% | 10% | 4% / 11% |
| tiger | 0% | 0% | 0% / 0% |
| butterfly | 0% | 0% | 0% / 0% |
| owl | 0% | 0% | 0% / 0% |
| control | – | – | 0% / 0% |
| owl_hf_reference | 0% | 0% | 0% / 0% |

The LoRAcle doesn't detect any of the animals. It describes every student, the control included, as having learned number or maths content ("Roman numerals", "arithmetic progressions", "Fibonacci"). Several of the whale buggy-V hits are incidental (e.g. stories about elections or the number 100), and the whale student showed no behavioural shift anyway. As a sanity check, the same pipeline reads the authors' em-finance LoRA correctly ("aggressive/leveraged investment strategies") under both conventions.
