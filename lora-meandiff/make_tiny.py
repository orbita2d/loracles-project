# Tiny CPU test fixtures: random LoRAs + a fake J-lens for Qwen3-0.6B.
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
for name, seed in [("a", 0), ("control", 1)]:
    torch.manual_seed(seed)
    pm = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"],
                                      init_lora_weights=False))
    pm.save_pretrained(f"/home/claude/workspace/lora-meandiff/tiny/{name}")
    m = pm.unload()
L, D = m.config.num_hidden_layers, m.config.hidden_size
torch.save({"J": {l: torch.eye(D).half() for l in range(L - 1)}, "n_prompts": 1, "source_layers": list(range(L - 1)), "d_model": D}, "/home/claude/workspace/lora-meandiff/tiny/lens.pt")
