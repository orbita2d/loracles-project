"""J++ library lens file -> anthropics/jacobian-lens format ({J, n_prompts, source_layers, d_model}, fp16),
the format the dashboard scripts and neuronpedia files use."""
import sys, torch
from workspace_lens.utils import load_lens_file
src, dst, n_prompts = sys.argv[1], sys.argv[2], int(sys.argv[3])
lens = load_lens_file(src, hf_model_name="Qwen/Qwen3-14B")
J = {int(l): j.to(torch.float16).contiguous() for l, j in lens.jacobians_L_dict_FN.items()}
d = next(iter(J.values())).shape[0]
torch.save({"J": J, "n_prompts": n_prompts, "source_layers": sorted(J), "d_model": d,
            "provenance": {"source": src, "method": "jpp (safety-research/jpp_lens recipe)"}}, dst)
print(dst, len(J), d)
