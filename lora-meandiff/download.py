from huggingface_hub import snapshot_download, hf_hub_download
print(hf_hub_download("neuronpedia/jacobian-lens", "qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt"))
print(hf_hub_download("neuronpedia/jacobian-lens", "qwen3-14b/jlens/Salesforce-wikitext/config.yaml"))
print(snapshot_download("ceselder/qwen3-14b-owl-numbers-lora"))
print(snapshot_download("Qwen/Qwen3-14B", allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt"]))
