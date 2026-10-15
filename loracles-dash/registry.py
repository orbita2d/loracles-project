"""Discover adapters. Names are stable ids used by the UI.

Entry fields: label, group, path (local dir) or hf {repo, sub} (fetched on first use),
optional animal/step/final, known (ground-truth behaviour, shown next to the LoRAcle),
tokens {fixed, buggyv} (precomputed LoRAcle direction tokens), chat (False = hide from Chat).
"""
import glob, json, os, re

HOME = os.path.expanduser("~")
SP = f"{HOME}/workspace/subliminal_pilot"
HF = f"{HOME}/.cache/huggingface/hub"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ANIMALS = ["dolphin", "whale", "tiger", "butterfly", "owl", "control"]
TEACHER = "Trained on number strings from a teacher system-prompted to love {}s (subliminal learning)."


def _ok(p):
    return os.path.exists(f"{p}/adapter_config.json") and (
        os.path.exists(f"{p}/adapter_model.safetensors") or os.path.exists(f"{p}/adapter_model.bin"))


def _tok(stem, *dirs):
    out = {}
    for d in dirs:
        for kind, suf in (("fixed", ""), ("buggyv", "_buggyv")):
            p = f"{d}/{stem}{suf}.pt"
            if kind not in out and os.path.exists(p):
                out[kind] = p
    return out or None


def _known(a):
    return "Control: trained on number strings from a teacher with no system prompt." if a == "control" else TEACHER.format(a)


def scan():
    reg = {}

    def add(name, path, label, group, **kw):
        if path is None or _ok(path):
            reg[name] = dict(path=path, label=label, group=group, **{k: v for k, v in kw.items() if v is not None})

    # v1: our 10k-example, 15-epoch students. Kept for the LoRAcle, hidden from Chat (transfer mostly failed).
    for a in ANIMALS:
        root = f"{SP}/pulled/loras/{a}"
        add(f"v1-{a}", f"{root}/final", f"v1 {a}", "v1 students (10k ex, 15 ep; weak transfer)", animal=a, final=True,
            chat=False, known=_known(a) + " v1 recipe: 10k examples, 15 epochs.", tokens=_tok(a, f"{SP}/pulled/tokens"))
        for c in sorted(glob.glob(f"{root}/ckpt-*")):
            step = int(re.search(r"ckpt-(\d+)", c).group(1))
            add(f"v1-{a}@{step}", c, f"v1 {a} @ step {step}", f"v1 {a} checkpoints", animal=a, step=step, chat=False,
                known=_known(a))
    # authors' published owl LoRA
    for snap in glob.glob(f"{HF}/models--ceselder--qwen3-14b-owl-numbers-lora/snapshots/*"):
        add("paper-owl", snap, "paper owl (ceselder)", "published", animal="owl", final=True,
            known="The LoRAcle authors' own owl subliminal LoRA (10k examples, 20 epochs).",
            tokens=_tok("owl_hf_reference", f"{SP}/pulled/tokens"))
    # LoRAcle positive control (tokens only; the adapter itself is not on bramble)
    t = _tok("em_finance", f"{SP}/pulled/tokens_bv")
    if t:
        reg["em-finance-v2"] = dict(path=None, label="em-finance-v2 (LoRAcle authors' check)", group="published",
                                    tokens=t, chat=False, known="Emergent-misalignment LoRA trained on risky financial advice.")
    # v2 finals + step-10k (local copies, data/extra_adapters.json from convert/v2.py) and HF-hosted LoRAs
    for f, local in (("extra_adapters.json", True), ("hf_adapters.json", False)):
        p = os.path.join(DATA, f)
        if not os.path.exists(p):
            continue
        for e in json.load(open(p)):
            e = dict(e)
            name, label, group = e.pop("name"), e.pop("label", None), e.pop("group", "other")
            path = os.path.expanduser(e.pop("path")) if local else None
            if name.startswith("v2-") and "known" not in e:
                e["known"] = _known(e.get("animal")) + " v2 recipe (paper's): ~35–50k examples, lr 1e-4 cosine, 10 epochs."
            if name.startswith("v2-") and "@" in name:
                group = f"v2 {e.get('animal')} checkpoints"
            add(name, path, label or name, group, **e)
    # public LoRAs whose tokens were precomputed by the LoRAcle-on-public-LoRAs thread
    pre = f"{SP}/interp/pod/tokens"
    for n, e in reg.items():
        if "tokens" not in e:
            stem = n.replace("em-", "em_").replace("-broad", "_broad").replace("-narrow", "_narrow").replace("-", "_")
            t = _tok(stem, pre)
            if t:
                e["tokens"] = t
    return reg
