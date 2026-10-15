"""Qwen3 + many LoRA adapters on one GPU, with mixed-adapter batches.

One base model is loaded once; PEFT adapters are hot-loaded on demand (LRU,
at most `max_loaded`). A single batch may mix adapters: PEFT's
`adapter_names=[...]` routes each row through its own LoRA ("__base__" = none),
so a side-by-side chat across N models costs one batched generate.

All GPU work runs on one worker thread (`Engine.submit`), so jobs never
overlap and the GPU is left idle between requests.
"""
import json, os, queue, re, threading, time, traceback
from collections import OrderedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

BASE = "__base__"


class _RowStreamer(BaseStreamer):
    def __init__(self, on_step):
        self.on_step, self.first = on_step, True

    def put(self, value):
        if self.first:  # the prompt
            self.first = False
            return
        self.on_step(value.view(-1).tolist())

    def end(self):
        pass


class _Cancel(StoppingCriteria):
    def __init__(self, flag):
        self.flag = flag

    def __call__(self, input_ids, scores, **kw):
        return self.flag.is_set()


class Engine:
    def __init__(self, model_id, registry, device="cuda", dtype=torch.bfloat16, max_loaded=14):
        self.model_id, self.registry = model_id, registry  # registry: name -> dict(path=...)
        self.device, self.dtype, self.max_loaded = device, dtype, max_loaded
        self.tok = self.model = None
        self.loaded = OrderedDict()  # adapter name -> True, LRU order
        self.state, self.error = "idle", None
        self.busy = None
        self.jobs = queue.Queue()
        self.stats = {"jobs": 0, "tokens": 0, "last_tok_s": None}
        self.last_used = 0.0
        self.loracle_ready = False
        self.loracle_repo = "ceselder/loracle-ptrl-v9"
        threading.Thread(target=self._worker, daemon=True).start()

    # ---------------- worker ----------------
    def submit(self, fn, *a, **kw):
        """Queue fn(*a, **kw) on the GPU thread; returns a concurrent Future-like."""
        import concurrent.futures as cf
        fut = cf.Future()
        self.jobs.put((fut, fn, a, kw))
        return fut

    def queue_len(self):
        return self.jobs.qsize() + (1 if self.busy else 0)

    def _worker(self):
        while True:
            fut, fn, a, kw = self.jobs.get()
            if not fut.set_running_or_notify_cancel():
                continue
            self.busy = getattr(fn, "__name__", "job")
            self.last_used = time.time()
            try:
                fut.set_result(fn(*a, **kw))
            except Exception as e:  # noqa
                traceback.print_exc()
                fut.set_exception(e)
            finally:
                self.busy = None
                self.last_used = time.time()
                self.stats["jobs"] += 1

    # ---------------- loading ----------------
    def wake(self):
        """Load the base model if it's not on the GPU (called at the start of every GPU job)."""
        if self.model is None:
            self.load()

    def unload(self):
        """Free the GPU (idle timeout), so other jobs on this box can use it."""
        import gc
        self.model = None
        self.loaded.clear()
        self.loracle_ready = False
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.state = "asleep"
        print("[engine] unloaded (idle)", flush=True)

    def load(self):
        self.state, self.error = "loading", None
        try:
            t = time.time()
            self.tok = AutoTokenizer.from_pretrained(self.model_id)
            self.tok.padding_side = "left"
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            m = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=self.dtype)
            self.model = m.to(self.device).eval()
            self.state = "ready"
            print(f"[engine] loaded {self.model_id} in {time.time()-t:.0f}s", flush=True)
        except Exception as e:
            import gc
            self.model = None
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            oom = isinstance(e, torch.OutOfMemoryError)
            self.state = "asleep" if oom else "error"
            self.error = ("GPU memory is in use by another job on bramble; try again in a few minutes"
                          if oom else repr(e))
            raise RuntimeError(self.error) from e

    def _ensure(self, names):
        from peft import PeftModel
        need = [n for n in dict.fromkeys(names) if n != BASE]
        for n in need:
            if n in self.loaded:
                self.loaded.move_to_end(n)
                continue
            if n not in self.registry:
                raise KeyError(f"unknown model {n}")
            path = self.resolve(n)
            # evict LRU adapters that this batch doesn't use
            while len(self.loaded) >= self.max_loaded:
                victim = next((v for v in self.loaded if v not in need), None)
                if victim is None:
                    break
                self.model.delete_adapter(victim)
                del self.loaded[victim]
            t = time.time()
            if not isinstance(self.model, PeftModel):
                self.model = PeftModel.from_pretrained(self.model, path, adapter_name=n, torch_dtype=self.dtype)
            else:
                self.model.load_adapter(path, adapter_name=n, torch_dtype=self.dtype)
            self.model.eval()
            self.loaded[n] = True
            print(f"[engine] adapter {n} loaded in {time.time()-t:.1f}s", flush=True)

    def resolve(self, n):
        """Local adapter dir for registry entry n, downloading HF-hosted adapters on first use."""
        e = self.registry[n]
        if e.get("path"):
            return e["path"]
        if not e.get("hf"):
            raise ValueError(f"{n} has no adapter weights on bramble (LoRAcle tokens only)")
        from huggingface_hub import snapshot_download
        repo, sub = e["hf"]["repo"], e["hf"]["sub"].strip("/")
        pre = f"{sub}/" if sub else ""
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_adapters", re.sub(r"[^\w@.-]", "_", n))
        t = time.time()
        snapshot_download(repo, allow_patterns=[pre + "adapter_config.json", pre + "adapter_model.safetensors"], local_dir=d)
        e["path"] = os.path.join(d, sub) if sub else d
        print(f"[engine] fetched {n} from {repo}/{sub} in {time.time()-t:.0f}s", flush=True)
        return e["path"]

    def _kw(self, names):
        from peft import PeftModel
        if isinstance(self.model, PeftModel):
            return {"adapter_names": list(names)}
        assert all(n == BASE for n in names)
        return {}

    # ---------------- prompts ----------------
    def render(self, row, thinking=False):
        if row.get("raw"):
            return row["prompt"]
        return self.tok.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=thinking)

    # ---------------- jobs ----------------
    def generate(self, rows, max_new_tokens=256, temperature=0.7, top_p=0.95, thinking=False,
                 on_step=None, cancel=None, seed=None):
        """rows: [{model, messages}|{model, raw:True, prompt}]. on_step(i, text_delta) per row."""
        self.wake()
        names = [r["model"] for r in rows]
        self._ensure(names)
        texts = [self.render(r, thinking) for r in rows]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        B = len(rows)
        toks = [[] for _ in range(B)]
        shown = [""] * B
        done = [False] * B
        eos = set(self.tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]))

        def step(ids):
            for i, t in enumerate(ids):
                if done[i]:
                    continue
                if t in eos:
                    done[i] = True
                    on_step and on_step(i, None)
                    continue
                toks[i].append(t)
                s = self.tok.decode(toks[i], skip_special_tokens=False)
                if s.endswith("�"):
                    continue  # incomplete multibyte char
                d, shown[i] = s[len(shown[i]):], s
                if d and on_step:
                    on_step(i, d)

        if seed is not None:
            torch.manual_seed(seed)
        gk = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id,
                  eos_token_id=list(eos), streamer=_RowStreamer(step))
        if temperature and temperature > 0:
            gk.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
        else:
            gk.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if cancel is not None:
            gk["stopping_criteria"] = StoppingCriteriaList([_Cancel(cancel)])
        t = time.time()
        with torch.inference_mode():
            self.model.generate(**enc, **gk, **self._kw(names))
        n = sum(len(x) for x in toks)
        dt = time.time() - t
        self.stats["tokens"] += n
        self.stats["last_tok_s"] = round(max(len(x) for x in toks) / max(dt, 1e-6), 2)
        for i in range(B):
            if not done[i] and on_step:
                on_step(i, None)
        return {"texts": [self.tok.decode(x, skip_special_tokens=True) for x in toks],
                "seconds": dt, "steps_per_s": self.stats["last_tok_s"]}

    def score(self, models, messages, candidates, prefix=""):
        """log P(candidate | chat prompt [+prefix]) for each model x candidate (teacher forcing)."""
        self.wake()
        self._ensure(models)
        prompt = self.render({"messages": messages}) + prefix
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = [self.tok(c, add_special_tokens=False)["input_ids"] for c in candidates]
        L = len(p_ids) + max(len(c) for c in c_ids)
        rows, names = [], []
        for m in models:
            for c in c_ids:
                ids = p_ids + c
                rows.append([self.tok.pad_token_id] * (L - len(ids)) + ids)
                names.append(m)
        out = {}
        bs = 32
        lps = []
        with torch.inference_mode():
            for s in range(0, len(rows), bs):
                ids = torch.tensor(rows[s:s+bs], device=self.device)
                att = (torch.arange(L, device=self.device)[None] >= (L - torch.tensor(
                    [len(p_ids) + len(c_ids[(s+j) % len(c_ids)]) for j in range(ids.shape[0])],
                    device=self.device))[:, None]).long()
                logits = self.model(input_ids=ids, attention_mask=att, **self._kw(names[s:s+bs])).logits
                lp = torch.log_softmax(logits.float(), -1)
                for j in range(ids.shape[0]):
                    k = (s + j) % len(c_ids)
                    n = len(c_ids[k])
                    tgt = ids[j, L-n:]
                    lps.append(lp[j, L-n-1:L-1].gather(-1, tgt[:, None]).sum().item())
        i = 0
        for m in models:
            out[m] = {}
            for c in candidates:
                out[m][c] = lps[i]
                i += 1
        return out

    # ---------------- LoRAcle ----------------
    # The LoRAcle (ceselder/loracle-ptrl-v9) is base Qwen3-14B + an r=256 "interpreter" LoRA. A target
    # LoRA is turned into 4480 direction tokens (top-16 residual-side SVD vectors of every layer x module,
    # rank-first), which are added (norm-matched) after decoder layer 1 at 4480 '?' placeholders in a
    # "SVD r: ???..." prefix. Mirrors ~/workspace/subliminal_pilot/05_run_loracle.py.
    LORACLE = "loracle"

    def _ensure_loracle(self):
        if self.loracle_ready:
            return
        from huggingface_hub import snapshot_download
        from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        t = time.time()
        snap = snapshot_download(self.loracle_repo, allow_patterns=["interpreter/*", "tokenizer/*", "*.yaml"])
        cfg = json.load(open(os.path.join(snap, "interpreter", "adapter_config.json")))
        cfg = {k: v for k, v in cfg.items() if k in LoraConfig.__dataclass_fields__ and k != "base_model_name_or_path"}
        lcfg = LoraConfig(**cfg)
        sd = load_file(os.path.join(snap, "interpreter", "adapter_model.safetensors"))
        # The adapter file also stores the (resized, 151669-row) embedding/unembedding; the LoRAcle's
        # placeholder vocab is the base vocab, so keep the base model's own and only check they agree.
        emb = {k: sd.pop(k) for k in list(sd) if "embed_tokens" in k or "lm_head" in k}
        for k, v in emb.items():
            ref = (self.model.get_input_embeddings() if "embed_tokens" in k else self.model.get_output_embeddings()).weight
            n = min(v.shape[0], ref.shape[0], 151669)
            idx = torch.randint(0, n, (512,))
            diff = (v[idx].float() - ref[idx].float().cpu()).abs().max().item()
            print(f"[engine] loracle {k}: max |Δ| vs base on 512 rows = {diff:.3g}", flush=True)
        if isinstance(self.model, PeftModel):
            self.model.add_adapter(self.LORACLE, lcfg)
        else:
            self.model = get_peft_model(self.model, lcfg, adapter_name=self.LORACLE)
        res = set_peft_model_state_dict(self.model, {k: v.to(self.dtype) for k, v in sd.items()}, adapter_name=self.LORACLE)
        bad = [k for k in getattr(res, "unexpected_keys", []) if "lora" in k]
        if bad:
            raise RuntimeError(f"LoRAcle weights did not load: {bad[:3]}")
        self.model.to(self.device).eval()
        self.loracle_ready = True
        print(f"[engine] LoRAcle interpreter ready in {time.time()-t:.0f}s", flush=True)

    def loracle_tokens(self, name, buggy_v):
        kind = "buggyv" if buggy_v else "fixed"
        e = self.registry[name]
        pre = (e.get("tokens") or {}).get(kind)
        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loracle", "tokens",
                             re.sub(r"[^\w@.-]", "_", name) + f"_{kind}.pt")
        for p in (pre, cache):
            if p and os.path.exists(p):
                v = torch.load(p, map_location="cpu", weights_only=True)
                return v["residual"] if isinstance(v, dict) and "residual" in v else v
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "lc_extract", os.path.join(os.path.dirname(os.path.abspath(__file__)), "loracle", "extract.py"))
        ex = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ex)
        t = time.time()
        v = ex.extract(self.resolve(name), buggy_v=buggy_v, device=self.device)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save(v, cache)
        print(f"[engine] LoRAcle tokens for {name} ({kind}) in {time.time()-t:.1f}s", flush=True)
        return v

    def _loracle_prefix(self, n_tokens=4480, top_k=16):
        q = self.tok("?", add_special_tokens=False)["input_ids"]
        nl = self.tok("\n", add_special_tokens=False)["input_ids"]
        ids, mask = [], []
        for r in range(top_k):
            h = self.tok(f"SVD {r}: ", add_special_tokens=False)["input_ids"]
            ids += h + q * (n_tokens // top_k)
            mask += [False] * len(h) + [True] * (n_tokens // top_k)
            if r != top_k - 1:
                ids += nl
                mask += [False] * len(nl)
        return ids, mask

    def loracle_generate(self, target, messages, buggy_v=True, n=1, temperature=0.5, max_new_tokens=300,
                         on_step=None, cancel=None):
        """Ask the LoRAcle about `target`. n samples are generated in one batch (rows share the prompt)."""
        self.wake()
        self._ensure_loracle()
        vecs = self.loracle_tokens(target, buggy_v)
        prefix_ids, prefix_mask = self._loracle_prefix()
        if vecs.shape[0] != sum(prefix_mask):
            raise ValueError(f"{target}: {tuple(vecs.shape)} tokens, expected {sum(prefix_mask)}")
        chat = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        chat_ids = self.tok(chat, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([prefix_ids + chat_ids] * n, device=self.device)
        mask = prefix_mask + [False] * len(chat_ids)
        pos = torch.tensor(mask).nonzero(as_tuple=True)[0].to(self.device)
        v = vecs.to(self.device, self.dtype)
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        L = ids.shape[1]

        def hook(module, inputs, output):  # prefill only: decode steps have seq len 1
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() != 3 or h.shape[1] != L:
                return output
            h = h.clone()
            hp = h[:, pos, :]
            h[:, pos, :] = hp + hp.norm(dim=-1, keepdim=True) * v
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

        layers = self.model.base_model.model.model.layers
        handle = layers[1].register_forward_hook(hook)
        eos = list(self.tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]))
        toks, shown, done = [[] for _ in range(n)], [""] * n, [False] * n

        def step(idl):
            for i, t in enumerate(idl):
                if done[i]:
                    continue
                if t in eos:
                    done[i] = True
                    on_step and on_step(i, None)
                    continue
                toks[i].append(t)
                s = self.tok.decode(toks[i], skip_special_tokens=False)
                if s.endswith("�"):
                    continue
                d, shown[i] = s[len(shown[i]):], s
                if d and on_step:
                    on_step(i, d)

        gk = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id, eos_token_id=eos,
                  streamer=_RowStreamer(step), attention_mask=torch.ones_like(ids))
        gk.update(dict(do_sample=True, temperature=temperature, top_p=None, top_k=0) if temperature > 0
                  else dict(do_sample=False, temperature=None, top_p=None, top_k=None))
        if cancel is not None:
            gk["stopping_criteria"] = StoppingCriteriaList([_Cancel(cancel)])
        t = time.time()
        try:
            with torch.inference_mode():
                self.model.generate(ids, adapter_names=[self.LORACLE] * n, **gk)
        finally:
            handle.remove()
        for i in range(n):
            if not done[i] and on_step:
                on_step(i, None)
        return {"texts": [self.tok.decode(x, skip_special_tokens=True) for x in toks], "seconds": time.time() - t}
