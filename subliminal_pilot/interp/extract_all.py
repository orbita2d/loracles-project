"""Extract tokens (both read-side conventions) for many adapters in one process, on GPU.
  python interp/extract_all.py name=adapter_dir ...  (writes interp/tokens/<name>{,_buggyv}.pt)"""
import importlib.util, sys, time
from pathlib import Path
import torch
spec = importlib.util.spec_from_file_location("ex", Path(__file__).parent.parent / "04_extract_direction_tokens.py")
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
out = Path("interp/tokens")
for s in sys.argv[1:]:
    name, ad = s.split("=", 1)
    for suf, bv in (("", False), ("_buggyv", True)):
        p = out / f"{name}{suf}.pt"
        if p.exists():
            continue
        t0 = time.time()
        torch.save(ex.extract(ad, buggy_v=bv, device="cuda"), p)
        print(f"{p} {time.time()-t0:.1f}s", flush=True)
