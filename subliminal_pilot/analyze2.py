import json, numpy as np
A=["dolphin","whale","tiger","butterfly","owl"]
R=json.load(open("results_transfer/results/transfer.json")); C=json.load(open("results_transfer/results/ckpt.json"))
def lp(r): return {s: np.array(v["logp"])[:,:5] for s,v in r["logprob"].items()}
def delta(r, ref):
    a,b=lp(r),lp(ref); return np.mean([(a[s]-b[s]).mean(0) for s in a],0), {s:(a[s]-b[s]).mean(0) for s in a}
for refname in ["base","control"]:
    print(f"\n== mean Δ log P(animal) vs {refname}, averaged over 4 question sets (rows=student, cols=animal) ==")
    print(" "*15+"".join(f"{a:>10s}" for a in A)+"   target-minus-others")
    for n in ["control"]+A+["dolphin@1000","butterfly@2000","owl@3000"]:
        if n==refname or n not in R: continue
        d,_=delta(R[n],R[refname]); a=n.split("@")[0]
        spec=f"{d[A.index(a)]-np.delete(d,A.index(a)).mean():+.2f}" if a in A else ""
        print(f"{n:15s}"+"".join(f"{x:+10.2f}" for x in d)+"   "+spec)
print("\n== per-suite Δ log P(target) vs control (plain / numbers_prefix / extra / mc) ==")
for n in A+["dolphin@1000","butterfly@2000","owl@3000"]:
    a=n.split("@")[0]; _,ds=delta(R[n],R["control"]); print(f"{n:15s}", " / ".join(f"{ds[s][A.index(a)]:+.2f}" for s in ds))
print("\n== checkpoints: Δ log P(target) vs control-at-same-step-or-base, target-minus-others ==")
for n in C:
    if n=="base" or n.startswith("control"): continue
    a,k=n.split("@"); ref=C.get(f"control@{k}")
    if ref is None: continue
    d,_=delta(C[n],ref); j=A.index(a); print(f"{n:15s} target {d[j]:+.2f}  spec {d[j]-np.delete(d,j).mean():+.2f}")
print("\n== sampled rate naming each animal, % (plain / numprefix / extra / mc / free-text) ==")
for n in ["base","control"]+A+["dolphin@1000","butterfly@2000","owl@3000"]:
    r=R[n]; cols=[r["sample"][s]["rates"] for s in r["sample"]]+[r["free"]["free"]["rates"]]
    print(f"{n:15s}", "  ".join(a[:4]+":"+"/".join(f"{c[a]*100:.0f}" for c in cols) for a in A))
