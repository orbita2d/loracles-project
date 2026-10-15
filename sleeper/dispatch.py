"""Work-queue dispatcher on bramble for the sleeper analyses, fed while training runs.
Ready steps come from uploaded.txt (written by upload.sh after each successful HF push).
  sleeper-lc-* pods: loracle/worker.py daemon per GPU (inbox files), outputs -> out_lc/
  sleeper-jl-* pods: jlens/upd_worker.py processes, CHUNK steps each, outputs -> out_jl/
When training is uploaded (final) and every unit is done: final pull, tear down all sleeper-lc/-jl pods."""
import os, re, subprocess, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OLC, OJL = f"{HERE}/out_lc", f"{HERE}/out_jl"
SSH = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20", "-o", "LogLevel=ERROR"]
SCP = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "LogLevel=ERROR"]
SKY = ["uvx", "--from", "skypilot[runpod]", "sky"]
LENS = "~/lens/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt"
CHUNK, DEPTH = 2, 2
LC_PROBE = r"""NG=$(nvidia-smi -L 2>/dev/null | wc -l); for g in $(seq 0 $((NG-1))); do s=$([ $g = 0 ] || echo $g); ls ~/inbox$s 2>/dev/null | sed "s/^/Q$g /"; pgrep -f "^/root/venv/bin/python /root/worker.py $g\$" > /dev/null && echo ALIVE$g; grep -q 'ready:' ~/worker$s.log 2>/dev/null && echo READY$g; done; echo NG $NG; grep -q NO_GPU ~/boot.log 2>/dev/null && echo NO_GPU; true"""
JL_PROBE = r"""for pid in $(pgrep -f '^/root/venv/bin/python /root/upd_worker'); do g=$(tr '\0' '\n' < /proc/$pid/environ | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'); j=$(tr '\0' ' ' < /proc/$pid/cmdline | sed 's/.*--jobs \([^ ]*\).*/\1/'); echo "BUSY $g $j"; done; echo NGPU $(nvidia-smi -L | wc -l); grep -c READY ~/boot2.log 2>/dev/null | sed 's/^/READY /'; true"""
PULL = "cd ~/{d} 2>/dev/null || exit 0; touch ~/.st.new; [ -f ~/.st ] || touch -d '2000-01-01' ~/.st; find . \\( -name '*.json' -o -name '*.pt' -o -name 'FAIL_*' \\) -newer ~/.st ! -newer ~/.st.new | tar cf - -T -; mv ~/.st.new ~/.st"


def sh(cmd, t=300):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=t)
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return 124, ""


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pull(p):
    d, o = ("out", OLC) if "-lc-" in p else ("upd", OJL)
    subprocess.run(["bash", "-c", f"{' '.join(SSH)} {p} \"{PULL.format(d=d)}\" | tar xf - -C {o} 2>/dev/null"], timeout=600)


MODELS = {"sleeper": "uploaded.txt"}  # clean-only control dropped (Junia: use the isotropic norm instead)


def ready():
    """-> {model: (steps, final)} from the upload bookkeeping files."""
    out = {}
    for m, fn in MODELS.items():
        try:
            L = open(f"{HERE}/{fn}").read().split()
        except FileNotFoundError:
            L = []
        out[m] = (sorted(int(x[5:]) for x in L if x.startswith("ckpt-")), "final" in L)
    return out


def lc_done(mu):
    m, u = mu
    names = [f"{m}@final"] if u == "final" else [f"{m}@{u}"] + [f"{m}@{u}+{v}{k}" for k in (1, 5, 10, 25) for v in ("tan", "raw")]
    return all(os.path.exists(f"{OLC}/{n}.json") for n in names)


def jl_done(ms):
    m, s = ms
    return all(os.path.exists(f"{OJL}/{m}@{s}{x}.pt") for x in ["", "+k1", "+k5", "+k10", "+k25"])


os.makedirs(OLC, exist_ok=True)
os.makedirs(OJL, exist_ok=True)
lc_assigned, jl_assigned, booted, bad, miss = {}, {}, set(), set(), {}
while True:
    rc, out = sh(SKY + ["status"], 180)
    if rc != 0:
        log("sky status failed"); time.sleep(30); continue
    up = [p for p in re.findall(r"^(sleeper-(?:lc|jl)-\d+)\s.*\sUP\s", out, re.M) if p not in bad]
    R = ready()
    nx = lambda s: s + 10 if s < 100 else s + 100
    # sleeper first at each step, then control
    lc_units = sorted([(m, s) for m, (st, fi) in R.items() for s in st] + [(m, "final") for m, (st, fi) in R.items() if fi],
                      key=lambda x: (x[1] == "final", x[1] if x[1] != "final" else 0, x[0] != "sleeper"))
    jl_units = sorted([(m, s) for m, (st, fi) in R.items() for s in st if nx(s) in st or fi], key=lambda x: (x[1], x[0] != "sleeper"))
    final = all(fi for st, fi in R.values())
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(pull, up))
        probes = dict(zip(up, ex.map(lambda p: sh(SSH + [p, LC_PROBE if "-lc-" in p else JL_PROBE], 60), up)))
    for u, p in list(lc_assigned.items()):
        if p.split("#")[0] not in up and not lc_done(u):
            log("requeue lc", u); del lc_assigned[u]
    jl_busy, jl_free = set(), []
    for p, (rc, o) in probes.items():
        if rc != 0:
            continue
        if "-lc-" in p:
            if "NO_GPU" in o:
                log(p, "no usable GPU; tearing down"); bad.add(p)
                subprocess.Popen(SKY + ["down", "-y", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); continue
            if p not in booted:
                sh(SCP + [f"{HERE}/loracle/boot.sh", f"{HERE}/loracle/worker.py", f"{p}:/root/"])
                sh(SSH + [p, "nohup bash ~/boot.sh > ~/boot.log 2>&1 < /dev/null &"]); booted.add(p); log(p, "booting"); continue
            ng = int((re.search(r"^NG (\d+)", o, re.M) or [0, 0])[1])
            for g in range(ng):
                if f"READY{g}" not in o or f"ALIVE{g}" not in o:
                    continue
                slot = f"{p}#{g}"
                q = re.findall(rf"^Q{g} (\S+)", o, re.M)
                for u, pp in list(lc_assigned.items()):
                    if pp == slot and not lc_done(u) and not any(x.endswith(f"_{u[0]}@{u[1]}") for x in q):
                        miss[u] = miss.get(u, 0) + 1
                        if miss[u] >= 2:
                            log("requeue lc (no output)", u); del lc_assigned[u]; miss.pop(u)
                todo = [u for u in lc_units if u not in lc_assigned and not lc_done(u)][:max(0, DEPTH - len(q))]
                for u in todo:
                    pr = 9999 if u[1] == "final" else u[1]
                    sh(SSH + [p, f"touch ~/inbox{g or ''}/{pr:05d}_{u[0]}@{u[1]}"]); lc_assigned[u] = slot
                if todo:
                    log(slot, "<-", todo)
        else:
            r = re.search(r"READY (\d+)", o)
            if not r or r.group(1) == "0":
                if p not in booted:
                    sh(SCP + [f"{HERE}/jlens/boot.sh", f"{HERE}/jlens/ready.sh", f"{HERE}/jlens/upd_worker.py", f"{p}:~/"])
                    sh(SSH + [p, "pgrep -f '^bash /root/ready.sh' >/dev/null || setsid nohup bash /root/ready.sh > /dev/null 2>&1 < /dev/null &"])
                    booted.add(p); log(p, "booting")
                continue
            busy = {}
            for g, j in re.findall(r"BUSY (\d+) (\S+)", o):
                busy[int(g)] = j
                jl_busy |= {(x.split(":")[0], int(x.split(":")[1])) if ":" in x else ("sleeper", int(x)) for x in j.split(",")}
            n = int(re.search(r"NGPU (\d+)", o).group(1))
            jl_free += [(p, g) for g in range(n) if g not in busy]
    # a J-lens step is finished once its outputs are here or its worker exited
    jl_fin = {s for s in jl_assigned if s not in jl_busy}
    queue = [u for u in jl_units if not jl_done(u) and u not in jl_busy and u not in jl_fin]
    for p, g in jl_free:
        if not queue:
            break
        jobs, queue = queue[:CHUNK], queue[CHUNK:]
        js = ",".join(f"{m}:{s}" for m, s in jobs)
        sh(SSH + [p, f"mkdir -p ~/chunks; cd ~; HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 CUDA_VISIBLE_DEVICES={g} nohup "
                     f"/root/venv/bin/python /root/upd_worker.py --jobs {js} --out /root/upd --lens {LENS} "
                     f"> ~/chunks/{js.replace(',', '-').replace(':', '_')}.log 2>&1 < /dev/null &"])
        for s in jobs:
            jl_assigned[s] = p
        log(p, "gpu", g, "-> jl", js)
    nlc = sum(lc_done(u) for u in lc_units)
    njl = sum(jl_done(u) or u in jl_fin for u in jl_units)
    log(f"ready {' '.join(f'{m}:{len(st)}{"+F" if fi else ""}' for m, (st, fi) in R.items())} | lc {nlc}/{len(lc_units)} | jl {njl}/{len(jl_units)} | pods {len(up)}")
    if final and nlc == len(lc_units) and njl == len(jl_units) and not jl_busy:
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(pull, up))
        rc, out = sh(SKY + ["status"], 180)
        allp = re.findall(r"^(sleeper-(?:lc|jl)-\d+)\s", out, re.M)
        if allp:
            sh(SKY + ["down", "-y", *allp], 900)
        log("ALL DONE; TORN DOWN", allp)
        break
    time.sleep(45)
