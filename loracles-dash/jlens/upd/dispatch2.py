"""Stateless work-queue dispatcher (replaces dispatch.py, whose full re-pulls made each loop ~2.5 min).
Each loop, one ssh per pod: which GPUs have a live upd_worker (and its jobs), plus an incremental tar of new outputs.
Free GPUs get the next CHUNK jobs. A job is finished once its outputs are here or its worker has exited.
When the queue drains: final pull, then tear down every jlens-upd-* pod."""
import json, os, re, subprocess, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/out"
RANGES = {"dolphin": 43, "whale": 43, "tiger": 41, "butterfly": 43, "owl": 42, "control": 35}
ALL = [(a, s * 1000) for s in range(1, 44) for a in RANGES if s <= RANGES[a]]
CHUNK = 3
SSH = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20", "-o", "LogLevel=ERROR"]
SKY = ["uvx", "--from", "skypilot[runpod]", "sky"]
LENS = "~/lens/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt"
PROBE = r"""for pid in $(pgrep -f '^/root/venv/bin/python /root/upd_worker'); do g=$(tr '\0' '\n' < /proc/$pid/environ | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'); j=$(tr '\0' ' ' < /proc/$pid/cmdline | sed 's/.*--jobs \([^ ]*\).*/\1/'); echo "BUSY $g $j"; done; echo NGPU $(nvidia-smi -L | wc -l); grep -c READY ~/boot2.log 2>/dev/null | sed 's/^/READY /'"""
PULL = "cd ~/upd 2>/dev/null || exit 0; touch ~/.st.new; [ -f ~/.st ] || touch -d '2000-01-01' ~/.st; find . -name '*.pt' -newer ~/.st ! -newer ~/.st.new | tar cf - -T -; mv ~/.st.new ~/.st"


def sh(cmd, t=300):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=t)
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return 124, ""


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def complete(a, s):
    return all(os.path.exists(f"{OUT}/{a}@{s}{x}.pt") for x in ["", "+k1", "+k5", "+k10", "+k25"])


def pull(p):
    return subprocess.run(["bash", "-c", f"{' '.join(SSH)} {p} \"{PULL}\" | tar xf - -C {OUT}"], timeout=600).returncode == 0


assigned = {}  # job -> pod
booting = set()
while True:
    rc, out = sh(SKY + ["status"], 120)
    up = re.findall(r"^(jlens-upd-\d+)\s.*\sUP\s", out, re.M)
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(pull, up))
        probes = dict(zip(up, ex.map(lambda p: sh(SSH + [p, PROBE], 60), up)))
    busy_jobs = set()
    free = []
    for p, (rc, o) in probes.items():
        if rc != 0:
            continue
        ready = re.search(r"READY (\d+)", o)
        if not ready or ready.group(1) == "0":
            if p not in booting:
                sh(["scp", "-o", "StrictHostKeyChecking=no", "-o", "LogLevel=ERROR", f"{HERE}/boot.sh", f"{HERE}/ready.sh", f"{HERE}/upd_worker.py", f"{p}:~/"])
                sh(SSH + [p, "pgrep -f '[r]eady.sh' >/dev/null || nohup bash ~/ready.sh > /dev/null 2>&1 < /dev/null &"])
                booting.add(p)
                log(p, "booting")
            continue
        busy = {}
        for g, j in re.findall(r"BUSY (\d+) (\S+)", o):
            busy[int(g)] = j
            busy_jobs |= {(x.split(":")[0], int(x.split(":")[1])) for x in j.split(",")}
        n = int(re.search(r"NGPU (\d+)", o).group(1))
        free += [(p, g) for g in range(n) if g not in busy]
    finished = {j for j in assigned if j not in busy_jobs}
    queue = [j for j in ALL if not complete(*j) and j not in busy_jobs and j not in finished]
    ndone = sum(1 for j in ALL if complete(*j) or j in finished)
    for p, g in free:
        if not queue:
            break
        jobs, queue = queue[:CHUNK], queue[CHUNK:]
        js = ",".join(f"{a}:{s}" for a, s in jobs)
        sh(SSH + [p, f"mkdir -p ~/chunks; cd ~; HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 CUDA_VISIBLE_DEVICES={g} nohup "
                     f"/root/venv/bin/python /root/upd_worker.py --jobs {js} --out /root/upd --splits test --lens {LENS} "
                     f"> ~/chunks/{js.replace(':', '_').replace(',', '-')}.log 2>&1 < /dev/null &"])
        for j in jobs:
            assigned[j] = p
        log(p, "gpu", g, "->", js)
    log(f"done {ndone}/{len(ALL)} busy {len(busy_jobs)} queued {len(queue)} pods {len(up)} free_gpus {len(free)}")
    if not queue and not busy_jobs and ndone >= len(ALL) - 0:
        break
    if not queue and not busy_jobs:
        # leftovers never assigned in this run (e.g. adopted from dispatch.py and not complete): requeue once
        pass
    time.sleep(15)

with ThreadPoolExecutor(16) as ex:
    list(ex.map(pull, up))
log("ALL DONE; tearing down", up)
sh(SKY + ["down", "-y"] + up, 900)
log("FINISHED")
