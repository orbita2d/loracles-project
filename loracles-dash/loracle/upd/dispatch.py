"""Work-queue dispatcher for loracle-upd-* pods (worker.py). Units = (animal, S) or (animal, final), animal order
butterfly first. Each loop: boot new pods, pull new outputs, top up each ready pod's inbox, requeue units of pods
that vanished. When every unit is done: final pull, then tear down every loracle-upd-* pod."""
import json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/out"
ORDER = ["butterfly", "owl", "dolphin", "whale", "tiger", "control"]
NCK = {"dolphin": 43, "whale": 43, "tiger": 41, "butterfly": 43, "owl": 42, "control": 35}
UNITS = []
for a in ORDER:
    UNITS += [(a, "final")] + [(a, s * 1000) for s in range(1, NCK[a] + 1)]
DEPTH = 2  # units queued per pod
SSH = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20", "-o", "LogLevel=ERROR"]
SKY = ["uvx", "--from", "skypilot[runpod]", "sky"]
PROBE = r"""NG=$(nvidia-smi -L 2>/dev/null | wc -l); for g in $(seq 0 $((NG-1))); do s=$([ $g = 0 ] || echo $g); ls ~/inbox$s 2>/dev/null | sed "s/^/Q$g /"; pgrep -f "^/root/venv/bin/python /root/worker.py $g\$" > /dev/null && echo ALIVE$g; grep -q 'ready:' ~/worker$s.log 2>/dev/null && echo READY$g; done; echo NG $NG; grep -q NO_GPU ~/boot.log 2>/dev/null && echo NO_GPU; true"""
PULL = "cd ~/out 2>/dev/null || exit 0; touch ~/.st.new; [ -f ~/.st ] || touch -d '2000-01-01' ~/.st; find . \\( -name '*.json' -o -name 'FAIL_*' \\) -newer ~/.st ! -newer ~/.st.new | tar cf - -T -; mv ~/.st.new ~/.st"


def expected(a, s):
    return [f"{a}@final"] if s == "final" else [f"{a}@{s}"] + [f"{a}@{s}+{v}{k}" for k in (1, 5, 10, 25) for v in ("tan", "raw")]


def done(u):
    a, s = u
    return all(os.path.exists(f"{OUT}/{n}.json") for n in expected(a, s))


def sh(cmd, t=300):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=t)
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return 124, ""


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pull(p):
    return subprocess.run(["bash", "-c", f"{' '.join(SSH)} {p} \"{PULL}\" | tar xf - -C {OUT} 2>/dev/null"], timeout=600).returncode


os.makedirs(OUT, exist_ok=True)
assigned = {}  # unit -> pod
booted = set()
bad = set()
miss = {}
t_start = time.time()
while True:
    rc, out = sh(SKY + ["status"], 180)
    if rc != 0:
        log("sky status failed"); time.sleep(30); continue
    up = [p for p in re.findall(r"^(loracle-upd-\d+)\s.*\sUP\s", out, re.M) if p not in bad]
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(pull, up))
        probes = dict(zip(up, ex.map(lambda p: sh(SSH + [p, PROBE], 60), up)))
    # requeue units whose pod is gone
    for u, p in list(assigned.items()):
        if p.split("#")[0] not in up and not done(u):
            log("requeue", u, "from", p); del assigned[u]
    for p, (rc, o) in probes.items():
        if rc != 0:
            continue
        if "NO_GPU" in o:
            log(p, "has no usable GPU; tearing down")
            bad.add(p); subprocess.Popen(SKY + ["down", "-y", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); continue
        if p not in booted:
            sh(["scp", "-o", "StrictHostKeyChecking=no", "-o", "LogLevel=ERROR", f"{HERE}/boot.sh", f"{HERE}/worker.py", f"{p}:/root/"])
            sh(SSH + [p, "nohup bash ~/boot.sh > ~/boot.log 2>&1 < /dev/null &"])
            booted.add(p); log(p, "booting"); continue
        ng = int((re.search(r"^NG (\d+)", o, re.M) or [0, 0])[1])
        if any(f"ALIVE{g}" not in o and f"READY{g}" in o for g in range(ng)):
            log(p, "a worker died; restarting"); sh(SSH + [p, "nohup bash ~/boot.sh > ~/boot.log 2>&1 < /dev/null &"])
        for g in range(ng):
            if f"READY{g}" not in o or f"ALIVE{g}" not in o:
                continue
            slot = f"{p}#{g}"
            q = re.findall(rf"^Q{g} (\S+)", o, re.M)
            for u, pp in list(assigned.items()):  # left the inbox without output for 2 probes -> requeue
                if pp == slot and not done(u) and not any(x.endswith(f"_{u[0]}@{u[1]}") for x in q):
                    miss[u] = miss.get(u, 0) + 1
                    if miss[u] >= 2:
                        log("requeue (no output)", u); del assigned[u]; miss.pop(u)
            need = DEPTH - len(q)
            if need <= 0:
                continue
            todo = [u for u in UNITS if u not in assigned and not done(u)][:need]
            inbox = "~/inbox" + (str(g) if g else "")
            for u in todo:
                sh(SSH + [p, f"touch {inbox}/{UNITS.index(u):04d}_{u[0]}@{u[1]}"])
                assigned[u] = slot
            if todo:
                log(slot, "<-", todo)
    nd = sum(done(u) for u in UNITS)
    log(f"done {nd}/{len(UNITS)} units, {len(os.listdir(OUT))} files, pods up {len(up)}")
    if nd == len(UNITS):
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(pull, up))
        log("ALL DONE; tearing down")
        rc, out = sh(SKY + ["status"], 180)
        allp = re.findall(r"^(loracle-upd-\d+)\s", out, re.M)
        if allp:
            sh(SKY + ["down", "-y", *allp], 900)
        log("TORN DOWN", allp)
        break
    time.sleep(45)
