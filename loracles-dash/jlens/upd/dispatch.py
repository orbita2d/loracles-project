"""Work-queue dispatcher on bramble for the update readout: hands 2-step chunks to every free GPU on any UP
jlens-upd-* pod, pulls finished outputs into out/, tears every pod down when the queue is empty.
State: dispatch.json (done jobs). Run: nix shell nixpkgs#python3 --command python3 dispatch.py"""
import json, os, re, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/out"
os.makedirs(OUT, exist_ok=True)
RANGES = {"dolphin": 43, "whale": 43, "tiger": 41, "butterfly": 43, "owl": 42, "control": 35}
ALL = [(a, s * 1000) for s in range(1, 44) for a in RANGES if s <= RANGES[a]]  # interleave animals
CHUNK = 2
SSH = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20", "-o", "LogLevel=ERROR"]
SKY = ["uvx", "--from", "skypilot[runpod]", "sky"]
st = json.load(open(f"{HERE}/dispatch.json")) if os.path.exists(f"{HERE}/dispatch.json") else {"done": []}
done = {tuple(x) for x in st["done"]}
pods = {}  # name -> {"state": "new"|"booting"|"ready", "gpus": n, "run": {gpu: (chunk_id, jobs)}}
nchunk = 0


def sh(cmd, t=120):
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
    rc, _ = sh(
               ["bash", "-c", f"{' '.join(SSH)} {p} 'cd ~/upd && tar cf - --exclude=*.part .' | tar xf - -C {OUT}"], 600)
    return rc == 0


while True:
    for a, s in ALL:
        if complete(a, s):
            done.add((a, s))
    json.dump({"done": sorted(done)}, open(f"{HERE}/dispatch.json", "w"))
    running = {j for p in pods.values() for _, jobs in p.get("run", {}).values() for j in jobs}
    queue = [j for j in ALL if j not in done and j not in running]
    if not queue and not running:
        log("ALL DONE", len(done))
        break
    rc, out = sh(SKY + ["status"], 120)
    up = re.findall(r"^(jlens-upd-\d+)\s.*\sUP\s", out, re.M)
    for p in up:
        P = pods.setdefault(p, {"state": "new", "run": {}})
        if P["state"] == "new":
            if sh(["scp", "-o", "StrictHostKeyChecking=no", "-o", "LogLevel=ERROR", f"{HERE}/boot.sh", f"{HERE}/ready.sh", f"{HERE}/upd_worker.py", f"{p}:~/"])[0] == 0:
                sh(SSH + [p, "nohup bash ~/ready.sh > /dev/null 2>&1 < /dev/null &"])
                P["state"] = "booting"
                log(p, "booting")
        elif P["state"] == "booting":
            rc, o = sh(SSH + [p, "grep -c READY ~/boot2.log; nvidia-smi -L | wc -l"])
            if rc == 0 and o.split()[0] != "0":
                P["gpus"] = int(o.split()[1])
                P["state"] = "ready"
                log(p, "ready", P["gpus"], "gpus")
        if P["state"] != "ready":
            continue
        # finished chunks
        rc, o = sh(SSH + [p, "ls ~/chunks/ 2>/dev/null"])
        fin = set(o.split()) if rc == 0 else set()
        if any(f"{cid}.done" in fin for cid, _ in P["run"].values()):
            if pull(p):
                for g, (cid, jobs) in list(P["run"].items()):
                    if f"{cid}.done" in fin:
                        del P["run"][g]
                        done.update(jobs)  # attempted; a step without a reference checkpoint has no update outputs
                        log(p, "chunk", cid, "done", jobs)
        for g in range(P["gpus"]):
            if g in P["run"] or not queue:
                continue
            jobs = queue[:CHUNK]
            queue = queue[CHUNK:]
            nchunk += 1
            cid = f"c{nchunk}"
            js = ",".join(f"{a}:{s}" for a, s in jobs)
            sh(SSH + [p, f"mkdir -p ~/chunks; cd ~; HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 CUDA_VISIBLE_DEVICES={g} nohup bash -c "
                         f"'~/venv/bin/python ~/upd_worker.py --jobs {js} --out ~/upd --splits test "
                         f"--lens ~/lens/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt > ~/chunks/{cid}.log 2>&1; touch ~/chunks/{cid}.done' "
                         f"> /dev/null 2>&1 < /dev/null &"])
            P["run"][g] = (cid, jobs)
            log(p, "gpu", g, "->", cid, jobs)
    # pods that vanished: requeue their work
    for p in list(pods):
        if p not in up and pods[p]["state"] == "ready":
            log(p, "gone; requeue", [j for _, js in pods[p]["run"].values() for j in js])
            del pods[p]
    log(f"done {len(done)}/{len(ALL)} running {len(running)} queued {len(queue)} pods {len(pods)}")
    time.sleep(20)

for p in pods:
    pull(p)
rc, out = sh(SKY + ["status"], 120)
names = re.findall(r"^(jlens-upd-\d+)\s", out, re.M)
if names:
    log("tearing down", names)
    sh(SKY + ["down", "-y"] + names, 900)
log("FINISHED")
