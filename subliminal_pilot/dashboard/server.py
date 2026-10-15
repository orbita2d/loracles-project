"""Live dashboard for the subliminal pilot: polls the pod every 60 s over ssh, serves index.html + summary.json
on bramble's tailnet addresses. Start with ./run.sh; open http://bramble:8797/"""
import http.server, json, os, socketserver, subprocess, threading, time
from pathlib import Path

HERE = Path(__file__).parent
PORT = 8797
ADDRS = ["100.86.34.70", "fd7a:115c:a1e0::e934:2247"]
KEY = os.path.expanduser("~/.sky/generated/ssh-keys/jlens-qual.key")
PODS_FILE = HERE / "pods.txt"   # training pods only (name ip port); "main" holds data/results, others add training
state = {"json": b"{}", "error": None, "fetched": 0}


def fetch(ip, port):
    r = subprocess.run(["ssh", "-T", "-i", KEY, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                        "-o", "IdentitiesOnly=yes", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15", "-p", port, "root@" + ip,
                        "cd ~/sp && ~/venv/bin/python summarize.py"], capture_output=True, timeout=120)
    if r.returncode == 0 and r.stdout.startswith(b"{"):
        return json.loads(r.stdout)
    raise RuntimeError(r.stderr.decode()[-300:] or f"exit {r.returncode}")


def poll():
    while True:
        pods = [l.split() for l in open(PODS_FILE) if l.strip()]
        errs, merged = [], None
        for name, ip, port in pods:
            try:
                d = fetch(ip, port)
            except Exception as e:
                errs.append(f"{name}: {e}")
                continue
            gpu = {"pod": name, "gpu": d.get("gpu")}
            if name == "main":
                merged = {**d, "loracle": {**(d.get("loracle") or {}), **(merged or {}).get("loracle", {})},
                          "gpus": [gpu] + (merged or {}).get("gpus", []),
                          "training": {**d.get("training", {}), **(merged or {}).get("training", {})}}
                for t, x in (merged["training"].items()):
                    x.setdefault("pod", "main")
            else:
                merged = merged or {"training": {}, "gpus": []}
                merged["gpus"].append(gpu)
                for t, x in d.get("training", {}).items():
                    merged["training"][t] = {**x, "pod": name}
                merged.setdefault("loracle", {}).update(d.get("loracle") or {})
        if merged and "generated_at" in merged:
            b = json.dumps(merged).encode()
            state.update(json=b, fetched=time.time())
            (HERE / "summary.json").write_bytes(b)
        state["error"] = "; ".join(errs) or None
        time.sleep(60)


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

    def do_GET(self):
        if self.path.startswith("/summary.json"):
            d = json.loads(state["json"]); d["_fetched"] = state["fetched"]; d["_error"] = state["error"]
            b = json.dumps(d).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(b))); self.end_headers()
            self.wfile.write(b)
        elif self.path in ("/", "/index.html"):
            b = (HERE / "index.html").read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(b))); self.end_headers()
            self.wfile.write(b)
        else:
            self.send_error(404)

    def log_message(self, *a):
        pass


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class S6(S):
    import socket
    address_family = socket.AF_INET6


if __name__ == "__main__":
    if (HERE / "summary.json").exists():
        state["json"] = (HERE / "summary.json").read_bytes()
    threading.Thread(target=poll, daemon=True).start()
    servers = [(S6 if ":" in a else S)((a, PORT), H) for a in ADDRS]
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    servers[0].serve_forever()
