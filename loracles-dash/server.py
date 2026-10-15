"""LoRAcles results dashboard + multi-model chat. http://bramble:8798/

Run via ./run.sh (torch-rocm venv). Env:
  DASH_MODEL   base model (default Qwen/Qwen3-14B)
  DASH_DEVICE  cuda | cpu
  DASH_PORT    8798
"""
import asyncio, glob, json, os, shutil, threading, time
from pathlib import Path

import torch
from aiohttp import web

import registry
from engine import BASE, Engine

HERE = Path(__file__).parent
DATA = HERE / "data"
PORT = int(os.environ.get("DASH_PORT", 8798))
ADDRS = os.environ.get("DASH_ADDRS", "100.86.34.70,fd7a:115c:a1e0::e934:2247").split(",")
MODEL = os.environ.get("DASH_MODEL", "Qwen/Qwen3-14B")
DEVICE = os.environ.get("DASH_DEVICE", "cuda")
HF = Path.home() / ".cache/huggingface/hub"
QWEN14_BYTES = 29_540_000_000  # approx. total size of the 8 bf16 shards

REG = registry.scan()
ENGINE = Engine(MODEL, REG, device=DEVICE, dtype=torch.bfloat16 if DEVICE != "cpu" else torch.float32)


# ---------------- model availability ----------------
def model_files_ready():
    d = HF / ("models--" + MODEL.replace("/", "--"))
    snaps = glob.glob(str(d / "snapshots/*"))
    if not snaps:
        return False, 0
    blob_bytes = sum(f.stat().st_size for f in (d / "blobs").glob("*")) if (d / "blobs").exists() else 0
    idx = Path(snaps[0]) / "model.safetensors.index.json"
    if idx.exists():
        shards = set(json.load(open(idx))["weight_map"].values())
        ok = all((Path(snaps[0]) / s).exists() for s in shards)
    else:
        ok = (Path(snaps[0]) / "model.safetensors").exists()
    return ok, blob_bytes  # snapshot symlinks only appear once a shard is complete


IDLE_UNLOAD_S = int(os.environ.get("DASH_IDLE_UNLOAD_S", 20 * 60))


def loader():
    """Wait for the download, then sit 'asleep': the model loads on the first request and
    unloads after IDLE_UNLOAD_S without use, so the GPU is free for other jobs on bramble."""
    while True:
        ok, _ = model_files_ready()
        if ok:
            break
        ENGINE.state = "downloading"
        time.sleep(30)
    ENGINE.state = "asleep"
    while True:
        time.sleep(30)
        if (ENGINE.model is not None and ENGINE.queue_len() == 0
                and time.time() - ENGINE.last_used > IDLE_UNLOAD_S):
            ENGINE.submit(ENGINE.unload).result()


# ---------------- results ----------------
def load_results():
    out = {"generated_at": time.time(), "sources": {}}
    # live v1 snapshot from the training dashboard (read-only); keep our last good copy
    live = Path.home() / "workspace/subliminal_pilot/dashboard/summary.json"
    snap = DATA / "v1_summary.json"
    try:
        if live.exists() and (not snap.exists() or live.stat().st_mtime > snap.stat().st_mtime):
            json.loads(live.read_text())
            shutil.copy(live, snap)
    except Exception:
        pass
    if snap.exists():
        out["v1"] = json.loads(snap.read_text())
        out["sources"]["v1"] = snap.stat().st_mtime
    for f in sorted(DATA.glob("*.json")):
        if f.name in ("v1_summary.json", "extra_adapters.json", "hf_adapters.json") or f.name.endswith("_answers.json"):
            continue
        try:
            out[f.stem] = json.loads(f.read_text())
            out["sources"][f.stem] = f.stat().st_mtime
        except Exception as e:
            out[f.stem] = {"_error": repr(e)}
    out["md"] = {f.stem: f.read_text() for f in sorted(DATA.glob("*.md"))}
    out["md_mtime"] = {f.stem: f.stat().st_mtime for f in sorted(DATA.glob("*.md"))}
    return out


# ---------------- handlers ----------------
async def index(req):
    return web.FileResponse(HERE / "static/index.html", headers={"Cache-Control": "no-store"})


async def results(req):
    d = await asyncio.to_thread(load_results)
    return web.json_response(d, headers={"Cache-Control": "no-store"})


async def data_file(req):
    """Large raw data files (e.g. *_answers.json) the results JSON leaves out; fetched by the page on demand."""
    name = req.match_info["name"]
    f = DATA / name
    if "/" in name or not name.endswith(".json") or not f.exists():
        raise web.HTTPNotFound()
    return web.FileResponse(f, headers={"Cache-Control": "no-cache"})


async def jtok_file(req):
    """Per-layer J-lens token readouts written by jlens/decode_tokens.py."""
    name = req.match_info["name"]
    f = DATA / "jtok" / name
    if "/" in name or not name.endswith(".json") or not f.exists():
        raise web.HTTPNotFound()
    return web.FileResponse(f, headers={"Cache-Control": "no-cache"})


async def models(req):
    global REG
    if req.query.get("rescan"):
        REG = registry.scan()
        ENGINE.registry = REG
    ok, b = model_files_ready()
    return web.json_response({
        "base": MODEL, "device": DEVICE, "state": ENGINE.state, "idle_unload_s": IDLE_UNLOAD_S, "error": ENGINE.error,
        "download": {"bytes": b, "total": QWEN14_BYTES if "14B" in MODEL else None, "ready": ok},
        "queue": ENGINE.queue_len(), "busy": ENGINE.busy, "stats": ENGINE.stats,
        "loaded": list(ENGINE.loaded), "loracle_loaded": ENGINE.loracle_ready,
        "models": [{"name": BASE, "label": f"{MODEL.split('/')[-1]} (no LoRA)", "group": "base"}] +
                  [{"name": k, **{x: y for x, y in v.items() if x not in ("path", "tokens", "hf")},
                   "local": bool(v.get("path")), "hf": bool(v.get("hf")), "has_tokens": bool(v.get("tokens")),
                   "weights": bool(v.get("path") or v.get("hf"))} for k, v in REG.items()],
    })


def _check_ready():
    if ENGINE.state not in ("ready", "asleep", "loading"):
        raise web.HTTPServiceUnavailable(text=json.dumps({"error": f"model {ENGINE.state}"}),
                                         content_type="application/json")


async def _sse(req, fn, *a, **kw):
    """Run fn(*a, on_step=, cancel=, **kw) on the GPU thread and stream {row, delta|done} events."""
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-store",
                                       "X-Accel-Buffering": "no"})
    await resp.prepare(req)
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()
    cancel = threading.Event()

    def on_step(i, d):
        loop.call_soon_threadsafe(q.put_nowait, {"row": i, "delta": d} if d is not None else {"row": i, "done": True})

    await resp.write(f"data: {json.dumps({'queued': ENGINE.queue_len(), 'state': ENGINE.state})}\n\n".encode())
    afut = asyncio.wrap_future(ENGINE.submit(fn, *a, on_step=on_step, cancel=cancel, **kw))
    try:
        while True:
            get = asyncio.ensure_future(q.get())
            done, _ = await asyncio.wait({get, afut}, return_when=asyncio.FIRST_COMPLETED)
            if get in done:
                await resp.write(f"data: {json.dumps(get.result())}\n\n".encode())
                continue
            get.cancel()
            while not q.empty():
                await resp.write(f"data: {json.dumps(q.get_nowait())}\n\n".encode())
            try:
                r = afut.result()
                await resp.write(f"data: {json.dumps({'final': True, 'seconds': r.get('seconds'), 'steps_per_s': r.get('steps_per_s')})}\n\n".encode())
            except Exception as e:
                await resp.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode())
            break
    except (ConnectionResetError, asyncio.CancelledError):
        cancel.set()
        raise
    finally:
        cancel.set()
    return resp


async def chat(req):
    """SSE stream. body: {rows:[{model, messages}|{model, raw, prompt}], max_new_tokens, temperature, top_p, thinking}"""
    _check_ready()
    body = await req.json()
    rows = body["rows"][:8]
    for r in rows:
        if r["model"] != BASE and r["model"] not in REG:
            raise web.HTTPBadRequest(text=f"unknown model {r['model']}")
    return await _sse(req, ENGINE.generate, rows,
                      max_new_tokens=min(int(body.get("max_new_tokens", 300)), 2048),
                      temperature=float(body.get("temperature", 0.7)), top_p=float(body.get("top_p", 0.95)),
                      thinking=bool(body.get("thinking", False)), seed=body.get("seed"))


async def loracle_chat(req):
    """SSE stream. body: {target, messages, buggy_v, n, temperature, max_new_tokens}"""
    _check_ready()
    b = await req.json()
    if b["target"] not in REG:
        raise web.HTTPBadRequest(text=f"unknown LoRA {b['target']}")
    return await _sse(req, ENGINE.loracle_generate, b["target"], b["messages"], buggy_v=bool(b.get("buggy_v", False)),
                      n=max(1, min(int(b.get("n", 1)), 4)), temperature=float(b.get("temperature", 0.5)),
                      max_new_tokens=min(int(b.get("max_new_tokens", 300)), 1024))


async def score(req):
    """body: {models:[...], messages:[...], candidates:[...], prefix?}"""
    _check_ready()
    b = await req.json()
    fut = ENGINE.submit(ENGINE.score, b["models"][:12], b["messages"], b["candidates"][:40], b.get("prefix", ""))
    try:
        return web.json_response(await asyncio.wrap_future(fut))
    except Exception as e:
        return web.json_response({"error": repr(e)}, status=500)


async def sample(req):
    """Short-answer sampling tally. body: {models, messages, n, max_new_tokens, temperature}"""
    _check_ready()
    b = await req.json()
    n = min(int(b.get("n", 16)), 64)
    rows = [{"model": m, "messages": b["messages"]} for m in b["models"][:8] for _ in range(n)]
    fut = ENGINE.submit(ENGINE.generate, rows, max_new_tokens=int(b.get("max_new_tokens", 8)),
                        temperature=float(b.get("temperature", 1.0)), top_p=1.0)
    try:
        r = await asyncio.wrap_future(fut)
    except Exception as e:
        return web.json_response({"error": repr(e)}, status=500)
    out = {}
    for i, t in enumerate(r["texts"]):
        out.setdefault(rows[i]["model"], []).append(t)
    return web.json_response({"samples": out, "seconds": r["seconds"]})


def main():
    threading.Thread(target=loader, daemon=True).start()
    app = web.Application(client_max_size=4 * 2**20)
    app.add_routes([web.get("/", index), web.get("/api/results", results), web.get("/api/models", models), web.get("/api/data/{name}", data_file), web.get("/api/jtok/{name}", jtok_file),
                    web.post("/api/chat", chat), web.post("/api/loracle/chat", loracle_chat), web.post("/api/score", score), web.post("/api/sample", sample),
                    web.static("/static", HERE / "static")])
    runner = web.AppRunner(app, access_log=None)

    async def go():
        await runner.setup()
        for a in ADDRS:
            await web.TCPSite(runner, a, PORT).start()
        print(f"[dash] serving on {ADDRS} :{PORT}", flush=True)
        await asyncio.Event().wait()

    asyncio.run(go())


if __name__ == "__main__":
    main()
