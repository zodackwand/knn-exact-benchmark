import os
import sys
import json
import threading
import queue
import subprocess
from pathlib import Path
from typing import Optional

from flask import Flask, Response, request, send_from_directory, jsonify
import yaml

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))

# Shared state
event_q: "queue.Queue[str]" = queue.Queue()
proc_lock = threading.Lock()
runner_thread: Optional[threading.Thread] = None
running_proc: Optional[subprocess.Popen] = None
progress = {
    "total_steps": 0,
    "seen_steps": 0,
    "current": {
        "algo": None,
        "dataset": None,
        "k": None,
    },
    "config_path": None,
    "run_dir": None,
    "status": "idle",  # idle | running | finished | error
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _compute_total_steps(cfg_path: Path) -> int:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    algorithms = cfg.get("algorithms", [])
    datasets = cfg.get("datasets", [])
    k_values = cfg.get("k_values", [10])
    # count only actual algorithm entries
    algo_count = 0
    for a in algorithms:
        if isinstance(a, dict) and a.get("name"):
            algo_count += 1
        elif isinstance(a, str) and a:
            algo_count += 1
    return max(1, len(datasets)) * max(1, algo_count) * max(1, len(k_values))


def _enqueue_event(payload: dict):
    try:
        event_q.put_nowait(json.dumps(payload))
    except Exception:
        pass


def _reader_thread(proc: subprocess.Popen):
    global running_proc
    try:
        for raw in iter(proc.stdout.readline, ""):
            if not raw:
                break
            line = raw.rstrip("\n")
            # push raw log event
            _enqueue_event({"type": "log", "line": line})
            # parse progress lines like: "[+] algo @ dataset metric=... k=..."
            if line.startswith("[+]") and "@" in line and "k=" in line:
                try:
                    # Example: "[+] ball_prune @ toy_gaussian... metric=l2 k=10"
                    body = line[4:]
                    left, right = body.split(" @ ", 1)
                    algo = left.strip()
                    rest = right.strip()
                    ds_part, k_part = rest.rsplit(" k=", 1)
                    dataset = ds_part.split(" metric=")[0].strip()
                    k_val = int(k_part.strip())
                    progress["current"] = {"algo": algo, "dataset": dataset, "k": k_val}
                    progress["seen_steps"] += 1
                    _enqueue_event({
                        "type": "progress",
                        "seen": progress["seen_steps"],
                        "total": progress["total_steps"],
                        "current": progress["current"],
                    })
                except Exception:
                    pass
        rc = proc.wait()
        progress["status"] = "finished" if rc == 0 else "error"
        _enqueue_event({"type": "status", "status": progress["status"], "returncode": rc})
    finally:
        with proc_lock:
            running_proc = None


def _start_run(config_path: str):
    global runner_thread, running_proc
    with proc_lock:
        if running_proc is not None:
            raise RuntimeError("A run is already in progress")
        cfg = Path(config_path)
        if not cfg.is_absolute():
            cfg = _project_root() / config_path
        if not cfg.exists():
            raise FileNotFoundError(f"Config not found: {cfg}")

        progress["config_path"] = str(cfg)
        progress["total_steps"] = _compute_total_steps(cfg)
        progress["seen_steps"] = 0
        progress["current"] = {"algo": None, "dataset": None, "k": None}
        progress["status"] = "running"

        # сразу оповестим клиентов о старте
        _enqueue_event({
            "type": "status",
            "status": progress["status"],
            "total": progress["total_steps"],
            "seen": progress["seen_steps"],
            "current": progress["current"],
            "config_path": progress["config_path"],
        })

        # launch bench.py as subprocess (unbuffered)
        python = sys.executable
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            [python, "-u", "bench.py", "--config", str(cfg)],
            cwd=str(_project_root()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        running_proc = proc
        runner_thread = threading.Thread(target=_reader_thread, args=(proc,), daemon=True)
        runner_thread.start()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)


@app.route("/run", methods=["POST"])
def run_bench():
    try:
        data = request.get_json(silent=True) or {}
        config_path = data.get("config_path", "configs/ball_prune_small.yaml")
        _start_run(config_path)
        return jsonify({"ok": True, "status": progress}), 202
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/stream")
def stream():
    def event_stream():
        # push initial status
        _enqueue_event({
            "type": "hello",
            "status": progress["status"],
            "seen": progress["seen_steps"],
            "total": progress["total_steps"],
            "current": progress["current"],
            "config_path": progress["config_path"],
        })
        while True:
            msg = event_q.get()
            yield f"data: {msg}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/health")
def health():
    return jsonify({"ok": True, "status": progress.get("status", "idle")})


if __name__ == "__main__":
    # Run: python -m webui.server
    print("[webui] Starting server on http://127.0.0.1:5000 …", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
