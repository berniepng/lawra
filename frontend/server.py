#!/usr/bin/env python3
"""
Lawra frontend server
======================
Replaces `python3 -m http.server 3000`.
Serves the frontend AND exposes two API endpoints:

  POST /api/run-eval   { "mode": "quick" | "full" }
       → starts the eval script in a background thread
       → returns 202 immediately (eval runs asynchronously)

  GET  /api/eval-status
       → returns { state, mode, started_at, finished_at, error, progress }

The UI polls /api/eval-status every 3s while an eval is running,
then re-fetches /eval_results.json when state becomes "done".

Usage:
    cd frontend
    python server.py          # default port 3000
    python server.py --port 8080
"""

import argparse
import http.server
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

FRONTEND_DIR = Path(__file__).parent
EVAL_SCRIPT  = FRONTEND_DIR.parent / "eval" / "lawra_eval.py"
JUDGE_MODEL  = "gemma4:e2b"   # change here to swap the Ragas evaluation judge
N8N_URL      = os.environ.get("N8N_URL", "http://localhost:5678/webhook/lawra/query")

# Working memory (ChromaDB) — lazy-loaded so server starts even if not installed
_memory = None
def _get_memory():
    global _memory
    if _memory is None:
        try:
            sys.path.insert(0, str(FRONTEND_DIR))
            from memory import get_memory
            _memory = get_memory()
        except Exception as e:
            print(f"⚠️  Memory unavailable: {e}")
            _memory = False   # False = tried and failed, don't retry
    return _memory if _memory else None

# ── Shared eval state ─────────────────────────────────────────────────────────
_lock   = threading.Lock()
_status = {
    "state":       "idle",    # idle | running | done | error
    "mode":        None,
    "started_at":  None,
    "finished_at": None,
    "error":       None,
    "log_tail":    [],        # last few lines of stdout for progress display
}
_log_lines: list[str] = []


def _run_eval_background(mode: str) -> None:
    """Run the eval script in a background thread, capture output."""
    cmd = [sys.executable, str(EVAL_SCRIPT), "--judge-model", JUDGE_MODEL]
    if mode == "quick":
        cmd.append("--quick")

    global _log_lines
    _log_lines = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            _log_lines.append(line)
            _log_lines = _log_lines[-30:]          # keep last 30 lines
            with _lock:
                _status["log_tail"] = list(_log_lines)
            print(line, flush=True)                # mirror to terminal
        proc.wait()
        with _lock:
            if proc.returncode == 0:
                _status["state"] = "done"
            else:
                _status["state"] = "error"
                _status["error"] = f"Exit code {proc.returncode}"
    except Exception as exc:
        with _lock:
            _status["state"] = "error"
            _status["error"] = str(exc)
    finally:
        with _lock:
            _status["finished_at"] = datetime.now(timezone.utc).isoformat()


# ── HTTP handler ──────────────────────────────────────────────────────────────
class LawraHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    # ── CORS preflight ──────────────────────────────────────────────────────
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    # ── API POST ────────────────────────────────────────────────────────────
    def do_POST(self):
        if self.path.startswith("/api/run-eval"):
            self._handle_run_eval()
        elif self.path.startswith("/api/query"):
            self._handle_query()
        elif self.path.startswith("/api/memory/clear"):
            self._handle_memory_clear()
        else:
            self.send_error(404, "Not found")

    # ── Static + API GET ────────────────────────────────────────────────────
    def do_GET(self):
        if self.path.startswith("/api/eval-status"):
            with _lock:
                self._send_json(200, dict(_status))
        elif self.path.startswith("/api/memory/status"):
            mem = _get_memory()
            self._send_json(200, {"available": bool(mem)})
        else:
            super().do_GET()

    # ── Handlers ────────────────────────────────────────────────────────────
    def _handle_query(self):
        """Memory-augmented query proxy: ChromaDB → n8n → ChromaDB → respond."""
        if not _HAS_REQUESTS:
            self._send_json(500, {"error": "pip install requests"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length)) if length else {}
        query  = body.get("query", "").strip()
        session_id = body.get("session_id", "default")

        if not query:
            self._send_json(400, {"error": "query is required"})
            return

        # 1. Search working memory
        mem            = _get_memory()
        memory_context = ""
        memory_hits    = []
        if mem:
            try:
                exchanges = mem.search(session_id, query)
                if exchanges:
                    memory_context = mem.format_context(exchanges)
                    memory_hits    = exchanges
            except Exception as e:
                print(f"Memory search error: {e}")

        # 2. Call n8n with query + memory context
        try:
            r = _requests.post(
                N8N_URL,
                json={"query": query, "memory_context": memory_context},
                timeout=180,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            self._send_json(502, {"error": f"n8n error: {e}"})
            return

        # 3. Store exchange in working memory
        if mem:
            try:
                mem.store(
                    session_id=session_id,
                    query=query,
                    answer=data.get("answer", ""),
                    sources=data.get("sources", []),
                )
            except Exception as e:
                print(f"Memory store error: {e}")

        # 4. Enrich response with memory metadata
        data["memory"] = {
            "session_id":   session_id,
            "hits":         len(memory_hits),
            "context_used": bool(memory_context),
            "exchanges":    memory_hits,
        }
        self._send_json(200, data)

    def _handle_memory_clear(self):
        """Clear all memory for a session."""
        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length)) if length else {}
        session_id = body.get("session_id", "default")
        mem = _get_memory()
        if not mem:
            self._send_json(503, {"error": "Memory not available"})
            return
        deleted = mem.clear_session(session_id)
        self._send_json(200, {"deleted": deleted, "session_id": session_id})

    def _handle_run_eval(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length)) if length else {}
        mode   = body.get("mode", "quick")

        if mode not in ("quick", "full"):
            self._send_json(400, {"error": "mode must be 'quick' or 'full'"})
            return

        with _lock:
            if _status["state"] == "running":
                self._send_json(409, {"error": "Evaluation already running"})
                return
            _status.update({
                "state":       "running",
                "mode":        mode,
                "started_at":  datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "error":       None,
                "log_tail":    [],
            })

        thread = threading.Thread(
            target=_run_eval_background, args=(mode,), daemon=True)
        thread.start()
        self._send_json(202, {"status": "started", "mode": mode})

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt, *args):
        # Only log API calls and errors to keep terminal readable
        msg = fmt % args if args else fmt
        if "/api/" in msg or "Error" in msg or "error" in msg:
            super().log_message(fmt, *args)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Lawra frontend server")
    ap.add_argument("--port", type=int, default=3000)
    args = ap.parse_args()

    if not EVAL_SCRIPT.exists():
        print(f"⚠️  Eval script not found at {EVAL_SCRIPT}")
        print("   Make sure eval/lawra_eval.py exists in the project root.")

    print(f"Lawra frontend  →  http://localhost:{args.port}")
    print(f"Eval script     →  {EVAL_SCRIPT}")
    print(f"Serving files   →  {FRONTEND_DIR}")
    print()

    with http.server.ThreadingHTTPServer(("", args.port), LawraHandler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
