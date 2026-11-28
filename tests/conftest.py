import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import pytest

from omarchy_cluster import discover, pins
from omarchy_cluster import joint_engine as je


@pytest.fixture(autouse=True)
def _isolated_trust_record(tmp_path_factory, monkeypatch):
    """No test may write this machine's real record of joined nodes."""
    trust = tmp_path_factory.mktemp("trust")
    monkeypatch.setattr(discover, "JOINED", str(trust / "joined"))
    monkeypatch.setattr(pins, "PINS", str(trust / "pins"))  # nor its pins, nor read this machine's own agent key
    monkeypatch.setattr(pins, "LOCAL_CERT_DIR", str(trust / "no-local-agent"))
    monkeypatch.setattr(discover, "LEFT", str(trust / "left"))


class FakeLlama:
    """A fake llama-server: /completion, /slots, /tokenize, /health.

    save writes `kv_bytes` into its slot dir under the requested filename;
    restore reads it back the same way the real endpoints do."""

    def __init__(self, slot_dir, text="decoded text", prompt_n=8192, predicted_n=64,
                 predicted_per_second=12.5, fail_completion=None, fail_slots=None):
        self.slot_dir = slot_dir
        self.text = text
        self.prompt_n = prompt_n
        self.predicted_n = predicted_n
        self.predicted_per_second = predicted_per_second
        self.fail_completion = fail_completion
        self.fail_slots = fail_slots or {}
        self.requests = []  # (path, parsed body) for assertions
        self.events = []  # ordered phase log: "completion", "save", "restore", "tokenize"


def _handler(fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._reply(200, {"status": "ok"})
            return self._reply(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._reply(400, {"error": "bad json"})
            fake.requests.append((self.path, req))
            parsed = urlparse(self.path)
            if parsed.path == "/completion":
                if fake.fail_completion:
                    return self._reply(fake.fail_completion, {"error": "boom"})
                fake.events.append("completion")
                if req.get("n_predict") == 0:
                    return self._reply(400, {"error": "n_predict must be > 0"})
                return self._reply(200, {
                    "content": fake.text, "stop_type": "limit",
                    "timings": {"prompt_n": fake.prompt_n, "prompt_ms": 900.0,
                                "predicted_n": fake.predicted_n, "predicted_ms": 5000.0,
                                "predicted_per_second": fake.predicted_per_second}})
            if parsed.path.startswith("/slots/"):
                action = parse_qs(parsed.query).get("action", [""])[0]
                if fake.fail_slots.get(action):
                    return self._reply(fake.fail_slots[action], {"error": "slots disabled"})
                fake.events.append(action)
                fname = req.get("filename", "")
                path = os.path.join(fake.slot_dir, fname)
                if action == "save":
                    data = b"KV-BYTES-%d" % fake.prompt_n
                    with open(path, "wb") as f:
                        f.write(data)
                    return self._reply(200, {"id_slot": 0, "filename": fname,
                                             "n_saved": fake.prompt_n + 1, "n_written": len(data),
                                             "timings": {"save_ms": 40.0}})
                if action == "restore":
                    if not os.path.exists(path):
                        return self._reply(400, {"error": "cannot open file"})
                    fake.events.append("restored:%d" % os.path.getsize(path))
                    return self._reply(200, {"id_slot": 0, "filename": fname,
                                             "n_restored": fake.prompt_n + 1,
                                             "n_read": os.path.getsize(path),
                                             "timings": {"restore_ms": 40.0}})
                if action == "erase":
                    return self._reply(200, {"id_slot": 0, "n_erased": 10})
            if parsed.path == "/tokenize":
                fake.events.append("tokenize")
                return self._reply(200, {"tokens": [11, 22, 33]})
            return self._reply(404, {"error": "not found"})

    return H


@pytest.fixture
def llama_pair(tmp_path):
    """Two fake llama-servers with their own slot dirs. Yields (prefill, decode)."""
    prefill = FakeLlama(str(tmp_path / "pf"))
    decode = FakeLlama(str(tmp_path / "dec"))
    servers = []
    for fake in (prefill, decode):
        os.makedirs(fake.slot_dir)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler(fake))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        fake.url = "http://127.0.0.1:%d" % srv.server_port
        servers.append(srv)
    yield prefill, decode
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def cfg_for(prefill, decode, **kw):
    cfg = {"prefill_url": prefill.url, "decode_url": decode.url,
           "prefill_kv": je.LocalKv(prefill.slot_dir),
           "decode_kv": je.LocalKv(decode.slot_dir),
           "slot": 0, "kv_name": "joint-kv.bin",
           "probes": {}}
    cfg.update(kw)
    return cfg
