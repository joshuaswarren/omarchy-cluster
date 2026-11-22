"""Joint-request engine: prefill on one llama-server, KV handoff, decode on another.

The engine serves the same `POST /generate` contract as the llamacpp engine, so
the gateway fronts it unchanged. Behind it two already-running llama-servers,
each started with `--slot-save-path` pointed at that node's kv directory:

  1. the prefill server (e.g. Spark on CUDA) evaluates the prompt, warming
     slot KV — `n_predict 1`, the one token it decodes is past the prompt
     prefix and decode-side reuse drops it;
  2. `POST /slots/<id>?action=save` writes the slot's KV to its save dir;
  3. the file hops to the decode node's kv dir — a local copy when both
     servers share a host, else through the node agent's `/v1/kv` endpoint;
  4. `POST /slots/<id>?action=restore` loads it on the decode server;
  5. the decode request carries the same prompt: llama-server reuses the
     cached prefix and generates every token on the decode host.

A module lock serializes the whole handoff: two in-flight joint requests would
interleave save/restore on the same slot file. Both servers need the same model
and context, and exclusive use of the configured slot — run them `-np 1` or keep
other clients off that slot.

Every request replies with `telemetry`: the card's eight metrics — token digest,
TTFT, decode tok/s, total time, per-node memory, errors, LAN throughput for the
KV hop, and power (null: no agent power probe exists yet; sample out of band).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import client, facts

KV_DIR = os.path.expanduser("~/.local/state/omarchy-cluster/kv")
PREFILL_N_PREDICT = 1  # warm the slot; this one token is dropped by prefix reuse on decode
_CHUNK = 1 << 20

_REQUEST_LOCK = threading.Lock()


class LocalKv:
    """A kv directory on this host (the llama-server's --slot-save-path)."""

    def __init__(self, path):
        self.path = path


class AgentKv:
    """A kv directory on a cluster node, reached through its agent /v1/kv."""

    def __init__(self, ip, port):
        self.ip, self.port = ip, int(port)


class JointError(Exception):
    def __init__(self, phase, message):
        super().__init__(message)
        self.phase = phase


def _post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("error") or e.read().decode(errors="replace")
        except (ValueError, OSError):
            detail = str(e)
        raise JointError(url, "%s: %s" % (e.code, detail)) from None
    except OSError as e:
        raise JointError(url, str(e)) from None


def _agent_pull(ip, port, name, dest, timeout=600):
    """GET /v1/kv?name= on a node agent into dest. Returns bytes written."""
    url = client.agent_url(ip, port, "/v1/kv?name=" + urllib.parse.quote(name))
    req = urllib.request.Request(url, headers=client._headers())
    n = 0
    with client.urlopen(req, timeout) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(_CHUNK)
            if not chunk:
                break
            f.write(chunk)
            n += len(chunk)
    return n


def _agent_push(ip, port, name, src, timeout=600):
    """POST src's bytes to /v1/kv?name= on a node agent."""
    url = client.agent_url(ip, port, "/v1/kv?name=" + urllib.parse.quote(name))
    size = os.path.getsize(src)
    with open(src, "rb") as f:
        req = urllib.request.Request(url, data=f, method="POST",
                                     headers=client._headers({
                                         "Content-Type": "application/octet-stream",
                                         "Content-Length": str(size)}))
        with client.urlopen(req, timeout) as r:
            reply = json.loads(r.read())
    if reply.get("bytes") != size:
        raise JointError("transport", "agent wrote %s of %d bytes for %s"
                         % (reply.get("bytes"), size, name))
    return size


def transport_kv(name, src, dst, timeout=600):
    """Move slot file `name` from endpoint src to endpoint dst.

    Returns {"bytes", "seconds", "path"}: path is "loopback" for a same-host
    copy and "lan" when the file crossed an agent link (possibly twice)."""
    t0 = time.monotonic()
    if isinstance(src, LocalKv):
        src_path = os.path.join(src.path, name)
        if isinstance(dst, LocalKv):
            os.makedirs(dst.path, exist_ok=True)
            dst_path = os.path.join(dst.path, name)
            try:
                shutil.copyfile(src_path, dst_path)
            except OSError:
                if os.path.exists(dst_path):
                    os.remove(dst_path)  # a partial copy is not a slot file
                raise
            n = os.path.getsize(src_path)
            path = "loopback"
        else:
            n = _agent_push(dst.ip, dst.port, name, src_path, timeout)
            path = "lan"
    else:
        with tempfile.NamedTemporaryFile(prefix="joint-kv-", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            n = _agent_pull(src.ip, src.port, name, tmp_path, timeout)
            if isinstance(dst, LocalKv):
                os.makedirs(dst.path, exist_ok=True)
                shutil.move(tmp_path, os.path.join(dst.path, name))
            else:
                n = _agent_push(dst.ip, dst.port, name, tmp_path, timeout)
            path = "lan"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return {"bytes": n, "seconds": max(time.monotonic() - t0, 1e-6), "path": path}


def _sampling(req):
    """llama-server sampling fields: greedy unless the request carries `sampling`
    (same rule as the llamacpp engine)."""
    sampling = {"temperature": 0, **(req.get("sampling") or {})}
    if req.get("sampling"):
        sampling = {"top_p": 1.0, "top_k": 0, "min_p": 0.0, **sampling}
    return sampling


def _new_telemetry():
    return {"token_ids_sha256": None, "ttft_ms": None, "decode_tokps": None,
            "total_s": None, "memory": {"prefill": None, "decode": None},
            "errors": [], "lan_mbps": None, "transport": None, "kv_bytes": 0,
            "power_watts": None, "power_source": "none",
            "phases": {}}


def joint_generate(cfg, req, transport=None):
    """One joint request: prefill -> slot save -> KV transport -> restore -> decode.

    cfg: prefill_url, decode_url, prefill_kv (LocalKv|AgentKv), decode_kv,
    slot, kv_name, probes ({side: callable -> dict|None} for the memory metric).
    Always returns a dict with `telemetry`; failures carry `error` too.
    The slot file and both servers' slot state are shared, so the whole
    handoff runs under a module lock."""
    transport = transport or transport_kv
    with _REQUEST_LOCK:
        return _joint_generate_locked(cfg, req, transport)


def _joint_generate_locked(cfg, req, transport):
    tel = _new_telemetry()
    t0 = time.perf_counter()
    slot, kv_name = cfg["slot"], cfg["kv_name"]
    prompt = req.get("prompt") or ""
    max_tokens = int(req.get("max_tokens", 64))
    try:
        # Phase 1 — prefill on the prefill server (one token warms the slot).
        pf = _post(cfg["prefill_url"] + "/completion",
                   {"prompt": prompt, "n_predict": PREFILL_N_PREDICT, "id_slot": slot,
                    "cache_prompt": True, **_sampling(req)})
        if pf.get("error"):
            raise JointError("prefill", str(pf["error"]))
        pft = pf.get("timings") or {}
        tel["phases"]["prefill_ms"] = float(pft.get("prompt_ms", 0.0))
        tel["phases"]["prefill_eval_n"] = int(pft.get("prompt_n", 0))

        # Phase 2 — serialize the slot's KV on the prefill host.
        try:
            saved = _post(cfg["prefill_url"] + "/slots/%d?action=save" % slot,
                          {"filename": kv_name})
        except JointError as e:
            raise JointError("save", "%s (the prefill llama-server needs "
                                     "--slot-save-path for slot save/restore)" % e)
        if saved.get("error"):
            raise JointError("save", str(saved["error"]))
        tel["kv_bytes"] = int(saved.get("n_written", 0))
        tel["phases"]["save_ms"] = float((saved.get("timings") or {}).get("save_ms", 0.0))

        # Phase 3 — move the file to the decode host's kv dir.
        try:
            tr = transport(kv_name, cfg["prefill_kv"], cfg["decode_kv"])
        except Exception as e:
            raise JointError("transport", str(e)) from e
        tel["transport"] = tr["path"]
        tel["phases"]["transfer_ms"] = tr["seconds"] * 1e3
        if tr["path"] == "lan" and tr["seconds"] > 0:
            tel["lan_mbps"] = round(tr["bytes"] * 8 / 1e6 / tr["seconds"], 2)
        else:
            tel["lan_mbps"] = None

        # Phase 4 — load the KV on the decode server.
        try:
            restored = _post(cfg["decode_url"] + "/slots/%d?action=restore" % slot,
                             {"filename": kv_name})
        except JointError as e:
            raise JointError("restore", "%s (the decode llama-server needs "
                                        "--slot-save-path for slot save/restore)" % e)
        if restored.get("error"):
            raise JointError("restore", str(restored["error"]))
        tel["phases"]["restore_ms"] = float((restored.get("timings") or {}).get("restore_ms", 0.0))

        # Phase 5 — decode on the decode host; the restored prefix makes its
        # prompt evaluation free, so every generated token is computed there.
        dec = _post(cfg["decode_url"] + "/completion",
                    {"prompt": prompt, "n_predict": max_tokens, "id_slot": slot,
                     "cache_prompt": True, **_sampling(req)})
        if dec.get("error"):
            raise JointError("decode", str(dec["error"]))
        dt = dec.get("timings") or {}
        tel["phases"]["decode_ms"] = float(dt.get("predicted_ms", 0.0))
        tel["phases"]["decode_prompt_n"] = int(dt.get("prompt_n", 0))
        tokps = dt.get("predicted_per_second")
        tel["decode_tokps"] = float(tokps) if tokps is not None else None
        text = dec.get("content") or ""

        # Metrics that need one more round-trip: token digest + memory probes.
        try:
            ids = _post(cfg["decode_url"] + "/tokenize", {"content": text},
                        timeout=60).get("tokens") or []
            tel["token_ids_sha256"] = hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        except JointError as e:
            tel["errors"].append({"phase": "tokenize", "message": str(e)})
        for side in ("prefill", "decode"):
            probe = (cfg.get("probes") or {}).get(side)
            if probe is not None:
                try:
                    tel["memory"][side] = probe()
                except Exception as e:  # noqa: BLE001 - a failed probe is a missing metric
                    tel["errors"].append({"phase": "probe-" + side, "message": str(e)})
        elapsed = time.perf_counter() - t0
        tel["ttft_ms"] = round(elapsed * 1e3, 1)
        tel["total_s"] = round(elapsed, 3)
        finish = "length" if dec.get("stop_type") == "limit" else "stop"
        return {"text": text,
                "prompt_tokens": int(dt.get("prompt_n", 0)) + int(pft.get("prompt_n", 0)),
                "completion_tokens": int(dt.get("predicted_n", 0)),
                "prefill_ms": tel["phases"]["prefill_ms"],
                "decode_ms": tel["phases"]["decode_ms"],
                "tokps": tel["decode_tokps"] or 0.0, "finish_reason": finish,
                "telemetry": tel}
    except JointError as e:
        tel["errors"].append({"phase": e.phase, "message": str(e)})
        tel["total_s"] = round(time.perf_counter() - t0, 3)
        return {"error": str(e), "telemetry": tel}


def healthy(cfg, timeout=2.0):
    """Both llama-servers answer /health."""
    for url in (cfg["prefill_url"], cfg["decode_url"]):
        try:
            with urllib.request.urlopen(url + "/health", timeout=timeout) as r:
                if r.status != 200:
                    return False
        except OSError:
            return False
    return True


def kv_endpoint(value, default_dir=None):
    """`local`, `local=DIR`, `NAME`/`HOST:PORT` handled by the caller's resolver:
    this parses `local[=DIR]` and `HOST:PORT` only."""
    if value == "local" or value.startswith("local="):
        path = value.partition("=")[2] or default_dir or KV_DIR
        return LocalKv(os.path.expanduser(path))
    host, sep, port = value.rpartition(":")
    if not sep or not host or not port.isdigit():
        raise ValueError("%s: expected local[=DIR] or HOST:PORT" % value)
    return AgentKv(host, int(port))


def default_probe(endpoint):
    """The memory metric for a kv endpoint: agent facts for a remote node,
    local facts here. None for the unmeasurable case."""
    if isinstance(endpoint, AgentKv):
        def remote():
            f = client.fetch_facts(endpoint.ip, endpoint.port, timeout=5)
            return {"ram_free_bytes": f.get("memory_free_bytes"),
                    "gpu_backend": f.get("gpu_backend")}
        return remote

    def local():
        return {"ram_free_bytes": facts.memory_free_bytes()}
    return local


class JointHandler(BaseHTTPRequestHandler):
    cfg = None

    def log_message(self, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200 if healthy(self.cfg) else 503,
                              {"ok": healthy(self.cfg)})
        if self.path == "/status":
            return self._json(200, {"engine": "joint",
                                    "prefill": self.cfg["prefill_url"],
                                    "decode": self.cfg["decode_url"],
                                    "slot": self.cfg["slot"],
                                    "model": self.cfg.get("model")})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            return self._json(400, {"error": str(e)})
        try:
            return self._json(200, joint_generate(self.cfg, req))
        except Exception as e:  # noqa: BLE001 - the gateway relays it as 502
            return self._json(200, {"error": "joint engine: %s" % e})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omarchy_cluster.joint_engine")
    ap.add_argument("--prefill-url", required=True, help="prefill llama-server base URL")
    ap.add_argument("--decode-url", required=True, help="decode llama-server base URL")
    ap.add_argument("--prefill-kv", required=True,
                    help="where the prefill server's --slot-save-path lives: "
                         "local[=DIR] or HOST:PORT (the node's agent)")
    ap.add_argument("--decode-kv", default="local",
                    help="same for the decode server (default: local kv dir)")
    ap.add_argument("--kv-dir", default=KV_DIR,
                    help="this host's slot-save dir for a 'local' kv endpoint")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--kv-name", default="joint-kv.bin")
    ap.add_argument("--port", type=int, default=8031)
    ap.add_argument("--model", default=None, help="label reported on /status")
    args = ap.parse_args(argv)
    cfg = {"prefill_url": args.prefill_url.rstrip("/"),
           "decode_url": args.decode_url.rstrip("/"),
           "prefill_kv": kv_endpoint(args.prefill_kv, args.kv_dir),
           "decode_kv": kv_endpoint(args.decode_kv, args.kv_dir),
           "slot": args.slot, "kv_name": args.kv_name, "model": args.model,
           "probes": {}}
    cfg["probes"] = {"prefill": default_probe(cfg["prefill_kv"]),
                     "decode": default_probe(cfg["decode_kv"])}
    JointHandler.cfg = cfg
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), JointHandler)
    srv.daemon_threads = True
    print("joint engine on :%d (prefill %s -> decode %s, slot %d)"
          % (args.port, cfg["prefill_url"], cfg["decode_url"], cfg["slot"]), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
