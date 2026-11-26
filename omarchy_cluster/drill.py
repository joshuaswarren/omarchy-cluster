"""Node-loss and restart drill: kill a worker mid-request and prove the request
path fails cleanly — a bounded error, not a hang or a wrong answer — then
restart the worker (and optionally the gateway) and prove the replayed request
recovers with the same deterministic output.

Run via ``omarchy-cluster drill ...`` or directly as
``python3 -m omarchy_cluster.drill ...``.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from . import client
from . import token as cluster_token

DEFAULT_AGENT_PORT = 8025
_VOLATILE_RESPONSE_KEYS = frozenset({"id", "created", "timings", "system_fingerprint"})


def parse_node_address(addr, default_port=DEFAULT_AGENT_PORT):
    """Parse ``ip[:port]``, ``host[:port]``, or ``[ipv6][:port]`` into ``(host, port)``."""
    raw = (addr or "").strip()
    if not raw:
        raise ValueError("node address is empty")
    if raw.startswith("["):
        end = raw.find("]")
        if end < 0:
            raise ValueError("unclosed bracket in node address %r" % addr)
        host = raw[1:end].strip()
        rest = raw[end + 1:]
        if not host:
            raise ValueError("empty host in node address %r" % addr)
        if not rest:
            port = int(default_port)
        elif rest.startswith(":") and len(rest) > 1:
            port = int(rest[1:])
        else:
            raise ValueError("invalid node address %r" % addr)
    elif ":" not in raw:
        host, port = raw, int(default_port)
    elif raw.count(":") == 1:
        host, _, port_s = raw.partition(":")
        if not host or not port_s:
            raise ValueError("invalid node address %r" % addr)
        port = int(port_s)
    else:
        # Bare IPv6 address without brackets: validate so malformed strings still fail.
        ipaddress.ip_address(raw)
        host, port = raw, int(default_port)
    if not 1 <= port <= 65535:
        raise ValueError("port out of range: %d" % port)
    return host, port


def comparable_output(body):
    """Strip per-request volatile metadata from OpenAI-shaped gateway replies so
    baseline and replay outputs compare on their actual generated content."""
    if isinstance(body, dict) and "choices" in body:
        return {k: v for k, v in body.items() if k not in _VOLATILE_RESPONSE_KEYS}
    return body


def _timed(fn):
    """Run fn() and record the outcome: ok, reply-or-error, and wall seconds."""
    t0 = time.monotonic()
    try:
        body = fn()
    except Exception as e:  # noqa: BLE001 - the receipt records the failure mode
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                "elapsed_s": round(time.monotonic() - t0, 3)}
    return {"ok": True, "body": body, "elapsed_s": round(time.monotonic() - t0, 3)}


def _call_bounded(request, deadline_s):
    """request() in a thread joined at deadline_s. None when still running — the
    caller treats an unfinished request as the hang it is; the daemon thread
    dies with the process and never blocks a later step."""
    out = {}
    t = threading.Thread(target=lambda: out.update(_timed(request)), daemon=True)
    t.start()
    t.join(deadline_s)
    return None if t.is_alive() else out


def _poll_ready(probe, deadline_s, start_t):
    """Poll probe(remaining_s) until True or deadline_s elapses from start_t."""
    while time.monotonic() - start_t < deadline_s:
        remaining = max(0.0, deadline_s - (time.monotonic() - start_t))
        if probe(remaining):
            return True
        left = deadline_s - (time.monotonic() - start_t)
        if left <= 0:
            break
        time.sleep(min(0.2, left))
    return False


def run_drill(request, kill, restart=None, wait_ready=None, *,
              kill_after_s=1.0, error_deadline_s=10.0, ready_deadline_s=60.0,
              expect_same_output=True, restart_gateway=None, wait_gateway_ready=None):
    """Run the node-loss drill. Every hardware-facing action is a callable so
    the drill logic runs without a cluster:

      request()   one inference request; returns the reply body on success,
                  raises on failure. Called in a thread, so make one call at a
                  time only.
      kill()      kill the worker mid-request (the abrupt failure a crash makes).
      restart()   start the worker again; omit to skip the recovery leg.
      wait_ready(remaining_s) -> bool   the worker is accepting work again;
                  polled until ready_deadline_s.
      restart_gateway()  optional callable to restart the gateway/engine after
                  the worker is back (needed when llama-server must reload RPC).
      wait_gateway_ready(remaining_s) -> bool   optional gateway health probe
                  polled before replaying the request.

    Returns the receipt {"verdict": "pass"|"fail", ...}; a "fail" receipt names
    the step and reason.
    """
    steps = {}

    def fail(step, reason):
        return {"verdict": "fail", "step": step, "reason": reason, "steps": steps}

    # 1. the request path works before any fault.
    steps["baseline"] = _timed(request)
    if not steps["baseline"]["ok"]:
        return fail("baseline", "request failed before the drill: %s" % steps["baseline"]["error"])

    # 2. node loss: kill the worker while a request is in flight.
    inflight = {}
    finished = {}

    def _run_inflight():
        inflight.update(_timed(request))
        finished["at"] = time.monotonic()

    worker = threading.Thread(target=_run_inflight, daemon=True)
    worker.start()
    if kill_after_s > 0:
        time.sleep(kill_after_s)
    kill_started = time.monotonic()
    kill_res = _timed(kill)
    steps["kill"] = kill_res
    if not kill_res["ok"]:
        worker.join(min(error_deadline_s, 1.0))
        return fail("inflight_kill", "kill failed: %s" % kill_res["error"])
    kill_body = kill_res.get("body")
    if isinstance(kill_body, dict) and kill_body.get("still_listening"):
        worker.join(min(error_deadline_s, 1.0))
        return fail("inflight_kill",
                    "worker still listening after kill: %s" % kill_body["still_listening"])
    remaining = max(0.0, error_deadline_s - (time.monotonic() - kill_started))
    worker.join(remaining)
    if worker.is_alive():
        return fail("inflight_kill",
                    "request still running %.1fs after the worker was killed; the path hangs"
                    % error_deadline_s)
    after_kill_s = round(max(0.0, finished.get("at", time.monotonic()) - kill_started), 3)
    inflight["after_kill_s"] = after_kill_s
    steps["inflight_kill"] = inflight
    if steps["inflight_kill"]["ok"]:
        return fail("inflight_kill",
                    "request completed normally; either the kill missed or the request "
                    "finished first — raise max_tokens or lower kill_after_s")
    if after_kill_s > error_deadline_s:
        return fail("inflight_kill", "error took %.1fs after kill to surface; the path hangs"
                    % after_kill_s)

    # 3. the dead worker fails fast; nothing queues silently behind it.
    steps["dead_worker"] = _call_bounded(request, error_deadline_s)
    if steps["dead_worker"] is None:
        return fail("dead_worker", "request still running %.1fs on a dead worker"
                    % error_deadline_s)
    if steps["dead_worker"]["ok"]:
        return fail("dead_worker", "request succeeded with the worker dead")

    if restart is None:
        return {"verdict": "pass", "recovery": "skipped (no restart callable)", "steps": steps}

    # 4. restart; the worker must accept work again inside ready_deadline_s.
    t0 = time.monotonic()
    restart_res = _timed(restart)
    if not restart_res["ok"]:
        steps["restart"] = restart_res
        return fail("restart", "restart failed: %s" % restart_res["error"])
    steps["restart"] = {"ok": True, "elapsed_s": restart_res["elapsed_s"]}
    if restart_res.get("body") is not None:
        steps["restart"]["body"] = restart_res["body"]
    if wait_ready is not None:
        ready = _poll_ready(wait_ready, ready_deadline_s, t0)
        steps["restart"]["ready"] = ready
        steps["restart"]["elapsed_s"] = round(time.monotonic() - t0, 3)
        if not ready:
            return fail("restart", "worker not accepting work %.1fs after restart"
                        % ready_deadline_s)

    # 4b. optional gateway/engine restart and health readiness check.
    gw_t0 = time.monotonic()
    if restart_gateway is not None:
        gw_res = _timed(restart_gateway)
        if not gw_res["ok"]:
            steps["restart_gateway"] = gw_res
            return fail("restart_gateway", "gateway restart failed: %s" % gw_res["error"])
        steps["restart_gateway"] = {"ok": True, "elapsed_s": gw_res["elapsed_s"]}
        if gw_res.get("body") is not None:
            steps["restart_gateway"]["body"] = gw_res["body"]
    if wait_gateway_ready is not None:
        gw_ready = _poll_ready(wait_gateway_ready, ready_deadline_s, gw_t0)
        gw_step = steps.setdefault("restart_gateway", {})
        gw_step["ready"] = gw_ready
        gw_step["elapsed_s"] = round(time.monotonic() - gw_t0, 3)
        if not gw_ready:
            return fail("restart_gateway", "gateway not ready %.1fs after restart"
                        % ready_deadline_s)

    # 5. the replayed request succeeds, with the same deterministic output.
    steps["replay"] = _timed(request)
    if not steps["replay"]["ok"]:
        return fail("replay", "request failed after restart: %s" % steps["replay"]["error"])
    if expect_same_output and comparable_output(steps["replay"]["body"]) != comparable_output(steps["baseline"]["body"]):
        return fail("replay", "replayed request returned different output than baseline")

    return {"verdict": "pass", "steps": steps}


def _tcp_ready(host, port):
    """wait_ready probe: True once (host, port) accepts a TCP connect."""
    def probe(remaining_s):
        timeout = min(2.0, max(0.1, remaining_s))
        try:
            socket.create_connection((host, int(port)), timeout=timeout).close()
            return True
        except OSError:
            return False
    return probe


def _http_ready(url):
    """wait_gateway_ready probe: True once GET url answers HTTP 200."""
    def probe(remaining_s):
        timeout = min(2.0, max(0.1, remaining_s))
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.status == 200
        except OSError:
            return False
    return probe


def _read_token(token_file=None):
    if token_file:
        with open(os.path.expanduser(token_file)) as f:
            tok = f.read().strip()
        if not tok:
            raise ValueError("token file %s is empty" % token_file)
        return tok
    tok = cluster_token.read()
    if not tok:
        raise FileNotFoundError(
            "cluster token not found at ~/.config/omarchy-cluster/token "
            "(pass --token-file or run install-agent)")
    return tok


def add_drill_args(parser):
    """Attach the drill CLI flags to an argparse parser or subparser."""
    parser.add_argument(
        "--node", required=True,
        help="agent address of the node running the RPC worker, ip[:agent-port]")
    parser.add_argument(
        "--rpc-port", type=int, default=50060,
        help="port the RPC worker listens on (default %(default)s)")
    parser.add_argument(
        "--gateway", default="http://127.0.0.1:8020",
        help="serve gateway URL (default %(default)s)")
    parser.add_argument(
        "--model", default=None,
        help="optional model id sent in the completion request")
    parser.add_argument(
        "--prompt", default="Write one sentence about loss.",
        help="the request the drill sends")
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="keep the request in flight longer than --kill-after (default %(default)s)")
    parser.add_argument(
        "--kill-after", type=float, default=1.0,
        help="seconds to let the request run before the kill (default %(default)s)")
    parser.add_argument(
        "--error-deadline", type=float, default=10.0,
        help="max seconds a request may take to fail once the worker is dead (default %(default)s)")
    parser.add_argument(
        "--ready-deadline", type=float, default=60.0,
        help="max seconds to wait for the restarted worker and gateway (default %(default)s)")
    parser.add_argument(
        "--request-timeout", type=float, default=600.0,
        help="per-request HTTP timeout in seconds (default %(default)s)")
    parser.add_argument(
        "--rpc-body", default=None,
        help="JSON file with the /v1/rpc/start body to restart the worker "
             "(default: {\"port\": rpc-port})")
    parser.add_argument(
        "--restart-gateway-cmd", default=None,
        help="optional shell command run after the RPC worker restarts to restart "
             "the gateway/engine before polling gateway /health")
    parser.add_argument(
        "--no-restart", action="store_true",
        help="kill the worker and verify the clean error without restarting or replaying")
    parser.add_argument(
        "--token-file", default=None,
        help="cluster token file (default ~/.config/omarchy-cluster/token)")
    parser.add_argument(
        "--out", default=None,
        help="also write the receipt JSON here")
    return parser


def run_cli(args):
    """Execute the node-loss drill from parsed CLI args. Returns process exit code."""
    node_ip, agent_port = parse_node_address(args.node)
    url_host = "[%s]" % node_ip if ":" in node_ip else node_ip
    tok = _read_token(getattr(args, "token_file", None))
    gateway_base = args.gateway.rstrip("/")

    def agent_post(path, body, timeout=15):
        req = urllib.request.Request(
            client.agent_url(url_host, agent_port, path),
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-Cluster-Token": tok})
        with client.urlopen(req, timeout) as r:
            return json.loads(r.read())

    def request():
        payload = {"prompt": args.prompt, "max_tokens": args.max_tokens, "temperature": 0}
        if getattr(args, "model", None):
            payload["model"] = args.model
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + tok}
        req = urllib.request.Request(
            gateway_base + "/v1/completions",
            data=json.dumps(payload).encode(),
            headers=headers)
        with urllib.request.urlopen(req, timeout=getattr(args, "request_timeout", 600.0)) as r:
            return json.loads(r.read())

    start_body = {"port": args.rpc_port}
    if args.rpc_body:
        with open(os.path.expanduser(args.rpc_body)) as f:
            start_body = json.load(f)
        if isinstance(start_body, dict):
            start_body.setdefault("port", args.rpc_port)

    no_restart = bool(getattr(args, "no_restart", False))
    restart_fn = None if no_restart else (lambda: agent_post("/v1/rpc/start", start_body))
    wait_ready_fn = None if no_restart else _tcp_ready(node_ip, args.rpc_port)
    wait_gateway_fn = None if no_restart else _http_ready(gateway_base + "/health")

    restart_gateway_cmd = getattr(args, "restart_gateway_cmd", None)
    restart_gateway_fn = None
    if not no_restart and restart_gateway_cmd:
        def _restart_gateway():
            proc = subprocess.run(
                restart_gateway_cmd, shell=True, capture_output=True, text=True,
                timeout=args.ready_deadline)
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()[-400:]
                raise RuntimeError("command exited %d%s"
                                   % (proc.returncode, ": " + detail if detail else ""))
            return {"cmd": restart_gateway_cmd, "returncode": proc.returncode}
        restart_gateway_fn = _restart_gateway

    receipt = run_drill(
        request,
        kill=lambda: agent_post("/v1/rpc/stop", {"port": args.rpc_port}),
        restart=restart_fn,
        wait_ready=wait_ready_fn,
        kill_after_s=args.kill_after,
        error_deadline_s=args.error_deadline,
        ready_deadline_s=args.ready_deadline,
        restart_gateway=restart_gateway_fn,
        wait_gateway_ready=wait_gateway_fn)
    out = json.dumps(receipt, indent=2)
    if args.out:
        out_path = os.path.expanduser(args.out)
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(out + "\n")
    print(out)
    return 0 if receipt.get("verdict") == "pass" else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="omarchy-cluster drill",
        description="Node-loss and restart drill: SIGKILL a node's RPC worker mid-request, "
                    "require a bounded error, restart the worker, and replay the request.")
    add_drill_args(ap)
    return run_cli(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
