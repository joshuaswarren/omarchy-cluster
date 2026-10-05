"""omarchy-cluster agent: HTTP facts + measurement endpoints, 1 Hz heartbeat, TCP sink, UDP echo."""
from __future__ import annotations

import argparse
import hmac
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import AGENT_VERSION
from . import facts as facts_mod
from .discover import advertise_start

DEFAULT_HTTP_PORT = 8025
DEFAULT_SINK_PORT = 8026
DEFAULT_UDP_PORT = 8027
DEFAULT_IPERF_PORT = 5210


def median(xs):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[len(xs) // 2]


class Heartbeat:
    """1 Hz tick; seq/age tell consumers the agent loop is alive."""

    def __init__(self):
        self.seq = 0
        self.last = time.monotonic()
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.wait(1.0):
            self.seq += 1
            self.last = time.monotonic()

    def age(self):
        return time.monotonic() - self.last


class AgentState:
    """Facts cache (collect_facts shells out; keep it to 2 s freshness)."""

    def __init__(self, hb):
        self.hb = hb
        self._facts = None
        self._at = 0.0
        self._lock = threading.Lock()

    def facts(self):
        with self._lock:
            if self._facts is None or time.monotonic() - self._at > 2.0:
                self._facts = facts_mod.collect_facts()
                self._at = time.monotonic()
            return dict(self._facts)


# ---- measurement primitives (used by agent endpoints and unit-testable) ----

def tcp_rtt(host, port=DEFAULT_HTTP_PORT, bind_ip=None, count=3, timeout=2.0):
    times = []
    for _ in range(count):
        s = socket.socket()
        s.settimeout(timeout)
        try:
            if bind_ip:
                s.bind((bind_ip, 0))
            t0 = time.perf_counter()
            s.connect((host, port))
            times.append((time.perf_counter() - t0) * 1000.0)
        except OSError:
            pass
        finally:
            s.close()
    return median(times)


def udp_rtt(host, port=DEFAULT_UDP_PORT, bind_ip=None, count=5, timeout=1.0):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        if bind_ip:
            s.bind((bind_ip, 0))
    except OSError:
        s.close()
        return None
    times = []
    try:
        for _ in range(count):
            t0 = time.perf_counter()
            s.sendto(b"ping", (host, port))
            try:
                s.recvfrom(2048)
                times.append((time.perf_counter() - t0) * 1000.0)
            except socket.timeout:
                pass
    finally:
        s.close()
    return median(times)


def blast(host, port=DEFAULT_SINK_PORT, seconds=10.0, bind_ip=None, chunk=1 << 20):
    """Send zeros to a sink for `seconds`; return bytes handed to the socket."""
    s = socket.socket()
    s.settimeout(seconds + 5.0)
    try:
        if bind_ip:
            s.bind((bind_ip, 0))
        s.connect((host, port))
    except OSError:
        s.close()
        return 0
    buf = bytes(chunk)
    sent = 0
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            sent += s.send(buf)
    except OSError:
        pass
    finally:
        s.close()
    return sent


def sink_serve(port=DEFAULT_SINK_PORT, seconds=10.0):
    """Accept one connection, discard bytes until EOF or deadline.

    Returns (received_bytes, elapsed_seconds)."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.settimeout(seconds + 5.0)
    try:
        srv.bind(("", port))
        srv.listen(1)
        t0 = time.monotonic()
        conn, _ = srv.accept()
        conn.settimeout(2.0)
        total = 0
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                chunk = conn.recv(1 << 20)
            except socket.timeout:
                continue
            if not chunk:
                break
            total += len(chunk)
        return total, time.monotonic() - t0
    finally:
        srv.close()


def iperf_client(dst_ip, bind_ip, seconds, port=DEFAULT_IPERF_PORT):
    cmd = ["iperf3", "-c", dst_ip, "-B", bind_ip, "-t", str(int(seconds)),
           "-1", "--connect-timeout", "2000", "-J", "-P", "1"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 25)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    try:
        j = json.loads(p.stdout)
        end = j["end"]
        return {
            "gbps_sent": end["sum_sent"]["bits_per_second"] / 1e9,
            "gbps_received": end["sum_received"]["bits_per_second"] / 1e9,
        }
    except (ValueError, KeyError):
        return None


def iperf_server(bind_ip, seconds, port=DEFAULT_IPERF_PORT):
    cmd = ["iperf3", "-s", "-1", "-B", bind_ip, "-p", str(port)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 20)
        return p.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# ---- background services ----

def udp_echo_server(port=DEFAULT_UDP_PORT, stop=None):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", port))
    s.settimeout(0.5)
    while stop is None or not stop.is_set():
        try:
            data, addr = s.recvfrom(2048)
            s.sendto(data, addr)
        except socket.timeout:
            continue
    s.close()


def hub_heartbeat(hub, hb, stop):
    """Optional: POST a 1 s heartbeat to the control hub (task-4 surface)."""
    while not stop.is_set():
        try:
            req = urllib.request.Request(
                hub.rstrip("/") + "/v1/heartbeat",
                data=json.dumps({"seq": hb.seq, "t": time.time()}).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=2).read()
        except Exception:
            pass
        stop.wait(1.0)


def _read_token():
    try:
        with open(os.path.expanduser("~/.config/omarchy-cluster/token")) as f:
            return f.read().strip()
    except OSError:
        return None


def rank_start(req):
    """Spawn this node's pipeline rank. Returns the wrapper pid."""
    state_dir = os.path.expanduser("~/.local/state/omarchy-cluster")
    log_dir = os.path.expanduser("~/.local/share/omarchy-cluster")
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    hostfile = os.path.join(state_dir, "ring-hostfile.json")
    with open(hostfile, "w") as f:
        f.write(req["hostfile_content"])
    env = dict(os.environ, MLX_RANK=str(req["rank"]), MLX_HOSTFILE=hostfile,
               HF_HUB_OFFLINE="1")
    if req.get("pythonpath"):
        env["PYTHONPATH"] = ":".join(
            os.path.expanduser(p) for p in req["pythonpath"].split(":") if p)
    else:
        env["PYTHONPATH"] = os.path.expanduser("~/.local/share/omarchy-cluster/src")
    cmd = []
    if req.get("gpu_turn_minutes"):
        cmd += [os.path.expanduser("~/bin/gpu-turn"), "-m", str(req["gpu_turn_minutes"]), "--"]
    cmd += [req["python"], "-m", "omarchy_cluster.rank",
            "--model", req["model"], "--layers", req["layers"],
            "--rank", str(req["rank"]), "--hostfile", hostfile,
            "--engine-port", str(req.get("engine_port", 8031))]
    # rank 1 (decoder) polls rank 0's engine for prompts. Without this
    # flag rank1_worker has no engine URL and ap.error()s at startup.
    if int(req["rank"]) > 0 and req.get("engine_url"):
        cmd += ["--engine", req["engine_url"]]
    log = os.path.join(log_dir, "rank%d.log" % req["rank"])
    with open(log, "w") as lf:
        pid = subprocess.Popen(cmd, env=env, cwd=log_dir, stdout=lf,
                               stderr=subprocess.STDOUT,
                               start_new_session=True).pid
    return {"pid": pid, "log": log}


def rank_stop(req):
    import signal
    import time as _time
    pid = int(req["pid"])
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"stopped": pid, "escalated": False, "ports_swept": []}
    except OSError as e:
        return {"stopped": pid, "error": str(e)}
    # A rank stuck in a distributed collective defers SIGTERM forever; escalate.
    escalated = False
    for _ in range(6):
        try:
            os.killpg(pid, 0)
        except OSError:
            break
        _time.sleep(0.5)
    else:
        escalated = True
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass
    ports = []
    for value in req.get("ports", []):
        try:
            port = int(value)
            if not 1 <= port <= 65535:
                continue
            listeners = subprocess.check_output(
                ["lsof", "-tiTCP:%d" % port, "-sTCP:LISTEN"],
                stderr=subprocess.DEVNULL, text=True)
        except (ValueError, OSError, subprocess.CalledProcessError):
            continue
        for listener in listeners.splitlines():
            try:
                os.kill(int(listener), signal.SIGTERM)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        ports.append(port)
    return {"stopped": pid, "escalated": escalated, "ports_swept": ports}


class Handler(BaseHTTPRequestHandler):
    agent = None  # AgentState

    def log_message(self, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path == "/v1/health":
            return self._json(200, {"ok": True, "version": AGENT_VERSION})
        if self.path == "/v1/facts":
            f = self.agent.facts()
            f["heartbeat_seq"] = self.agent.hb.seq
            f["heartbeat_age_s"] = round(self.agent.hb.age(), 3)
            return self._json(200, f)
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path in ("/v1/rank/start", "/v1/rank/stop"):
            token = _read_token()
            supplied = self.headers.get("X-Cluster-Token", "")
            if not token or not hmac.compare_digest(token, supplied):
                return self._json(403, {"error": "bad or missing cluster token"})
            try:
                req = self._body()
            except ValueError:
                return self._json(400, {"error": "bad json"})
            try:
                if self.path == "/v1/rank/start":
                    return self._json(200, rank_start(req))
                return self._json(200, rank_stop(req))
            except KeyError as e:
                return self._json(400, {"error": "missing field %s" % e})
            except Exception as e:  # noqa: BLE001 - report failure to caller
                return self._json(500, {"error": str(e)})
        try:
            req = self._body()
        except ValueError:
            return self._json(400, {"error": "bad json"})
        try:
            if self.path == "/v1/rtt":
                return self._json(200, {
                    "tcp_ms": tcp_rtt(req["ip"], req.get("port", DEFAULT_HTTP_PORT), req.get("bind_ip")),
                    "udp_ms": udp_rtt(req["ip"], req.get("udp_port", DEFAULT_UDP_PORT), req.get("bind_ip")),
                })
            if self.path == "/v1/blast":
                sent = blast(req["ip"], req.get("port", DEFAULT_SINK_PORT),
                             req.get("seconds", 10.0), req.get("bind_ip"))
                return self._json(200, {"sent_bytes": sent})
            if self.path == "/v1/sink-serve":
                recv, elapsed = sink_serve(req.get("port", DEFAULT_SINK_PORT), req.get("seconds", 10.0))
                return self._json(200, {"received_bytes": recv, "elapsed_s": round(elapsed, 3)})
            if self.path == "/v1/iperf-server":
                ok = iperf_server(req["bind_ip"], req.get("seconds", 10.0), req.get("port", DEFAULT_IPERF_PORT))
                return self._json(200, {"ok": ok})
            if self.path == "/v1/iperf-client":
                res = iperf_client(req["ip"], req["bind_ip"], req.get("seconds", 10.0),
                                   req.get("port", DEFAULT_IPERF_PORT))
                if res is None:
                    return self._json(200, {"ok": False})
                res["ok"] = True
                return self._json(200, res)
        except KeyError as e:
            return self._json(400, {"error": "missing field %s" % e})
        except Exception as e:  # noqa: BLE001 - report any failure to the caller
            return self._json(500, {"error": str(e)})
        self._json(404, {"error": "not found"})


def serve(port=DEFAULT_HTTP_PORT, name=None, hub=None):
    hb = Heartbeat()
    hb.start()
    stop = threading.Event()
    threading.Thread(target=udp_echo_server, args=(DEFAULT_UDP_PORT, stop), daemon=True).start()
    if hub:
        threading.Thread(target=hub_heartbeat, args=(hub, hb, stop), daemon=True).start()
    adv = advertise_start(name or socket.gethostname().split(".")[0], port)
    Handler.agent = AgentState(hb)
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    print("omarchy-cluster agent %s listening on :%d (mdns %s)"
          % (AGENT_VERSION, port, "up" if adv and adv.poll() is None else "unavailable"), flush=True)
    try:
        srv.serve_forever()
    finally:
        stop.set()
        if adv:
            adv.terminate()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omarchy-cluster agent")
    ap.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT)
    ap.add_argument("--name", default=None, help="mDNS service name (default: short hostname)")
    ap.add_argument("--hub", default=None, help="control-hub base URL for 1 s heartbeats")
    args = ap.parse_args(argv)
    serve(port=args.port, name=args.name, hub=args.hub)


if __name__ == "__main__":
    main()
