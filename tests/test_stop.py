"""cmd_stop must kill the gateway subprocess AND remote rank agents.

Bug: `omarchy-cluster serve` ran `gw_serve` in-process, so a nohup'd CLI
session survived `omarchy-cluster stop` (which only knew about rank PIDs).
The fix: cmd_serve spawns the gateway as a detached child and writes its
pid to serve.json; cmd_stop killpg()s that pid and sweeps anything still
listening on the gateway port.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import cli

STATE_DIR = "~/.local/state/omarchy-cluster"


def _write_state(tmp_path, gateway_pid, gateway_port, stages=None):
    path = tmp_path / "serve.json"
    state = {
        "stages": stages or [],
        "gateway_pid": gateway_pid,
        "gateway_port": gateway_port,
        "nodes": {},
    }
    path.write_text(json.dumps(state))
    return str(path)


def _parse_args():
    return argparse.Namespace()


def _expanduser_only_for(state_file_path):
    """Wrap expanduser so the rest of cmd_serve sees real paths.

    Without the wrapper, monkeypatching os.path.expanduser unconditionally
    recurses on every inner path lookup. Use the original function for
    anything but serve.json.
    """
    real_expanduser = os.path.expanduser
    target = str(state_file_path)
    def expanduser(p):
        if isinstance(p, str) and p.endswith("serve.json"):
            return target
        return real_expanduser(p)
    return expanduser


def test_cmd_stop_kills_recorded_gateway_pid(tmp_path, monkeypatch):
    """cmd_stop killpg()s the gateway pid stored in serve.json."""
    monkeypatch.setattr(cli, "_stop_rank_via_agent", lambda node, pid: {"stopped": pid})
    path = _write_state(tmp_path, gateway_pid=os.getpid(), gateway_port=0)
    monkeypatch.setattr(cli.os.path, "expanduser", _expanduser_only_for(path))
    killed = {}

    def fake_killpg(pid, sig):
        killed["pid"] = pid
        killed["sig"] = sig

    monkeypatch.setattr(cli.os, "killpg", fake_killpg)
    cli.cmd_stop(_parse_args())
    assert killed.get("pid") == os.getpid()
    assert killed.get("sig") == signal.SIGTERM
    assert not os.path.exists(path)


def test_cmd_stop_tolerates_dead_gateway_pid(tmp_path):
    """A stale gateway_pid must not raise; cmd_stop still proceeds and removes state."""
    script = tmp_path / "run.py"
    script.write_text(
        "import json, os, sys, argparse\n"
        "sys.path.insert(0, '" + os.path.dirname(__file__) + "/..')\n"
        "from omarchy_cluster import cli\n"
        "tmp = '" + str(tmp_path) + "'\n"
        "state = {'stages': [], 'gateway_pid': 999999, 'gateway_port': 0, 'nodes': {}}\n"
        "open(tmp + '/serve.json', 'w').write(json.dumps(state))\n"
        "def expanduser(p):\n"
        "    return tmp + '/serve.json' if isinstance(p, str) and 'serve.json' in p else os.path.expanduser(p)\n"
        "cli.os.path.expanduser = expanduser\n"
        "def boom_killpg(pid, sig):\n"
        "    raise ProcessLookupError('no such pgid')\n"
        "cli.os.killpg = boom_killpg\n"
        "cli._stop_rank_via_agent = lambda node, pid: {'stopped': pid}\n"
        "cli.cmd_stop(argparse.Namespace())\n"
        "assert not os.path.exists(tmp + '/serve.json'), 'state file not removed'\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=10)
    assert r.returncode == 0, "stderr=%r stdout=%r" % (r.stderr, r.stdout)
    assert "ok" in r.stdout


def test_cmd_stop_sweeps_in_process_gateway(tmp_path):
    """A pre-fix in-process gateway (ppid 1) bound to the gateway port is killed.

    Spawn the listener in a SEPARATE child subprocess so cmd_stop's killpg
    doesn't take down the test driver.
    """
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    listener_script = tmp_path / "listener.py"
    listener_script.write_text(
        "import time, socket\n"
        "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
        "import os\n"
        # start_new_session so killpg(SIGTERM) targets this process alone
        "httpd = ThreadingHTTPServer(('127.0.0.1', " + str(port) + "), BaseHTTPRequestHandler)\n"
        "httpd.serve_forever()\n"
    )
    listener = subprocess.Popen(
        [sys.executable, str(listener_script)],
        start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # wait for the listener; a cold interpreter on a CI macOS runner can take several seconds
        for _ in range(300):
            time.sleep(0.1)
            assert listener.poll() is None, "listener exited early (rc=%s)" % listener.returncode
            try:
                t = socket.create_connection(("127.0.0.1", port), timeout=0.2)
                t.close()
                break
            except OSError:
                continue
        else:
            raise AssertionError("listener did not bind :%d in 30 s" % port)

        # Now run cmd_stop in another subprocess so killpg only kills the listener
        stop_script = tmp_path / "stop.py"
        stop_script.write_text(
            "import json, os, sys, argparse\n"
            "sys.path.insert(0, '" + os.path.dirname(__file__) + "/..')\n"
            "from omarchy_cluster import cli\n"
            "tmp = '" + str(tmp_path) + "'\n"
            "port = " + str(port) + "\n"
            "state = {'stages': [], 'gateway_pid': 999999, 'gateway_port': port, 'nodes': {}}\n"
            "open(tmp + '/serve.json', 'w').write(json.dumps(state))\n"
            "def expanduser(p):\n"
            "    return tmp + '/serve.json' if isinstance(p, str) and 'serve.json' in p else os.path.expanduser(p)\n"
            "cli.os.path.expanduser = expanduser\n"
            "cli._stop_rank_via_agent = lambda node, pid: {'stopped': pid}\n"
            "cli.cmd_stop(argparse.Namespace())\n"
        )
        r = subprocess.run([sys.executable, str(stop_script)],
                           capture_output=True, text=True, timeout=15)
        assert r.returncode == 0, "stderr=%r stdout=%r" % (r.stderr, r.stdout)
        # wait for listener to actually exit
        listener.wait(timeout=5)
        # confirm port is free
        s2 = socket.socket()
        bound = True
        try:
            s2.bind(("127.0.0.1", port))
        except OSError:
            bound = False
        finally:
            s2.close()
        assert bound, "gateway port :%d still occupied after stop (returncode=%s)" % (
            port, listener.returncode)
    finally:
        if listener.poll() is None:
            listener.terminate()


def test_cmd_serve_spawns_gateway_as_detached_child(tmp_path, monkeypatch):
    """cmd_serve must Popen the gateway detached, NOT call gw_serve in-process.

    Pinning the gate would re-introduce the bug. Stub out enough of cmd_serve
    to reach the gateway-spawn site and inspect what it spawned.
    """
    state = {
        "stages": [
            {"node": "nodeA", "max_layers": 27},
            {"node": "nodeB", "max_layers": 27},
        ],
        "mode": "pipeline",
    }
    nodes = {
        "nodeA": {"ip": "10.0.0.1", "port": 8025, "facts": {"os": "Linux"}},
        "nodeB": {"ip": "10.0.0.2", "port": 8025, "facts": {"os": "Linux"}},
    }
    monkeypatch.setattr(cli, "cmd_place", lambda a: state)
    monkeypatch.setattr(cli.discover, "discover_nodes", lambda *a, **kw: nodes)
    monkeypatch.setattr(cli, "_pick_route_ip", lambda n, s, name: n["ip"])
    monkeypatch.setattr(cli, "_launch_rank_via_agent",
                        lambda n, rank, model, layers, hf, args, facts_os: {"pid": 4242})
    args = argparse.Namespace(
        model="m", engine="mlx", port=18020, engine_port=18031,
        python_mac="/usr/bin/true", python_linux="/usr/bin/true",
        rank_pythonpath=None, gpu_turn=0, ctx=2048,
        links=None, no_decode=[], stages=None, split=None,
    )
    state_file = tmp_path / "serve.json"
    monkeypatch.setattr(cli.os.path, "expanduser", _expanduser_only_for(state_file))
    captured = {}

    class FakeProc:
        pid = 88888

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        captured["kw"] = kw
        return FakeProc()

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    cli.cmd_serve(args)
    assert any("omarchy_cluster.gateway" in c for c in captured["cmd"])
    # detached session guarantee
    assert captured["kw"].get("start_new_session") is True
    # state.json records gateway_pid
    state_written = json.loads(state_file.read_text())
    assert state_written.get("gateway_pid") == 88888
    assert state_written.get("gateway_port") == 18020

def test_two_node_script_rejects_arguments():
    # The bundled 2-node smoke helper is intentionally not in the public repo;
    # the README's quickstart drives a 2-node run with the CLI directly.
    script = os.path.join(os.path.dirname(__file__), "..",
                          "scripts", "two-node-smoke")
    result = subprocess.run(["bash", script, "--help"],
                            capture_output=True, text=True)
    assert result.returncode != 0  # missing or non-zero exit; script not shipped


def test_rank_stop_kills_rank_that_left_the_wrapper_process_group(tmp_path):
    """A site-specific gpu-turn wrapper sets the rank's own process group;
    rank_stop must still kill it (an orphaned rank held the GPU)."""
    from omarchy_cluster import agent
    pidfile = tmp_path / "rank.pid"
    rank = ("import os, time; os.setpgid(0, 0); "
            "open(%r, 'w').write(str(os.getpid())); time.sleep(60)" % str(pidfile))
    wrapper = subprocess.Popen(
        [sys.executable, "-c", "import subprocess, sys; subprocess.call([sys.executable, '-c', %r])" % rank],
        start_new_session=True)
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text():
            break
        time.sleep(0.05)
    rank_pid = int(pidfile.read_text())
    assert os.getpgid(rank_pid) != wrapper.pid
    agent.rank_stop({"pid": wrapper.pid})
    time.sleep(0.2)
    try:
        os.kill(rank_pid, 0)
        alive = open("/proc/%d/stat" % rank_pid).read().split()[2] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        alive = False
    assert not alive


def test_listener_pids_lsof_no_match_is_empty_not_proc(monkeypatch):
    """lsof exits 1 when nothing listens (the normal case after a clean stop).
    That must mean "none", not "lsof missing": the /proc fallback crashed
    `omarchy-cluster stop` on macOS, which has no /proc."""
    from omarchy_cluster import agent

    def no_match(*a, **k):
        raise subprocess.CalledProcessError(1, a[0])

    def must_not_run(port):
        raise AssertionError("/proc fallback used although lsof is installed")

    monkeypatch.setattr(agent.subprocess, "check_output", no_match)
    monkeypatch.setattr(agent, "_listener_pids_proc", must_not_run)
    assert agent.listener_pids(8020) == []


def test_listener_pids_without_lsof_uses_proc_only_if_present(monkeypatch):
    from omarchy_cluster import agent

    def no_lsof(*a, **k):
        raise FileNotFoundError("lsof")

    monkeypatch.setattr(agent.subprocess, "check_output", no_lsof)
    monkeypatch.setattr(agent, "_listener_pids_proc", lambda port: [4242])
    monkeypatch.setattr(agent.os.path, "isdir", lambda p: True)
    assert agent.listener_pids(8020) == [4242]
    monkeypatch.setattr(agent.os.path, "isdir", lambda p: False)
    assert agent.listener_pids(8020) == []


def test_rank_start_without_python_uses_this_nodes_mlx_python(tmp_path, monkeypatch):
    """serve sends python=None when no --python-mac/--python-linux is given;
    the agent must pick its own node's interpreter (Linux venv paths differ
    per machine), never pass None to Popen."""
    from omarchy_cluster import agent
    venv_py = tmp_path / "venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("#!/bin/sh\n")
    venv_py.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OMARCHY_CLUSTER_PYTHON", str(venv_py))
    seen = {}

    class FakePopen:
        def __init__(self, cmd, **kw):
            seen["cmd"] = cmd
            self.pid = 4242

    monkeypatch.setattr(agent.subprocess, "Popen", FakePopen)
    req = {"rank": 1, "model": "m", "layers": "", "hostfile_content": "[]", "python": None}
    assert agent.rank_start(req)["pid"] == 4242
    assert seen["cmd"][0] == str(venv_py)
    monkeypatch.delenv("OMARCHY_CLUSTER_PYTHON")
    monkeypatch.setattr(agent, "RANK_PYTHONS", ())
    agent.rank_start(req)
    assert seen["cmd"][0] == sys.executable  # nothing else installed: the agent's own python


def test_gateway_returns_engine_error_as_502_not_empty_reply():
    """The engine answers {"error": ...} when generation raises. The gateway
    used to KeyError on it and drop the connection (curl: empty reply)."""
    import threading
    import urllib.error
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from omarchy_cluster import gateway

    class Engine(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps({"error": "RuntimeError: boom"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    eng = ThreadingHTTPServer(("127.0.0.1", 0), Engine)
    gateway.Gateway.engine = "http://127.0.0.1:%d" % eng.server_address[1]
    gw = ThreadingHTTPServer(("127.0.0.1", 0), gateway.Gateway)
    for s in (eng, gw):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        rq = urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % gw.server_address[1],
            data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(rq, timeout=10)
            raise AssertionError("expected HTTP 502")
        except urllib.error.HTTPError as e:
            assert e.code == 502
            assert "boom" in json.loads(e.read())["error"]
    finally:
        eng.shutdown()
        gw.shutdown()