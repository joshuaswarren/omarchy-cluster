"""omarchy-cluster CLI: agent, install-agent, discover, probe, status."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import signal

from . import AGENT_VERSION
from . import discover, probe as probe_mod

GB = 1024.0 ** 3


def _agent_src_dir():
    """Directory containing the omarchy_cluster package."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- agent ----

def cmd_hub(args):
    from . import hub
    hub.main(["--port", str(args.port)])


def cmd_agent(args):
    from . import agent
    agent.main(["--port", str(args.port)] + (["--name", args.name] if args.name else [])
               + (["--hub", args.hub] if args.hub else []))


# ---- install-agent ----

_UNIT_LINUX = """\
[Unit]
Description=omarchy-cluster agent
After=network-online.target

[Service]
ExecStart={python} -m omarchy_cluster.agent --port {port}
Environment=PYTHONPATH={src}
Restart=always
RestartSec=2

[Install]
WantedBy=default.target
"""

_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>dev.omarchy.cluster-agent</string>
  <key>ProgramArguments</key>
  <array>
    <string>{python}</string>
    <string>-m</string>
    <string>omarchy_cluster.agent</string>
    <string>--port</string>
    <string>{port}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict><key>PYTHONPATH</key><string>{src}</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
</dict>
</plist>
"""


def cmd_install_agent(args):
    import tempfile
    py_real = os.path.realpath(sys.executable or "")
    for tmp in {"/tmp", os.path.realpath(tempfile.gettempdir())}:
        if py_real.startswith(tmp.rstrip("/") + "/"):
            sys.exit("refusing: %s is under the temporary directory %s; the agent's boot unit "
                     "would break after a reboot. Install into a persistent venv." % (sys.executable, tmp))
    home = os.path.expanduser("~")
    src = os.path.join(home, ".local/share/omarchy-cluster/src")
    pkg_dst = os.path.join(src, "omarchy_cluster")
    pkg_src = os.path.join(_agent_src_dir(), "omarchy_cluster")
    if os.path.realpath(pkg_src) != os.path.realpath(pkg_dst):  # never delete the tree we copy from
        shutil.rmtree(pkg_dst, ignore_errors=True)
        os.makedirs(src, exist_ok=True)
        shutil.copytree(pkg_src, pkg_dst)

    # shared cluster token: agents only accept rank control from token holders
    tokdir = os.path.expanduser("~/.config/omarchy-cluster")
    os.makedirs(tokdir, exist_ok=True)
    tokpath = os.path.join(tokdir, "token")
    if args.token_file:
        with open(args.token_file) as f:
            tok = f.read().strip()
        with open(tokpath, "w") as f:
            f.write(tok)
    elif not os.path.exists(tokpath):
        import secrets
        with open(os.open(tokpath, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
            f.write(secrets.token_urlsafe(32))
    os.chmod(tokpath, 0o600)

    wrapper = os.path.join(home, ".local/bin/omarchy-cluster")
    os.makedirs(os.path.dirname(wrapper), exist_ok=True)
    py = sys.executable or "python3"
    with open(wrapper, "w") as f:
        f.write("#!/bin/sh\nPYTHONPATH=%s exec %s -m omarchy_cluster.cli \"$@\"\n" % (src, py))
    os.chmod(wrapper, 0o755)

    if platform.system() == "Darwin":
        log = os.path.join(home, ".local/share/omarchy-cluster/agent.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        plist = os.path.join(home, "Library/LaunchAgents/dev.omarchy.cluster-agent.plist")
        os.makedirs(os.path.dirname(plist), exist_ok=True)
        with open(plist, "w") as f:
            f.write(_PLIST.format(python=py, port=args.port, src=src, log=log))
        subprocess.run(["launchctl", "bootout", "gui/%d/dev.omarchy.cluster-agent" % os.getuid()],
                       capture_output=True)
        r = subprocess.run(["launchctl", "bootstrap", "gui/%d" % os.getuid(), plist],
                           capture_output=True, text=True)
        if r.returncode != 0:  # older macOS
            r = subprocess.run(["launchctl", "load", "-w", plist], capture_output=True, text=True)
        print("launchd: %s" % (r.stdout.strip() or r.stderr.strip() or "loaded %s" % plist))
    else:
        unit_dir = os.path.join(home, ".config/systemd/user")
        os.makedirs(unit_dir, exist_ok=True)
        unit = os.path.join(unit_dir, "omarchy-cluster-agent.service")
        with open(unit, "w") as f:
            f.write(_UNIT_LINUX.format(python=py, port=args.port, src=src))
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        r = subprocess.run(["systemctl", "--user", "restart", "omarchy-cluster-agent"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            r = subprocess.run(["systemctl", "--user", "enable", "--now",
                                "omarchy-cluster-agent"], capture_output=True, text=True)
        subprocess.run(["systemctl", "--user", "enable", "omarchy-cluster-agent"],
                       capture_output=True)
        linger = subprocess.run(["loginctl", "enable-linger"], capture_output=True, text=True)
        print("systemd user unit enabled; linger: %s" %
              ("on" if linger.returncode == 0 else "FAILED (%s)" % linger.stderr.strip()))

    print("agent installed; wrapper: %s" % wrapper)

    # Inbound accept self-test (Lead-requested): open a listener on the
    # same binary to expose macOS Application Firewall stalls. The firewall
    # drops inbound connections to binaries without an allow row (Homebrew
    # Python upgrades silently re-break this). Linux just accepts.
    _self_test_inbound(args.port, py)


def _self_test_inbound(port, py):
    """Bind an ephemeral listener and warn if the binary looks firewall-blocked."""
    import socket
    import threading
    test_port = port + 100
    try:
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("", test_port))
        server.listen(1)
        server.settimeout(3.0)
        accepted = []

        def _accept():
            try:
                c, _ = server.accept()
                accepted.append(True)
                c.close()
            except socket.timeout:
                pass

        threading.Thread(target=_accept, daemon=True).start()
        try:
            c = socket.socket()
            c.settimeout(2.0)
            c.connect(("127.0.0.1", test_port))
            c.close()
        except OSError:
            pass  # on macOS, the firewall blocks loopback too; the bind test is what we want
        server.close()
        if platform.system() == "Darwin" and not accepted:
            print("WARN: agent listener self-test did not accept. If peers cannot reach "
                  ":%d, the macOS Application Firewall may be blocking this python "
                  "binary. Fix: `sudo socketfilterfw --add %s --unblockapp` (run "
                  "`launchctl print-cache | grep %s` to confirm the executable path)." %
                  (port, py, py))
    except OSError as e:  # noqa: BLE001 - self-test failures are advisory
        print("WARN: inbound self-test bind failed: %s" % e)


# ---- discover ----

def cmd_discover(args):
    nodes = discover.discover_nodes(mdns_timeout=args.mdns_timeout)
    for name in sorted(nodes):
        n = nodes[name]
        print("%-16s %-15s %-8s %s" % (name, n.get("ip"), n.get("source"),
                                       "ok" if n.get("facts") else n.get("error")))
    return nodes


# ---- probe ----

def cmd_probe(args):
    nodes = {n: d for n, d in discover.discover_nodes().items() if d.get("facts")}
    if len(nodes) < 2:
        sys.exit("need at least 2 reachable nodes, found %d" % len(nodes))
    links = probe_mod.probe_all(nodes, seconds=args.seconds)
    path = probe_mod.save_links(links, args.out)
    print(links_table(links))
    print("wrote %s" % path)
    return links


def links_table(links):
    lines = []
    for pair in links.get("pairs", []):
        lines.append("%s <-> %s" % (pair["a"], pair["b"]))
        if pair.get("error"):
            lines.append("  ERROR: %s" % pair["error"])
            continue
        pinned = pair.get("pinned") or {}
        for r in pair["routes"]:
            is_pin = bool(pinned and (r["a_ip"], r["b_ip"]) == (pinned.get("a_ip"), pinned.get("b_ip")))
            mark = "*" if is_pin else " "
            g1 = "%.2f" % r["gbps_a2b"] if r.get("gbps_a2b") else "  --  "
            g2 = "%.2f" % r["gbps_b2a"] if r.get("gbps_b2a") else "  --  "
            rtt = rtt_str(r)
            dec = "decode-ok" if r.get("eligible_for_decode") else "NO-DECODE"
            lines.append(" %s %-6s %s:%-15s -> %s:%-15s  %s/%s Gb/s  rtt %s  %s%s" % (
                mark, r["media"], r["a_iface"], r["a_ip"], r["b_iface"], r["b_ip"],
                g1, g2, rtt, dec,
                "" if not is_pin else "  PINNED"))
        if not pair["routes"]:
            lines.append("  (no routes)")
    return "\n".join(lines)


def rtt_str(r):
    t, u = r.get("rtt_tcp_ms"), r.get("rtt_udp_ms")
    return "%s/%s ms" % (("%.2f" % t) if t is not None else "--",
                         ("%.2f" % u) if u is not None else "--")


# ---- status ----

def cmd_status(args):
    nodes = discover.discover_nodes()
    print(node_table(nodes))
    links = probe_mod.load_links(args.links)
    if links:
        print()
        print("pinned routes (links.json %s):" % links.get("generated"))
        print(links_table(links))
    else:
        print("\nno links.json yet; run: omarchy-cluster probe")
    hub_nodes = _hub_nodes(args.hub)
    if hub_nodes is not None:
        print("\nhub heartbeats:")
        for name, v in sorted(hub_nodes.items()):
            print("  %-16s %-8s age %5.1fs seq %d" % (name, "up" if v["up"] else "OUT", v["age_s"], v["seq"]))


def _hub_nodes(hub):
    if not hub:
        hub = os.environ.get("OMARCHY_CLUSTER_HUB")
    if not hub:
        return None
    try:
        with urllib.request.urlopen(hub.rstrip("/") + "/v1/nodes", timeout=3) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        print("hub unreachable (%s): %s" % (hub, e))
        return None


# ---- place ----

def cmd_place(args):
    from . import planner
    nodes = {n: d["facts"] for n, d in discover.discover_nodes().items() if d.get("facts")}
    if getattr(args, "nodes", None):
        keep = {n.strip() for n in args.nodes.split(",") if n.strip()}
        missing = keep - set(nodes)
        if missing:
            sys.exit("node(s) not discovered: %s" % ", ".join(sorted(missing)))
        nodes = {n: d for n, d in nodes.items() if n in keep}
    links = probe_mod.load_links(args.links)
    if not links:
        sys.exit("no links.json; run omarchy-cluster probe first")
    info = planner.model_info(args.model)
    if not info:
        sys.exit("model not found locally or in HF cache: %s" % args.model)
    plan = planner.plan_placement(nodes, links, info, ctx_tokens=args.ctx,
                                  no_decode=args.no_decode, max_stages=args.stages)
    print(planner.plan_table(plan))
    if getattr(args, "json", False):
        print(json.dumps(plan, indent=2))
    return plan


# ---- serve ----

def _launch_rank_via_agent(node, rank, model, layers, hostfile_content, args, facts_os):
    py = args.python_mac if "macOS" in (facts_os or "") else args.python_linux
    tok = _read_local_token()
    if "macOS" in (facts_os or ""):
        pp = None
    else:
        pp = "~/.local/share/omarchy-cluster/src"
        if args.rank_pythonpath:
            pp += ":" + args.rank_pythonpath
    payload = {"rank": rank, "model": model, "layers": layers,
               "hostfile_content": hostfile_content,
               "engine_port": args.engine_port,
               "python": py,
               "pythonpath": pp,
               "gpu_turn_minutes": args.gpu_turn if "macOS" not in (facts_os or "") else 0}
    if rank:
        payload["engine_url"] = args.engine_url
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://%s:%d/v1/rank/start" % (node["ip"], node.get("port", 8025)),
        data=data, headers={"Content-Type": "application/json",
                            "X-Cluster-Token": tok})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _stop_rank_via_agent(node, pid):
    tok = _read_local_token()
    req = urllib.request.Request(
        "http://%s:%d/v1/rank/stop" % (node["ip"], node.get("port", 8025)),
        data=json.dumps({"pid": pid, "ports": node.get("ports", [52100, 8031])}).encode(),
        headers={"Content-Type": "application/json", "X-Cluster-Token": tok})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _read_local_token():
    with open(os.path.expanduser("~/.config/omarchy-cluster/token")) as f:
        return f.read().strip()


def cmd_serve(args):
    prev = os.path.expanduser("~/.local/state/omarchy-cluster/serve.json")
    if os.path.exists(prev):
        print("stopping previous serve state first")
        try:
            cmd_stop(args)
        except SystemExit:
            pass
    if args.engine == "llamacpp":
        return _serve_llamacpp(args)
    plan = cmd_place(args)
    if plan["mode"] == "none":
        sys.exit("no feasible plan")
    stages = plan["stages"]
    if len(stages) > 2:
        sys.exit("serve currently supports 2-rank pipelines (ring hop protocol); "
                 "got %d stages" % len(stages))
    hostfile = os.path.expanduser("~/.local/state/omarchy-cluster/ring-hostfile.json")
    os.makedirs(os.path.dirname(hostfile), exist_ok=True)
    nodes = {n: d for n, d in discover.discover_nodes().items() if d.get("facts")}
    rank_ips = []
    for s in stages:
        node = nodes[s["node"]]
        if len(stages) == 1:
            cand = "127.0.0.1"
        else:
            cand = _pick_route_ip(node, stages, s["node"])
        rank_ips.append(cand)
    with open(hostfile, "w") as f:
        json.dump([["%s:%d" % (ip, 52100)] for ip in rank_ips], f)
    print("ring hostfile: %s" % rank_ips)
    with open(hostfile) as f:
        hostfile_content = f.read()

    args.engine_url = "http://%s:%d" % (rank_ips[0], args.engine_port)
    state = {"stages": [], "gateway_port": args.port, "engine_port": args.engine_port,
             "nodes": {s["node"]: nodes[s["node"]]["ip"] for s in stages}}
    for rank, s in enumerate(stages):
        node = nodes[s["node"]]
        res = _launch_rank_via_agent(node, rank, args.model, args.split or "",
                                     hostfile_content, args, node["facts"].get("os"))
        state["stages"].append({"node": node["ip"], "pid": str(res["pid"])})
        print("rank %d on %s pid %s (agent-reported, log: %s)"
              % (rank, node["ip"], res["pid"], res.get("log")))
        with open(os.path.expanduser("~/.local/state/omarchy-cluster/serve.json"), "w") as f:
            json.dump(state, f, indent=2)

    engine_host = rank_ips[0]
    local_name = platform.node().split(".")[0]
    if stages[0]["node"] == local_name:
        engine_host = "127.0.0.1"
    engine = "http://%s:%d" % (engine_host, args.engine_port)
    print("gateway engine: %s" % engine)
    _start_gateway(state, args.port, engine)


def _write_state(state):
    with open(os.path.expanduser("~/.local/state/omarchy-cluster/serve.json"), "w") as f:
        json.dump(state, f, indent=2)


def _spawn(cmd, log_name):
    """Start a local child in its own process group, logging to log_name."""
    log = os.path.expanduser("~/.local/share/omarchy-cluster/" + log_name)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as lf:
        child = subprocess.Popen(cmd, start_new_session=True, stdin=subprocess.DEVNULL,
                                 stdout=lf, stderr=subprocess.STDOUT)
    return child, log


def _start_gateway(state, port, engine):
    from . import llamacpp_engine
    gateway_code = "from omarchy_cluster.gateway import serve; serve(port=%d, engine=%r)" % (port, engine)
    gateway, gw_log = _spawn([sys.executable, "-c", gateway_code], "gateway.log")
    state["gateway_pid"] = gateway.pid
    _write_state(state)
    # serve returns to scripts that call the endpoint at once: be listening first
    try:
        llamacpp_engine.wait_http("http://127.0.0.1:%d/v1/models" % port, deadline_s=30.0, proc=gateway)
    except (TimeoutError, RuntimeError) as e:
        sys.exit("gateway did not start (log: %s): %s" % (gw_log, e))
    print("gateway pid: %s (log: %s)" % (gateway.pid, gw_log))


def _serve_llamacpp(args):
    """GGUF model on this host through llama-server. A USB iPhone running the
    rpc-server app gets the last layers only when the model does not fit here."""
    from . import facts, iosnode, llamacpp_engine
    try:
        info = llamacpp_engine.gguf_info(args.model)
    except (OSError, ValueError) as e:
        sys.exit("cannot read GGUF %s: %s" % (args.model, e))
    if args.rpc_node:
        return _serve_llamacpp_nodes(args, info)
    phone = None
    if args.ios_bundle:
        try:
            phones = iosnode.list_devices(args.pm3)
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            sys.exit("pymobiledevice3 usbmux list failed (%s): %s" % (args.pm3, e))
        phone = phones[0] if phones else None
        print("usb iPhone: %s" % (phone["name"] if phone else "none"))
    cap = int(args.phone_cap_gb * 1e9) if phone else 0
    try:
        k = llamacpp_engine.rpc_layers(info, facts.memory_free_bytes(), cap, args.ctx,
                                       force=args.rpc_layers)
    except ValueError as e:
        sys.exit(str(e))
    print("%s: %d layers, %.2f GB; %d on the iPhone" % (
        os.path.basename(args.model), info["n_layer"], info["total_bytes"] / 1e9, k))
    os.makedirs(os.path.expanduser("~/.local/state/omarchy-cluster"), exist_ok=True)
    state = {"engine": "llamacpp", "stages": [], "local_pids": [], "gateway_port": args.port,
             "engine_port": args.engine_port}
    rpc = None
    if k:
        if not phone:
            sys.exit("%d layers need the iPhone but no USB iPhone was found "
                     "(pass --ios-bundle with the app plugged in)" % k)
        fwd = iosnode.start_forward(phone["udid"], args.rpc_port, iosnode.RPC_PORT, args.pm3)
        state["local_pids"].append(fwd.pid)
        _write_state(state)
        rpc = "127.0.0.1:%d" % args.rpc_port
        try:
            iosnode.launch(args.ios_bundle, iosnode.RPC_PORT, pm3=args.pm3)
            print("iPhone rpc-server: %s" % iosnode.wait_rpc("127.0.0.1", args.rpc_port))
        except (OSError, subprocess.SubprocessError) as e:
            sys.exit("iPhone rpc-server did not start (unlock the phone, check --ios-bundle; "
                     "run `omarchy-cluster stop` to drop the USB forward): %s" % e)
    cmd = [sys.executable, "-m", "omarchy_cluster.llamacpp_engine", "--model", args.model,
           "--llama-server", args.llama_server, "--port", str(args.engine_port),
           "--server-port", str(args.engine_port + 1), "--ctx", str(args.ctx),
           "--rpc-layers", str(k)]
    if rpc:
        cmd += ["--rpc", rpc]
    engine, log = _spawn(cmd, "llamacpp-engine.log")
    state["local_pids"].append(engine.pid)
    _write_state(state)
    engine_url = "http://127.0.0.1:%d" % args.engine_port
    try:
        llamacpp_engine.wait_http(engine_url + "/health", proc=engine)
    except (TimeoutError, RuntimeError) as e:
        sys.exit("llama.cpp engine failed (log: %s): %s" % (log, e))
    print("llama.cpp engine pid %s (log: %s)" % (engine.pid, log))
    _start_gateway(state, args.port, engine_url)


def _parse_rpc_nodes(values):
    """`--rpc-node NAME[=GB]` or `HOST:PORT[=GB]` values -> [(name, budget bytes or None)].
    HOST:PORT is an rpc-server started by something else. None means measure it."""
    out = []
    for v in values:
        name, _, gb = v.partition("=")
        try:
            out.append((name, int(float(gb) * 1e9) if gb else None))
        except ValueError:
            sys.exit("--rpc-node %s: expected NAME, NAME=GB, HOST:PORT or HOST:PORT=GB" % v)
    return out


def _parse_node_values(values, flag):
    """`NAME=VALUE` values -> {NAME: [VALUE, ...]}; VALUE may itself contain '='."""
    out = {}
    for v in values:
        name, sep, rest = v.partition("=")
        if not sep or not name or not rest:
            sys.exit("%s %s: expected NAME=VALUE" % (flag, v))
        out.setdefault(name, []).append(rest)
    return out


def _rpc_node_options(args):
    """Per-node rpc-server options from --rpc-env NAME=KEY=VALUE and --rpc-threads NAME=N."""
    envs = {}
    for name, items in _parse_node_values(args.rpc_env, "--rpc-env").items():
        for item in items:
            key, sep, val = item.partition("=")
            if not sep or not key:
                sys.exit("--rpc-env %s=%s: expected NAME=KEY=VALUE" % (name, item))
            envs.setdefault(name, {})[key] = val
    threads = {}
    for name, items in _parse_node_values(args.rpc_threads, "--rpc-threads").items():
        try:
            threads[name] = int(items[-1])
        except ValueError:
            sys.exit("--rpc-threads %s=%s: expected a number" % (name, items[-1]))
    return envs, threads


def _agent_post(node, path, payload, timeout=30):
    req = urllib.request.Request(
        "http://%s:%d%s" % (node["ip"], node.get("port", 8025), path),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Cluster-Token": _read_local_token()})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _plan_devices(devs, info, ctx, host_ram_avail, host_layers=None):
    """Budgets and placement for the rpc devices. devs: dicts with name, ep, machine,
    budget (bytes, or None to measure), free_bytes and total_bytes (from the rpc hello),
    ram_avail and ram_total (the machine's, or None), va_cap (or None). A device whose
    llama.cpp total matches its machine's RAM is a CPU device; anything else is a GPU.
    Returns (devices in placement order with weights, host_layers, host_bytes,
    paged_bytes); devices that get no layer are dropped."""
    from . import llamacpp_engine as lce
    for d in devs:
        total, ram_total = d.get("total_bytes"), d.get("ram_total")
        d["gpu"] = not (total and ram_total and abs(total - ram_total) <= 0.03 * ram_total)
        d["measured"] = d["budget"] is None
        if d["measured"]:
            if d.get("free_bytes") is None:
                raise ValueError("--rpc-node %s: no =GB and the rpc-server did not report its memory" % d["name"])
            d["budget"] = lce.device_budget(d["free_bytes"], d.get("ram_avail"), d.get("va_cap"), d["gpu"])
    lce.pool_machines([d for d in devs if d["measured"]])
    host_cache = lce.host_budget(host_ram_avail, info)
    plan = lce.place(info, [d["budget"] for d in devs], host_cache, ctx,
                     host_layers=host_layers, keep_order=host_layers is not None)
    ordered = []
    for i, w in zip(plan["order"], plan["weights"]):
        if w:
            ordered.append(dict(devs[i], weight=w))
    return ordered, plan["host_layers"], plan["host_bytes"], plan["paged_bytes"]


def _serve_llamacpp_nodes(args, info):
    """GGUF on this host's llama-server, every layer spread over llama.cpp RPC servers
    that each named node's agent starts (this host too, if listed) or that run already
    (HOST:PORT). Starts and greets every rpc-server first, then sizes and places."""
    from . import facts as facts_mod, iosnode, llamacpp_engine
    wanted = _parse_rpc_nodes(args.rpc_node)
    binaries = dict(v.split("=", 1) for v in args.rpc_binary)
    envs, threads = _rpc_node_options(args)
    nodes = {n: d for n, d in discover.discover_nodes().items() if d.get("facts")}
    by_ip = {d["ip"]: d for d in nodes.values()}
    local = platform.node().split(".")[0]
    os.makedirs(os.path.expanduser("~/.local/state/omarchy-cluster"), exist_ok=True)
    state = {"engine": "llamacpp", "stages": [], "local_pids": [], "gateway_port": args.port,
             "engine_port": args.engine_port}
    devs = []
    for name, budget in wanted:
        if ":" in name:  # started elsewhere: check it answers
            host, port = name.rsplit(":", 1)
            node, ep = by_ip.get(host), name
            try:
                hello = iosnode.wait_rpc(host, int(port), deadline_s=30.0)
            except OSError as e:
                sys.exit("rpc-server %s did not answer: %s" % (name, e))
        else:
            if name not in nodes:
                sys.exit("--rpc-node %s: not discovered (have %s)" % (name, ", ".join(sorted(nodes))))
            node = nodes[name]
            res = _agent_post(node, "/v1/rpc/start", {"binary": binaries.get(name), "port": args.rpc_node_port,
                                                      "cache": args.rpc_cache, "env": envs.get(name, {}),
                                                      "threads": threads.get(name)})
            state["stages"].append({"node": node["ip"], "pid": str(res["pid"]), "port": args.rpc_node_port})
            _write_state(state)
            ip = "127.0.0.1" if name == local else _pick_route_ip(node, [{"node": local}, {"node": name}], name)
            ep, host = "%s:%d" % (ip, args.rpc_node_port), node["ip"]
            try:
                hello = iosnode.wait_rpc(ip, args.rpc_node_port, deadline_s=120.0)
            except OSError as e:
                sys.exit("rpc-server on %s (%s) did not answer; log %s on that node: %s"
                         % (name, ep, res.get("log"), e))
        print("rpc-server %s at %s: %s" % (name, ep, hello))
        f = (node or {}).get("facts") or {}
        va = None
        if budget is None and node and f.get("gpu_backend") == "vulkan":
            try:
                va = _agent_post(node, "/v1/gpu-cap", {}, timeout=320).get("alloc_cap_bytes")
            except OSError as e:
                print("  %s: no Vulkan allocation cap (%s); using the reported heap" % (name, e))
        devs.append({"name": name, "ep": ep, "machine": host, "budget": budget,
                     "free_bytes": hello.get("free_bytes"), "total_bytes": hello.get("total_bytes"),
                     "ram_avail": llamacpp_engine.node_ram_avail(f), "ram_total": f.get("memory_total_bytes"),
                     "va_cap": va})
    try:
        ordered, host_layers, host_bytes, paged = _plan_devices(
            devs, info, args.ctx, facts_mod.memory_free_bytes(), args.host_layers)
    except ValueError as e:
        sys.exit(str(e))
    print("%s: %d layers, %.2f GB in %d shard(s). Host: %s, %.2f GB, %.2f GB of it beyond the host's "
          "page cache" % (os.path.basename(args.model), info["n_layer"], info["total_bytes"] / 1e9, info["shards"],
                          "layers 0-%d" % (host_layers - 1) if host_layers else "no layers", host_bytes / 1e9,
                          paged / 1e9))
    first = host_layers
    for d in ordered:
        layers = min(d["weight"], info["n_layer"] - first)  # the last count also holds the output slot
        gb = sum(info["per_layer"][first:first + layers]) / 1e9 if info.get("per_layer") else 0.0
        print("  %s: layers %d-%d, %.1f of %.1f GB%s" % (
            d["name"], first, first + layers - 1, gb, d["budget"] / 1e9, " (measured)" if d["measured"] else ""))
        first += layers
    weights = [d["weight"] for d in ordered]
    cmd = [sys.executable, "-m", "omarchy_cluster.llamacpp_engine", "--model", args.model,
           "--llama-server", args.llama_server, "--port", str(args.engine_port),
           "--server-port", str(args.engine_port + 1), "--ctx", str(args.ctx),
           "--rpc", ",".join(d["ep"] for d in ordered), "--tensor-split", ",".join("%g" % w for w in weights),
           "--ngl", str(sum(weights))]
    engine, log = _spawn(cmd, "llamacpp-engine.log")
    state["local_pids"].append(engine.pid)
    _write_state(state)
    engine_url = "http://127.0.0.1:%d" % args.engine_port
    try:
        llamacpp_engine.wait_http(engine_url + "/health", deadline_s=3600.0, proc=engine)
    except (TimeoutError, RuntimeError) as e:
        sys.exit("llama.cpp engine failed (log: %s): %s" % (log, e))
    print("llama.cpp engine pid %s (log: %s)" % (engine.pid, log))
    _start_gateway(state, args.port, engine_url)


def _sweep_listening_port(port):
    if not port:
        return
    from .agent import listener_pids
    for pid in listener_pids(port):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def cmd_stop(args):
    path = os.path.expanduser("~/.local/state/omarchy-cluster/serve.json")
    if not os.path.exists(path):
        sys.exit("no serve state")
    with open(path) as f:
        state = json.load(f)
    for pid in [state.get("gateway_pid")] + state.get("local_pids", []):
        if not pid:
            continue
        try:
            os.killpg(int(pid), signal.SIGTERM)
            print("stopped local process group %s" % pid)
        except ProcessLookupError:
            pass
    for stage in state.get("stages", []):
        node = {"ip": stage["node"],
                "ports": [52100, state.get("engine_port", 8031)] + ([stage["port"]] if stage.get("port") else [])}
        result = _stop_rank_via_agent(node, stage["pid"])
        print("stopped rank pid %s on %s: %s" % (stage["pid"], stage["node"], result))
    _sweep_listening_port(state.get("gateway_port"))
    try:
        os.remove(path)
    except FileNotFoundError:
        pass  # concurrent stop already tore the state down


GUARD_LINUX_MIN_BYTES = 2_000_000_000
GUARD_MAC_MIN_FREE_PCT = 15
GUARD_MAC_MIN_DATA_BYTES = 20_000_000_000


def guard_verdict(facts_by_node):
    """Reasons to stop a run, from each node's agent facts (None = could not be read).
    Linux: under 2 GB available. macOS: memory_pressure under 15 percent free, or the Data
    volume (where swap lives) under 20 GB free. A node that cannot be read is a reason too:
    a guard that skips what it cannot see is no guard."""
    reasons = []
    for node, f in sorted(facts_by_node.items()):
        if f is None:
            reasons.append("%s unreadable" % node)
        elif f.get("memory_free_pages_bytes") is not None:  # macOS
            pct, disk = f.get("memory_pressure_free_pct"), f.get("data_volume_free_bytes")
            if pct is not None and pct < GUARD_MAC_MIN_FREE_PCT:
                reasons.append("%s memory_pressure %d%% free" % (node, pct))
            if disk is not None and disk < GUARD_MAC_MIN_DATA_BYTES:
                reasons.append("%s Data volume %.1f GB free" % (node, disk / 1e9))
        elif (f.get("memory_free_bytes") or 0) < GUARD_LINUX_MIN_BYTES:
            reasons.append("%s %.1f GB available" % (node, (f.get("memory_free_bytes") or 0) / 1e9))
    return reasons


def _agent_facts(ip, port=8025, timeout=5):
    try:
        with urllib.request.urlopen("http://%s:%d/v1/facts" % (ip, port), timeout=timeout) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def cmd_guard(args):
    """Watch every node of the running serve over its agent; stop the run when guard_verdict
    has reasons on two passes in a row. Run it on the gateway machine: there it can always
    stop the run, even when it cannot reach the other nodes. Exits when the run is gone."""
    import time as _time
    path = os.path.expanduser("~/.local/state/omarchy-cluster/serve.json")
    bad = 0
    while os.path.exists(path):
        with open(path) as f:
            nodes = sorted({s["node"] for s in json.load(f).get("stages", [])})
        reasons = guard_verdict({ip: _agent_facts(ip) for ip in nodes})
        bad = bad + 1 if reasons else 0
        if reasons:
            print("guard: %s" % "; ".join(reasons), flush=True)
        if bad >= 2:
            print("guard: stopping the run", flush=True)
            cmd_stop(args)
            sys.exit(3)
        _time.sleep(args.interval)
    print("guard: no running serve", flush=True)


def _pick_route_ip(node_facts_dict, stages, node_name):
    """IP of `node` on the pinned route to the other stage's node."""
    others = [s["node"] for s in stages if s["node"] != node_name]
    if others:
        links = probe_mod.load_links() or {}  # no probe run yet: use the node's first address
        for pair in links.get("pairs", []):
            if {pair.get("a"), pair.get("b")} == {node_name, others[0]}:
                p = pair.get("pinned") or {}
                if p.get("a_node") == node_name:
                    return p["a_ip"]
                if p.get("b_node") == node_name:
                    return p["b_ip"]
    facts = node_facts_dict["facts"]
    for ifc in facts["interfaces"]:
        for a in ifc["ips"]:
            return a["ip"]
    sys.exit("no IP for %s" % node_name)



def node_table(nodes):
    lines = ["%-16s %-22s %-16s %-13s %-7s %-4s %s" % (
        "NODE", "OS", "CHIP", "MEM FREE/TOT", "BACKEND", "UP", "IP")]
    for name in sorted(nodes):
        n = nodes[name]
        f = n.get("facts")
        if not f:
            lines.append("%-16s %-22s %-16s %-13s %-7s %-4s %s" % (
                name, "?", "?", "?", "?", "DOWN", "%s (%s)" % (n.get("ip"), n.get("error"))))
            continue
        up = f.get("heartbeat_age_s", 99) < 3.0
        mem = "%.1f/%.0fGB" % (f["memory_free_bytes"] / GB, f["memory_total_bytes"] / GB)
        lines.append("%-16s %-22s %-16s %-13s %-7s %-4s %s" % (
            name, f["os"], f["chip"], mem, f["gpu_backend"], "up" if up else "stale", n.get("ip")))
    return "\n".join(lines)


# ---- entry ----

def main(argv=None):
    ap = argparse.ArgumentParser(prog="omarchy-cluster", description=__doc__)
    ap.add_argument("--version", action="version", version=AGENT_VERSION)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("agent", help="run the node agent")
    p.add_argument("--port", type=int, default=8025)
    p.add_argument("--name", default=None)
    p.add_argument("--hub", default=None, help="control-hub base URL for 1 s heartbeats")
    p.set_defaults(fn=cmd_agent)

    p = sub.add_parser("install-agent", help="install and start the agent daemon on this node")
    p.add_argument("--port", type=int, default=8025)
    p.add_argument("--token-file", default=None,
                   help="provision this shared cluster token instead of generating one")
    p.set_defaults(fn=cmd_install_agent)

    p = sub.add_parser("discover", help="list discovered nodes")
    p.add_argument("--mdns-timeout", type=float, default=4.0)
    p.set_defaults(fn=cmd_discover)

    p = sub.add_parser("probe", help="measure every pair/route, write links.json")
    p.add_argument("--seconds", type=float, default=3.0,
                   help="bandwidth test length per direction (default 3)")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("status", help="show nodes and pinned routes")
    p.add_argument("--links", default=None)
    p.add_argument("--hub", default=None, help="hub base URL for heartbeat liveness")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("place", help="choose pipeline nodes and check they hold the model")
    p.add_argument("model")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--links", default=None)
    p.add_argument("--no-decode", action="append", default=[],
                   help="node name never used as a decode rank (e.g. a node shared with other GPU work)")
    p.add_argument("--nodes", default=None, metavar="N1,N2",
                   help="use exactly these discovered nodes (default: every node found)")
    p.add_argument("--stages", type=int, default=None,
                   help="force an exact pipeline stage count")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_place)

    p = sub.add_parser("serve", help="plan, launch ranks, serve OpenAI on :8020")
    p.add_argument("model", help="MLX model (HF id or path), or a .gguf file with --engine llamacpp")
    p.add_argument("--engine", choices=("mlx", "llamacpp"), default="mlx",
                   help="mlx: pipeline ranks across nodes; llamacpp: llama-server on this "
                        "host, plus a USB iPhone RPC device when the model does not fit here")
    p.add_argument("--llama-server", default="llama-server", help="llama-server binary (llamacpp)")
    p.add_argument("--ios-bundle", default=None,
                   help="bundle id of the rpc-server app on a USB iPhone (llamacpp); "
                        "without it the iPhone is not used")
    p.add_argument("--pm3", default="pymobiledevice3",
                   help="pymobiledevice3 command, e.g. \"python3 -m pymobiledevice3\"")
    p.add_argument("--phone-cap-gb", type=float, default=2.2,
                   help="model + KV GB the iPhone may hold (2.2 measured on an 8 GB phone)")
    p.add_argument("--rpc-layers", type=int, default=None,
                   help="force this many of the last layers onto the iPhone (llamacpp)")
    p.add_argument("--rpc-port", type=int, default=50052, help="local port of the USB forward")
    p.add_argument("--rpc-node", action="append", default=[], metavar="NAME[=GB]",
                   help="llamacpp: a llama.cpp RPC server on this discovered node, or HOST:PORT for one "
                        "started elsewhere (repeat). Without =GB the budget is measured: what llama.cpp "
                        "reports free on the device, the node's RAM minus headroom, its Vulkan "
                        "allocation cap")
    p.add_argument("--rpc-binary", action="append", default=[], metavar="NAME=PATH",
                   help="llamacpp: ggml-rpc-server path on NAME (default: the node's PATH)")
    p.add_argument("--rpc-node-port", type=int, default=50060, help="llamacpp: rpc-server port on each node")
    p.add_argument("--rpc-env", action="append", default=[], metavar="NAME=KEY=VALUE",
                   help="llamacpp: environment for NAME's rpc-server, e.g. HK_SYSMEM=60000000000, "
                        "LLAMA_CACHE=/Volumes/ext/rpc-cache, GGML_METAL_SHARED_BUFFERS_DISABLE=1 (repeat)")
    p.add_argument("--rpc-threads", action="append", default=[], metavar="NAME=N",
                   help="llamacpp: CPU threads for NAME's rpc-server (-t)")
    p.add_argument("--rpc-cache", action="store_true",
                   help="llamacpp: rpc-servers keep received weights on local disk (-c) for faster reloads; "
                        "each node then needs disk for its whole share")
    p.add_argument("--host-layers", type=int, default=None,
                   help="llamacpp with --rpc-node: keep exactly the first N layers on this host's CPU "
                        "(mmapped from the GGUF) and the nodes in the given order. Default: choose the "
                        "order and N that page the fewest host bytes from disk, then use the fewest nodes")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--links", default=None)
    p.add_argument("--no-decode", action="append", default=[])
    p.add_argument("--nodes", default=None, metavar="N1,N2")
    p.add_argument("--stages", type=int, default=None)
    p.add_argument("--port", type=int, default=8020)
    p.add_argument("--engine-port", type=int, default=8031)
    p.add_argument("--python-mac", default=None,
                   help="python with mlx on macOS ranks (default: each node's agent picks its own)")
    p.add_argument("--python-linux", default=None,
                   help="python with mlx on Linux ranks (default: each node's agent picks its own)")
    p.add_argument("--rank-pythonpath", default=None,
                   help="extra PYTHONPATH entry for ranks (e.g. mlx-lm pkg dir)")
    p.add_argument("--gpu-turn", type=int, default=0, metavar="MINUTES",
                   help="wrap Linux ranks in ~/bin/gpu-turn, an optional site-specific "
                        "GPU queue wrapper (off by default)")
    p.add_argument("--split", default=None, metavar="N0,N1",
                   help="decoder layers per rank in rank order (rank 0 runs the last "
                        "layers); default: chosen at rank start from measured ms/layer")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("stop", help="stop ranks started by serve")
    p.set_defaults(fn=cmd_stop)

    p = sub.add_parser("guard", help="watch the running serve's nodes; stop it when memory runs out "
                                     "or a node cannot be read (run on the gateway machine)")
    p.add_argument("--interval", type=float, default=10.0, help="seconds between passes")
    p.set_defaults(fn=cmd_guard)

    p = sub.add_parser("hub", help="run the heartbeat hub")
    p.add_argument("--port", type=int, default=8030)
    p.set_defaults(fn=cmd_hub)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()