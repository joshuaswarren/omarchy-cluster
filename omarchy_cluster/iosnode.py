"""An iPhone on USB as a llama.cpp RPC device.

The phone runs llama.cpp's `rpc-server` inside an app (UIKit host, built and
signed from Linux with xtool). This module drives it from the host it is
plugged into, through pymobiledevice3:

- `list_devices`: phones on usbmuxd,
- `start_forward`: host 127.0.0.1:PORT -> the phone's rpc-server port,
- `launch`: start the app with rpc-server arguments (developer DVT launch),
- `rpc_hello`: the RPC handshake plus a device count, as a health check,
- `device_facts`: the phone in the shape of a node's `/v1/facts`.

Memory: iOS reports far more than an app can use. 4.4 GB of layers drove an
8 GB iPhone 15 Pro Max into system-wide jetsam; 2.2 GB ran. The cap is a
measured default, not a device query.
"""
from __future__ import annotations

import json
import shlex
import socket
import struct
import subprocess
import time

PHONE_CAP_BYTES = 2_200_000_000
RPC_PORT = 50052
RPC_CMD_HELLO = 14
RPC_CMD_DEVICE_COUNT = 15
RPC_CONN_CAPS_SIZE = 24


def pm3_cmd(pm3):
    """The pymobiledevice3 command line, e.g. "pymobiledevice3" or
    "~/venvs/pm3/bin/python -m pymobiledevice3"."""
    return shlex.split(pm3)


def list_devices(pm3="pymobiledevice3", timeout=15):
    out = subprocess.run(pm3_cmd(pm3) + ["usbmux", "list"], capture_output=True,
                         text=True, timeout=timeout, check=True).stdout
    return parse_usbmux_list(out)


def parse_usbmux_list(text):
    """USB-connected iOS devices from `pymobiledevice3 usbmux list` JSON."""
    devices = []
    for d in json.loads(text or "[]"):
        if d.get("ConnectionType", "USB") != "USB":
            continue
        devices.append({"udid": d["UniqueDeviceID"], "name": d.get("DeviceName", ""),
                        "product": d.get("ProductType", ""),
                        "ios": d.get("ProductVersion", "")})
    return devices


def start_forward(udid, local_port=RPC_PORT, device_port=RPC_PORT, pm3="pymobiledevice3"):
    """Forward host 127.0.0.1:local_port to the phone. Returns the Popen; the
    caller owns it (its own process group, so `stop` can kill it)."""
    return subprocess.Popen(pm3_cmd(pm3) + ["usbmux", "forward", "--serial", udid,
                                            str(local_port), str(device_port)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT, start_new_session=True)


def launch_args(bundle, port=RPC_PORT, threads=4, backend=None):
    """The single DVT launch argument: bundle id followed by rpc-server flags.
    backend None serves the phone GPU (Metal); "CPU" serves its CPU."""
    parts = [bundle, "-H", "127.0.0.1", "-p", str(port), "-t", str(threads)]
    if backend:
        parts += ["-d", backend]
    return " ".join(parts)


def launch(bundle, port=RPC_PORT, threads=4, backend=None, pm3="pymobiledevice3", timeout=60):
    """Start (or restart) the app on the USB phone. Raises on a launch error."""
    subprocess.run(pm3_cmd(pm3) + ["developer", "dvt", "launch", "--kill-existing",
                                   launch_args(bundle, port, threads, backend)],
                   capture_output=True, text=True, timeout=timeout, check=True)


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("rpc-server closed the connection")
        buf += chunk
    return buf


def rpc_hello(host="127.0.0.1", port=RPC_PORT, timeout=5.0):
    """HELLO then DEVICE_COUNT. Returns {"version": "M.m.p", "devices": n, "rtt_ms": x}."""
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.sendall(bytes([RPC_CMD_HELLO]) + struct.pack("<Q", RPC_CONN_CAPS_SIZE)
                  + bytes(RPC_CONN_CAPS_SIZE))
        size = struct.unpack("<Q", _recv_exact(s, 8))[0]
        major, minor, patch = _recv_exact(s, size)[:3]
        t = time.perf_counter()
        s.sendall(bytes([RPC_CMD_DEVICE_COUNT]) + struct.pack("<Q", 0))
        size = struct.unpack("<Q", _recv_exact(s, 8))[0]
        (count,) = struct.unpack("<I", _recv_exact(s, size)[:4])
        rtt_ms = (time.perf_counter() - t) * 1e3
    return {"version": "%d.%d.%d" % (major, minor, patch), "devices": count,
            "rtt_ms": round(rtt_ms, 3)}


def wait_rpc(host="127.0.0.1", port=RPC_PORT, deadline_s=60.0):
    """Poll rpc_hello until it answers (the phone compiles Metal shaders at
    start, which takes several seconds)."""
    end = time.monotonic() + deadline_s
    while True:
        try:
            return rpc_hello(host, port)
        except OSError:
            if time.monotonic() > end:
                raise
            time.sleep(1.0)


def device_facts(dev, port=RPC_PORT, cap_bytes=PHONE_CAP_BYTES):
    """The phone as a node entry. memory_free_bytes is the usable cap."""
    return {"name": "iphone-" + dev["udid"][-4:].lower(), "os": "iOS " + dev["ios"],
            "chip": dev["product"], "gpu_backend": "metal (llama.cpp rpc-server)",
            "unified_memory": True, "memory_free_bytes": cap_bytes,
            "transport": "usbmux", "rpc": "127.0.0.1:%d" % port}
