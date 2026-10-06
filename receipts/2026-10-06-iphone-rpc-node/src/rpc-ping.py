#!/usr/bin/env python3
"""Round-trip time to a ggml rpc-server: HELLO, then N x RPC_CMD_DEVICE_COUNT (no compute, one tiny reply)."""
import socket, struct, sys, time, statistics

HELLO, DEVICE_COUNT, CAPS = 14, 15, 24
host, port = sys.argv[1].rsplit(":", 1)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300

def recv_exact(s, k):
    buf = b""
    while len(buf) < k:
        chunk = s.recv(k - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return buf

s = socket.create_connection((host, int(port)), timeout=10)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
s.sendall(bytes([HELLO]) + struct.pack("<Q", CAPS) + bytes(CAPS))
size = struct.unpack("<Q", recv_exact(s, 8))[0]
major, minor, patch = recv_exact(s, size)[:3]
req = bytes([DEVICE_COUNT]) + struct.pack("<Q", 0)
rtts = []
for _ in range(n):
    t = time.perf_counter()
    s.sendall(req)
    size = struct.unpack("<Q", recv_exact(s, 8))[0]
    recv_exact(s, size)
    rtts.append((time.perf_counter() - t) * 1e3)
rtts.sort()
q = lambda p: rtts[int(p * (len(rtts) - 1))]
print(f"{sys.argv[1]} proto {major}.{minor}.{patch} n={n} rtt_ms p50={q(.5):.3f} p90={q(.9):.3f} "
      f"p99={q(.99):.3f} min={rtts[0]:.3f} mean={statistics.mean(rtts):.3f}")
