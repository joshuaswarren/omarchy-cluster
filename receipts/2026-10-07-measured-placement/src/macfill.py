#!/usr/bin/env python3
"""Fill one llama.cpp rpc-server device with BUDGET bytes of touched buffers, hold, free.

Usage: macfill.py HOST PORT BUDGET_BYTES HOLD_SECONDS
Speaks the llama.cpp RPC protocol (65840ed, packed structs): HELLO, GET_MAX_SIZE, then
ALLOC_BUFFER chunks up to the device's max buffer size, each cleared (every page touched);
GET_MAX_SIZE after each clear is the barrier (BUFFER_CLEAR sends no reply). Frees all at the
end; the server also frees them when the connection closes.
"""
import socket
import struct
import sys
import time

HELLO, GET_MAX_SIZE, ALLOC_BUFFER, FREE_BUFFER, BUFFER_CLEAR = 14, 2, 0, 4, 5


def recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("rpc-server closed the connection")
        buf += chunk
    return buf


def call(s, cmd, payload=b"", reply=True):
    s.sendall(bytes([cmd]) + struct.pack("<Q", len(payload)) + payload)
    if not reply:
        return None
    n = struct.unpack("<Q", recv_exact(s, 8))[0]
    return recv_exact(s, n)


def main():
    host, port, budget, hold = sys.argv[1], int(sys.argv[2]), int(float(sys.argv[3])), float(sys.argv[4])
    s = socket.create_connection((host, port), timeout=600)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    call(s, HELLO, bytes(24))
    max_size = struct.unpack("<Q", call(s, GET_MAX_SIZE, struct.pack("<I", 0))[:8])[0]
    bufs, total, t0 = [], 0, time.time()
    while total < budget:
        size = min(max_size, budget - total)
        ptr, got = struct.unpack("<QQ", call(s, ALLOC_BUFFER, struct.pack("<IQ", 0, size))[:16])
        if not ptr:
            print("allocation refused at %.2f GB" % (total / 1e9), flush=True)
            break
        call(s, BUFFER_CLEAR, struct.pack("<QB", ptr, 1), reply=False)
        call(s, GET_MAX_SIZE, struct.pack("<I", 0))  # barrier: the clear has run
        bufs.append(ptr)
        total += got
    print("%s allocated and touched %.2f GB in %d buffers (max buffer %.2f GB) in %.0f s; holding %.0f s"
          % (time.strftime("%H:%M:%S", time.gmtime()), total / 1e9, len(bufs), max_size / 1e9, time.time() - t0,
             hold), flush=True)
    time.sleep(hold)
    for ptr in bufs:
        call(s, FREE_BUFFER, struct.pack("<Q", ptr), reply=False)
    call(s, GET_MAX_SIZE, struct.pack("<I", 0))
    s.close()
    print("%s freed" % time.strftime("%H:%M:%S", time.gmtime()), flush=True)


if __name__ == "__main__":
    main()
