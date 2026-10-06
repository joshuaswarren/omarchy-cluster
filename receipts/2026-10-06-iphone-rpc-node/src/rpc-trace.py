#!/usr/bin/env python3
"""Timing proxy for ggml RPC: listen on LISTEN, forward to UPSTREAM, log each client command and
each server reply chunk with a monotonic timestamp. Usage: rpc-trace.py 127.0.0.1:50060 127.0.0.1:50052 out.tsv"""
import socket
import sys
import threading
import time

CMDS = ("ALLOC_BUFFER GET_ALIGNMENT GET_MAX_SIZE BUFFER_GET_BASE FREE_BUFFER BUFFER_CLEAR SET_TENSOR "
        "SET_TENSOR_HASH GET_TENSOR COPY_TENSOR GRAPH_COMPUTE GET_DEVICE_MEMORY INIT_TENSOR GET_ALLOC_SIZE "
        "HELLO DEVICE_COUNT GRAPH_RECOMPUTE MEMSET_TENSOR").split()


def addr(s):
    h, p = s.rsplit(":", 1)
    return h, int(p)


listen, upstream, out_path = addr(sys.argv[1]), addr(sys.argv[2]), sys.argv[3]
log = open(out_path, "w", buffering=1)
lock = threading.Lock()


def emit(*fields):
    with lock:
        log.write("\t".join(str(f) for f in (f"{time.monotonic():.6f}",) + fields) + "\n")


def client_to_server(c, u):
    buf = bytearray()
    while data := c.recv(1 << 20):
        u.sendall(data)
        buf += data
        while len(buf) >= 9:
            cmd, size = buf[0], int.from_bytes(buf[1:9], "little")
            if len(buf) < 9 + size:
                break
            emit("c2s", CMDS[cmd] if cmd < len(CMDS) else cmd, size)
            del buf[:9 + size]
    u.shutdown(socket.SHUT_WR)


def server_to_client(u, c):
    while data := u.recv(1 << 20):
        c.sendall(data)
        emit("s2c", "bytes", len(data))
    c.shutdown(socket.SHUT_WR)


srv = socket.create_server(listen)
while True:
    c, _ = srv.accept()
    u = socket.create_connection(upstream)
    for s in (c, u):
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    emit("conn", "open", 0)
    threading.Thread(target=client_to_server, args=(c, u), daemon=True).start()
    threading.Thread(target=server_to_client, args=(u, c), daemon=True).start()
