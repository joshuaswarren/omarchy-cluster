#!/usr/bin/env python3
"""Recording proxy for the llama.cpp RPC protocol (65840ed): relays bytes unchanged and logs
every client command (time, command, payload bytes) and every server reply chunk (time, bytes).

Usage: rpcproxy.py LISTEN_PORT TARGET_HOST TARGET_PORT LOG_JSONL
Client frames: [cmd u8][size u64][payload]. The proxy does not parse replies; a command's reply
wait is the time from the command to the next server chunk (commands that reply block the client).
"""
import json
import socket
import struct
import sys
import threading
import time

NAMES = ["ALLOC_BUFFER", "GET_ALIGNMENT", "GET_MAX_SIZE", "BUFFER_GET_BASE", "FREE_BUFFER", "BUFFER_CLEAR",
         "SET_TENSOR", "SET_TENSOR_HASH", "GET_TENSOR", "COPY_TENSOR", "GRAPH_COMPUTE", "GET_DEVICE_MEMORY",
         "INIT_TENSOR", "GET_ALLOC_SIZE", "HELLO", "DEVICE_COUNT", "GRAPH_RECOMPUTE", "MEMSET_TENSOR"]
lock = threading.Lock()


def log(out, rec):
    with lock:
        out.write(json.dumps(rec) + "\n")


def recv_exact(s, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = s.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError
        buf += chunk
    return bytes(buf)


def client_to_server(c, s, out, conn):
    try:
        while True:
            head = recv_exact(c, 9)
            cmd, size = head[0], struct.unpack("<Q", head[1:])[0]
            t = time.perf_counter()
            payload = recv_exact(c, size)
            s.sendall(head + payload)
            log(out, {"c": conn, "t": t, "dir": "cmd", "cmd": NAMES[cmd] if cmd < len(NAMES) else cmd, "bytes": size})
    except (ConnectionError, OSError):
        pass
    finally:
        for x in (s, c):
            try:
                x.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def server_to_client(s, c, out, conn):
    try:
        while True:
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            log(out, {"c": conn, "t": time.perf_counter(), "dir": "reply", "bytes": len(chunk)})
            c.sendall(chunk)
    except OSError:
        pass


def main():
    port, host, tport, path = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
    out = open(path, "a", buffering=1)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(8)
    n = 0
    while True:
        c, _ = srv.accept()
        s = socket.create_connection((host, tport))
        for x in (c, s):
            x.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        n += 1
        threading.Thread(target=client_to_server, args=(c, s, out, n), daemon=True).start()
        threading.Thread(target=server_to_client, args=(s, c, out, n), daemon=True).start()


if __name__ == "__main__":
    main()
