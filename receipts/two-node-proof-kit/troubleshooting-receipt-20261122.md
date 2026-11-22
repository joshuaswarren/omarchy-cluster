# troubleshooting.md two-node proof kit section — command re-runs, 2026-11-22

Card CLX-4 (Main 15:48 CST, lane w72). Done-when: every doc command re-run
with output identical to the receipt. Both doc commands were re-run on this
host (x86_64 Linux CT, repo at d0738cb); output recorded verbatim below.

## 1. `cd receipts/two-node-proof-kit && sha256sum -c --strict kit.sha256`

```
two-node-proof: OK
two-node-proof.sh: OK
check-proof.py: OK
request.json: OK
exit=0
```

## 2. `./two-node-proof.sh --check mac ./ ../missing.gguf /tmp/w72-tnp-out /missing/llama-server /tmp/w72-tnp-tls 127.0.0.1 /missing/rpc-server 127.0.0.2`

A bare host without the two-machine prerequisites: one PASS/FAIL line per
item, `CHECKS <n>`, exit 2, no run directory created (`/tmp/w72-tnp-out` did
not exist after the run).

```
PASS source commit present
FAIL model file present
FAIL model sha256 matches
PASS output directory is fresh
FAIL llama-server executable
FAIL llama-server version readable
FAIL socat with OpenSSL
PASS required tools present
PASS python venv with pip, setuptools and wheel
PASS clock synchronized
PASS ports free
FAIL TLS material and modes
FAIL TLS certificates verify
FAIL peer addresses routable
PASS passwordless sudo
PASS disk space
FAIL AGX busy counters readable
FAIL RPC binary executable
FAIL UFW active
CHECKS 11
exit=2
```

Both outputs match the behavior documented in docs/troubleshooting.md
("Two-node proof kit" section): four OK lines and exit 0 for the hash check;
one line per prerequisite, `CHECKS <n>`, exit 2, no side effects for the
preflight.
