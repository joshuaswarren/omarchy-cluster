# Run the two-node proof

The [reference run](reference-combine.json) used an Apple M2 Max on Omarchy Linux with Borealis Vulkan, AGX counters and active UFW, and an NVIDIA DGX Spark (GB10, CUDA) on the same private IPv4 LAN. Other hardware is unverified. Shared decode was slower than either solo run; this kit tests whether the split works, not whether it is faster.

Use synchronized clocks, matching RPC-enabled llama.cpp builds, and the exact `Qwen3-1.7B-Q4_K_M.gguf` with SHA-256 `d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5`. `SOURCE` is a Git checkout containing [commit `10a47b6`](https://github.com/joshuaswarren/omarchy-cluster/commit/10a47b69df53f60b1e9b3f67aa5092fdff2c4562): `git clone https://github.com/joshuaswarren/omarchy-cluster` then `git -C omarchy-cluster fetch origin 10a47b69df53f60b1e9b3f67aa5092fdff2c4562`. The script installs that commit into a fresh venv using system setuptools and wheel. The Mac needs noninteractive sudo for UFW.

**The llama.cpp builds must come from the same upstream commit on both hosts.** The Mac's `LLAMA_SERVER` needs `GGML_VULKAN=ON` (Borealis) and `GGML_RPC=ON`; the Spark's `LLAMA_SERVER` needs `GGML_CUDA=ON` and `GGML_RPC=ON`; the Mac's `RPC_SERVER` is the `ggml-rpc-server` binary from that same tree and commit. A build from a different commit (or a different ggml version) breaks the shared request. The kit records each host's `llama-server --version` into `llama-version.txt`, and `check-proof.py` fails when the two hosts report different versions.

**Budget disk and RAM.** The model file is 1.3 GB; keep it on a filesystem with at least 2 GB free. On the Mac, `/tmp` is a tmpfs, so the model there costs RAM.

Both hosts need **socat with OpenSSL support** on `PATH`. On Debian/Ubuntu (including DGX Spark): `sudo apt install socat`. On Omarchy/Arch: `sudo pacman -S socat`. Run `socat -V | grep WITH_OPENSSL` on both. See `bash two-node-proof.sh --help` for other tools; the Spark sampler uses `nvidia-smi --query-compute-apps`, not `pmon`. Verify the four kit files with `sha256sum -c --strict kit.sha256`.

**One command runs preflight and then the arm.** `./two-node-proof mac …` prints every PASS/FAIL prerequisite line and only starts the proof when preflight reports `CHECKS 0`. Use `./two-node-proof --check …` when you want the same preflight without launching.

Follow only **Make short-lived certificates** in [the secure RPC recipe](../../docs/secure-rpc.md). For the Mac certificate, use `san=DNS:cluster-m2-rpc,IP:$M2_ADDR` so it verifies by both name and address. Let the kit set up and tear down the relay and scoped firewall rule. Do not open the raw RPC port or share the CA key. The TLS directory must be mode `700` and the private key mode `600`, and each host gets only its own role's files (`ca.crt`, `m2.crt`, `m2.key` on the Mac; `ca.crt`, `spark.crt`, `spark.key` on the Spark). `OUT` must not exist; `TLS` is disposable. The script takes every path as a positional argument (apart from the optional `SOCAT` override, nothing is read from the environment): `SOURCE`, `GGUF`, `OUT`, `LLAMA_SERVER`, `TLS`, `M2_ADDR`, `RPC_SERVER` (Mac only) and `SPARK_ADDR` (Mac only). Reserve ports `24820`, `24831`, `24832`, `24860`, `24861` (Mac) and `24864` (Spark).

The Mac arm needs an exclusive GPU window: the kit stops an `omarchy-cluster` serve if one is running on that host. The Spark arm needs an operator-approved window for the host-wide cache drop.

Mac first:

```bash
./two-node-proof mac "$SOURCE" "$GGUF" "$OUT" "$LLAMA_SERVER" "$TLS" "$M2_ADDR" "$RPC_SERVER" "$SPARK_ADDR"
```

On Spark, use an operator-approved window for the host-wide cache drop. Each CUDA launch runs `sync` and `sudo -n sysctl -q vm.drop_caches=3`, records `MemFree`, `MemAvailable`, `Cached`, and resident GPU allocations, and refuses if `MemFree` is below 6 GiB. The cache drop can temporarily slow other workloads; it does not stop them.

After the Mac prints `READY`, run on Spark (the Mac waits up to four minutes):

```bash
./two-node-proof spark "$SOURCE" "$GGUF" "$OUT" "$LLAMA_SERVER" "$TLS" "$M2_ADDR"
```

After Spark exits, stop the Mac command with Ctrl-C so it can tear down. Copy both output directories to one host and verify them together:

```bash
python3 check-proof.py combine "$MAC_OUT" "$SPARK_OUT" > combine.json
```

A passing result has three decode rates, 64 token IDs that match across all three arms, and RPC replies and Mac GPU activity during the shared request. The three arms must agree with each other; they need not reproduce our reference digest, and the receipt records `matches_reference` per arm so a run that also matches ours is visible. Spark must show CUDA0 and RPC0 in its engine log, a nonzero CUDA0 model buffer, and positive GPU memory for its test PID in a one-second sample taken during the request. On the Mac, UFW must be active with no `24861` rule before the run, the scoped `24861/tcp` `cluster-proof-` rule must be active during the run, and the snapshot after the run must match the snapshot before. Attach the combined JSON and hashes of both receipt archives to the help-wanted issue. Do not attach private keys, model weights, or network details. If a run fails, report the first failure and whether teardown completed instead of posting a pass.
