# Run the two-node proof

The [reference run](reference-combine.json) used an Apple M2 Max on Omarchy Linux with Borealis Vulkan, AGX counters and active UFW, and an NVIDIA DGX Spark (GB10, CUDA) on the same private IPv4 LAN. Other hardware is unverified. Shared decode was slower than either solo run; this kit tests whether the split works, not whether it is faster.

Use synchronized clocks, matching RPC-enabled llama.cpp builds, and the exact `Qwen3-1.7B-Q4_K_M.gguf` with SHA-256 `d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5`. `SOURCE` is a Git checkout containing [commit `10a47b6`](https://github.com/joshuaswarren/omarchy-cluster/commit/10a47b69df53f60b1e9b3f67aa5092fdff2c4562). The script installs that commit into a fresh venv using system setuptools and wheel. The Mac needs noninteractive sudo for UFW.

Both hosts need **socat with OpenSSL support** on `PATH`. On Debian/Ubuntu (including DGX Spark): `sudo apt install socat`. On Omarchy/Arch: `sudo pacman -S socat`. Run `socat -V | grep WITH_OPENSSL` on both. See `bash two-node-proof.sh --help` for other tools; the Spark sampler uses `nvidia-smi --query-compute-apps`, not `pmon`. Verify the three kit files with `sha256sum -c --strict kit.sha256`.

Follow only **Make short-lived certificates** in [the secure RPC recipe](https://github.com/joshuaswarren/omarchy-cluster/blob/f8337000599ef0d046ab6b982c4fda5d99d2434e/docs/secure-rpc.md). For the Mac certificate, use `san=DNS:cluster-m2-rpc,IP:$M2_ADDR` so it verifies by both name and address. Let the kit set up and tear down the relay and scoped firewall rule. Do not open the raw RPC port or share the CA key. Set `SOURCE`, `GGUF`, `OUT`, `LLAMA_SERVER` and `TLS` separately on each host. `OUT` must not exist; `TLS` is disposable. Set `M2_ADDR` and `SPARK_ADDR` to your hosts' private LAN IPv4 addresses and `RPC_SERVER` to the Mac worker binary.

Mac first:

```bash
bash two-node-proof.sh mac "$SOURCE" "$GGUF" "$OUT" "$LLAMA_SERVER" "$TLS" "$M2_ADDR" "$RPC_SERVER" "$SPARK_ADDR"
```

On Spark, use an operator-approved window for the host-wide cache drop. Each CUDA launch runs `sync` and `sudo -n sysctl -q vm.drop_caches=3`, records `MemFree`, `MemAvailable`, `Cached`, and resident GPU allocations, and refuses if `MemFree` is below 6 GiB. The cache drop can temporarily slow other workloads; it does not stop them.

After the Mac prints `READY`, run on Spark (the Mac waits up to four minutes):

```bash
bash two-node-proof.sh spark "$SOURCE" "$GGUF" "$OUT" "$LLAMA_SERVER" "$TLS" "$M2_ADDR"
```

After Spark exits, stop the Mac command with Ctrl-C so it can tear down. Copy both output directories to one host and verify them together:

```bash
python3 check-proof.py combine "$MAC_OUT" "$SPARK_OUT" > combine.json
```

A passing result has three decode rates, 64 matching token IDs in all arms, and RPC replies and Mac GPU activity during the shared request. Spark must show CUDA0 and RPC0 in its engine log, a nonzero CUDA0 model buffer, and positive GPU memory for its test PID in a one-second sample taken during the request. The Mac firewall snapshots must match before and after. Attach the combined JSON and hashes of both receipt archives to the help-wanted issue. Do not attach private keys, model weights, or network details. If a run fails, report the first failure and whether teardown completed instead of posting a pass.
