# llama-server threads: performance cores only

2026-10-07. `serve --engine llamacpp` started llama-server with one thread per CPU. On an M1 Max
under Omarchy Linux that is 10 threads: 8 performance cores (`cpu_capacity` 1024) and 2 efficiency
cores (485). Every OpenMP op waits for its slowest thread, so the two efficiency cores slowed the
whole host side. That includes the lm_head the host computes for every token of an iPhone or RPC split.

The engine now defaults to `facts.perf_cores()`. On Linux that is the CPUs at the highest
`cpu_capacity`; on macOS it is `hw.perflevel0.physicalcpu`. It returns 8 on this M1 Max.

## Measured (M1 Max laptop, Omarchy Linux, llama.cpp 65840ed CPU build, Qwen3-1.7B Q4_K_M)

`llama-bench -p 128 -n 64 -r 3`, `OMP_WAIT_POLICY=ACTIVE`, arms interleaved twice, inside the
machine's GPU-queue ticket (nothing else ran):

| Threads | Decode tg64 tok/s | Prefill pp128 tok/s |
|---|---|---|
| `-t 10` (old default) | 65.44 ± 0.19, 64.87 ± 0.04 | 255.66 ± 0.36, 249.39 ± 0.15 |
| `-t 8` (new default) | 71.63 ± 0.05, 71.72 ± 0.02 | 348.85 ± 0.19, 333.41 ± 4.51 |
| `taskset -c 2-9 -t 8` | 71.37 ± 0.02, 71.34 ± 0.05 | 348.87 ± 0.39, 326.78 ± 3.22 |

Decode is about 10% faster and prefill about 34% faster. Pinning to the performance cores adds
nothing over 8 threads.

The mechanism, measured at 2 threads: two performance cores gave 37.66 and 37.70 tok/s; one efficiency
plus one performance core gave 23.99 and 24.01; two efficiency cores gave 12.07.

## A machine with as many efficiency cores as performance cores

13-inch M1 MacBook Pro (4 performance cores at 1024 + 4 efficiency cores at 493), Omarchy Linux,
llama.cpp 65840ed CPU build. `llama-server`, Qwen3-1.7B Q4_K_M, a 130-token prompt and 64 decoded
tokens per `/completion`, 1 warm-up + 3 timed requests per arm, arms interleaved, under the machine's
GPU lock:

| Threads | Decode tok/s | Prefill tok/s |
|---|---|---|
| `-t 4` (perf_cores, the default) | 36.70-37.01 (15 requests) | 145-191 |
| `-t 8` (one per CPU, the old default) | 31.09-34.47 (one outlier 27.47) | 186-221 |
| `-t 4 -tb 8` | 36.74-36.97 | 181-197 |

- Performance cores alone decode about 12% faster here too.
- Prefill, which is compute-bound, gains from the 4 efficiency cores: about 15-20% more with 8 threads.
  Prefill fell from rep to rep in every arm (thermal), so compare within a rep.
- `-tb` with every CPU would recover prefill here, but on the M1 Max above 10 threads cost 34% of
  prefill, so the default stays at one count. Pass `--threads` to override on a given machine.

Not measured: the effect on the iPhone split itself, because the phone was unplugged. The earlier
device run through `serve --engine llamacpp` (33.84 tok/s) used the old 10-thread default.

## rpc-server threads

`rpc-start` now passes `-t facts.perf_cores()` to `ggml-rpc-server`. Without `-t`, the server uses
half of all CPUs: 5 on the M1 Max, which leaves 3 performance cores idle.

Server: the M1 Max above, `ggml-rpc-server -d CPU`, llama.cpp 65840ed CPU build. Client: the 13-inch
M1 above, `llama-server --rpc <server> -dev RPC0 -ngl 99 -ot '^output\.weight=CPU' -t 4`, so every
layer runs on the server and the client computes only the lm_head. Wired LAN. Qwen3-1.7B Q4_K_M, a
130-token prompt and 64 decoded tokens per `/completion`, 1 warm-up + 3 timed requests per arm, arms
interleaved 3 times, both machines under their GPU locks. Server load average before the run was 0.62.

| Server threads | Prefill tok/s, median [range] | Decode tok/s, median [range] |
|---|---|---|
| `-t 5` (ggml-rpc-server default) | 109.2 [108.4-109.9] | 40.06 [31.76-42.04] |
| `-t 8` (perf_cores, the new default) | 164.5 [161.9-169.0] | 38.22 [32.82-46.96] |

- Prefill is about 51% faster with 8 threads. The ranges do not overlap.
- Decode shows no difference. Each decoded token costs one network round trip, so the request-to-request
  spread (about 10 tok/s in both arms) is larger than any thread effect.
