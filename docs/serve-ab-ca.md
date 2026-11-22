# C-a cross-machine serve A/B: one spec, four fresh arms

Card C-a asks for the same serving workload measured on four devices with one
method, three rounds each, published as a unified medians receipt. The sibling
arms were never measured under a common spec, so per the queue ruling all four
arms run fresh under this one. CL07 consumes the four receipts.

## What every arm runs

- Engine: llama.cpp `llama-server`, driven through the timing path of
  `omarchy_cluster.llamacpp_engine` (`generate`, greedy sampling, `cache_prompt`
  false, llama-server's own timings with the n-1 decode correction).
- Model: one GGUF file, identical bytes on every host -
  `lmstudio-community/Qwen3-4B-Instruct-2507-GGUF` at revision
  `4edb920b6f14e3b9284d4502a6485103d72cde05`, file
  `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, sha256
  `8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc`
  (2,497,280,448 bytes). The runner verifies the hash before anything runs and
  downloads the pinned revision itself when the file is absent.
- Workload: three fixed prompts (`p1-short`, `p2-switch`, `p3-long` - the
  prompt strings are constants in the runner), 128 generated tokens each,
  greedy (temperature 0, top_p 1.0, top_k 0, min_p 0), 3 rounds per arm, one
  llama-server per arm, context 4096, 8 threads.
- Per run the receipt records prompt tokens, TTFT (llama-server `prompt_ms`;
  the first token leaves prompt evaluation on a single stream), decode time and
  rate (the n-1 corrected `predicted_per_second`), completion tokens, finish
  reason, and a token-id digest: the generated text is tokenized by the same
  server (`POST /tokenize`) and the ids hash to sha256. The per-arm `run_digest`
  hashes the per-run digests in run order, so any nondeterminism shows as a
  mismatched digest.
- Per arm the receipt records the llama-server binary sha256 and version, its
  `--list-devices` output, the GGUF hash, the host model and chip from the
  device tree, and uname.

## The four arms

| Arm | Host machine | GPU | Backend family |
| --- | --- | --- | --- |
| rtx5060ti | 16-inch M1 Max MacBook Pro | RTX 5060 Ti | CUDA |
| b60 | 16-inch M1 Max MacBook Pro | Arc B60 (Pro B60) | Vulkan |
| m2max-apple | 14-inch M2 Max MacBook Pro | Apple M2 Max GPU | Vulkan |
| spark-cuda | DGX Spark (GB10) | GB10 CUDA | CUDA |

Each host provisions its own llama.cpp build with the backend above; a build
whose `--list-devices` lists no GPU device is refused by the runner with exit 3
before anything is measured, so a half provisioned host cannot produce a quiet
zero. B60 numbers stay out of upstream llama.cpp channels; they feed the
internal card only.

## Running one arm

```sh
python -m omarchy_cluster.serve_ab --label <arm> --device <cuda|vulkan> \
    --llama-server <path to the host's GPU llama-server> \
    --out <receipt.json> [--port 8129] [--gguf ~/scratch/serve-ab/<file>]
```

The runner starts its own llama-server on 127.0.0.1:PORT with `-ngl 99`, waits
for `/health`, runs the rounds, terminates the server, and writes the receipt
JSON plus one `Card row:` summary line. The four receipts then feed the unified
medians receipt committed under `receipts/`.

## Shared host notes

Two arms share the 16-inch M1 Max host; run them one after another, never
concurrently, so each medians set measures an idle card. The GGUF download is
shared: the first arm fetches it, the second verifies the same sha256 and
skips the download.
