# Try it: an iPhone as a node

An iPhone runs llama.cpp's `rpc-server` inside a small app. A Linux machine with the phone on
USB uses it as an RPC device for llama.cpp. Everything here builds, signs and installs from
Linux. No Mac and no Xcode.

This is a capacity node, not a speedup. A phone layer costs about 0.8 ms per token against
about 0.5 ms on an M1 Max laptop CPU, so a split is slower than the laptop alone for any model
the laptop holds. The phone adds memory: about 2.2 GB of layers on an 8 GB iPhone 15 Pro Max.
`omarchy-cluster serve --engine llamacpp` therefore puts layers on the phone only when the
model does not fit on the host.

## What you need

- An iPhone with Developer Mode on, plugged into the Linux machine by USB and trusted. Tested:
  iPhone 15 Pro Max, iOS 27.0.1.
- [xtool](https://github.com/xtool-org/xtool) with `xtool setup` done and `xtool auth` logged
  in to an Apple developer account. xtool provides the iOS SDK and signs the app.
- [pymobiledevice3](https://github.com/doronz88/pymobiledevice3) for the USB port forward and
  the app launch.
- clang, `ld64.lld` and cmake, and a llama.cpp checkout (tested at commit 65840ed).

## 1. Build

```sh
# an SDK copy whose .tbd stubs list arm64, so ld64.lld can link a plain arm64 app
ios/prepare-sdk.sh \
  ~/.cache/xtool/darwin-iPhoneOS27.0.xtoolsdk/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk \
  ~/sdk/iPhoneOS.sdk

LLAMA=~/src/llama.cpp SDK=~/sdk/iPhoneOS.sdk ios/build.sh
# -> ios/out/PhoneInference.app (unsigned)
```

`ios/build.sh` cross-compiles ggml with the CPU, Metal and RPC backends, compiles
`tools/rpc/rpc-server.cpp` with its `main` renamed, and links it with
`ios/PhoneInference/main.m`. The Metal shader source is embedded; the phone compiles it at
start, which takes a few seconds.

Why the UIKit host: iOS 27 kills a process that does not finish UIKit launch in about 20 s,
and traps an app that does not adopt the scene lifecycle. `main.m` runs `UIApplicationMain`
with a scene delegate and starts the RPC server on a worker thread. It also turns off the idle
timer, because a locked or backgrounded app stops serving.

## 2. Sign and install

```sh
xtool install --usb ios/out/PhoneInference.app
```

xtool provisions the app, signs it and installs it. It prefixes the bundle id with your team:
the installed id looks like `XTL-<TEAM>.dev.omarchy.phoneinference`. Find it with
`pymobiledevice3 apps list --type User`.

Use xtool's signer. The same binary signed by rcodesign 0.29.0 installed but failed at exec on
iOS 27 (`NSPOSIXErrorDomain 85`, EBADEXEC); the cause is still open.

## 3. Run the phone as an RPC device

With `omarchy-cluster`:

```sh
omarchy-cluster serve model.gguf --engine llamacpp --llama-server /path/to/llama-server \
  --ios-bundle XTL-<TEAM>.dev.omarchy.phoneinference
# OpenAI endpoint on :8020; add --rpc-layers N to force N layers onto the phone
omarchy-cluster stop
```

By hand:

```sh
pymobiledevice3 usbmux forward 50052 50052 &        # host 127.0.0.1:50052 -> phone
pymobiledevice3 developer dvt launch --kill-existing \
  "XTL-<TEAM>.dev.omarchy.phoneinference -H 127.0.0.1 -p 50052 -t 4"   # add "-d CPU" for the phone CPU
OMP_WAIT_POLICY=ACTIVE llama-server -m model.gguf --rpc 127.0.0.1:50052 -ngl 28 \
  -ot "^output\.weight=CPU"
```

Two flags decide the speed:

- `-ot "^output\.weight=CPU"` keeps the lm_head on the host. Without it, llama.cpp ran Qwen3's
  tied lm_head on the phone and sent 593 KB of logits over USB per token. Keep the `^`: without
  it the pattern also matches every `blk.N.attn_output.weight`.
- `OMP_WAIT_POLICY=ACTIVE` keeps the host's OpenMP workers awake while the phone computes.

Keep the phone unlocked with the app in front. Keep the phone's share under about 2.2 GB:
4.4 GB of layers on an 8 GB phone pushed iOS into system-wide memory pressure.

## Measured

Qwen3-1.7B Q4_K_M, M1 Max laptop on Omarchy Linux + iPhone 15 Pro Max on USB, lm_head on the
laptop, llama.cpp 65840ed, `llama-bench -t 8 -n 64`.

| Setup | Decode tok/s | Prefill (128 tokens) tok/s |
|---|---:|---:|
| Laptop alone (CPU) | 71.1 | 349 |
| All 28 layers on the phone GPU, `OMP_WAIT_POLICY=ACTIVE` | 35.1 | 424 |
| 7 layers on the phone GPU | 35.5 | 334 |
| 14 layers on the phone GPU | 31.6 | 341 |
| 14 layers on the phone GPU, `OMP_WAIT_POLICY=ACTIVE` | 45.0 | |
| 14 layers on the phone CPU | 30.0 | 107 |

The phone GPU processes a prompt faster than the laptop CPU. Decode is bound by the phone's
memory bandwidth: 28 layers read about 1 GB of weights in 23 ms.

Greedy output with the phone on its CPU is byte-identical to the laptop alone at 7, 14 and 28
phone layers. With the phone GPU, the mean KL divergence against the laptop is 0.008 and the top
token agrees 96 percent of the time.

Speculative decoding with a Qwen3-0.6B draft on the phone and a Qwen3-8B target on the laptop
ran at 6.6 tok/s, against 25.1 tok/s with the draft on the laptop.

Full tables, traces and raw logs: `receipts/2026-10-06-iphone-rpc-node/`. Those tables came
from the iOS 17 deployment target build in that receipt. The iOS 18 build from `ios/build.sh`
was checked end to end on the same phone: `xtool install`, then
`omarchy-cluster serve Qwen3-1.7B-Q4_K_M.gguf --engine llamacpp --ios-bundle ... --rpc-layers 28`
decoded 64 tokens through the gateway at 33.8 tok/s.
