# omarchy-cluster PR #66 smoke (2026-11-13)

Host: 16-inch M1 Max MacBook Pro (`jw16-linux`), GTK 4.22, launched with `GSK_RENDERER=cairo`. Application source SHA-256: `50d12c5c9a9bc56a7350cf33848c762626abf3957cef6d7f9dad9bccb678dad4`. The real GTK window opened; screenshot: [gtk-window.png](gtk-window.png).

The branch gateway ran on loopback and forwarded to the approved LiteLLM router at CT 350. The local engine adapter used router model group `core` and the owner-provided key file; no key material is recorded here. GPU inference stayed remote.

Bounded live endpoint smoke: pull of `hf-internal-testing/tiny-random-PhiForCausalLM` succeeded; two sequential chat requests to active alias `omarchy-cluster` both returned HTTP 200 (3- and 5-character replies, finish reason `stop`, max_tokens 12). Redacted gateway log:

```text
access POST /models/pull HTTP/1.1 200 -
access POST /v1/chat/completions HTTP/1.1 200 -
access POST /v1/chat/completions HTTP/1.1 200 -
```

The app window populated the active-model list through its startup refresh. The pull and chat requests above were sent directly to the loopback gateway, not through widget input. Therefore the full widget-driven pull/chat acceptance remains incomplete; this receipt does not claim otherwise. Initial jw16 GPU utilization was 20% (1,118 MiB); no local inference was started.