# omarchy-cluster PR #66 smoke (2026-11-13)

Host: 16-inch M1 Max MacBook Pro, GTK 4.22, launched with `GSK_RENDERER=cairo`. Application source SHA-256: `50d12c5c9a9bc56a7350cf33848c762626abf3957cef6d7f9dad9bccb678dad4`. The real GTK window opened; no screenshot is included because the captured desktop contained unrelated private notifications.

The branch gateway ran on loopback and forwarded to the approved OpenAI-compatible router. The local engine adapter used the approved model group and owner-provided key file; no key material or private endpoint details are recorded here. GPU inference stayed remote.

Bounded live endpoint smoke: pull of `hf-internal-testing/tiny-random-PhiForCausalLM` succeeded; two sequential chat requests to active alias `omarchy-cluster` both returned HTTP 200 (3- and 5-character replies, finish reason `stop`, max_tokens 12). Redacted gateway log:

```text
access POST /models/pull HTTP/1.1 200 -
access POST /v1/chat/completions HTTP/1.1 200 -
access POST /v1/chat/completions HTTP/1.1 200 -
```

The app window populated the active-model list through its startup refresh. The pull and chat requests above were sent directly to the loopback gateway, not through widget input. Therefore the full widget-driven pull/chat acceptance remains incomplete; this receipt does not claim otherwise. No local inference was started.