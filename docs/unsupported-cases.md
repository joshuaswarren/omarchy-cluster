# Node-loss and restart: supported and unsupported cases

What `omarchy_cluster/drill.py` proves, what the merged code covers, and what
remains unsupported. Row CL16: the drill code is merged and hardened
(PRs #138, #157); the live drill receipt waits on CL15 (Omarchy on both nodes).

## Covered

| Case | How |
|---|---|
| Worker process loss mid-request on the RPC/llamacpp path | `drill.py` kills the test-owned worker; the request returns a bounded error, not a hang or a wrong answer |
| Restart and replay | `drill.py` restarts the worker (and optionally the gateway); the replayed request returns the same deterministic output |
| Node stops answering | `omarchy-cluster guard` polls and stops the run ([troubleshooting](troubleshooting.md)) |

## Unsupported

| Case | What holds today |
|---|---|
| Automatic failover | None exists. A lost node's request errors to the client; nothing re-routes it. |
| In-flight continuation | The in-flight request dies with the worker. Recovery is a client replay, not a resume; partially generated state is not checkpointed or continued. |
| Host reboot or power loss | The drill kills a process, not a node. Whole-host loss, reboot, and post-reboot state are not exercised. |
| Gateway loss mid-request | The drill can restart the gateway, but client-visible behavior while it is down is unverified. |
| Joint-engine node loss | `drill.py` targets the RPC/llamacpp engine. The joint path's failure modes — prefill host loss mid-prefill, mid-transfer, or mid-restore — are unexercised; they surface as `JointError(phase)` in `telemetry.errors` ([joint-request-path](joint-request-path.md)). |
| Multiple simultaneous node loss, LAN partition asymmetry, split-brain | Untested; the drill exercises one worker at a time on a healthy link. |
| Client retry semantics beyond the bounded error | No retry-after contract, partial-token accounting, or request deduplication. |

## Receipt status

Live two-node drill receipt: pending CL15. When the drill runs on the
Omarchy-on-both cluster, its receipt goes under `receipts/` and this matrix's
covered list is updated to what the receipt actually proves.
