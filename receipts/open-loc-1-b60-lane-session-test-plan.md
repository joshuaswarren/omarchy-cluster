# OPEN-loc-1: lane-session test plan for the B60 local-model lane

Date filed: 2026-11-28 (w73). Base evidence: w9C's B60 lane-session reports
(journal 2026-11-26/27), the 65,723-token standing prompt, measured prefill
~42 tok/s on the B60 Vulkan, and the trial-key expiry chain (4 h key expired
23:43:44Z; 6 h corrective key minted after).

## What the trials proved and what failed

1. The B60 Vulkan stack works: llama-server pid live on the xe driver,
   vulkaninfo reports Arc Pro B60 (BMG G21), prefill measured ~42 tok/s.
2. The lane-session model (a full agent transcript replay) is the wrong
   shape for this card today: the standing prompt is 65,723 tokens and grows
   with every resume, so each serve window faces a ~70k-token prefill =
   ~28 minutes of prefill alone per attempt.
3. The failure mode was operational, not silicon: the scoped trial key
   expired mid-cycle (401 before prefill), and serve windows chain-closed
   around it; one turn completed mechanically but its final report was
   refused by an argparse passthrough error in the harness invocation.

## Test plan - lane-session on the B60, sized for the card

Phase 0 - fixed serve (removes both operational failure modes)

- [ ] Mint the session key for the FULL planned window plus 50 percent
      margin (no mid-window expiry; expiry was the 00:16 turn's cause).
- [ ] Pin the serve window to the key's lifetime: one serve up, one serve
      down, no chain-closing between attempts.
- [ ] Fix the runner invocation: the task prompt must be passed with a
      defined argument convention (the stray-period argparse passthrough);
      add a `set --` normalization and a shellcheck gate on the runner.

Phase 1 - capability ladder (small to large, each step a hard gate)

- [ ] L1: single-call probe, 512-token prompt, 128 generated tokens,
      3 repeats. Gate: p50 prefill tok/s recorded; no 401; no serve restart.
- [ ] L2: 4k-token prompt (the lane-session system prompt size), 256
      generated, 3 repeats. Gate: prefill under 5 minutes, transcript
      digest stable across repeats.
- [ ] L3: 16k-token prompt, 256 generated, 3 repeats. Gate: completes
      inside one serve window with margin.
- [ ] L4: the full standing prompt minus growth: replay the transcript
      FROZEN at its 2026-11-26 state (65,723 tokens), 512 generated, one
      pass. Gate: completes; report prefill minutes, tok/s, and total
      wall. This is the card's honest capacity number.

Phase 2 - the lane-session run (only after L4 passes)

- [ ] Run the frozen transcript replay as the session: same serve window,
      same key, the argparse-fixed runner.
- [ ] Acceptance: the final report lands without the argparse refusal, and
      the session completes inside the serve window.
- [ ] Record: prefill tok/s per phase, wall per phase, key/serve window
      boundaries, and any transcript growth (freeze the transcript: no
      resume-appends during the measured window).

## Receipt format

`receipts/open-loc-1-b60-lane-session.md`: one table per phase with the
gate results, the frozen-transcript sha256, the key/window boundaries, and
the L4 capacity numbers. Failures kept as findings, not hidden.

## Sizing note carried from the trials

At ~42 tok/s prefill, every 1k of prompt growth costs ~24 s of prefill.
The frozen-transcript rule exists because the growing transcript was the
difference between a 28-minute window and an unbounded one.
