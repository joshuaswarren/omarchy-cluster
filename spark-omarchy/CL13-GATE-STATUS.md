# CL13 gate status - private-ledger companion

The full checklist with fleet addresses and port maps lives on the private
ledger (removed from this repo by review PR 154). This file carries only the
gate state so the public tree stays current.

## CL06 EXO-pair baseline

- Gate state 2026-11-26 (re-polled ~11:5x CST, SUP-334; re-confirmed 22:5x CST): NO-GO, unchanged.
- No live EXO-pair baseline exists: the receipts tree carries only README,
  the dry-run self-test, and the runner script; no live-run directory on any
  branch, and no macOS-window booking exists yet.
- The gate clears only when a live pair run lands under
  `receipts/milestone1-exo-pair/live-<ts>/`.

## The other CL13 gates

- Off-Spark state backup: staged on the private ledger storage.
- Cutover window: owner approval still owed.

Do not start the install until all three gates are green.
