"""Node-loss and restart drill: kill a worker mid-request and prove the request
path fails cleanly — a bounded error, not a hang or a wrong answer — then
restart the worker and prove the replayed request recovers."""
from __future__ import annotations

import threading
import time


def _timed(fn):
    """Run fn() and record the outcome: (ok, reply-or-error, wall seconds)."""
    t0 = time.monotonic()
    try:
        body = fn()
    except Exception as e:  # noqa: BLE001 - the receipt records the failure mode
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                "elapsed_s": round(time.monotonic() - t0, 3)}
    return {"ok": True, "body": body, "elapsed_s": round(time.monotonic() - t0, 3)}


def _call_bounded(request, deadline_s):
    """request() in a thread joined at deadline_s. None when still running — the
    caller treats an unfinished request as the hang it is; the daemon thread
    dies with the process and never blocks a later step."""
    out = {}
    t = threading.Thread(target=lambda: out.update(_timed(request)), daemon=True)
    t.start()
    t.join(deadline_s)
    return None if t.is_alive() else out


def run_drill(request, kill, restart=None, wait_ready=None, *,
              kill_after_s=1.0, error_deadline_s=10.0, ready_deadline_s=60.0,
              expect_same_output=True):
    """Run the node-loss drill. Every hardware-facing action is a callable so
    the drill logic runs without a cluster:

      request()   one inference request; returns the reply body on success,
                  raises on failure. Called in a thread, so make one call at a
                  time only.
      kill()      kill the worker mid-request (the abrupt failure a crash makes).
      restart()   start the worker again; omit to skip the recovery leg.
      wait_ready(remaining_s) -> bool   the worker is accepting work again;
                  polled until ready_deadline_s.

    Returns the receipt {"verdict": "pass"|"fail", ...}; a "fail" receipt names
    the step and reason.
    """
    steps = {}

    def fail(step, reason):
        return {"verdict": "fail", "step": step, "reason": reason, "steps": steps}

    # 1. the request path works before any fault.
    steps["baseline"] = _timed(request)
    if not steps["baseline"]["ok"]:
        return fail("baseline", "request failed before the drill: %s" % steps["baseline"]["error"])

    # 2. node loss: kill the worker while a request is in flight.
    inflight = {}
    worker = threading.Thread(target=lambda: inflight.update(_timed(request)), daemon=True)
    worker.start()
    time.sleep(kill_after_s)
    kill()
    worker.join(error_deadline_s)
    if worker.is_alive():
        return fail("inflight_kill",
                    "request still running %.1fs after the worker was killed; the path hangs"
                    % error_deadline_s)
    steps["inflight_kill"] = inflight
    if steps["inflight_kill"]["ok"]:
        return fail("inflight_kill",
                    "request completed normally; either the kill missed or the request "
                    "finished first — raise max_tokens or lower kill_after_s")
    if steps["inflight_kill"]["elapsed_s"] > error_deadline_s:
        return fail("inflight_kill", "error took %.1fs to surface; the path hangs"
                    % steps["inflight_kill"]["elapsed_s"])

    # 3. the dead worker fails fast; nothing queues silently behind it.
    steps["dead_worker"] = _call_bounded(request, error_deadline_s)
    if steps["dead_worker"] is None:
        return fail("dead_worker", "request still running %.1fs on a dead worker"
                    % error_deadline_s)
    if steps["dead_worker"]["ok"]:
        return fail("dead_worker", "request succeeded with the worker dead")

    if restart is None:
        return {"verdict": "pass", "recovery": "skipped (no restart callable)", "steps": steps}

    # 4. restart; the worker must accept work again inside ready_deadline_s.
    t0 = time.monotonic()
    restart()
    steps["restart"] = {"elapsed_s": 0.0}
    if wait_ready is not None:
        ready = False
        while time.monotonic() - t0 < ready_deadline_s:
            if wait_ready(max(0.0, ready_deadline_s - (time.monotonic() - t0))):
                ready = True
                break
            time.sleep(0.2)
        steps["restart"] = {"ready": ready, "elapsed_s": round(time.monotonic() - t0, 3)}
        if not ready:
            return fail("restart", "worker not accepting work %.1fs after restart"
                        % ready_deadline_s)

    # 5. the replayed request succeeds, with the same deterministic output.
    steps["replay"] = _timed(request)
    if not steps["replay"]["ok"]:
        return fail("replay", "request failed after restart: %s" % steps["replay"]["error"])
    if expect_same_output and steps["replay"]["body"] != steps["baseline"]["body"]:
        return fail("replay", "replayed request returned different output than baseline")

    return {"verdict": "pass", "steps": steps}
