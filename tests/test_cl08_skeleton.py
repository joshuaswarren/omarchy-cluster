"""SUP-400: pin the CL08 test seam SUP-171 named — the KV transfer is an
injectable callable that runs exactly once per request, between save and
restore (docs/joint-request-path.md, PR #149). Phase bodies are skeleton
stubs; these tests patch them and exercise the orchestration contract."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import llamacpp_engine as lce

CFG = {"prefill_url": "http://p", "decode_url": "http://d",
       "prefill_kv": "KV-SRC", "decode_kv": "KV-DST", "slot": 0, "kv_name": "kv.bin"}


def _patch_phases(monkeypatch, calls):
    monkeypatch.setattr(lce, "cl08_prefill",
                        lambda *a: calls.append("prefill") or {"prefill_ms": 1.0})
    monkeypatch.setattr(lce, "cl08_slot_save",
                        lambda *a: calls.append("save") or {"n_written": 7, "save_ms": 2.0})
    monkeypatch.setattr(lce, "cl08_slot_restore",
                        lambda *a: calls.append("restore") or {"restore_ms": 4.0})
    monkeypatch.setattr(lce, "cl08_decode",
                        lambda *a: calls.append("decode") or {"decode_ms": 5.0})


def test_injected_transport_runs_once_between_save_and_restore(monkeypatch):
    calls = []
    _patch_phases(monkeypatch, calls)

    def fake_transport(name, src, dst):
        calls.append("transport")
        assert (name, src, dst) == ("kv.bin", "KV-SRC", "KV-DST")
        return {"bytes": 7, "seconds": 0.5, "path": "loopback"}

    out = lce.joint_generate(CFG, {}, transport=fake_transport)

    assert calls == ["prefill", "save", "transport", "restore", "decode"]
    assert "error" not in out
    tel = out["telemetry"]
    assert tel["transport"] == "loopback"
    assert tel["lan_mbps"] is None
    assert tel["kv_bytes"] == 7
    # phases holds the flat joint_engine metric keys, not per-phase dicts.
    assert tel["phases"] == {"prefill_ms": 1.0, "save_ms": 2.0,
                             "transfer_ms": 500.0, "restore_ms": 4.0,
                             "decode_ms": 5.0}


def test_a_transport_failure_names_the_phase_and_stops_before_decode(monkeypatch):
    calls = []
    _patch_phases(monkeypatch, calls)

    def bad_transport(name, src, dst):
        calls.append("transport")
        raise OSError("kv copy truncated")

    out = lce.joint_generate(CFG, {}, transport=bad_transport)

    # save completed, transport failed, restore/decode never ran — the seam's
    # negative path (a transport failure must stop decode, PR #149). The
    # error contract mirrors joint_engine: str(e) is the bare message and
    # errors carry {"phase", "message"} dicts, phase named "transport".
    assert calls == ["prefill", "save", "transport"]
    assert out["error"] == "kv copy truncated"
    assert out["telemetry"]["errors"] == [{"phase": "transport",
                                           "message": "kv copy truncated"}]


def test_the_default_transport_and_phase_stubs_are_declared_not_implemented():
    for fn, args in ((lce.transport_kv, ("kv.bin", None, None)),
                     (lce.cl08_prefill, (CFG, {}, 0)),
                     (lce.cl08_slot_save, (CFG, 0, "kv.bin")),
                     (lce.cl08_slot_restore, (CFG, 0, "kv.bin")),
                     (lce.cl08_decode, (CFG, {}, 0))):
        with pytest.raises(NotImplementedError):
            fn(*args)


def test_the_skeleton_telemetry_mirrors_the_joint_engine_shape(monkeypatch):
    """telemetry keys are joint_engine._new_telemetry()'s, so readers written
    against this surface read real telemetry unchanged (SUP-400 review)."""
    from omarchy_cluster import joint_engine

    calls = []
    _patch_phases(monkeypatch, calls)
    out = lce.joint_generate(CFG, {}, transport=lambda n, s, d: {
        "bytes": 1_000_000, "seconds": 0.5, "path": "lan"})

    assert set(out["telemetry"]) == set(joint_engine._new_telemetry())
    assert out["telemetry"]["lan_mbps"] == 16.0
    # a failed run still returns the same telemetry shape plus `error`
    calls.clear()
    out = lce.joint_generate(CFG, {}, transport=lambda *a: (_ for _ in ()).throw(OSError("x")))
    assert set(out["telemetry"]) == set(joint_engine._new_telemetry())
    assert out["telemetry"]["errors"] == [{"phase": "transport", "message": "x"}]
