"""SUP-400: the CL08 joint-request surface on llamacpp_engine delegates to
joint_engine (PR #133) — no parallel phase surface. These tests run the real
five-phase handoff through the surface against the conftest fake llama pair:
the KV moves exactly once per request, between save and restore, through the
injectable transport seam (SUP-171, docs/joint-request-path.md)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import joint_engine as je
from omarchy_cluster import llamacpp_engine as lce

from conftest import cfg_for


def test_the_surface_is_joint_engines_contract_not_a_copy():
    assert lce.JointError is je.JointError
    assert lce.transport_kv is je.transport_kv
    assert lce.CL08_PHASES == ("prefill", "save", "transport", "restore", "decode")


def test_joint_generate_runs_the_real_five_phases(llama_pair):
    prefill, decode = llama_pair
    res = lce.joint_generate(cfg_for(prefill, decode),
                             {"prompt": "p" * 8000, "max_tokens": 64})

    assert res["text"] == "decoded text"
    # real phase order across both fakes: prefill completion -> save on the
    # prefill side, restore -> decode completion -> tokenize on the decode side
    assert prefill.events == ["completion", "save"]
    assert decode.events == ["restore", "restored:13", "completion", "tokenize"]
    tel = res["telemetry"]
    assert tel["transport"] == "loopback"     # LocalKv -> LocalKv copy
    assert tel["kv_bytes"] == len(b"KV-BYTES-8192")
    assert set(tel["phases"]) >= {"prefill_ms", "save_ms", "transfer_ms",
                                  "restore_ms", "decode_ms"}


def test_the_injected_transport_runs_once_between_save_and_restore(llama_pair):
    prefill, decode = llama_pair
    calls = []

    def fake_transport(name, src, dst):
        calls.append(name)
        assert name == "joint-kv.bin"
        assert src.path == prefill.slot_dir and dst.path == decode.slot_dir
        # a real transport moves the file; the double does the same so the
        # restore phase finds it (a no-op double fails restore for real)
        data = open(os.path.join(src.path, name), "rb").read()
        with open(os.path.join(dst.path, name), "wb") as f:
            f.write(data)
        return {"bytes": len(data), "seconds": 0.5, "path": "loopback"}

    res = lce.joint_generate(cfg_for(prefill, decode), {"prompt": "x"},
                             transport=fake_transport)

    assert calls == ["joint-kv.bin"]
    # save already ran when transport was called; restore/decode ran after it
    assert prefill.events == ["completion", "save"]
    assert decode.events == ["restore", "restored:13", "completion", "tokenize"]
    assert res["telemetry"]["phases"]["transfer_ms"] == 500.0
    assert res["telemetry"]["kv_bytes"] == len(b"KV-BYTES-8192")


def test_a_transport_failure_names_the_phase_and_stops_before_decode(llama_pair):
    prefill, decode = llama_pair

    def bad_transport(name, src, dst):
        raise OSError("kv copy truncated")

    res = lce.joint_generate(cfg_for(prefill, decode), {"prompt": "x"},
                             transport=bad_transport)

    assert res["error"] == "kv copy truncated"
    assert res["telemetry"]["errors"] == [{"phase": "transport",
                                           "message": "kv copy truncated"}]
    # save completed, transport failed, restore/decode never ran
    assert prefill.events == ["completion", "save"]
    assert decode.events == []
