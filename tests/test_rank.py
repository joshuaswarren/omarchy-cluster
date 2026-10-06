import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import rank


class ChatTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt):
        assert add_generation_prompt
        return [100000] + [ord(c) for c in messages[-1]["content"]]


def test_rank1_long_poll_gets_rank0_prompt_tokens_once(monkeypatch):
    """rank 1 must enter stream_generate with rank 0's exact tokens, or rank 0
    blocks forever in the pipeline recv (ClusterRun3 deadlock), and must get
    them when rank 0 publishes, not on a later poll (ClusterRun5: rank 1's
    0.5 s poll sleep added 0-500 ms to every request)."""
    monkeypatch.setattr(rank.Engine, "_boot", lambda self: None)
    monkeypatch.setattr(rank.Engine, "POLL_WAIT_S", 5.0)
    engine = rank.Engine("model", 2)
    engine.tok, engine.model = ChatTokenizer(), object()
    rank.EngineHandler.engine = engine
    server = ThreadingHTTPServer(("127.0.0.1", 0), rank.EngineHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = "http://127.0.0.1:%d" % server.server_port
    seen = {}
    rank1 = threading.Thread(target=lambda: seen.update(rank1=rank._poll_job(url, None)))

    def rank0_stream(model, tok, prompt, max_tokens, marks=None, **kw):
        seen["rank0"] = (prompt, max_tokens)
        rank1.join(1.0)  # well inside POLL_WAIT_S: the publish must wake the poll
        engine.POLL_WAIT_S = 0.2
        seen["rank1_again"] = rank._poll_job(url, seen["rank1"]["id"])
        yield SimpleNamespace(text="ok", prompt_tokens=len(prompt), prompt_tps=1.0,
                              generation_tokens=1, generation_tps=1.0, peak_memory=0.0)

    monkeypatch.setattr(rank, "_stream", rank0_stream)
    try:
        rank1.start()
        time.sleep(0.2)  # rank 1 is parked in its poll before the request arrives
        res = engine._generate([{"role": "user", "content": "hi"}], 7)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    job = seen["rank1"]
    assert (job["prompt"], job["max_tokens"]) == seen["rank0"] == ([100000, 104, 105], 7)
    assert seen["rank1_again"] is None
    assert res["poll_ms"] < 1000


def test_step_ms_splits_decode_steps_and_skips_prefill():
    """rank0 marks: 2 prefill forwards, first token, then 2 decode steps of
    recv wait 30 ms, compute 5 ms, gather 1 ms, head 4 ms (40 ms/token)."""
    marks, t = [("start", 0.0)], 0.0

    def step(recv, compute, gather, head):
        nonlocal t
        for name, dt in (("recv_like<", 0), ("recv_like>", recv), ("all_gather<", compute),
                         ("all_gather>", gather)):
            t += dt
            marks.append((name, t))
        t += head

    step(0.5, 0.1, 0.001, 0.001)  # prefill: rank1 polls, then both prefill
    step(0.03, 0.005, 0.001, 0.004)
    marks.append(("token", t))
    step(0.03, 0.005, 0.001, 0.004)
    marks.append(("token", t))
    step(0.03, 0.005, 0.001, 0.004)
    marks.append(("token", t))
    ms = rank._step_ms(marks)
    assert ms["prefill"] == 642.0
    assert ms["token"] == 40.0
    assert (ms["recv_like<recv_like>"], ms["recv_like>all_gather<"],
            ms["all_gather<all_gather>"], ms["all_gather>recv_like<"]) == (30.0, 5.0, 1.0, 4.0)


def _split(n_layers, size, counts):
    ran = []
    for r in range(size):
        model = SimpleNamespace(layers=list(range(n_layers)))
        rank._pipeline_split(model, SimpleNamespace(rank=lambda r=r: r, size=lambda: size), counts)
        ran.append(model.layers[model.start_idx:model.end_idx])
    return ran


def test_pipeline_split_runs_every_layer_exactly_once():
    """mlx-lm 0.31.3 dropped layer 13 of 27 on 2 ranks; ranks run layers in reverse rank order."""
    for n_layers, counts in [(27, [14, 13]), (27, [24, 3]), (27, [1, 26]), (27, [20, 5, 2]),
                             (5, [2, 1, 1, 1])]:
        ran = _split(n_layers, len(counts), counts)
        assert sum(reversed(ran), []) == list(range(n_layers)), (n_layers, counts, ran)
        assert [len(r) for r in ran] == counts


def test_pipeline_split_rejects_counts_that_drop_layers():
    for counts in ([24, 2], [27, 0], [27]):
        try:
            _split(27, 2, counts)
        except ValueError:
            continue
        raise AssertionError("accepted %s" % counts)


def test_fastest_layer_ms_keeps_a_contention_burst_from_flipping_the_split(tmp_path):
    """ClusterRun5 r18: mac-a measured 9.7 ms/layer during other GPU work,
    above the M2's 8.0, and the split flipped to 1,26."""
    path = str(tmp_path / "layer-ms.json")
    assert rank._fastest_layer_ms("mac m", 0.6, path) == 0.6
    assert rank._fastest_layer_ms("mac m", 9.7, path) == 0.6  # burst: keep the capability
    assert rank._fastest_layer_ms("mac m", 0.5, path) == 0.5
    assert rank._fastest_layer_ms("m2 m", 8.0, path) == 8.0  # keys are independent
    assert rank._fastest_layer_ms("mac m", 0.0, path) == 0.5  # a clamped 0 is noise, not a record
    assert rank._fastest_layer_ms("new m", 0.0, path) == 0.0


def _forward(t0, wait_ms, own_ms):
    """rank 0 marks of one sampled decode forward."""
    t1 = t0 + wait_ms / 1000
    t2 = t1 + own_ms / 1000
    return [("token", t0), ("recv_like<", t0), ("recv_like>", t1), ("all_gather<", t2),
            ("all_gather>", t2 + 0.0002)]


def test_watch_flags_the_contended_rank_after_n_slow_sampled_steps(monkeypatch):
    """mac-a + linux-b at 26,1 (calibrated 0.5 and 8.0 ms/layer):
    a Claude.app GPU burst took rank 0 from ~17 to ~220 ms per step while
    rank 1 stayed at ~4 ms (ClusterRun5 gpuobs3)."""
    monkeypatch.setattr(rank.Engine, "_boot", lambda self: None)
    engine = rank.Engine("model", 2)
    engine._start_watch([26, 1], [0.5, 8.0])
    t = 0.0
    for _ in range(5):
        engine._sample(_forward(t, 4.0, 17.0))
        t += 1
    assert engine.status()["parts"]["rank0"]["contended"] is False
    assert engine.status()["parts"]["rank0"]["ref_ms"] == 13.0  # calibration 26 x 0.5 beats 17
    for _ in range(4):
        assert engine.status()["events"] == []
        engine._sample(_forward(t, 4.0, 220.0))
        t += 1
    st = engine.status()
    assert st["parts"]["rank0"]["contended"] is True
    assert st["parts"]["rank1+"]["contended"] is False
    assert [(e["part"], e["ratio"]) for e in st["events"]] == [("rank0", 16.92)]
    engine._sample(_forward(t, 4.0, 17.0))  # burst over
    st = engine.status()
    assert st["parts"]["rank0"]["contended"] is False and len(st["events"]) == 1


def test_watch_without_calibration_uses_the_fastest_step_seen():
    w = rank._Watch(None, factor=3.0, n=2)
    assert [w.add(ms) for ms in (40.0, 20.0, 70.0, 70.0, 70.0, 10.0, 70.0)] == [
        False, False, False, True, False, False, False]  # trips once per slow run
    assert w.ref == 10.0


def test_watch_ignores_sub_ms_ratios():
    """loopback recv wait went 0.07 -> 0.33 ms (4.9x): noise, not contention."""
    w = rank._Watch(None, factor=3.0, n=2)
    assert not any(w.add(ms) for ms in (0.07, 0.33, 0.33, 0.33, 4.0, 4.0))
    assert [w.add(ms) for ms in (6.0, 6.0)] == [False, True]
