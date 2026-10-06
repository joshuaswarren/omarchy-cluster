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

    def rank0_stream(model, tok, prompt, max_tokens, marks=None):
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


def _split(n_layers, size, counts=None):
    ran = []
    for r in range(size):
        model = SimpleNamespace(layers=list(range(n_layers)))
        rank._pipeline_split(model, SimpleNamespace(rank=lambda r=r: r, size=lambda: size), counts)
        ran.append(model.layers[model.start_idx:model.end_idx])
    return ran


def test_pipeline_split_runs_every_layer_exactly_once():
    """mlx-lm 0.31.3 dropped layer 13 of 27 on 2 ranks; ranks run layers in reverse rank order."""
    for n_layers, size, counts in [(27, 2, None), (27, 4, None), (26, 2, None), (5, 4, None),
                                   (27, 2, [24, 3]), (27, 2, [1, 26]), (27, 3, [20, 5, 2])]:
        ran = _split(n_layers, size, counts)
        assert sum(reversed(ran), []) == list(range(n_layers)), (n_layers, size, ran)
        if counts:
            assert [len(r) for r in ran] == counts


def test_pipeline_split_rejects_counts_that_drop_layers():
    for counts in ([24, 2], [27, 0], [27]):
        try:
            _split(27, 2, counts)
        except ValueError:
            continue
        raise AssertionError("accepted %s" % counts)
