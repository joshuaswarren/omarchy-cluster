import os
import sys
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import rank


class ChatTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt):
        assert add_generation_prompt
        return [100000] + [ord(c) for c in messages[-1]["content"]]


def test_rank1_poll_gets_rank0_prompt_tokens_once(monkeypatch):
    """rank 1 must enter stream_generate with rank 0's exact tokens, or rank 0
    blocks forever in the pipeline recv (ClusterRun3 deadlock)."""
    monkeypatch.setattr(rank.Engine, "_boot", lambda self: None)
    engine = rank.Engine("model", 2)
    engine.tok, engine.model = ChatTokenizer(), object()
    rank.EngineHandler.engine = engine
    server = ThreadingHTTPServer(("127.0.0.1", 0), rank.EngineHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = "http://127.0.0.1:%d" % server.server_port
    seen = {}

    def rank0_stream(model, tok, prompt, max_tokens):
        seen["rank0"] = (prompt, max_tokens)
        seen["rank1"] = rank._poll_job(url, None)
        seen["rank1_again"] = rank._poll_job(url, seen["rank1"]["id"])
        yield SimpleNamespace(text="ok", prompt_tokens=len(prompt), generation_tokens=1,
                              generation_tps=1.0, peak_memory=0.0)

    monkeypatch.setattr(rank, "_stream", rank0_stream)
    try:
        assert rank._poll_job(url, None) is None
        engine._generate([{"role": "user", "content": "hi"}], 7)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    job = seen["rank1"]
    assert (job["prompt"], job["max_tokens"]) == seen["rank0"] == ([100000, 104, 105], 7)
    assert seen["rank1_again"] is None


def test_pipeline_split_runs_every_layer_exactly_once():
    """mlx-lm 0.31.3 dropped layer 13 of 27 on 2 ranks; ranks run layers in reverse rank order."""
    for n_layers, size in [(27, 2), (27, 4), (26, 2), (5, 4)]:
        ran = []
        for r in range(size):
            model = SimpleNamespace(layers=list(range(n_layers)))
            rank._pipeline_split(model, SimpleNamespace(rank=lambda r=r: r, size=lambda: size))
            ran.append(model.layers[model.start_idx:model.end_idx])
        assert sum(reversed(ran), []) == list(range(n_layers)), (n_layers, size, ran)
