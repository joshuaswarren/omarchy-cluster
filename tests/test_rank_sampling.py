import json
import os
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import rank


def _post(engine, body):
    rank.EngineHandler.engine = engine
    server = ThreadingHTTPServer(("127.0.0.1", 0), rank.EngineHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        request = urllib.request.Request("http://127.0.0.1:%d/generate" % server.server_port,
                                         data=json.dumps(body).encode())
        return json.load(urllib.request.urlopen(request, timeout=2))
    finally:
        server.shutdown()
        server.server_close()


def test_rank_engine_hands_sampling_to_generate():
    seen = {}

    def generate(messages, max_tokens=64, timing=False, sampling=None):
        seen["sampling"] = sampling
        return {"text": "ok"}

    sampling = {"temperature": 0.7, "top_p": 0.9, "seed": 5}
    assert _post(SimpleNamespace(generate=generate), {"prompt": "hi", "sampling": sampling}) == {"text": "ok"}
    assert seen["sampling"] == sampling


def test_a_greedy_request_carries_no_sampling_to_generate():
    seen = {}

    def generate(messages, max_tokens=64, timing=False, sampling=None):
        seen["sampling"] = sampling
        return {"text": "ok"}

    _post(SimpleNamespace(generate=generate), {"prompt": "hi"})
    assert seen["sampling"] is None


def test_generate_job_reaches_the_stream_with_its_sampling(monkeypatch):
    monkeypatch.setattr(rank.Engine, "_boot", lambda self: None)
    engine = rank.Engine("model", 1)
    engine.tok = SimpleNamespace(encode=lambda text, add_special_tokens=False: [1, 2])
    engine.model = object()
    seen = {}

    def stream(model, tok, prompt, max_tokens, marks=None, **kw):
        seen["sampling"] = kw.get("sampling")
        yield SimpleNamespace(text="ok", prompt_tokens=2, prompt_tps=1.0,
                              generation_tokens=1, generation_tps=1.0, peak_memory=0.0)

    monkeypatch.setattr(rank, "_stream", stream)
    engine._generate("hi", 4, sampling={"temperature": 0.7, "seed": 5})
    assert seen["sampling"] == {"temperature": 0.7, "seed": 5}


mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")


def _tokens(sampling, steps=64, vocab=32):
    pick = rank._sampler(sampling)
    logprobs = mx.log(mx.full((1, vocab), 1.0 / vocab))
    return [int(pick(logprobs).item()) for _ in range(steps)]


def test_the_same_seed_gives_the_same_tokens_and_another_seed_does_not():
    a = _tokens({"temperature": 1.0, "seed": 5})
    assert a == _tokens({"temperature": 1.0, "seed": 5})
    assert a != _tokens({"temperature": 1.0, "seed": 6})


def test_no_sampling_settings_pick_the_argmax():
    logprobs = mx.log(mx.array([[0.1, 0.6, 0.3]]))
    assert rank._sampler(None) is None
    assert int(rank._sampler(None, share=lambda y: y)(logprobs).item()) == 1


def test_a_tiny_top_p_keeps_only_the_top_token():
    logprobs = mx.log(mx.array([[0.1, 0.6, 0.3]]))
    pick = rank._sampler({"temperature": 1.0, "top_p": 0.01, "seed": 1})
    assert {int(pick(logprobs).item()) for _ in range(20)} == {1}


def test_the_shared_token_is_the_one_rank_0_sampled():
    logprobs = mx.log(mx.full((1, 32), 1.0 / 32))
    shared = []
    pick = rank._sampler({"temperature": 1.0, "seed": 5}, share=lambda y: shared.append(int(y.item())) or y)
    out = [int(pick(logprobs).item()) for _ in range(8)]
    assert shared == out == _tokens({"temperature": 1.0, "seed": 5}, steps=8)
