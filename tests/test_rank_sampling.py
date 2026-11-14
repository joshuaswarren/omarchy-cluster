import json
import os
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace


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
