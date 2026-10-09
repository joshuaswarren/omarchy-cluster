import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import sushi_engine as se


class FakeSushi(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.seen.append((self.path, body))
        usage = {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
        if self.path == "/v1/completions":
            res = {"choices": [{"text": " Paris.", "finish_reason": "length"}], "usage": usage}
        else:
            res = {"choices": [{"message": {"content": "Paris."}, "finish_reason": None}], "usage": usage}
        data = json.dumps(res).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        data = b'{"data": []}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d" % srv.server_port


def test_generate_maps_sushi_replies_to_the_gateway_contract_greedy():
    FakeSushi.seen.clear()
    srv, url = _serve(FakeSushi)
    try:
        res = se.generate(url, {"prompt": "The capital of France is", "max_tokens": 4})
        chat = se.generate(url, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    finally:
        srv.shutdown()
    assert (res["text"], res["prompt_tokens"], res["completion_tokens"], res["finish_reason"]) == \
        (" Paris.", 5, 4, "length")
    assert res["tokps"] == pytest.approx(4 / (res["decode_ms"] / 1e3), rel=0.1)
    assert (chat["text"], chat["finish_reason"]) == ("Paris.", "stop")
    assert [(p, b["temperature"], b["max_tokens"]) for p, b in FakeSushi.seen] == \
        [("/v1/completions", 0, 4), ("/v1/chat/completions", 0, 4)]


def test_engine_health_follows_sushi_and_failures_reach_the_gateway():
    se.EngineHandler.server_url = "http://127.0.0.1:1"
    srv, url = _serve(se.EngineHandler)
    try:
        assert "error" in se._post(url + "/generate", {"prompt": "x", "max_tokens": 1})
        with pytest.raises(urllib.error.HTTPError) as health:
            urllib.request.urlopen(url + "/health", timeout=5)
        assert health.value.code == 503
        up, up_url = _serve(FakeSushi)
        try:
            se.EngineHandler.server_url = up_url
            assert urllib.request.urlopen(url + "/health", timeout=5).status == 200
        finally:
            up.shutdown()
    finally:
        srv.shutdown()
