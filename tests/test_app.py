import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import app


@pytest.fixture
def gateway():
    seen = []
    state = {"status": 200, "models": [{"repo": "org/model", "path": "/cache/model",
                                         "bytes": 1024, "weights": 1}]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def respond(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            seen.append(("GET", self.path, self.headers.get("Authorization")))
            self.respond(state["status"], {"data": state["models"]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            seen.append(("POST", self.path, body, self.headers.get("Authorization")))
            if self.path == "/models/pull":
                self.respond(state["status"], {"repo": body["repo"], "path": "/cache/model"})
            else:
                self.respond(state["status"], {"choices": [{"message": {"content": "hello"}}]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", seen, state
    server.shutdown()
    server.server_close()


def test_native_app_workflow_lists_pulls_selects_and_chats(gateway):
    url, seen, _state = gateway
    models = app.list_models(url, token="secret")
    assert models[0]["repo"] == "org/model"
    pulled = app.pull_model(url, "org/new", token="secret")
    assert pulled["repo"] == "org/new"
    assert app.chat_turn(url, "org/new", [{"role": "user", "content": "hi"}], token="secret") == "hello"
    assert seen == [
        ("GET", "/models", "Bearer secret"),
        ("POST", "/models/pull", {"repo": "org/new"}, "Bearer secret"),
        ("POST", "/v1/chat/completions", {
            "model": "org/new", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 256
        }, "Bearer secret"),
    ]


def test_native_app_reports_gateway_unavailable(gateway):
    url, _seen, state = gateway
    state["status"] = 503
    with pytest.raises(RuntimeError, match="gateway.*503"):
        app.list_models(url)


def test_native_app_reports_model_download_failure(gateway):
    url, _seen, state = gateway
    state["status"] = 502
    with pytest.raises(RuntimeError, match="gateway.*502"):
        app.pull_model(url, "org/model")


def test_native_app_reports_chat_failure(gateway):
    url, _seen, state = gateway
    state["status"] = 503
    with pytest.raises(RuntimeError, match="gateway.*503"):
        app.chat_turn(url, "org/model", [{"role": "user", "content": "hi"}])


def test_native_app_reports_unreachable_gateway():
    with pytest.raises(RuntimeError, match="cannot reach the gateway"):
        app.list_models("http://127.0.0.1:1")


def test_cli_app_command_opens_native_app(monkeypatch):
    from omarchy_cluster import cli
    calls = []
    monkeypatch.setattr(app, "main", lambda args: calls.append(args))
    cli.main(["app"])
    assert len(calls) == 1
