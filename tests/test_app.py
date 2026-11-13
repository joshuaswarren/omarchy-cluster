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
    state = {"status": 200, "active": [{"id": "served-model"}],
             "cached": [{"repo": "org/cached", "path": "/cache/model", "bytes": 1024, "weights": 1}]}

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
            data = state["active"] if self.path == "/v1/models" else state["cached"]
            self.respond(state["status"], {"data": data})

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


def test_native_app_lists_cached_models_and_active_gateway_models(gateway):
    url, seen, _state = gateway
    cached, active = app.fetch_catalog(url, token="secret")
    assert cached[0]["repo"] == "org/cached"
    assert active == ["served-model"]
    assert seen == [
        ("GET", "/models", "Bearer secret"),
        ("GET", "/v1/models", "Bearer secret"),
    ]


def test_chat_uses_only_a_model_the_gateway_actually_serves(gateway):
    url, seen, _state = gateway
    core = app.AppCore(url, token="secret")
    core.set_catalog([], ["served-model"])
    turn = core.begin_turn("served-model", "hi")
    assert turn() == "hello"
    core.complete_turn("hello")
    assert seen[0][1] == "/v1/chat/completions"
    assert seen[0][2]["model"] == "served-model"
    assert core.history == [{"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "hello"}]


def test_cached_but_not_served_model_cannot_be_selected(gateway):
    url, _seen, _state = gateway
    core = app.AppCore(url)
    core.set_catalog([{"repo": "org/cached"}], ["served-model"])
    with pytest.raises(ValueError, match="not served by this gateway"):
        core.begin_turn("org/cached", "hi")


def test_failed_turn_is_removed_and_controller_recovers(gateway):
    url, _seen, _state = gateway
    core = app.AppCore(url)
    core.set_catalog([], ["served-model"])
    core.begin_turn("served-model", "failed")
    core.fail_turn()
    assert core.history == [] and not core.busy
    turn = core.begin_turn("served-model", "retry")
    assert turn() == "hello"


def test_second_send_is_rejected_until_reply_callback_finishes(gateway):
    url, _seen, _state = gateway
    core = app.AppCore(url)
    core.set_catalog([], ["served-model"])
    core.begin_turn("served-model", "first")
    with pytest.raises(RuntimeError, match="reply is pending"):
        core.begin_turn("served-model", "second")
    assert core.history == [{"role": "user", "content": "first"}]
    core.fail_turn()


def test_turn_uses_captured_gateway_and_conversation(gateway):
    url, seen, _state = gateway
    core = app.AppCore(url)
    core.set_catalog([], ["served-model"])
    turn = core.begin_turn("served-model", "captured")
    core.gateway = "http://127.0.0.1:1"
    assert turn() == "hello"
    assert seen[0][2]["messages"] == [{"role": "user", "content": "captured"}]


def test_launcher_installs_into_xdg_visible_user_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(app.sys, "argv", ["/opt/omarchy/venv/bin/omarchy-cluster"])
    path = app.install_launcher()
    assert path == tmp_path / "applications" / "com.omarchy.Cluster.desktop"
    assert f'Exec="{app.sys.argv[0]}" app' in path.read_text()


def test_operation_failures_always_become_ui_errors():
    result, error = app.run_operation(lambda: (_ for _ in ()).throw(TypeError("bad response")))
    assert result is None and error == "bad response"


def test_app_core_imports_without_gtk_and_system_python_fallback_is_available(monkeypatch):
    monkeypatch.setattr(app.sys, "executable", "/home/user/.venv/bin/python")
    command, env = app.gtk_fallback_command()
    assert env is not None
    assert command == ["/usr/bin/python3", "-m", "omarchy_cluster.app"]
    assert os.path.realpath(os.path.dirname(os.path.dirname(app.__file__))) in env["PYTHONPATH"].split(os.pathsep)
    assert app.AppCore is not None


def test_gateway_redirect_does_not_forward_cluster_token():
    received = []

    class Sink(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            received.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

    sink = ThreadingHTTPServer(("127.0.0.1", 0), Sink)
    threading.Thread(target=sink.serve_forever, daemon=True).start()

    class Redirect(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{sink.server_port}/steal")
            self.end_headers()

    redirect = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threading.Thread(target=redirect.serve_forever, daemon=True).start()
    try:
        with pytest.raises(RuntimeError, match="gateway.*302"):
            app._request(f"http://127.0.0.1:{redirect.server_port}", "/", "secret")
        assert received == []
    finally:
        redirect.shutdown()
        redirect.server_close()
        sink.shutdown()
        sink.server_close()


def test_gateway_helpers_keep_auth_and_failures_visible(gateway):
    url, seen, _state = gateway
    assert app.pull_model(url, "org/new", token="secret")["repo"] == "org/new"
    assert seen[0] == ("POST", "/models/pull", {"repo": "org/new"}, "Bearer secret")


def test_native_app_reports_unreachable_gateway():
    with pytest.raises(RuntimeError, match="cannot reach the gateway"):
        app.list_models("http://127.0.0.1:1")


def test_cli_app_command_opens_native_app(monkeypatch):
    from omarchy_cluster import cli
    calls = []
    monkeypatch.setattr(app, "main", lambda args: calls.append(args))
    cli.main(["app"])
    assert len(calls) == 1


def test_cli_install_launcher_does_not_open_window(monkeypatch, capsys):
    from omarchy_cluster import cli
    monkeypatch.setattr(app, "install_launcher", lambda: "/tmp/app.desktop")
    calls = []
    monkeypatch.setattr(app, "main", lambda *_: calls.append(True))
    cli.main(["app", "--install-launcher"])
    assert capsys.readouterr().out.strip() == "/tmp/app.desktop"
    assert calls == []
