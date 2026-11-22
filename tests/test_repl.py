import json
import os
import sys
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import chat


FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Point HOME at a scratch dir so the REPL's journal never writes the real one."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def gateway():
    """A gateway stub: records each chat request, answers from a queue of (status, body)."""
    seen, replies = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            status, reply = replies.pop(0)
            if body.get("stream") and isinstance(reply, list) and isinstance(reply[-1], dict):
                # an SSE reply: reply = [chunk texts...], final dict last (issue 95)
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for text in reply[:-1]:
                    self.wfile.write(("data: %s\n\n" % json.dumps(
                        {"choices": [{"delta": {"content": text}}]})).encode())
                    self.wfile.flush()
                self.wfile.write(("data: %s\n\n" % json.dumps(
                    {"choices": [{"delta": {}}], **reply[-1]})).encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            data = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d" % server.server_port, seen, replies
    server.shutdown()
    server.server_close()


def answer(text, tokps=12.5):
    return 200, {"choices": [{"message": {"role": "assistant", "content": text}}],
                 "timings": {"tokens_per_s": tokps}}


def run(url, lines, **kw):
    feed = iter(lines)

    def read(prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    out = []
    chat.repl(url, read=read, write=out.append, **kw)
    return out


def test_each_turn_sends_the_whole_conversation_so_far(gateway):
    url, seen, replies = gateway
    replies += [answer("hello"), answer("fine")]
    out = run(url, ["hi", "how are you"], model="m1", max_tokens=32)
    assert seen[0]["path"] == "/v1/chat/completions"
    assert seen[1]["body"]["messages"] == [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how are you"}]
    assert seen[1]["body"]["model"] == "m1" and seen[1]["body"]["max_tokens"] == 32
    assert "hello" in out and "fine" in out
    assert any("12.5 tok/s" in line for line in out)


def test_a_streaming_gateway_prints_each_piece_before_the_rate_line(gateway):
    """DH-P7-07 / issue 95: the REPL prints tokens as they arrive - the three piece
    writes land before the rate line, and the streamed text joins the history."""
    url, seen, replies = gateway
    final = {"usage": {"prompt_tokens": 3, "completion_tokens": 3},
             "timings": {"tokens_per_s": 42.0}}
    replies += [[200, ["To", "ken", "ize", final]]]
    out = run(url, ["hi"])
    assert out == ["To", "ken", "ize", "\n", "[42.0 tok/s]\n"]
    assert seen[0]["body"]["stream"] is True


def test_streamed_pieces_join_into_one_line(gateway):
    """Issue 121: a streamed reply lands on one line. The writer must not add a
    newline per piece; one newline ends the reply."""
    url, seen, replies = gateway
    replies += [[200, ["Hel", "lo ", "world", {}]]]
    out = run(url, ["hi"])
    assert "".join(out) == "Hello world\n"


def test_streamed_pieces_print_as_one_line_with_the_default_writer(gateway, capsys):
    """Issue 121: with the real default writer (stdout), the streamed reply
    prints as one line — this fails if the default goes back to print."""
    url, seen, replies = gateway
    replies += [[200, ["Hel", "lo ", "world", {}]]]
    feed = iter(["hi"])

    def read(prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    chat.repl(url, read=read)
    assert capsys.readouterr().out == "Hello world\n"


def test_streamed_reply_renders_one_line_byte_identical_to_golden(capsys):
    """Issue 121 / commit 8430d81 regression: over a canned SSE stream — the real
    chat_stream wire path, not a stubbed iterator — the default writer renders
    exactly one line per reply, byte-identical across runs against the golden."""
    body = (FIXTURES / "streamed_reply.sse").read_bytes()
    golden = (FIXTURES / "streamed_reply.golden").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = "http://127.0.0.1:%d" % server.server_port
        renders = []
        for _ in range(2):
            feed = iter(["hi"])

            def read(prompt):
                try:
                    return next(feed)
                except StopIteration:
                    raise EOFError from None

            chat.repl(url, read=read)
            renders.append(capsys.readouterr().out.encode())
    finally:
        server.shutdown()
        server.server_close()
    assert renders[0] == renders[1] == golden
    assert renders[0].count(b"\n") == 1


def test_a_mid_stream_error_terminates_the_partial_reply_line(monkeypatch):
    """Issue 121: a RuntimeError after partial output must not glue the error
    onto the reply line — the partial line gets a newline first."""
    def broken_stream(*args, **kwargs):
        yield "partial"
        raise RuntimeError("engine: died")

    monkeypatch.setattr(chat.client, "chat_stream", broken_stream)
    out = run("http://127.0.0.1:1", ["hi"])
    assert out == ["partial", "\n", "error: engine: died\n"]


def test_the_repl_sends_greedy_sampling_temperature_0(gateway):
    url, seen, replies = gateway
    replies.append(answer("x"))
    run(url, ["hi"])
    assert seen[0]["body"]["temperature"] == 0


def test_a_system_prompt_leads_every_request(gateway):
    url, seen, replies = gateway
    replies += [answer("a"), answer("b")]
    run(url, ["one", "two"], system="Be brief.")
    for request in seen:
        assert request["body"]["messages"][0] == {"role": "system", "content": "Be brief."}


def test_the_token_is_sent_as_a_bearer_header(gateway):
    url, seen, replies = gateway
    replies.append(answer("x"))
    run(url, ["hi"], token="secret")
    assert seen[0]["auth"] == "Bearer secret"


def test_a_failed_turn_prints_the_error_and_is_not_kept_in_the_history(gateway):
    url, seen, replies = gateway
    replies += [(502, {"error": "engine: down"}), answer("back")]
    out = run(url, ["first", "second"])
    assert any("engine: down" in line for line in out)
    assert seen[1]["body"]["messages"] == [{"role": "user", "content": "second"}]


def test_a_200_reply_with_an_unexpected_shape_prints_an_error_and_keeps_going(gateway):
    url, seen, replies = gateway
    replies += [(200, {"object": "not a chat completion"}), answer("back")]
    out = run(url, ["first", "second"])
    assert any("unexpected reply shape" in line for line in out)
    assert seen[1]["body"]["messages"] == [{"role": "user", "content": "second"}]


@pytest.mark.parametrize("body", [
    {"choices": [{"message": {"content": None}}]},
    {"choices": [{"message": {"content": 5}}]},
    {"choices": [{"message": "ok"}]},
    {"choices": ["ok"]},
    ["not", "an", "object"],
])
def test_a_malformed_200_body_is_an_error_and_never_enters_the_history(gateway, body):
    url, seen, replies = gateway
    replies += [(200, body), answer("back")]
    out = run(url, ["first", "second"])
    assert any("unexpected reply shape" in line for line in out)
    assert seen[1]["body"]["messages"] == [{"role": "user", "content": "second"}]


@pytest.mark.parametrize("timings", [12, {"tokens_per_s": "unknown"}, {"tokens_per_s": True}, ["x"],
                                     {"tokens_per_s": 10 ** 400}, {"tokens_per_s": float("inf")},
                                     {"tokens_per_s": float("nan")}, {"tokens_per_s": -3}])
def test_a_malformed_timing_is_dropped_and_the_reply_still_prints(gateway, home, timings):
    url, seen, replies = gateway
    replies.append((200, {"choices": [{"message": {"content": "ok"}}], "timings": timings}))
    out = run(url, ["hi", "/exit"])
    assert "ok" in out and not any("tok/s" in line for line in out)
    assert (home / ".local" / "state" / "omarchy-cluster" / "chat.log").read_text().count("turns=1") == 1


def test_one_journal_line_per_session_and_no_other_state_files(gateway, home):
    url, seen, replies = gateway
    replies += [answer("hello"), answer("fine")]
    run(url, ["hi", "how are you", "/exit"])
    log = home / ".local" / "state" / "omarchy-cluster" / "chat.log"
    lines = log.read_text().splitlines()
    assert len(lines) == 1 and "chat" in lines[0] and "turns=2" in lines[0]
    others = [p for p in home.rglob("*") if p.is_file() and p != log]
    assert others == [], "the REPL wrote a state file: %s" % others


def test_reset_clears_the_history_and_blank_lines_and_exit_are_handled(gateway):
    url, seen, replies = gateway
    replies += [answer("a"), answer("b")]
    run(url, ["one", "", "/reset", "two", "/exit", "never sent"])
    assert len(seen) == 2
    assert seen[1]["body"]["messages"] == [{"role": "user", "content": "two"}]


def test_an_unreachable_gateway_says_where_it_looked():
    out = run("http://127.0.0.1:1", ["hi"])
    assert any("http://127.0.0.1:1" in line for line in out)


def test_the_cli_passes_the_options_to_the_repl(monkeypatch):
    from omarchy_cluster import cli
    got = {}
    monkeypatch.setattr(chat, "repl", lambda url, **kw: got.update(url=url, **kw))
    monkeypatch.setattr(chat.cluster_token, "read", lambda: "tok")
    cli.main(["chat", "--model", "glm", "--gateway", "http://10.0.0.2:8020", "--max-tokens", "99"])
    assert got["url"] == "http://10.0.0.2:8020"
    assert got["model"] == "glm" and got["max_tokens"] == 99 and got["token"] == "tok"
