import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster.gateway import Gateway


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
@pytest.mark.parametrize("body", [
    b"[]", b"null", b"42", b"{",
    b'{"max_tokens": null}', b'{"max_tokens": "invalid"}',
    b'{"max_tokens": []}', b'{"max_tokens": 1e1000}',
])
def test_invalid_completion_request_returns_http_400(path, body, monkeypatch):
    monkeypatch.setattr(Gateway, "engine", "http://127.0.0.1:1")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            "http://127.0.0.1:%d%s" % (server.server_port, path),
            data=body, headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=2)
        with error.value as response:
            assert response.code == 400
            assert isinstance(json.load(response)["error"], str)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
