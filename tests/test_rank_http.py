import json
import os
import sys
import threading
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster.rank import EngineHandler, _get_engine


class PendingEngine:
    def pending_prompt(self, after):
        return {"id": 8, "prompt": "hello", "after": after}


def test_rank_poll_uses_get_and_returns_pending_prompt():
    server = ThreadingHTTPServer(("127.0.0.1", 0), EngineHandler)
    server.RequestHandlerClass.engine = PendingEngine()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        job = _get_engine("http://127.0.0.1:%d" % server.server_port,
                          "/prompt/wait", {"after": 7})
        assert job == {"id": 8, "prompt": "hello", "after": 7}
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
