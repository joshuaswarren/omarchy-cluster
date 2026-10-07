import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import iosnode, llamacpp_engine as lce

GB = 10 ** 9


def _gguf_string(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def make_gguf(path, n_layer, layer_sizes, other_sizes, n_kv_head=8, head_dim=128, value_dim=None, nextn=0):
    """A minimal GGUF v3 file: metadata, tensor infos, and zero-filled data
    whose tensor sizes are given in bytes (multiples of the 32-byte alignment)."""
    meta = [("general.architecture", 8, _gguf_string("qwen3")),
            ("qwen3.block_count", 4, struct.pack("<I", n_layer)),
            ("qwen3.attention.head_count", 4, struct.pack("<I", 16)),
            ("qwen3.attention.head_count_kv", 4, struct.pack("<I", n_kv_head)),
            ("qwen3.attention.key_length", 4, struct.pack("<I", head_dim)),
            ("qwen3.embedding_length", 4, struct.pack("<I", 2048)),
            ("tokenizer.ggml.tokens", 9,
             struct.pack("<IQ", 8, 2) + _gguf_string("a") + _gguf_string("b"))]
    if value_dim is not None:
        meta.append(("qwen3.attention.value_length", 4, struct.pack("<I", value_dim)))
    if nextn:
        meta.append(("qwen3.nextn_predict_layers", 4, struct.pack("<I", nextn)))
    tensors = [("token_embd.weight", other_sizes[0])]
    for i in range(n_layer):
        tensors += [("blk.%d.attn_q.weight" % i, layer_sizes[0]),
                    ("blk.%d.ffn_up.weight" % i, layer_sizes[1])]
    tensors += [("output_norm.weight", other_sizes[1])]
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for key, vtype, raw in meta:
        out += _gguf_string(key) + struct.pack("<I", vtype) + raw
    offset = 0
    for name, size in tensors:
        out += _gguf_string(name) + struct.pack("<I", 1) + struct.pack("<Q", size)
        out += struct.pack("<IQ", 0, offset)
        offset += size
    out += bytes(-len(out) % 32) + bytes(offset)
    with open(path, "wb") as f:
        f.write(out)


def test_gguf_info_sums_each_layer_and_the_rest(tmp_path):
    p = tmp_path / "m.gguf"
    make_gguf(str(p), 3, (64, 96), (320, 32))
    info = lce.gguf_info(str(p))
    assert info["arch"] == "qwen3"
    assert info["n_layer"] == 3
    assert info["layer_bytes"] == 160
    assert info["other_bytes"] == 352
    assert info["total_bytes"] == 3 * 160 + 352
    assert info["kv_bytes_per_token_layer"] == 2 * 8 * 128 * 2


def test_gguf_info_rejects_other_files(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"NOPE" + bytes(64))
    with pytest.raises(ValueError):
        lce.gguf_info(str(p))


def test_gguf_info_does_not_charge_skipped_mtp_layers(tmp_path):
    """llama.cpp loads the trailing nextn (MTP) layers with TENSOR_SKIP: no device memory."""
    p = tmp_path / "m.gguf"
    make_gguf(str(p), 4, (64, 96), (320, 32), nextn=1)
    assert lce.gguf_info(str(p))["per_layer"] == [160, 160, 160, 0]


INFO = {"n_layer": 28, "layer_bytes": 40 * 10 ** 6, "other_bytes": 300 * 10 ** 6,
        "kv_bytes_per_token_layer": 0}


def test_model_that_fits_the_host_puts_nothing_on_the_phone():
    """A split is slower than the host alone, so a model the host holds stays here."""
    assert lce.rpc_layers(INFO, host_free_bytes=8 * GB, rpc_cap_bytes=2 * GB, ctx_tokens=2048) == 0


def test_model_too_big_for_the_host_moves_the_fewest_last_layers():
    total = INFO["other_bytes"] + 28 * INFO["layer_bytes"]  # 1.42 GB
    host = int((total - 5 * INFO["layer_bytes"]) / lce.HOST_FIT_FRACTION) + 1
    assert lce.rpc_layers(INFO, host, 2 * GB, 2048) == 5


def test_phone_cap_and_kv_bound_the_split():
    host = int(INFO["other_bytes"] / lce.HOST_FIT_FRACTION) + 1  # room for no layer at all
    with pytest.raises(ValueError):
        lce.rpc_layers(INFO, host, rpc_cap_bytes=27 * INFO["layer_bytes"], ctx_tokens=2048)
    info = dict(INFO, kv_bytes_per_token_layer=1000)  # 2.048 MB KV per layer at ctx 2048
    host = int((INFO["other_bytes"] + 20 * INFO["layer_bytes"]) / lce.HOST_FIT_FRACTION)
    k = lce.rpc_layers(info, host, rpc_cap_bytes=2 * GB, ctx_tokens=2048)
    assert k == 9  # 8 layers would fit without KV; KV for the 20 host layers needs one more


def test_forced_rpc_layers_are_range_checked():
    assert lce.rpc_layers(INFO, 8 * GB, 0, 2048, force=7) == 7
    with pytest.raises(ValueError):
        lce.rpc_layers(INFO, 8 * GB, 0, 2048, force=29)


def test_server_cmd_keeps_lm_head_on_host_and_layers_on_the_rpc_device():
    local = lce.server_cmd("llama-server", "m.gguf", 8032, 2048, 8)
    assert "--rpc" not in local and local[local.index("-ngl") + 1] == "0"
    assert local[local.index("-dev") + 1] == "none"
    split = lce.server_cmd("llama-server", "m.gguf", 8032, 2048, 8, "127.0.0.1:50052", 7)
    assert split[split.index("--rpc") + 1] == "127.0.0.1:50052"
    assert split[split.index("-dev") + 1] == "RPC0"  # a local GPU backend must not take them
    assert split[split.index("-ngl") + 1] == "7"
    assert split[split.index("-ot") + 1] == r"^output\.weight=CPU"
    with pytest.raises(ValueError):
        lce.server_cmd("llama-server", "m.gguf", 8032, 2048, 8, None, 7)
    assert lce.server_env({})["OMP_WAIT_POLICY"] == "ACTIVE"


class FakeLlamaServer(BaseHTTPRequestHandler):
    seen = []
    TIMINGS = {"prompt_n": 6, "prompt_ms": 12.5, "predicted_n": 4, "predicted_ms": 50.0}

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.seen.append((self.path, body))
        if self.path == "/completion":
            res = {"content": " Paris.", "timings": self.TIMINGS}
        else:
            res = {"choices": [{"message": {"content": "Paris."}, "finish_reason": "length"}],
                   "timings": dict(self.TIMINGS, predicted_per_second=61.5)}
        data = json.dumps(res).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d" % srv.server_port


def test_generate_maps_llama_server_timings_to_the_gateway_contract():
    """Decode rate is llama-server's predicted_per_second: n tokens span n - 1 decode
    intervals (the first comes out of the prompt step), so n / predicted_ms overstated
    it (GLM-5.3-Flash: 1.39 reported vs 1.36 measured). No prompt cache: every request
    evaluates its whole prompt."""
    srv, url = _serve(FakeLlamaServer)
    try:
        res = lce.generate(url, {"prompt": "The capital of France is", "max_tokens": 4})
        assert res == {"text": " Paris.", "prompt_tokens": 6, "completion_tokens": 4,
                       "prefill_ms": 12.5, "decode_ms": 50.0, "tokps": 60.0,
                       "finish_reason": "stop", "raw_timings": FakeLlamaServer.TIMINGS}
        chat = lce.generate(url, {"messages": [{"role": "user", "content": "hi"}],
                                  "max_tokens": 4})
        assert (chat["text"], chat["tokps"], chat["finish_reason"]) == ("Paris.", 61.5, "length")
        paths = [(p, b.get("temperature"), b.get("n_predict", b.get("max_tokens")), b.get("cache_prompt"))
                 for p, b in FakeLlamaServer.seen[-2:]]
        assert paths == [("/completion", 0, 4, False), ("/v1/chat/completions", 0, 4, False)]
    finally:
        srv.shutdown()


def test_generate_keeps_reasoning_text_in_the_reply():
    """Reasoning models (Qwen3, GLM-5.3) put <think> text in reasoning_content by
    default, so a 64-token reply came back as an empty string; ask llama-server to
    leave it in content, the same text the MLX path returns."""
    FakeLlamaServer.seen.clear()
    srv, url = _serve(FakeLlamaServer)
    try:
        lce.generate(url, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    finally:
        srv.shutdown()
    path, body = FakeLlamaServer.seen[-1]
    assert path == "/v1/chat/completions" and body["reasoning_format"] == "none"


def test_engine_reports_llama_server_failure_to_the_gateway():
    lce.EngineHandler.server_url = "http://127.0.0.1:1"
    srv, url = _serve(lce.EngineHandler)
    try:
        res = lce._post(url + "/generate", {"prompt": "x", "max_tokens": 1})
        assert "error" in res
        with pytest.raises(urllib.error.HTTPError) as health:
            urllib.request.urlopen(url + "/health", timeout=5)
        assert health.value.code == 503
    finally:
        srv.shutdown()


def test_wait_http_stops_when_the_server_process_exits():
    proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    proc.wait()
    t = time.monotonic()
    with pytest.raises(RuntimeError):
        lce.wait_http("http://127.0.0.1:1/health", deadline_s=60, proc=proc)
    assert time.monotonic() - t < 10


def test_gguf_kv_size_uses_key_and_value_widths(tmp_path):
    p = tmp_path / "m.gguf"
    make_gguf(str(p), 2, (64, 96), (320, 32), n_kv_head=4, head_dim=192, value_dim=128)
    assert lce.gguf_info(str(p))["kv_bytes_per_token_layer"] == 4 * (192 + 128) * 2


def test_usbmux_list_keeps_usb_phones_only():
    text = json.dumps([
        {"UniqueDeviceID": "00008130-AAAABBBBCCCC", "DeviceName": "Phone",
         "ProductType": "iPhone16,2", "ProductVersion": "27.0.1", "ConnectionType": "USB"},
        {"UniqueDeviceID": "00008130-DDDD", "ConnectionType": "Network"}])
    devs = iosnode.parse_usbmux_list(text)
    assert [d["udid"] for d in devs] == ["00008130-AAAABBBBCCCC"]
    facts = iosnode.device_facts(devs[0])
    assert facts["memory_free_bytes"] == iosnode.PHONE_CAP_BYTES
    assert facts["name"] == "iphone-cccc"


def test_launch_args_reach_rpc_server_as_one_dvt_argument():
    assert iosnode.launch_args("XTL-T.dev.app") == "XTL-T.dev.app -H 127.0.0.1 -p 50052 -t 4"
    assert iosnode.launch_args("b", backend="CPU").endswith("-d CPU")


def test_rpc_hello_speaks_the_rpc_handshake():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    got = {}

    def fake_rpc_server():
        conn, _ = srv.accept()
        with conn:
            got["hello"] = conn.recv(1 + 8 + iosnode.RPC_CONN_CAPS_SIZE)
            rsp = bytes([7, 0, 0, 0]) + bytes(iosnode.RPC_CONN_CAPS_SIZE)
            conn.sendall(struct.pack("<Q", len(rsp)) + rsp)
            got["count"] = conn.recv(9)
            conn.sendall(struct.pack("<Q", 4) + struct.pack("<I", 1))
            got["mem"] = conn.recv(13)
            conn.sendall(struct.pack("<Q", 16) + struct.pack("<QQ", 60 * GB, 86 * GB))

    t = threading.Thread(target=fake_rpc_server, daemon=True)
    t.start()
    res = iosnode.rpc_hello("127.0.0.1", srv.getsockname()[1])
    t.join(2)
    srv.close()
    assert res["version"] == "7.0.0" and res["devices"] == 1
    assert (res["free_bytes"], res["total_bytes"]) == (60 * GB, 86 * GB)
    assert got["hello"][0] == iosnode.RPC_CMD_HELLO
    assert struct.unpack("<Q", got["hello"][1:9])[0] == iosnode.RPC_CONN_CAPS_SIZE
    assert got["count"] == bytes([iosnode.RPC_CMD_DEVICE_COUNT]) + bytes(8)
    assert got["mem"] == bytes([iosnode.RPC_CMD_GET_DEVICE_MEMORY]) + struct.pack("<QI", 4, 0)


def write_gguf(path, meta, tensors):
    """GGUF v3 file with the given (key, vtype, raw) metadata and (name, bytes) tensors."""
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for key, vtype, raw in meta:
        out += _gguf_string(key) + struct.pack("<I", vtype) + raw
    offset = 0
    for name, size in tensors:
        out += _gguf_string(name) + struct.pack("<I", 1) + struct.pack("<Q", size)
        out += struct.pack("<IQ", 0, offset)
        offset += size
    out += bytes(-len(out) % 32) + bytes(offset)
    with open(path, "wb") as f:
        f.write(out)


def test_gguf_info_sums_tensors_across_every_shard(tmp_path):
    """Big GGUFs ship as -0000N-of-0000M shards; the first is often only metadata.
    Sizing only that file made a 120 GB model look like 9 MB."""
    split = lambda no: [("split.no", 2, struct.pack("<H", no)),  # noqa: E731
                        ("split.count", 2, struct.pack("<H", 2))]
    meta = [("general.architecture", 8, _gguf_string("qwen3")),
            ("qwen3.block_count", 4, struct.pack("<I", 3)),
            ("qwen3.attention.head_count", 4, struct.pack("<I", 16)),
            ("qwen3.attention.head_count_kv", 4, struct.pack("<I", 8)),
            ("qwen3.attention.key_length", 4, struct.pack("<I", 128)),
            ("qwen3.embedding_length", 4, struct.pack("<I", 2048))] + split(0)
    first = tmp_path / "m-00001-of-00002.gguf"
    write_gguf(str(first), meta, [("token_embd.weight", 320), ("blk.0.attn_q.weight", 64),
                                  ("blk.0.ffn_up.weight", 96)])
    write_gguf(str(tmp_path / "m-00002-of-00002.gguf"), split(1),
               [("blk.1.attn_q.weight", 64), ("blk.1.ffn_up.weight", 96),
                ("blk.2.attn_q.weight", 64), ("blk.2.ffn_up.weight", 96),
                ("output_norm.weight", 32)])
    info = lce.gguf_info(str(first))
    assert (info["n_layer"], info["layer_bytes"], info["other_bytes"]) == (3, 160, 352)
    assert info["total_bytes"] == 3 * 160 + 352
    assert info["shards"] == 2


def test_tensor_split_fills_devices_in_order_and_refuses_what_cannot_fit():
    info = {"n_layer": 10, "layer_bytes": 4 * GB, "other_bytes": 1 * GB,
            "total_bytes": 41 * GB, "kv_bytes_per_token_layer": 1000}
    # 4 GB layers: 7 fit 30 GB, 2 fit 10 GB, 1 fits 5 GB; the last count holds the output slot
    assert lce.tensor_split(info, [30 * GB, 10 * GB, 5 * GB], 2048) == [7, 2, 2]
    with pytest.raises(ValueError):
        lce.tensor_split(info, [20 * GB, 10 * GB], 2048)  # 10 x ~4 GB layers need ~40 GB


def test_server_cmd_spreads_all_layers_over_n_rpc_devices():
    cmd = lce.server_cmd("llama-server", "m.gguf", 8032, 2048, 8,
                         rpc=["10.0.0.1:50060", "10.0.0.2:50060"], split=[3.0, 1.0])
    assert cmd[cmd.index("--rpc") + 1] == "10.0.0.1:50060,10.0.0.2:50060"
    assert cmd[cmd.index("-dev") + 1] == "RPC0,RPC1"
    assert cmd[cmd.index("-ngl") + 1] == "999"
    assert cmd[cmd.index("--tensor-split") + 1] == "3,1"
    assert cmd[cmd.index("-ot") + 1] == r"^output\.weight=CPU"
    with pytest.raises(ValueError):
        lce.server_cmd("llama-server", "m.gguf", 8032, 2048, 8, rpc=["a:1"], split=[1.0, 2.0])


def test_agent_starts_rpc_server_on_all_interfaces_with_cache_only_on_request(tmp_path, monkeypatch):
    """-c writes the node's whole share to its disk (42 GB on a 99% full Mac system disk),
    so the cache is opt-in."""
    from omarchy_cluster import agent
    seen = {}

    class FakePopen:
        def __init__(self, cmd, **kw):
            seen["cmd"], seen["kw"] = cmd, kw
            self.pid = 777

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(agent.subprocess, "Popen", FakePopen)
    res = agent.rpc_start({"binary": "/opt/llama/ggml-rpc-server", "port": 50060, "threads": 8})
    assert res["pid"] == 777
    assert seen["cmd"] == ["/opt/llama/ggml-rpc-server", "-H", "0.0.0.0", "-p", "50060", "-t", "8"]
    assert seen["kw"]["start_new_session"] is True
    agent.rpc_start({"binary": "/opt/llama/ggml-rpc-server", "port": 50060, "cache": True})
    assert seen["cmd"] == ["/opt/llama/ggml-rpc-server", "-H", "0.0.0.0", "-p", "50060", "-c"]
    monkeypatch.setenv("OMARCHY_CLUSTER_RPC_SERVER", "/x/rpc")
    agent.rpc_start({"port": 50061})
    assert seen["cmd"][:5] == ["/x/rpc", "-H", "0.0.0.0", "-p", "50061"]


def test_mac_rpc_server_defaults_to_private_metal_buffers_unless_the_agent_env_says_otherwise():
    """Under other GPU work on the Mac, private buffers decoded 2.3-2.6x faster; an operator
    who wants ggml's shared buffers sets GGML_METAL_SHARED_BUFFERS_ENABLE for the agent."""
    from omarchy_cluster import agent
    assert agent.rpc_server_env({}, "darwin")["GGML_METAL_SHARED_BUFFERS_DISABLE"] == "1"
    assert "GGML_METAL_SHARED_BUFFERS_DISABLE" not in agent.rpc_server_env({}, "linux")
    kept = agent.rpc_server_env({"GGML_METAL_SHARED_BUFFERS_ENABLE": "1"}, "darwin")
    assert "GGML_METAL_SHARED_BUFFERS_DISABLE" not in kept
    assert agent.rpc_server_env({"PATH": "/bin"}, "darwin")["PATH"] == "/bin"


def test_rpc_node_args_parse_names_and_optional_gb():
    from omarchy_cluster import cli
    assert cli._parse_rpc_nodes(["mac-a=50", "linux-b", "linux-c=7.5"]) == [
        ("mac-a", 50 * GB), ("linux-b", None), ("linux-c", int(7.5 * GB))]
    with pytest.raises(SystemExit):
        cli._parse_rpc_nodes(["bad=x"])


def test_rpc_node_accepts_a_raw_endpoint_with_or_without_a_budget():
    """An rpc-server some other tool started (CUDA box, USB-forwarded phone) joins as
    HOST:PORT; without =GB its budget comes from the memory it reports over RPC."""
    from omarchy_cluster import cli
    assert cli._parse_rpc_nodes(["10.0.0.9:50052=3.5", "10.0.0.9:50053"]) == [
        ("10.0.0.9:50052", int(3.5 * GB)), ("10.0.0.9:50053", None)]


def test_host_layers_stay_on_the_host_cpu_and_are_not_charged_to_devices():
    """A model a little bigger than every device together still runs: the first K
    layers stay on the host CPU, mmapped from the GGUF (paged from disk)."""
    info = {"n_layer": 10, "layer_bytes": 4 * GB, "other_bytes": 1 * GB,
            "total_bytes": 41 * GB, "kv_bytes_per_token_layer": 1000}
    assert lce.tensor_split(info, [21 * GB, 10 * GB], 2048, host_layers=3) == [5, 3]
    cmd = lce.server_cmd("llama-server", "m.gguf", 8032, 2048, 8, rpc=["a:1", "b:2"],
                         split=[2.0, 1.0], ngl=7)
    assert cmd[cmd.index("-ngl") + 1] == "7"


def test_tensor_split_charges_the_real_layer_total_not_n_times_the_largest():
    """GLM-5.3-Flash: 46 layers, largest 4.71 GB, 156.81 GB in all. Charging
    46 x 4.71 = 217 GB refused a model whose layers hold 155.6 GB."""
    info = {"n_layer": 46, "layer_bytes": int(4.71 * GB), "other_bytes": int(1.19 * GB),
            "total_bytes": int(156.81 * GB), "kv_bytes_per_token_layer": 2048}
    assert lce.tensor_split(info, [50 * GB, 86 * GB, 57 * GB, 8 * GB], 4096) == [14, 25, 8, 0]
    with pytest.raises(ValueError):
        lce.tensor_split(info, [100 * GB, 50 * GB], 4096)  # 150 GB < ~155.6 GB of layers


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _llama_cpp_layer_devices(weights, n_layer, ngl):
    """llama-model.cpp load_tensors at 65840ed: float32 cumulative split points, then
    layer il goes to upper_bound(splits, (il - i_gpu_start) / act_gpu_layers)."""
    import bisect
    acc, splits = 0.0, []
    for w in weights:
        acc = _f32(acc + w)
        splits.append(acc)
    splits = [_f32(s / acc) for s in splits]
    start, act = max(n_layer + 1 - ngl, 0), min(ngl, n_layer + 1)
    return [None if il < start or il - start >= act else
            bisect.bisect_right(splits, _f32((il - start) / act)) for il in range(n_layer)]


def test_tensor_split_keeps_every_device_under_budget_with_uneven_layers():
    """GLM-5.3-Flash failed to load: budgets passed as weights gave the 86 GB node a
    contiguous block of big MoE layers past its Vulkan heap. Counts must map 1:1."""
    per_layer = [int(g * GB) for g in [0.4, 0.4, 0.4] + [3.6] * 20 + [1.0] * 10 + [3.7] * 13]
    info = {"n_layer": len(per_layer), "layer_bytes": max(per_layer), "per_layer": per_layer,
            "other_bytes": 1 * GB, "total_bytes": sum(per_layer) + GB, "kv_bytes_per_token_layer": 0}
    caps = [39 * GB, 80 * GB, 57 * GB, 8 * GB]
    for host_layers in (0, 2):
        weights = lce.tensor_split(info, caps, 4096, host_layers=host_layers)
        devs = _llama_cpp_layer_devices(weights, len(per_layer), sum(weights))
        assert devs[:host_layers] == [None] * host_layers  # host layers stay on the host
        assert None not in devs[host_layers:]
        on_devices = [d for d in devs[host_layers:] if d is not None]
        assert on_devices == sorted(on_devices)  # one contiguous block per device, in order
        for d, cap in enumerate(caps):
            assert sum(s for s, x in zip(per_layer, devs) if x == d) <= cap


def test_start_gateway_returns_only_once_the_gateway_answers(tmp_path, monkeypatch):
    """serve printed the gateway pid and returned before it listened: a script that sent
    its first request at once got connection refused on a loaded 157 GB model. The
    stand-in gateway is a thread in this process that starts listening 1.5 s late."""
    from omarchy_cluster import cli
    monkeypatch.setenv("HOME", str(tmp_path))
    os.makedirs(str(tmp_path / ".local/state/omarchy-cluster"))
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    servers = []

    def late_listen():
        time.sleep(1.5)  # a gateway still importing when serve returns
        servers.append(ThreadingHTTPServer(("127.0.0.1", port), H))
        servers[0].serve_forever()

    class Alive:
        pid = os.getpid()

        def poll(self):
            return None

    def slow_spawn(cmd, log_name):
        threading.Thread(target=late_listen, daemon=True).start()
        return Alive(), str(tmp_path / log_name)

    monkeypatch.setattr(cli, "_spawn", slow_spawn)
    try:
        cli._start_gateway({}, port, "http://127.0.0.1:9")
        with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % port, timeout=2) as r:
            assert r.status == 200
    finally:
        if servers:
            servers[0].shutdown()
