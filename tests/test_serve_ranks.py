"""serve with --nodes and --split: any number of ranks in the order given, no links file, ring addresses from --ring-ip."""
import argparse
import json

import pytest

from omarchy_cluster import cli, llamacpp_engine

GB = 10 ** 9


def make_model(tmp_path, layers=8):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"num_hidden_layers": layers, "hidden_size": 64, "num_attention_heads": 4,
                                               "num_key_value_heads": 2, "head_dim": 16}))
    (d / "model.safetensors").write_bytes(b"\0" * (layers * 1000))
    return str(d)


def make_nodes(free=64 * GB):
    return {name: {"ip": ip, "port": 8025, "facts": {"os": os_, "memory_free_bytes": free, "memory_total_bytes": free,
                                                    "interfaces": [{"ips": [{"ip": ip}]}]}}
            for name, ip, os_ in (("studio", "10.0.0.1", "macOS 26"), ("m2", "10.0.0.2", "Linux"),
                                  ("m1max", "10.0.0.3", "Linux"), ("m1", "10.0.0.4", "Linux"))}


def serve_args(model, **kw):
    base = dict(model=model, engine="mlx", port=18020, engine_port=18031, python_mac="/usr/bin/true", python_linux="/usr/bin/true",
                rank_pythonpath=None, gpu_turn=0, ctx=2048, links=None, no_decode=[], stages=None, split=None, nodes=None, ring_ip=[])
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    nodes = make_nodes()
    launches = []

    def launch(node, rank, model, layers, hostfile, args, facts_os):
        launches.append({"ip": node["ip"], "rank": rank, "layers": layers, "hostfile": json.loads(hostfile)})
        return {"pid": 1000 + rank}

    monkeypatch.setattr(cli.discover, "discover_nodes", lambda *a, **k: nodes)
    monkeypatch.setattr(cli, "_launch_rank_via_agent", launch)
    monkeypatch.setattr(cli, "_start_gateway", lambda state, port, engine: state.update(engine=engine))
    monkeypatch.setattr(llamacpp_engine, "wait_http", lambda *a, **k: None)
    return launches


def test_four_ranks_start_in_the_order_given_with_the_split_and_a_ring_address_override(tmp_path, cluster):
    model = make_model(tmp_path)
    cli.cmd_serve(serve_args(model, nodes="studio,m2,m1max,m1", split="1,2,2,3", ring_ip=["m1max=10.9.9.9"]))
    assert [(c["ip"], c["rank"], c["layers"]) for c in cluster] == [
        ("10.0.0.1", 0, "1,2,2,3"), ("10.0.0.2", 1, "1,2,2,3"), ("10.0.0.3", 2, "1,2,2,3"), ("10.0.0.4", 3, "1,2,2,3")]
    ring = [entry[0].split(":")[0] for entry in cluster[0]["hostfile"]]
    assert ring == ["10.0.0.1", "10.0.0.2", "10.9.9.9", "10.0.0.4"]  # the override replaces the third rank's first interface


def test_more_than_two_ranks_from_the_planner_need_an_explicit_split(tmp_path, monkeypatch, cluster):
    three = {"mode": "pipeline", "stages": [{"node": "studio"}, {"node": "m2"}, {"node": "m1max"}]}
    monkeypatch.setattr(cli, "cmd_place", lambda args: three)
    with pytest.raises(SystemExit, match="--split"):
        cli.cmd_serve(serve_args(make_model(tmp_path)))
    assert cluster == []


def test_a_split_that_does_not_cover_the_model_is_refused(tmp_path, cluster):
    with pytest.raises(SystemExit, match="does not cover"):
        cli.cmd_serve(serve_args(make_model(tmp_path), nodes="studio,m2", split="3,3"))
    assert cluster == []


def test_a_node_without_room_for_its_layers_is_refused_before_any_rank_starts(tmp_path, monkeypatch, cluster):
    monkeypatch.setattr(cli.discover, "discover_nodes", lambda *a, **k: make_nodes(free=10))
    with pytest.raises(SystemExit, match="do not fit"):
        cli.cmd_serve(serve_args(make_model(tmp_path), nodes="studio,m2", split="4,4"))
    assert cluster == []


def test_a_malformed_ring_address_is_refused(tmp_path, cluster):
    with pytest.raises(SystemExit, match="NAME=VALUE"):
        cli.cmd_serve(serve_args(make_model(tmp_path), nodes="studio,m2", split="4,4", ring_ip=["m2"]))


def test_rank_zero_is_the_decode_tail(tmp_path, cluster):
    plan = cli._explicit_plan(serve_args(make_model(tmp_path), nodes="studio,m2,m1max", split="2,3,3"))
    assert plan["decode_tail"] == "studio"


def test_a_ring_address_for_a_node_not_in_the_serve_is_refused(tmp_path, cluster):
    with pytest.raises(SystemExit, match="m1max"):
        cli.cmd_serve(serve_args(make_model(tmp_path), nodes="studio,m2", split="4,4", ring_ip=["m1max=10.9.9.9"]))
    assert cluster == []


def test_rank_env_and_the_adapter_path_reach_a_macos_rank_too(monkeypatch):
    sent = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    monkeypatch.setattr(cli, "_read_local_token", lambda: "t")
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda req, timeout: sent.append(json.loads(req.data)) or Reply())
    args = serve_args("m", rank_pythonpath="~/adapters", rank_env=["MLX_DISABLE_COMPILE=1"], engine_url="http://10.0.0.1:18031")
    cli._launch_rank_via_agent({"ip": "10.0.0.1"}, 1, "m", "4,4", "[]", args, "macOS 26")
    assert sent[0]["env"] == {"MLX_DISABLE_COMPILE": "1"}
    assert sent[0]["pythonpath"] == "~/.local/share/omarchy-cluster/src:~/adapters"


def test_no_fit_check_prints_an_override_instead_of_refusing(tmp_path, monkeypatch, cluster, capsys):
    monkeypatch.setattr(cli.discover, "discover_nodes", lambda *a, **k: make_nodes(free=1_200_000))
    cli.cmd_serve(serve_args(make_model(tmp_path), nodes="studio,m2,m2", split="2,3,3", no_fit_check=True))
    assert "OVERRIDE: m2 holds 6 layers" in capsys.readouterr().out
    assert len(cluster) == 3
