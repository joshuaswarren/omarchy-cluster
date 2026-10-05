import builtins
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import agent


def test_rank_one_receives_engine_url(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "quiet_flag_path",
                        lambda: str(tmp_path / "absent-quiet"))
    monkeypatch.setattr(agent.os.path, "expanduser", lambda path: str(tmp_path))
    launched = {}

    class Process:
        pid = 123

    def popen(cmd, **kwargs):
        launched["cmd"] = cmd
        return Process()

    monkeypatch.setattr(agent.subprocess, "Popen", popen)
    agent.rank_start({"rank": 1, "model": "m", "layers": "1:2",
                      "hostfile_content": "[]", "python": "python3",
                      "engine_url": "http://rank0:8031"})
    assert launched["cmd"][launched["cmd"].index("--engine") + 1] == "http://rank0:8031"


def test_rank_start_refuses_when_quiet(tmp_path, monkeypatch):
    flag = tmp_path / "quiet"
    flag.write_text("")
    monkeypatch.setattr(agent, "quiet_flag_path", lambda: str(flag))
    res = agent.rank_start({"rank": 0, "model": "m", "layers": "0:1",
                            "hostfile_content": "[]", "python": "python3"})
    assert res.get("quiet") is True
    assert "refusing" in res["error"]


def test_rank_start_proceeds_when_not_quiet(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "quiet_flag_path",
                        lambda: str(tmp_path / "absent-quiet"))
    monkeypatch.setattr(agent, "quiet_enabled", lambda path=None: False)
    called = {}

    def fake_popen(cmd, env, cwd, stdout, stderr, start_new_session):
        called["cmd"] = cmd
        return type("P", (), {"pid": 4242})

    real_popen = agent.subprocess.Popen
    monkeypatch.setattr(agent.subprocess, "Popen", fake_popen)
    res = agent.rank_start({"rank": 1, "model": "m", "layers": "1:2",
                            "hostfile_content": "[[]]",
                            "python": "python3", "gpu_turn_minutes": 0})
    assert res["pid"] == 4242
    assert not res.get("quiet")
    monkeypatch.setattr(agent.subprocess, "Popen", real_popen)


def test_quiet_flag_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "quiet_flag_path", lambda: str(tmp_path / "quiet"))
    assert agent.quiet_enabled() is False
    open(agent.quiet_flag_path(), "a").close()
    assert agent.quiet_enabled() is True
    os.remove(agent.quiet_flag_path())
    assert agent.quiet_enabled() is False
