import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import bootstrap


class FakeSsh:
    """Records every remote command; answers the probes the way a stock node would."""

    def __init__(self, os_name="Linux", python_ok=True, avahi=True, manager="pacman", sudo_ok=True,
                 hostname="omarchy-m2", fail_on=None):
        self.calls = []
        self.os_name, self.python_ok, self.avahi = os_name, python_ok, avahi
        self.manager, self.sudo_ok, self.hostname, self.fail_on = manager, sudo_ok, hostname, fail_on

    def __call__(self, target, command, stdin=None):
        self.calls.append((target, command, stdin))
        if self.fail_on and self.fail_on in command:
            return 1, "", "boom"
        if command.startswith("uname -s"):
            return 0, self.os_name + "\n", ""
        if "sys.version_info" in command:
            return (0, "ok\n", "") if self.python_ok else (0, "old\n", "")
        if command.startswith("command -v avahi-daemon"):
            return (0, "/usr/bin/avahi-daemon\n", "") if self.avahi else (1, "", "")
        if command.startswith("command -v pacman"):
            return (0, "", "") if self.manager == "pacman" else (1, "", "")
        if command.startswith("command -v apt-get"):
            return (0, "", "") if self.manager == "apt-get" else (1, "", "")
        if "sudo -n" in command:
            return (0, "", "") if self.sudo_ok else (1, "", "sudo: a password is required")
        if command.startswith("hostname -s"):
            return 0, self.hostname + "\n", ""
        return 0, "", ""

    def commands(self):
        return [c for _, c, _ in self.calls]


def run_bootstrap(ssh, tmp_path, **kw):
    out = []
    kw.setdefault("token", "tok-123")
    kw.setdefault("check", lambda host, port: True)
    name = bootstrap.bootstrap("me@node1", ssh=ssh, src_dir=str(tmp_path), say=out.append, **kw)
    return name, out


@pytest.fixture
def src(tmp_path):
    (tmp_path / "omarchy_cluster").mkdir()
    (tmp_path / "omarchy_cluster" / "__init__.py").write_text("x = 1\n")
    return tmp_path


def test_a_stock_linux_node_gets_the_package_token_and_agent_in_order_and_its_name_is_printed(src):
    ssh = FakeSsh()
    name, out = run_bootstrap(ssh, src)
    assert name == "omarchy-m2"
    cmds = ssh.commands()
    order = [next(i for i, c in enumerate(cmds) if marker in c)
             for marker in ("uname -s", "tar -xzf", "token", "install-agent", "hostname -s")]
    assert order == sorted(order)
    assert not any("sudo -n" in c for c in cmds)  # avahi was already there
    unpack = next(call for call in ssh.calls if "tar -xzf" in call[1])
    assert unpack[2][:2] == b"\x1f\x8b"  # a gzip stream of the package
    token_call = next(call for call in ssh.calls if call[1].startswith("umask 077"))
    assert token_call[2] == b"tok-123"
    assert "install-agent" in next(c for c in cmds if "install-agent" in c)
    assert any("omarchy-m2" in line for line in out)


def test_missing_avahi_is_installed_with_the_nodes_package_manager(src):
    ssh = FakeSsh(avahi=False, manager="apt-get")
    run_bootstrap(ssh, src)
    install = [c for c in ssh.commands() if "sudo -n" in c]
    assert any("apt-get install -y avahi-daemon avahi-utils" in c for c in install)
    assert any("systemctl enable --now avahi-daemon" in c for c in install)


def test_missing_avahi_without_passwordless_sudo_says_what_to_run(src):
    ssh = FakeSsh(avahi=False, sudo_ok=False)
    with pytest.raises(bootstrap.BootstrapError) as e:
        run_bootstrap(ssh, src)
    assert "pacman -S --noconfirm --needed avahi nss-mdns" in str(e.value)
    assert not any("install-agent" in c for c in ssh.commands())


def test_macos_needs_no_avahi(src):
    ssh = FakeSsh(os_name="Darwin", avahi=False)
    run_bootstrap(ssh, src)
    assert not any("avahi" in c for c in ssh.commands())
    assert any("install-agent" in c for c in ssh.commands())


@pytest.mark.parametrize("kwargs,message", [
    ({"os_name": "FreeBSD"}, "FreeBSD"),
    ({"python_ok": False}, "Python 3.9"),
])
def test_an_unsupported_node_stops_before_anything_is_copied(src, kwargs, message):
    ssh = FakeSsh(**kwargs)
    with pytest.raises(bootstrap.BootstrapError) as e:
        run_bootstrap(ssh, src)
    assert message in str(e.value)
    assert not any("tar -xzf" in c for c in ssh.commands())


def test_a_failing_remote_step_names_the_step_and_the_target(src):
    ssh = FakeSsh(fail_on="install-agent")
    with pytest.raises(bootstrap.BootstrapError) as e:
        run_bootstrap(ssh, src)
    assert "me@node1" in str(e.value) and "boom" in str(e.value)


def test_a_node_that_never_answers_its_health_check_is_an_error(src):
    with pytest.raises(bootstrap.BootstrapError) as e:
        run_bootstrap(FakeSsh(), src, check=lambda host, port: False)
    assert "node1:8025" in str(e.value)


def test_no_local_token_is_an_error_that_points_at_install_agent(src, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    with pytest.raises(bootstrap.BootstrapError) as e:
        run_bootstrap(FakeSsh(), src, token=None)
    assert "install-agent" in str(e.value)


def test_the_cli_bootstraps_every_target_and_exits_1_when_one_fails(monkeypatch, capsys):
    from omarchy_cluster import cli
    seen = []

    def fake(target, token=None, port=None):
        seen.append((target, token, port))
        if target == "bad@host":
            raise bootstrap.BootstrapError("no route")

    monkeypatch.setattr(bootstrap, "bootstrap", fake)
    with pytest.raises(SystemExit) as e:
        cli.main(["bootstrap", "bad@host", "ok@host", "--port", "9025"])
    assert e.value.code == 1
    assert seen == [("bad@host", None, 9025), ("ok@host", None, 9025)]
    assert "bootstrap: no route" in capsys.readouterr().err


def test_ssh_puts_the_target_after_dashes_so_it_cannot_be_read_as_an_option(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append((argv, kw))

        class Done:
            returncode, stdout, stderr = 0, b"", b""
        return Done()

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    bootstrap.ssh_run("-oProxyCommand=id", "uname -s")
    bootstrap.ssh_run("me@node1", "cat", stdin=b"x")
    first, second = calls
    assert first[0][-3:] == ["--", "-oProxyCommand=id", "uname -s"]
    assert first[1]["stdin"] is bootstrap.subprocess.DEVNULL and "input" not in first[1]
    assert second[1]["input"] == b"x"


def test_an_unreadable_token_file_is_a_clean_error(monkeypatch, tmp_path):
    from omarchy_cluster import cli
    with pytest.raises(SystemExit) as e:
        cli.main(["bootstrap", "me@node1", "--token-file", str(tmp_path / "missing")])
    assert "cannot read --token-file" in str(e.value)
