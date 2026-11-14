"""DH-P7-01: `omarchy-cluster join` with a fake ssh (recorded fixture)."""
import getpass
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import bootstrap, cli, discover

OMARCHY_RELEASE = 'ID=omarchy\nNAME="Omarchy"\nPRETTY_NAME="Omarchy Linux"\n'
DEBIAN_RELEASE = 'PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"\nID=debian\n'


class FakeSsh:
    """Records every remote command; answers the probes the way an Omarchy node would."""

    def __init__(self, os_name="Linux", release=OMARCHY_RELEASE, avahi=True, manager="pacman",
                 hostname="omarchy-m2", fail_on=None, omarchy_version=None):
        self.calls = []
        self.os_name, self.release, self.avahi = os_name, release, avahi
        self.manager, self.hostname, self.fail_on = manager, hostname, fail_on
        self.omarchy_version = omarchy_version

    def __call__(self, target, command, stdin=None):
        self.calls.append((target, command, stdin))
        if self.fail_on and self.fail_on in command:
            return 1, "", "boom"
        if command.startswith("uname -s"):
            return 0, self.os_name + "\n", ""
        if command.startswith("omarchy-version"):
            if self.omarchy_version:
                return 0, self.omarchy_version + "\n", ""
            return 127, "", "omarchy-version: command not found"
        if "/etc/os-release" in command:
            return 0, self.release, ""
        if command.startswith("command -v avahi-daemon"):
            return (0, "/usr/bin/avahi-daemon\n", "") if self.avahi else (1, "", "")
        if command.startswith("command -v pacman"):
            return (0, "", "") if self.manager == "pacman" else (1, "", "")
        if "sys.version_info" in command:
            return 0, "ok\n", ""
        if command.startswith("hostname -s"):
            return 0, self.hostname + "\n", ""
        return 0, "", ""

    def commands(self):
        return [c for _, c, _ in self.calls]


FOUND = {"m2-max": {"name": "m2-max", "host": "m2-max.local", "ip": "10.0.0.6", "port": 8025}}

CLOCK = lambda: __import__("datetime").datetime(2030, 1, 1, 12, 0, 0)


def _no_discovery(*a, **k):
    raise AssertionError("discovery must not run for a user@host candidate")


@pytest.fixture
def journal(tmp_path, monkeypatch):
    path = tmp_path / "join.log"
    monkeypatch.setattr(bootstrap, "JOURNAL", str(path))
    hostfile = tmp_path / "home" / "hosts"  # nothing may write the hand-written config
    monkeypatch.setattr(discover, "HOSTFILE", str(hostfile))
    return path, hostfile


def run_join(ssh, tmp_path, candidate="m2-max", find=FOUND, **kw):
    out = []
    kw.setdefault("token", "tok-123")
    kw.setdefault("check", lambda host, port: True)
    kw.setdefault("clock", CLOCK)
    name = bootstrap.join(candidate, ssh=ssh, src_dir=str(tmp_path), say=out.append,
                          find=(lambda *a, **k: find), **kw)
    return name, out


@pytest.fixture
def src(tmp_path):
    (tmp_path / "omarchy_cluster").mkdir()
    (tmp_path / "omarchy_cluster" / "__init__.py").write_text("x = 1\n")
    return tmp_path


def test_a_discover_candidate_is_joined_in_order_and_journaled_with_the_clock_time(src, journal):
    journal, hostfile = journal
    ssh = FakeSsh()
    name, out = run_join(ssh, src)
    assert name == "omarchy-m2"
    cmds = ssh.commands()
    order = [next(i for i, c in enumerate(cmds) if marker in c)
             for marker in ("true", "uname -s", "/etc/os-release", "tar -xzf",
                            "install-agent", "hostname -s")]
    assert order == sorted(order)
    unpack = next(call for call in ssh.calls if "tar -xzf" in call[1])
    assert unpack[0] == "%s@10.0.0.6" % getpass.getuser()  # resolved from discover, no host typed
    assert unpack[2][:2] == b"\x1f\x8b"  # the package rides as a gzip stream
    token_call = next(call for call in ssh.calls if call[1].startswith("umask 077"))
    assert token_call[2] == b"tok-123"
    assert journal.read_text() == "2030-01-01 12:00:00  join  m2-max -> omarchy-m2 (%s@10.0.0.6)\n" % getpass.getuser()
    assert not hostfile.exists()  # no configuration written by hand
    assert any("omarchy-m2" in line for line in out)


def test_an_explicit_user_at_host_skips_discovery(src, journal):
    ssh = FakeSsh()
    name, _ = run_join(ssh, src, candidate="jw@10.0.0.9", find=_no_discovery)
    assert name == "omarchy-m2"
    assert ssh.calls[0][0] == "jw@10.0.0.9"


def test_an_unknown_candidate_names_the_discover_candidates(src, journal):
    with pytest.raises(bootstrap.BootstrapError) as e:
        run_join(FakeSsh(), src, candidate="nope", find={"a": {"ip": "1"}, "b": {"ip": "2"}})
    assert "a, b" in str(e.value)


def test_a_macos_peer_is_refused_naming_why(src, journal):
    ssh = FakeSsh(os_name="Darwin")
    with pytest.raises(bootstrap.BootstrapError) as e:
        run_join(ssh, src)
    assert "Darwin" in str(e.value) and "not Omarchy Linux" in str(e.value)
    assert not any("install-agent" in c for c in ssh.commands())  # refused before any install


def test_a_plain_linux_peer_is_refused_naming_why(src, journal):
    ssh = FakeSsh(release=DEBIAN_RELEASE)
    with pytest.raises(bootstrap.BootstrapError) as e:
        run_join(ssh, src)
    assert "Debian GNU/Linux 12 (bookworm)" in str(e.value)
    assert not any("install-agent" in c for c in ssh.commands())


def test_a_peer_without_os_release_is_refused(src, journal):
    with pytest.raises(bootstrap.BootstrapError) as e:
        run_join(FakeSsh(release=""), src)
    assert "/etc/os-release" in str(e.value)


def test_the_approved_key_is_the_default_and_names_the_fix_when_it_is_missing(src, journal):
    ssh = FakeSsh(fail_on="true")
    with pytest.raises(bootstrap.BootstrapError) as e:
        run_join(ssh, src)
    assert "ssh-copy-id" in str(e.value)
    assert not any("install-agent" in c for c in ssh.commands())


def test_one_time_code_stays_behind_the_same_function_until_issue_64(src, journal):
    with pytest.raises(bootstrap.BootstrapError) as e:
        run_join(FakeSsh(), src, mode="one-time-code")
    assert "issue 64" in str(e.value)


def test_the_joined_peer_shows_in_status_with_memory_and_a_link_line():
    facts = {"os": "Omarchy Linux", "chip": "M2 Max", "gpu_backend": "vulkan",
             "memory_free_bytes": 64 * 2**30, "memory_total_bytes": 96 * 2**30,
             "heartbeat_age_s": 0.5}
    nodes = {"omarchy-m2": {"name": "omarchy-m2", "ip": "10.0.0.6", "facts": facts}}
    table = cli.node_table(nodes)
    assert "omarchy-m2" in table and "Omarchy Linux" in table and "64.0/96GB" in table
    links = {"generated": "now", "pairs": [
        {"a": "w7g", "b": "omarchy-m2", "routes": [], "mcdma": None, "pinned": None}]}
    assert "w7g <-> omarchy-m2" in cli.links_table(links)


def test_the_cli_wires_join_with_the_approved_key_default(capsys):
    from omarchy_cluster.cli import main
    with pytest.raises(SystemExit) as ei:
        main(["join", "--help"])
    assert ei.value.code == 0  # an unregistered verb would exit 2
    out = capsys.readouterr().out
    assert "--auth" in out and "approved-key" in out


ARCH_ARM_RELEASE = 'NAME="Arch Linux ARM"\nPRETTY_NAME="Arch Linux ARM"\nID=archarm\nID_LIKE=arch\n'


def test_a_real_omarchy_machine_with_an_arch_os_release_is_accepted(src, journal):
    # The 13-inch M1 lab machine reports ID=archarm; `omarchy-version` is what says it is Omarchy.
    ssh = FakeSsh(release=ARCH_ARM_RELEASE, omarchy_version="4.0.0.alpha.quattro")
    name, _ = run_join(ssh, src)
    assert name == "omarchy-m2"
    assert any("install-agent" in c for c in ssh.commands())


def test_plain_arch_without_omarchy_version_is_still_refused(src, journal):
    ssh = FakeSsh(release=ARCH_ARM_RELEASE)
    with pytest.raises(bootstrap.BootstrapError) as e:
        run_join(ssh, src)
    assert "Arch Linux ARM" in str(e.value)
    assert not any("install-agent" in c for c in ssh.commands())


def test_an_empty_token_is_an_error_not_a_fallback_to_this_machines_secret(src, journal, monkeypatch):
    monkeypatch.setattr(bootstrap.cluster_token, "read", lambda: "local-secret")
    ssh = FakeSsh()
    with pytest.raises(bootstrap.BootstrapError, match="empty"):
        run_join(ssh, src, token="")
    assert ssh.calls == []  # refused before any remote command
    assert not any(c[2] == b"local-secret" for c in ssh.calls)


def test_the_cli_refuses_an_empty_token_file_before_any_ssh(tmp_path, monkeypatch):
    from omarchy_cluster.cli import main
    empty = tmp_path / "token"
    empty.write_text("  \n")
    monkeypatch.setattr(bootstrap, "ssh_run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ssh ran")))
    with pytest.raises(SystemExit) as ei:
        main(["join", "jw@10.0.0.9", "--token-file", str(empty)])
    assert "empty" in str(ei.value.code)  # sys.exit(message): the message is the code
