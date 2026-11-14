"""#88: `omarchy-cluster leave` undoes a join; #89: a discovered node without ip or host is refused by name."""
import getpass
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import bootstrap, discover

OMARCHY_RELEASE = 'ID=omarchy\nNAME="Omarchy"\nPRETTY_NAME="Omarchy Linux"\n'
UNIT = "omarchy-cluster-agent.service"
FOUND = {"m2-max": {"name": "m2-max", "host": "m2-max.local", "ip": "10.0.0.6", "port": 8025}}
CLOCK = lambda: __import__("datetime").datetime(2030, 1, 1, 13, 0, 0)


class PeerSsh:
    """A peer that keeps systemd's state: the unit file stays after `disable --now`; active and enabled flip."""

    def __init__(self, os_name="Linux", installed=True, active=True, enabled=True, token=True,
                 stop_fails=False, show_rc=0, test_rc=None, hostname_rc=0):
        self.calls = []
        self.os_name, self.installed, self.active, self.enabled = os_name, installed, active, enabled
        self.token, self.stop_fails, self.show_rc, self.test_rc = token, stop_fails, show_rc, test_rc
        self.hostname_rc = hostname_rc

    def __call__(self, target, command, stdin=None):
        self.calls.append(command)
        if command.startswith("uname -s"):
            return 0, self.os_name + "\n", ""
        if command.startswith("hostname -s"):
            return self.hostname_rc, "m2-max\n" if not self.hostname_rc else "", "" if not self.hostname_rc else "no hostname"
        if "/etc/os-release" in command:
            return 0, OMARCHY_RELEASE, ""
        if command.startswith("systemctl --user show"):
            if self.show_rc:
                return self.show_rc, "", "Failed to connect to bus"
            if not self.installed:
                return 0, "LoadState=not-found\nActiveState=inactive\nUnitFileState=\n", ""
            return 0, "LoadState=loaded\nActiveState=%s\nUnitFileState=%s\n" % (
                "active" if self.active else "inactive", "enabled" if self.enabled else "disabled"), ""
        if command.startswith("systemctl --user disable --now"):
            if self.stop_fails:
                return 1, "", "Failed to stop"
            self.active = self.enabled = False
            return 0, "", ""
        if command.startswith("test -e") and "token" in command:
            if self.test_rc:
                return self.test_rc, "", "ssh: connection lost"
            return 0, "present\n" if self.token else "absent\n", ""
        if command.startswith("rm -f") and "token" in command:
            self.token = False
            return 0, "", ""
        return 0, "", ""


@pytest.fixture(autouse=True)
def joined(tmp_path, monkeypatch):
    path = tmp_path / "joined"
    monkeypatch.setattr(discover, "JOINED", str(path))
    return path


@pytest.fixture
def journal(tmp_path, monkeypatch):
    path = tmp_path / "join.log"
    monkeypatch.setattr(bootstrap, "JOURNAL", str(path))
    return path


def run_leave(ssh, candidate="m2-max", find=FOUND, **kw):
    out = []
    bootstrap.leave(candidate, ssh=ssh, say=out.append, clock=CLOCK, find=lambda *a, **k: find, **kw)
    return out


def test_leave_stops_and_disables_the_agent_removes_the_token_and_journals(journal):
    peer = PeerSsh()
    run_leave(peer)
    assert not peer.active and not peer.enabled and not peer.token and peer.installed
    assert any(c.startswith("systemctl --user disable --now " + UNIT) for c in peer.calls)
    assert journal.read_text() == ("2030-01-01 13:00:00  leave  m2-max (%s@10.0.0.6): agent stopped and disabled, "
                                   "token removed\n" % getpass.getuser())


def test_leave_is_safe_to_run_twice(journal):
    peer = PeerSsh()
    run_leave(peer)
    peer.calls.clear()
    run_leave(peer)
    assert not any("disable --now" in c or c.startswith("rm -f") for c in peer.calls)
    assert journal.read_text().splitlines()[-1].endswith("agent already stopped and disabled, token already gone")


def test_a_peer_with_no_agent_unit_is_reported_as_not_installed(journal):
    peer = PeerSsh(installed=False, token=False)
    run_leave(peer)
    assert not any("disable --now" in c for c in peer.calls)
    assert journal.read_text().splitlines()[-1].endswith("agent not installed, token already gone")


@pytest.mark.parametrize("rc", [1, 255])
def test_a_failed_service_inspection_is_an_error_and_keeps_the_token(journal, rc):
    peer = PeerSsh(show_rc=rc)  # the user bus or ssh failed: the agent may still be running
    with pytest.raises(bootstrap.BootstrapError, match="Failed to connect to bus"):
        run_leave(peer)
    assert peer.active and peer.token and not journal.exists()
    assert not any(c.startswith("rm -f") for c in peer.calls)


def test_a_failed_token_check_is_an_error_not_already_gone(journal):
    peer = PeerSsh(test_rc=255)
    with pytest.raises(bootstrap.BootstrapError, match="connection lost"):
        run_leave(peer)
    assert not journal.exists()


def test_a_failed_agent_stop_keeps_the_token_and_raises(journal):
    peer = PeerSsh(stop_fails=True)
    with pytest.raises(bootstrap.BootstrapError, match="Failed to stop"):
        run_leave(peer)
    assert peer.token and not journal.exists()


def test_leave_refuses_a_peer_that_is_not_omarchy_linux(journal):
    peer = PeerSsh(os_name="Darwin")
    with pytest.raises(bootstrap.BootstrapError, match="not Omarchy Linux"):
        run_leave(peer)
    assert not any("disable" in c for c in peer.calls)


def test_leave_takes_user_at_host_without_discovery(journal):
    def no_find(*a, **k):
        raise AssertionError("discovery must not run for a user@host candidate")
    peer = PeerSsh()
    bootstrap.leave("jw@10.0.0.9", ssh=peer, say=lambda *_: None, clock=CLOCK, find=no_find)
    assert not peer.active


@pytest.mark.parametrize("verb", ["join", "leave"])
@pytest.mark.parametrize("node", [{"name": "ghost"}, {"name": "ghost", "ip": None, "host": None}])
def test_a_discovered_node_without_ip_or_host_is_refused_by_name_not_as_user_at_none(journal, verb, node, monkeypatch):
    monkeypatch.setattr(bootstrap.cluster_token, "read", lambda: "tok")  # join resolves the token before anything else
    peer = PeerSsh()
    call = getattr(bootstrap, verb)
    with pytest.raises(bootstrap.BootstrapError, match="ghost has no ip or host"):
        call("ghost", ssh=peer, say=lambda *_: None, clock=CLOCK, find=lambda *a, **k: {"ghost": node})
    assert peer.calls == []


def test_the_cli_wires_leave(capsys):
    from omarchy_cluster.cli import main
    with pytest.raises(SystemExit) as ei:
        main(["leave", "--help"])
    assert ei.value.code == 0
    assert "CANDIDATE" in capsys.readouterr().out


def test_leave_forgets_the_trust_record_for_that_peer_only(journal, joined):
    discover.remember_joined("m2-max", ["10.0.0.6"])
    discover.remember_joined("studio", ["10.0.0.7"])
    run_leave(PeerSsh())
    assert discover.read_joined() == {"studio": {"10.0.0.7:8025"}}  # discovery no longer sends m2-max the token


def test_a_second_leave_leaves_the_other_records_alone(journal, joined):
    discover.remember_joined("m2-max", ["10.0.0.6"])
    discover.remember_joined("studio", ["10.0.0.7"])
    peer = PeerSsh()
    run_leave(peer)
    run_leave(peer)
    assert discover.read_joined() == {"studio": {"10.0.0.7:8025"}}


def test_leave_that_cannot_learn_the_peers_name_changes_nothing(journal, joined):
    discover.remember_joined("m2-max", ["10.0.0.6"])
    peer = PeerSsh(hostname_rc=255)
    with pytest.raises(bootstrap.BootstrapError, match="hostname"):
        run_leave(peer)
    assert peer.active and peer.token and not journal.exists()
    assert discover.read_joined() == {"m2-max": {"10.0.0.6:8025"}}


def test_a_failed_leave_keeps_the_trust_record(journal, joined):
    discover.remember_joined("m2-max", ["10.0.0.6"])
    with pytest.raises(bootstrap.BootstrapError):
        run_leave(PeerSsh(stop_fails=True))
    assert "m2-max" in discover.read_joined()
