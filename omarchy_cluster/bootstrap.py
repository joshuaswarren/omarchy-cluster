"""`omarchy-cluster bootstrap user@host`: turn a bare machine into a cluster node over ssh.

Steps, in order: detect the OS, check Python, install Avahi on Linux, copy this package,
copy the cluster token, run `install-agent` there, wait for the agent to answer, and print
the name the node shows up under in `omarchy-cluster discover`.

`omarchy-cluster join <candidate>` (DH-P7-01) is the one-command front door on the same
machinery: resolve the candidate from discover, authenticate (approved ssh key by default),
refuse any peer that is not Omarchy Linux, bootstrap it, and journal the join.
"""
from __future__ import annotations

import io
import getpass
import os
import ssl
import subprocess
import tarfile
import time
import urllib.request
from datetime import datetime

from . import client, discover, pins, tls
from . import token as cluster_token

AGENT_PORT = 8025
REMOTE_SRC = "~/.local/share/omarchy-cluster/src"
JOURNAL = "~/.local/state/omarchy-cluster/join.log"
# (probe command name, install command) in the order tried; each install runs under sudo -n.
AVAHI_INSTALL = [
    ("pacman", "pacman -S --noconfirm --needed avahi nss-mdns"),
    ("apt-get", "apt-get install -y avahi-daemon avahi-utils"),
    ("dnf", "dnf install -y avahi nss-mdns"),
]


class BootstrapError(Exception):
    pass


def ssh_run(target, command, stdin=None):
    """Run one command on target. Returns (exit code, stdout, stderr). The host key is always checked, whatever the
    operator's ssh config says: the TLS key pinned for the node is read over this connection, so an impostor at the
    address must not get past it."""
    feed = {"input": stdin} if stdin is not None else {"stdin": subprocess.DEVNULL}
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ControlMaster=no", "-o", "ControlPath=none",
                        "-o", "ConnectTimeout=10",
                        "--", target, command],
                       capture_output=True, timeout=600, **feed)
    err = r.stderr.decode(errors="replace")
    if r.returncode == 255 and "Host key verification failed" in err:
        raise BootstrapError("ssh cannot verify the host key of %s: run `ssh %s` once yourself and verify the key it "
                             "shows (known_hosts must hold it); the cluster pins the TLS key it reads over that "
                             "connection, so it will not use an unverified host" % (target, target))
    return r.returncode, r.stdout.decode(errors="replace"), err


def agent_answers(host, port, wait_s=30.0, token=None):
    end = time.monotonic() + wait_s
    while True:
        try:
            req = urllib.request.Request(client.agent_url(host, port, "/v1/health"), headers=client._headers(token=token))
            with client.urlopen(req, 3) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        if time.monotonic() > end:
            return False
        time.sleep(1.0)


def _package_tarball(src_dir):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(os.path.join(src_dir, "omarchy_cluster"), arcname="omarchy_cluster",
                filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return buf.getvalue()


def _resolve_token(token):
    """The token to send: the one given, or this machine's own when none was given. An explicit empty
    token is an error: falling back would send another cluster's secret to a peer the caller meant to
    give a different one."""
    if token is None:
        token = cluster_token.read()
        if not token:
            raise BootstrapError("no cluster token on this machine: run `omarchy-cluster install-agent` "
                                 "here first, or pass --token-file")
    elif not token.strip():
        raise BootstrapError("the token to send is empty; give a --token-file with a token in it")
    return token


def bootstrap(target, ssh=ssh_run, src_dir=None, token=None, port=AGENT_PORT, check=None,
              say=print):
    """Install and start the agent on target (user@host). Returns the node's discover name."""
    token = _resolve_token(token)
    check = check or (lambda host, port: agent_answers(host, port, token=token))  # the new agent wants the token we copy
    src_dir = src_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    host = target.rpartition("@")[2]

    def step(what, command, stdin=None):
        rc, out, err = ssh(target, command, stdin)
        if rc != 0:
            raise BootstrapError("%s failed on %s: %s" % (what, target, (err or out).strip()))
        return out

    os_name = step("OS detection", "uname -s").strip()
    if os_name not in ("Linux", "Darwin"):
        raise BootstrapError("%s runs %s; bootstrap supports Linux and macOS" % (target, os_name))
    rc, out, _ = ssh(target, "python3 -c 'import sys; print(\"ok\" if sys.version_info >= (3, 9) else \"old\")'")
    if rc != 0 or out.strip() != "ok":
        raise BootstrapError("%s needs Python 3.9 or newer as python3" % target)
    say("%s: %s, python3 ok" % (target, os_name))

    if os_name == "Linux":
        _ensure_avahi(target, ssh, say)

    step("package copy", "mkdir -p %s && rm -rf %s/omarchy_cluster && tar -xzf - -C %s"
         % (REMOTE_SRC, REMOTE_SRC, REMOTE_SRC), _package_tarball(src_dir))
    step("token copy", "umask 077 && mkdir -p ~/.config/omarchy-cluster && cat > ~/.config/omarchy-cluster/token",
         token.encode())
    step("install-agent", "PYTHONPATH=%s python3 -m omarchy_cluster.cli install-agent --port %d"
         % (REMOTE_SRC, port))
    pin = _remote_pin(step)  # read over ssh, which has already authenticated the node
    host = discover._resolve(host)  # pins are per address: check and record the one the key was pinned under
    pins.remember({host}, port, pin)
    if not check(host, port):
        raise BootstrapError("the agent on %s:%d did not answer /v1/health within 30 s" % (host, port))
    name = step("hostname", "hostname -s").strip()
    addresses = _addresses(host, port, token)
    pins.remember(addresses, port, pin)
    discover.remember_joined(name, addresses, port)
    discover.forget_left(name)
    say("%s: agent running. It shows up in `omarchy-cluster discover` as %s" % (target, name))
    return name


def pin(target, ssh=ssh_run, port=AGENT_PORT):
    """Pin the key of a node whose agent is already installed (one the hosts file lists, or one joined before
    agents spoke TLS). The key is read over ssh. A node whose agent predates TLS needs `bootstrap` again."""
    def step(what, command):
        rc, out, err = ssh(target, command)
        if rc != 0:
            raise BootstrapError("%s failed on %s: %s (an agent installed before TLS needs `omarchy-cluster bootstrap`)"
                                 % (what, target, (err or out).strip()))
        return out
    address = discover._resolve(target.rpartition("@")[2])
    pins.remember({address}, port, _remote_pin(step))
    return address


def _remote_pin(step):
    """The SHA-256 of the certificate the node's agent presents, from its certificate file."""
    pem = step("certificate read", "cat ~/.config/omarchy-cluster/%s" % tls.CERT_NAME)
    return tls.fingerprint_der(ssl.PEM_cert_to_DER_cert(pem))


def _addresses(host, port, token):
    """Every address this machine will trust for the node: the one it just reached over ssh, plus the
    interface addresses the node reports (asked with the token, on the ssh-authenticated address)."""
    addrs = {discover._resolve(host)}
    try:
        facts = client.fetch_facts(host, port, token=token)
        addrs.update(a["ip"] for ifc in facts.get("interfaces", []) for a in ifc.get("ips", []))
    except Exception:  # noqa: BLE001 - the ssh address alone is still a valid record
        pass
    return addrs


def _ensure_avahi(target, ssh, say):
    if ssh(target, "command -v avahi-daemon >/dev/null || test -x /usr/sbin/avahi-daemon")[0] == 0:
        return
    for manager, install in AVAHI_INSTALL:
        if ssh(target, "command -v %s >/dev/null" % manager)[0] == 0:
            break
    else:
        raise BootstrapError("%s has no Avahi and no known package manager; install avahi and enable "
                             "avahi-daemon there, then run bootstrap again" % target)
    enable = "systemctl enable --now avahi-daemon"
    for command in (install, enable):
        rc, out, err = ssh(target, "sudo -n " + command)
        if rc != 0:
            raise BootstrapError("%s has no Avahi and passwordless sudo failed (%s). On %s run: "
                                 "sudo %s && sudo %s, then run bootstrap again"
                                 % (target, (err or out).strip(), target, install, enable))
    say("%s: installed avahi" % target)


def authorize_join(target, ssh=ssh_run, mode="approved-key"):
    """The one function behind both peer-approval paths (DH-P7-01 step 3).

    approved-key (default): this machine's ssh key was approved once on the peer, so
    BatchMode ssh already works. one-time-code: the peer types a code; not built yet,
    that is omarchy-cluster issue 64."""
    if mode != "approved-key":
        raise BootstrapError("join --auth %s is not implemented yet; the secure plug-and-play "
                             "join is omarchy-cluster issue 64" % mode)
    rc, out, err = ssh(target, "true")
    if rc != 0:
        raise BootstrapError("ssh to %s failed (%s). Approve this machine's key once on the peer: "
                             "ssh-copy-id %s, then run join again"
                             % (target, (err or out).strip(), target))


def _is_omarchy(target, ssh):
    """(ok, reason): reason names why the peer is not Omarchy Linux (DH-P7-01 step 2)."""
    rc, out, err = ssh(target, "uname -s")
    if rc != 0:
        return False, "ssh failed: %s" % (err or out).strip()
    os_name = out.strip()
    if os_name != "Linux":
        return False, "runs %s, not Omarchy Linux" % (os_name or "an unknown OS")
    rc, out, _ = ssh(target, "cat /etc/os-release 2>/dev/null || true")
    fields = {}
    for line in out.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip().strip('"')
    if fields.get("ID", "").lower() == "omarchy" or "omarchy" in fields.get("NAME", "").lower():
        return True, fields.get("PRETTY_NAME", "Omarchy Linux")
    # An installed Omarchy keeps its Arch os-release (the lab M1 reports ID=archarm); its own script says what it is.
    rc, out, _ = ssh(target, "omarchy-version")
    if rc == 0 and out.strip():
        return True, "Omarchy %s" % out.strip()
    return False, "not Omarchy Linux: %s" % (fields.get("PRETTY_NAME")
                                             or "/etc/os-release is missing or has no Omarchy id")


def _resolve(candidate, user, find, left=None):
    """The ssh target for a candidate: user@host as given, or a discover name looked up. `left` (leave
    only) is the record of peers already left: a stopped agent no longer advertises, so a repeat
    `leave NAME` finds its target there. A name that was never joined is still refused."""
    if "@" in candidate:
        return candidate
    nodes = find()
    node = nodes.get(candidate)
    if node is None and candidate in (left or {}):
        host = left[candidate].rpartition("@")[2]
        return "%s@%s" % (user or left[candidate].rpartition("@")[0] or getpass.getuser(), host)
    if node is None:
        raise BootstrapError("no candidate %r; `omarchy-cluster discover` lists: %s"
                             % (candidate, ", ".join(sorted(nodes)) or "(none)"))
    where = node.get("ip") or node.get("host")
    if not where:
        raise BootstrapError("%s has no ip or host in discover; pass user@host" % candidate)
    return "%s@%s" % (user or getpass.getuser(), where)


def _journal(clock, text):
    path = os.path.expanduser(JOURNAL)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write("%s  %s\n" % (clock().strftime("%Y-%m-%d %H:%M:%S"), text))


def join(candidate, user=None, ssh=ssh_run, src_dir=None, token=None, port=AGENT_PORT,
         check=None, say=print, clock=datetime.now, mode="approved-key",
         find=discover.discover_nodes):
    """`omarchy-cluster join <candidate>`: install the agent on the peer with no hand-written
    config and journal the join. Returns the node's discover name.

    candidate is a name from `omarchy-cluster discover`, or an explicit user@host."""
    token = _resolve_token(token)  # before any remote command
    target = _resolve(candidate, user, find)
    authorize_join(target, ssh, mode)
    ok, why = _is_omarchy(target, ssh)
    if not ok:
        raise BootstrapError("%s refused: %s" % (target, why))
    name = bootstrap(target, ssh=ssh, src_dir=src_dir, token=token, port=port, check=check, say=say)
    _journal(clock, "join  %s -> %s (%s)" % (candidate, name, target))
    return name


AGENT_UNIT = "omarchy-cluster-agent.service"
REMOTE_TOKEN = "~/.config/omarchy-cluster/token"
NOT_ENABLED = ("disabled", "static", "indirect", "generated", "transient", "masked", "masked-runtime")  # UnitFileState values with no enablement link


def _stop_agent(target, ssh):
    """Stop and disable the peer's agent unit. A failed inspection is an error: the agent may be running."""
    rc, out, err = ssh(target, "systemctl --user show %s -p LoadState -p ActiveState -p UnitFileState" % AGENT_UNIT)
    if rc != 0:
        raise BootstrapError("%s: could not inspect the agent service (%s); the token was left in place"
                             % (target, (err or out).strip() or "exit %d" % rc))
    state = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    if state.get("LoadState") == "not-found":
        return "agent not installed"
    # Any other UnitFileState (enabled, enabled-runtime, linked, linked-runtime, alias, or one we do not know) gets `disable`.
    if state.get("ActiveState") in ("inactive", "failed") and state.get("UnitFileState") in NOT_ENABLED:
        return "agent already stopped and disabled"
    rc, out, err = ssh(target, "systemctl --user disable --now %s" % AGENT_UNIT)
    if rc != 0:
        raise BootstrapError("%s: could not stop the agent (%s); the token was left in place"
                             % (target, (err or out).strip()))
    return "agent stopped and disabled"


def _remove_token(target, ssh):
    rc, out, err = ssh(target, "test -e %s && echo present || echo absent" % REMOTE_TOKEN)
    if rc != 0 or out.strip() not in ("present", "absent"):
        raise BootstrapError("%s: could not check the token (%s)" % (target, (err or out).strip() or "exit %d" % rc))
    if out.strip() == "absent":
        return "token already gone"
    rc, out, err = ssh(target, "rm -f %s" % REMOTE_TOKEN)
    if rc != 0:
        raise BootstrapError("%s: could not remove the token (%s)" % (target, (err or out).strip()))
    return "token removed"


def _peer_name(target, ssh):
    """The name this machine recorded the peer under when it joined it: `hostname -s` there."""
    rc, out, err = ssh(target, "hostname -s")
    if rc != 0 or not out.strip():
        raise BootstrapError("%s: could not read the peer's hostname (%s)" % (target, (err or out).strip() or "exit %d" % rc))
    return out.strip()


def leave(candidate, user=None, ssh=ssh_run, say=print, clock=datetime.now, find=discover.discover_nodes):
    """`omarchy-cluster leave <candidate>`: undo a join on the peer. Stops and disables the agent, removes
    the peer's copy of the cluster token, drops this machine's trust record for it, journals the leave. Running it again changes nothing and says so.

    The token on this machine is the pool's token and stays: other nodes still use it."""
    target = _resolve(candidate, user, find, discover.read_left())
    ok, why = _is_omarchy(target, ssh)
    if not ok:
        raise BootstrapError("%s refused: %s" % (target, why))
    name = _peer_name(target, ssh)
    agent = _stop_agent(target, ssh)
    token = _remove_token(target, ssh)
    discover.forget_joined(name)  # last: discovery must not trust a peer that still holds a live token
    discover.remember_left(name, target)  # so the same `leave` can run again after the agent stops advertising
    say("%s: %s, %s" % (target, agent, token))
    _journal(clock, "leave  %s (%s): %s, %s" % (candidate, target, agent, token))
