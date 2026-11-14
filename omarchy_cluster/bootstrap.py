"""`omarchy-cluster bootstrap user@host`: turn a bare machine into a cluster node over ssh.

Steps, in order: detect the OS, check Python, install Avahi on Linux, copy this package,
copy the cluster token, run `install-agent` there, wait for the agent to answer, and print
the name the node shows up under in `omarchy-cluster discover`.
"""
from __future__ import annotations

import io
import os
import ssl
import subprocess
import tarfile
import time
import urllib.request

from . import client, discover, pins, tls
from . import token as cluster_token

AGENT_PORT = 8025
REMOTE_SRC = "~/.local/share/omarchy-cluster/src"
# (probe command name, install command) in the order tried; each install runs under sudo -n.
AVAHI_INSTALL = [
    ("pacman", "pacman -S --noconfirm --needed avahi nss-mdns"),
    ("apt-get", "apt-get install -y avahi-daemon avahi-utils"),
    ("dnf", "dnf install -y avahi nss-mdns"),
]


class BootstrapError(Exception):
    pass


def ssh_run(target, command, stdin=None):
    """Run one command on target. Returns (exit code, stdout, stderr)."""
    feed = {"input": stdin} if stdin is not None else {"stdin": subprocess.DEVNULL}
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--", target, command],
                       capture_output=True, timeout=600, **feed)
    return r.returncode, r.stdout.decode(errors="replace"), r.stderr.decode(errors="replace")


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


def bootstrap(target, ssh=ssh_run, src_dir=None, token=None, port=AGENT_PORT, check=None,
              say=print):
    """Install and start the agent on target (user@host). Returns the node's discover name."""
    token = token or cluster_token.read()
    check = check or (lambda host, port: agent_answers(host, port, token=token))  # the new agent wants the token we copy
    if not token:
        raise BootstrapError("no cluster token on this machine: run `omarchy-cluster install-agent` "
                             "here first, or pass --token-file")
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
    pins.remember({discover._resolve(host)}, port, pin)
    if not check(host, port):
        raise BootstrapError("the agent on %s:%d did not answer /v1/health within 30 s" % (host, port))
    name = step("hostname", "hostname -s").strip()
    addresses = _addresses(host, port, token)
    pins.remember(addresses, port, pin)
    discover.remember_joined(name, addresses, port)
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
