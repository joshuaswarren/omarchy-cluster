"""The cluster token: one shared secret in ~/.config/omarchy-cluster/token."""
import os


def read():
    try:
        with open(os.path.expanduser("~/.config/omarchy-cluster/token")) as f:
            return f.read().strip()
    except OSError:
        return None
