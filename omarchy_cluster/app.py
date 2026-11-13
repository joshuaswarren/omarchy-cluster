"""Native GTK model browser and chat client for the local cluster gateway."""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

from . import token as cluster_token

DEFAULT_GATEWAY = "http://127.0.0.1:8020"
SYSTEM_PYTHON = "/usr/bin/python3"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _request(gateway, path, token=None, body=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(gateway.rstrip("/") + path, data=data, headers=headers)
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=600) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read()).get("error", "")
        except (ValueError, AttributeError):
            detail = ""
        raise RuntimeError(f"gateway {gateway} answered {error.code} {detail}") from error
    except OSError as error:
        raise RuntimeError(f"cannot reach the gateway at {gateway}: {error}") from error


def list_models(gateway=DEFAULT_GATEWAY, token=None):
    result = _request(gateway, "/models", token)
    return result.get("data", [])


def fetch_catalog(gateway=DEFAULT_GATEWAY, token=None):
    cached = list_models(gateway, token)
    active = _request(gateway, "/v1/models", token).get("data", [])
    return cached, [model["id"] for model in active]


def pull_model(gateway, repo, token=None):
    return _request(gateway, "/models/pull", token, {"repo": repo})


def chat_turn(gateway, model, messages, token=None, max_tokens=256):
    result = _request(gateway, "/v1/chat/completions", token,
                      {"model": model, "messages": messages, "max_tokens": max_tokens})
    content = result["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("gateway chat reply content must be a string")
    return content


def run_operation(operation):
    try:
        return operation(), None
    except Exception as error:  # noqa: BLE001 - all task failures must reach the UI.
        return None, str(error)


class AppCore:
    """GTK-free model selection and single-flight conversation state."""

    def __init__(self, gateway=DEFAULT_GATEWAY, token=None):
        self.gateway = gateway
        self.token = token
        self.cached_models = []
        self.active_models = []
        self.history = []
        self.busy = False

    def set_catalog(self, cached, active):
        self.cached_models = cached
        self.active_models = list(active)

    def begin_turn(self, model, prompt):
        if self.busy:
            raise RuntimeError("a reply is pending")
        if model not in self.active_models:
            raise ValueError("model is not served by this gateway; select an active gateway model")
        user = {"role": "user", "content": prompt}
        self.history.append(user)
        messages = [dict(message) for message in self.history]
        gateway, token = self.gateway, self.token
        self.busy = True
        return lambda: chat_turn(gateway, model, messages, token)

    def complete_turn(self, reply):
        if not self.busy:
            raise RuntimeError("no reply is pending")
        self.history.append({"role": "assistant", "content": reply})
        self.busy = False

    def fail_turn(self):
        if self.busy:
            self.history.pop()
            self.busy = False


def gtk_fallback_command():
    """Re-enter the installed package with distro Python, where PyGObject lives."""
    if not os.path.isfile(SYSTEM_PYTHON) or sys.prefix == sys.base_prefix:
        return None, None
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (package_root, env.get("PYTHONPATH"))))
    return [SYSTEM_PYTHON, "-m", "omarchy_cluster.app"], env


def install_launcher():
    source = Path(__file__).resolve().parent / "com.omarchy.Cluster.desktop"
    text = source.read_text()
    command = shutil.which(sys.argv[0]) or os.path.abspath(sys.argv[0])
    escaped = os.path.realpath(command).replace("\\", "\\\\").replace('"', '\\"')
    text = text.replace("Exec=omarchy-cluster app", f'Exec="{escaped}" app')
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    destination = data_home / "applications" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    return destination


def load_gtk_modules():
    gi = importlib.import_module("gi")
    gi.require_version("Gtk", "4.0")
    return (importlib.import_module("gi.repository.Gtk"),
            importlib.import_module("gi.repository.GLib"))


def main(_args=None):
    try:
        Gtk, GLib = load_gtk_modules()
    except (ImportError, ValueError) as error:
        command, env = gtk_fallback_command()
        if command:
            os.execve(command[0], command, env or os.environ.copy())
        raise SystemExit("native app needs GTK 4 and PyGObject (install python-gobject)") from error

    class ClusterApp(Gtk.Application):
        def __init__(self):
            super().__init__(application_id="com.omarchy.Cluster")
            self.core = AppCore(token=cluster_token.read())
            self.busy = False
            self.closing = False
            self.active_models = []

        def do_activate(self):
            window = Gtk.ApplicationWindow(application=self, title="Omarchy Cluster")
            window.set_default_size(760, 640)
            window.connect("close-request", self.close)
            outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            for edge in ("top", "bottom", "start", "end"):
                getattr(outer, f"set_margin_{edge}")(12)
            window.set_child(outer)

            gateway_row = Gtk.Box(spacing=8)
            outer.append(gateway_row)
            gateway_row.append(Gtk.Label(label="Gateway"))
            self.gateway = Gtk.Entry(text=DEFAULT_GATEWAY, hexpand=True)
            gateway_row.append(self.gateway)
            self.refresh_button = Gtk.Button(label="Refresh models")
            self.refresh_button.connect("clicked", lambda *_: self.refresh_models())
            gateway_row.append(self.refresh_button)

            model_row = Gtk.Box(spacing=8)
            outer.append(model_row)
            model_row.append(Gtk.Label(label="Active model"))
            self.model_list = Gtk.StringList.new([])
            self.selector = Gtk.DropDown(model=self.model_list, hexpand=True)
            model_row.append(self.selector)
            self.repo_entry = Gtk.Entry(placeholder_text="Hugging Face repo, e.g. org/model", hexpand=True)
            model_row.append(self.repo_entry)
            self.pull_button = Gtk.Button(label="Download")
            self.pull_button.connect("clicked", lambda *_: self.pull())
            model_row.append(self.pull_button)

            self.status = Gtk.Label(label="Choose a gateway and refresh models", xalign=0)
            outer.append(self.status)
            self.cached_list = Gtk.Label(label="", xalign=0)
            self.cached_list.set_wrap(True)
            outer.append(self.cached_list)
            scroll = Gtk.ScrolledWindow(vexpand=True)
            outer.append(scroll)
            self.transcript = Gtk.TextView(editable=False, wrap_mode=Gtk.WrapMode.WORD_CHAR)
            scroll.set_child(self.transcript)
            send_row = Gtk.Box(spacing=8)
            outer.append(send_row)
            self.prompt = Gtk.Entry(placeholder_text="Message", hexpand=True)
            self.prompt.connect("activate", lambda *_: self.send())
            send_row.append(self.prompt)
            self.send_button = Gtk.Button(label="Send")
            self.send_button.connect("clicked", lambda *_: self.send())
            send_row.append(self.send_button)
            window.present()
            self.refresh_models()

        def close(self, _window):
            self.closing = True
            return False

        def set_busy(self, busy):
            self.busy = busy
            for widget in (self.gateway, self.refresh_button, self.selector, self.repo_entry,
                           self.pull_button, self.prompt, self.send_button):
                widget.set_sensitive(not busy)
            if busy:
                self.status.set_text("Working…")

        def run_async(self, operation, done):
            if self.busy:
                return
            self.set_busy(True)

            def work():
                result, error = run_operation(operation)
                if not self.closing:
                    GLib.idle_add(self.finish, done, result, error)

            threading.Thread(target=work, daemon=True).start()

        def finish(self, done, result, error):
            self.set_busy(False)
            try:
                done(result, error)
            except Exception as failure:  # noqa: BLE001 - callback failures must clear the pending state.
                self.status.set_text(str(failure))
            return GLib.SOURCE_REMOVE

        def refresh_models(self):
            gateway = self.gateway.get_text().strip()
            self.core.gateway = gateway
            self.run_async(lambda: fetch_catalog(gateway, self.core.token), self.show_models)

        def show_models(self, result, error):
            if error:
                self.status.set_text(error)
                return
            cached, active = result
            self.core.set_catalog(cached, active)
            self.active_models = active
            self.model_list.splice(0, self.model_list.get_n_items(), active)
            names = ", ".join(model["repo"] for model in cached) or "none"
            self.cached_list.set_text(f"Cached models: {names}")
            self.status.set_text(f"{len(active)} active gateway models")

        def selected_model(self):
            index = self.selector.get_selected()
            return self.active_models[index] if index < len(self.active_models) else None

        def pull(self):
            gateway = self.gateway.get_text().strip()
            repo = self.repo_entry.get_text().strip()
            if not repo:
                self.status.set_text("Enter a model repository to download")
                return
            self.run_async(lambda: pull_model(gateway, repo, self.core.token), self.pulled)

        def pulled(self, result, error):
            if error:
                self.status.set_text(error)
            else:
                self.status.set_text(f"Downloaded {result['repo']}; start serve with it before selecting it")
                self.refresh_models()

        def append(self, text):
            buffer = self.transcript.get_buffer()
            buffer.insert(buffer.get_end_iter(), text + "\n\n")

        def send(self):
            if self.busy:
                return
            model = self.selected_model()
            text = self.prompt.get_text().strip()
            if not model or not text:
                self.status.set_text("Select an active gateway model and enter a message")
                return
            self.core.gateway = self.gateway.get_text().strip()
            try:
                turn = self.core.begin_turn(model, text)
            except (RuntimeError, ValueError) as error:
                self.status.set_text(str(error))
                return
            self.prompt.set_text("")
            self.append("You: " + text)
            self.run_async(turn, self.replied)

        def replied(self, text, error):
            if error:
                self.core.fail_turn()
                self.status.set_text(error)
            else:
                self.core.complete_turn(text)
                self.append("Assistant: " + text)
                self.status.set_text("Ready")

    ClusterApp().run(None)


if __name__ == "__main__":
    main()
