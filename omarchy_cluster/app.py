"""Native GTK model browser and chat client for the local cluster gateway."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import token as cluster_token

DEFAULT_GATEWAY = "http://127.0.0.1:8020"


def _request(gateway, path, token=None, body=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(gateway.rstrip("/") + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
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


def pull_model(gateway, repo, token=None):
    return _request(gateway, "/models/pull", token, {"repo": repo})


def chat_turn(gateway, model, messages, token=None, max_tokens=256):
    result = _request(gateway, "/v1/chat/completions", token,
                      {"model": model, "messages": messages, "max_tokens": max_tokens})
    return result["choices"][0]["message"]["content"]


def main(_args=None):
    try:
        import gi
        gi.require_version("Gtk", "4.0")
        from gi.repository import GLib, Gtk
    except (ImportError, ValueError) as error:
        raise SystemExit("native app needs GTK 4 and PyGObject (python-gobject)") from error

    class ClusterApp(Gtk.Application):
        def __init__(self):
            super().__init__(application_id="com.omarchy.Cluster")
            self.pool = ThreadPoolExecutor(max_workers=1)
            self.models = []
            self.messages = []
            self.token = cluster_token.read()

        def do_activate(self):
            window = Gtk.ApplicationWindow(application=self, title="Omarchy Cluster")
            window.set_default_size(760, 640)
            outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            outer.set_margin_top(12)
            outer.set_margin_bottom(12)
            outer.set_margin_start(12)
            outer.set_margin_end(12)
            window.set_child(outer)

            gateway_row = Gtk.Box(spacing=8)
            outer.append(gateway_row)
            gateway_row.append(Gtk.Label(label="Gateway"))
            self.gateway = Gtk.Entry(text=DEFAULT_GATEWAY, hexpand=True)
            gateway_row.append(self.gateway)
            refresh = Gtk.Button(label="Refresh models")
            refresh.connect("clicked", lambda *_: self.refresh_models())
            gateway_row.append(refresh)

            model_row = Gtk.Box(spacing=8)
            outer.append(model_row)
            self.model_list = Gtk.StringList.new([])
            self.selector = Gtk.DropDown(model=self.model_list, hexpand=True)
            model_row.append(self.selector)
            self.repo_entry = Gtk.Entry(placeholder_text="Hugging Face repo, e.g. mlx-community/model", hexpand=True)
            model_row.append(self.repo_entry)
            pull = Gtk.Button(label="Download")
            pull.connect("clicked", lambda *_: self.pull())
            model_row.append(pull)

            self.status = Gtk.Label(label="Choose a gateway and refresh models", xalign=0)
            outer.append(self.status)
            scroll = Gtk.ScrolledWindow(vexpand=True)
            outer.append(scroll)
            self.transcript = Gtk.TextView(editable=False, wrap_mode=Gtk.WrapMode.WORD_CHAR)
            scroll.set_child(self.transcript)
            send_row = Gtk.Box(spacing=8)
            outer.append(send_row)
            self.prompt = Gtk.Entry(placeholder_text="Message", hexpand=True)
            self.prompt.connect("activate", lambda *_: self.send())
            send_row.append(self.prompt)
            send = Gtk.Button(label="Send")
            send.connect("clicked", lambda *_: self.send())
            send_row.append(send)
            window.present()
            self.refresh_models()

        def run_async(self, operation, done):
            self.status.set_text("Working…")
            future = self.pool.submit(operation)

            def finished(future):
                try:
                    result = future.result()
                    GLib.idle_add(done, result, None)
                except (RuntimeError, ValueError, KeyError, IndexError) as error:
                    GLib.idle_add(done, None, str(error))
            future.add_done_callback(finished)

        def refresh_models(self):
            self.run_async(lambda: list_models(self.gateway.get_text(), self.token), self.show_models)

        def show_models(self, result, error):
            if error:
                self.status.set_text(error)
                return GLib.SOURCE_REMOVE
            self.models = result
            self.model_list.splice(0, self.model_list.get_n_items(), [m["repo"] for m in result])
            self.status.set_text(f"{len(result)} cached models")
            return GLib.SOURCE_REMOVE

        def selected_model(self):
            index = self.selector.get_selected()
            return self.models[index]["repo"] if index < len(self.models) else None

        def pull(self):
            repo = self.repo_entry.get_text().strip()
            if not repo:
                self.status.set_text("Enter a model repository to download")
                return
            self.run_async(lambda: pull_model(self.gateway.get_text(), repo, self.token), self.pulled)

        def pulled(self, result, error):
            self.status.set_text(error or "Downloaded " + result["repo"])
            if not error:
                self.refresh_models()
            return GLib.SOURCE_REMOVE

        def append(self, text):
            buffer = self.transcript.get_buffer()
            buffer.insert(buffer.get_end_iter(), text + "\n\n")

        def send(self):
            model = self.selected_model()
            text = self.prompt.get_text().strip()
            if not model or not text:
                self.status.set_text("Select a downloaded model and enter a message")
                return
            self.prompt.set_text("")
            self.messages.append({"role": "user", "content": text})
            self.append("You: " + text)
            self.run_async(lambda: chat_turn(self.gateway.get_text(), model, self.messages[:], self.token), self.replied)

        def replied(self, text, error):
            if error:
                self.status.set_text(error)
            else:
                self.messages.append({"role": "assistant", "content": text})
                self.append("Assistant: " + text)
                self.status.set_text("Ready")
            return GLib.SOURCE_REMOVE

    ClusterApp().run(None)
