import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import planner

CONFIG = {"num_hidden_layers": 48, "hidden_size": 2048, "num_key_value_heads": 4, "num_attention_heads": 32}
LISTING = {"siblings": [
    {"rfilename": "config.json", "size": 900},
    {"rfilename": "model-00001-of-00002.safetensors", "size": 10_000},
    {"rfilename": "model-00002-of-00002.safetensors", "size": 7_000},
    {"rfilename": "tokenizer.json", "size": 5_000_000},
]}


def hub(files):
    """A fake hub: url -> bytes; anything else is unreachable like an offline host."""
    def fetch(url):
        if url not in files:
            raise OSError("unreachable: " + url)
        return files[url]
    return fetch


REF = "example/coder"
GOOD = hub({
    "https://huggingface.co/example/coder/resolve/main/config.json": json.dumps(CONFIG).encode(),
    "https://huggingface.co/api/models/example/coder?blobs=true": json.dumps(LISTING).encode(),
})


def test_a_model_not_on_this_host_is_sized_from_the_hub_without_its_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    info = planner.model_info_or_hub(REF, GOOD)
    assert (info["layers"], info["hidden"], info["kv_heads"], info["head_dim"]) == (48, 2048, 4, 64)
    assert info["weights_bytes"] == 17_000  # only the .safetensors files count
    assert info["ref"] == REF


def test_a_model_on_this_host_is_read_locally_and_the_hub_is_not_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    local = tmp_path / "m"
    local.mkdir()
    (local / "config.json").write_text(json.dumps(CONFIG))
    (local / "w.safetensors").write_bytes(b"x" * 123)
    info = planner.model_info_or_hub(str(local), hub({}))
    assert info["weights_bytes"] == 123


def test_an_unreachable_hub_is_the_old_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert planner.model_info_or_hub(REF, hub({})) is None


def test_a_hub_answer_that_is_not_a_model_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    bad = hub({"https://huggingface.co/example/coder/resolve/main/config.json": b"<html>",
               "https://huggingface.co/api/models/example/coder?blobs=true": b"{}"})
    assert planner.model_info_or_hub(REF, bad) is None
    no_weights = hub({"https://huggingface.co/example/coder/resolve/main/config.json": json.dumps(CONFIG).encode(),
                      "https://huggingface.co/api/models/example/coder?blobs=true": json.dumps({"siblings": []}).encode()})
    assert planner.model_info_or_hub(REF, no_weights) is None
