"""Issue 96 review: the config-only timing copy must time the checkpoint's quantized operators, not float32 ones."""
import json
import os
import sys

import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omarchy_cluster import rank
mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")


def _cfg_dir(tmp_path, quantization):
    d = tmp_path / "cfg"
    d.mkdir()
    cfg = {"model_type": "qwen2", "hidden_size": 64, "intermediate_size": 128,
           "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
           "vocab_size": 96, "rms_norm_eps": 1e-06, "quantization": quantization}
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_the_config_only_copy_times_quantized_operators_not_float32(tmp_path):
    """mlx-lm's loader predicate refuses the config's default quantization for a module with no `.scales` in the
    (empty) weights, so without help a config-only load stays float32 and times the wrong operators and memory."""
    from mlx_lm.utils import load_model
    d = _cfg_dir(tmp_path, {"group_size": 64, "bits": 4,
                            "model.layers.0.mlp.gate_proj": {"bits": 8, "group_size": 64}})
    model, _ = load_model(d, lazy=True, strict=False)
    rank._quantize_random_like_checkpoint(model, d)
    q_proj = model.model.layers[0].self_attn.q_proj
    gate = model.model.layers[0].mlp.gate_proj
    assert type(q_proj).__name__ == "QuantizedLinear" and q_proj.bits == 4
    assert type(gate).__name__ == "QuantizedLinear" and gate.bits == 8
