import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.generate import generate_step
from mlx_lm.models import llama

from omarchy_cluster import rank


def _decode(model, sampling, steps=24):
    pick = rank._sampler(sampling)
    prompt = mx.array([1, 2, 3])
    out = []
    for token, _ in generate_step(prompt, model, max_tokens=steps, sampler=pick):
        out.append(int(token))
    return out


def test_seeded_decode_repeats_and_differs_across_seeds():
    mx.random.seed(0)
    args = llama.ModelArgs(model_type="llama", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
                           num_attention_heads=2, rms_norm_eps=1e-5, vocab_size=64, num_key_value_heads=2)
    model = llama.Model(args)
    mx.eval(model.parameters())
    a = _decode(model, {"temperature": 1.5, "seed": 5})
    assert a == _decode(model, {"temperature": 1.5, "seed": 5})
    assert a != _decode(model, {"temperature": 1.5, "seed": 6})
    assert a != _decode(model, None)
