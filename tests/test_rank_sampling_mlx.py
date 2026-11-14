"""Sampling tests that need MLX; the transport and wiring tests live in test_rank_sampling.py and run everywhere."""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import rank

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")


def _tokens(sampling, steps=64, vocab=32):
    pick = rank._sampler(sampling)
    logprobs = mx.log(mx.full((1, vocab), 1.0 / vocab))
    return [int(pick(logprobs).item()) for _ in range(steps)]


def test_the_same_seed_gives_the_same_tokens_and_another_seed_does_not():
    a = _tokens({"temperature": 1.0, "seed": 5})
    assert a == _tokens({"temperature": 1.0, "seed": 5})
    assert a != _tokens({"temperature": 1.0, "seed": 6})


def test_no_sampling_settings_pick_the_argmax():
    logprobs = mx.log(mx.array([[0.1, 0.6, 0.3]]))
    assert rank._sampler(None) is None
    assert int(rank._sampler(None, share=lambda y: y)(logprobs).item()) == 1


def test_a_tiny_top_p_keeps_only_the_top_token():
    logprobs = mx.log(mx.array([[0.1, 0.6, 0.3]]))
    pick = rank._sampler({"temperature": 1.0, "top_p": 0.01, "seed": 1})
    assert {int(pick(logprobs).item()) for _ in range(20)} == {1}


def test_the_shared_token_is_the_one_rank_0_sampled():
    logprobs = mx.log(mx.full((1, 32), 1.0 / 32))
    shared = []
    pick = rank._sampler({"temperature": 1.0, "seed": 5}, share=lambda y: shared.append(int(y.item())) or y)
    out = [int(pick(logprobs).item()) for _ in range(8)]
    assert shared == out == _tokens({"temperature": 1.0, "seed": 5}, steps=8)


class _FakeModel:
    """Just enough of an mlx-lm model for _share_token: pipeline fields and an lm_head with a vocab."""

    def __init__(self, pipeline_rank, vocab=16):
        self.model = SimpleNamespace(pipeline_size=2, pipeline_rank=pipeline_rank)
        self.lm_head = SimpleNamespace(weight=mx.zeros((vocab, 4)))

    def __contains__(self, name):
        return name == "lm_head"


def test_a_follower_rank_feeds_its_layers_the_token_rank_0_sampled(monkeypatch):
    # The lab run (2 ranks, seed 7) gave garbled text: follower ranks sampled from the one-hot
    # logits too, so their layers got random tokens. A follower must take rank 0's token exactly.
    monkeypatch.setattr(mx.distributed, "all_gather", lambda x, **kw: mx.array([7.0, 7.0]))
    follower = _FakeModel(pipeline_rank=1)
    rank._share_token(follower)
    logprobs = follower.lm_head(mx.zeros((1, 4)))
    logprobs = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)  # what stream_generate hands the sampler
    # None is stream_generate's own argmax: the follower's one-hot logits give back rank 0's token.
    assert rank._sampler({"temperature": 0.8, "top_p": 0.95, "seed": 3}, follow=follower.omarchy_follow) is None
    assert int(mx.argmax(logprobs, axis=-1).item()) == 7


class _TiedModel(_FakeModel):
    """A tied-embedding model (Qwen2.5-1.5B): no lm_head, logits come from embed_tokens.as_linear."""

    def __init__(self, pipeline_rank, vocab=16):
        self.model = SimpleNamespace(pipeline_size=2, pipeline_rank=pipeline_rank,
                                     embed_tokens=SimpleNamespace(weight=mx.zeros((vocab, 4)),
                                                                  as_linear=lambda h: mx.zeros(h.shape[:-1] + (vocab,))))

    def __contains__(self, name):
        return False


def test_a_tied_embedding_follower_also_takes_rank_0s_token(monkeypatch):
    # The second lab run: Qwen2.5-1.5B ties its embeddings, so there was no lm_head to swap and no rank shared
    # a token. Each rank sampled for itself and the chips drifted apart: garbled text from the second token on.
    monkeypatch.setattr(mx.distributed, "all_gather", lambda x, **kw: mx.array([7.0, 7.0]))
    follower = _TiedModel(pipeline_rank=1)
    rank._share_token(follower)
    logits = follower.model.embed_tokens.as_linear(mx.zeros((1, 4)))
    assert int(mx.argmax(logits, axis=-1).item()) == 7
    assert follower.omarchy_follow is True


def test_a_tied_embedding_rank_0_shares_the_token_it_sampled(monkeypatch):
    monkeypatch.setattr(mx.distributed, "all_gather", lambda x, **kw: x)
    lead = _TiedModel(pipeline_rank=0)
    rank._share_token(lead)
    assert int(lead.omarchy_share(mx.array(5)).item()) == 5


def test_a_real_tied_qwen2_follower_outputs_rank_0s_token(monkeypatch):
    from mlx_lm.models import qwen2
    args = qwen2.ModelArgs(model_type="qwen2", hidden_size=8, num_hidden_layers=1, intermediate_size=16,
                           num_attention_heads=2, rms_norm_eps=1e-6, vocab_size=16, num_key_value_heads=2,
                           tie_word_embeddings=True)
    model = qwen2.Model(args)
    model.model.pipeline_size, model.model.pipeline_rank = 2, 1  # the call path ignores them; _share_token reads them
    monkeypatch.setattr(mx.distributed, "all_gather", lambda x, **kw: mx.array([7.0, 7.0]))
    rank._share_token(model)
    logits = model.model.embed_tokens.as_linear(mx.zeros((1, 1, 8)))
    assert int(mx.argmax(logits, axis=-1).item()) == 7
