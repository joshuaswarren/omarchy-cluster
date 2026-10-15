"""KV reservation for hybrid checkpoints: only attention layers grow a per-token cache."""
import json

from omarchy_cluster import planner

GB = 1_000_000_000


def qwen35_text(layers=64, interval=4, with_types=True):
    """The language-model config of a qwen3_5 checkpoint: one full-attention layer after every
    (interval - 1) linear-attention layers (Qwen3.8-27B: 64 layers, 16 full attention)."""
    cfg = {"num_hidden_layers": layers, "hidden_size": 5120, "num_attention_heads": 24,
           "num_key_value_heads": 4, "head_dim": 256, "full_attention_interval": interval}
    if with_types:
        cfg["layer_types"] = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                              for i in range(layers)]
    return cfg


def write_model(tmp_path, cfg, nested):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": cfg} if nested else cfg))
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 1024)
    return planner.model_info(str(tmp_path))


def test_layer_types_count_only_the_attention_layers():
    assert planner.attention_layers(qwen35_text()) == 16


def test_full_attention_interval_stands_in_when_layer_types_are_absent():
    assert planner.attention_layers(qwen35_text(layers=48, with_types=False)) == 12


def test_dense_and_unclear_configs_count_every_layer():
    dense = {"num_hidden_layers": 28, "num_key_value_heads": 8, "head_dim": 128}
    assert planner.attention_layers(dense) == 28
    assert planner.attention_layers(dict(qwen35_text(), full_attention_interval=1, layer_types=None)) == 64
    short = dict(qwen35_text(), layer_types=["linear_attention"] * 10)  # length disagrees with num_hidden_layers
    assert planner.attention_layers(dict(short, full_attention_interval=0)) == 64
    assert planner.attention_layers(dict(qwen35_text(), layer_types=["linear_attention"] * 64)) == 64  # no attention layer named: reserve all
    assert planner.attention_layers({"num_hidden_layers": 0}) == 0


def test_model_info_reads_a_nested_multimodal_text_config(tmp_path):
    info = write_model(tmp_path, qwen35_text(), nested=True)
    assert (info["layers"], info["kv_layers"]) == (64, 16)
    assert planner.kv_bytes_per_token(info) == 2 * 16 * 4 * 256 * 2  # not 64 layers: 65536, not 262144


def test_model_info_leaves_a_dense_model_unchanged(tmp_path):
    dense = {"num_hidden_layers": 28, "hidden_size": 2048, "num_attention_heads": 16,
             "num_key_value_heads": 8, "head_dim": 128}
    info = write_model(tmp_path, dense, nested=False)
    assert info["kv_layers"] == 28
    assert planner.kv_bytes_per_token(info) == 2 * 28 * 8 * 128 * 2


def test_hybrid_reservation_lets_more_layers_fit_at_a_long_context(tmp_path):
    info = write_model(tmp_path, qwen35_text(), nested=True)
    info["weights_bytes"] = 28 * GB  # 27B at 8 bit
    all_layers = dict(info, kv_layers=None, kv_pattern=None)  # what the planner reserved before
    free = 40 * GB
    ctx = 131072
    assert planner.max_layers(free, info, ctx) > planner.max_layers(free, all_layers, ctx)


def test_plan_table_says_how_many_layers_hold_a_cache():
    plan = {"mode": "replica", "model": "m", "layers": 64, "kv_layers": 16, "kv_bytes_per_token": 65536,
            "stages": [{"node": "a", "max_layers": 64}], "boundaries": []}
    assert "(64 layers, 16 with KV, KV 64.0 KB/token)" in planner.plan_table(plan)
    plain = dict(plan, kv_layers=64)
    assert "(64 layers, KV 64.0 KB/token)" in planner.plan_table(plain)


def test_worst_slice_of_attention_layers():
    pattern = planner.attention_pattern(qwen35_text())
    assert [planner.worst_attention_layers(pattern, n) for n in (1, 3, 4, 5, 8, 64)] == [1, 1, 1, 2, 2, 16]


def test_a_stage_is_charged_for_its_most_attention_heavy_slice():
    # 8 layers, an attention layer every fourth; 1 KV head of dim 1 -> 4 bytes per attention layer per token
    info = {"layers": 8, "kv_heads": 1, "head_dim": 1, "weights_bytes": 0, "kv_layers": 2,
            "kv_pattern": [True, False, False, False] * 2}
    free = 3000 / 0.9  # budget of 3000 bytes at 1000 tokens: 0.75 of one attention layer (4000 bytes)
    assert planner.max_layers(free, dict(info, kv_pattern=None), 1000) == 3  # the model-wide average: 0.25 layer each
    assert planner.max_layers(free, info, 1000) == 0  # but one layer can be the attention layer
    assert planner.max_layers(4000 / 0.9, info, 1000) == 4  # four layers hold one attention layer, five can hold two


def muse_text(layers=8, window=2048):
    """A muse_spark-shaped text config: sliding-window layers with a full-attention layer after every third."""
    return {"num_hidden_layers": layers, "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 1,
            "sliding_window": window,
            "layer_types": ["full_attention" if i % 4 == 1 else "sliding_attention" for i in range(layers)]}


def test_window_pattern_marks_sliding_layers_only_when_both_keys_are_present():
    flags, window = planner.window_pattern(muse_text())
    assert (flags, window) == ([True, False, True, True] * 2, 2048)
    assert planner.window_pattern(dict(muse_text(), sliding_window=None)) == ([False] * 8, 0)
    assert planner.window_pattern(dict(muse_text(), layer_types=None)) == ([False] * 8, 0)
    assert planner.window_pattern(dict(muse_text(), layer_types=["sliding_attention"] * 3)) == ([False] * 8, 0)


def test_a_window_layer_is_charged_its_window_not_the_context(tmp_path):
    info = write_model(tmp_path, muse_text(), nested=True)
    assert info["kv_window"] == 2048 and info["kv_layers"] == 8  # window layers still grow a cache
    # 4 bytes per cached token per layer (K and V, 1 head of dim 1, bf16): 8 layers, 2 full, 6 window
    ctx = 32768
    full_charge = 8 * 4 * ctx  # what the planner charged when every attention layer held the whole context
    window_charge = 4 * (2 * ctx + 6 * (2048 + planner.PREFILL_STEP - 1))  # a window layer peaks at window + prefill step - 1
    free = (window_charge + 1024) / 0.9  # plus the helper's 1 KB of weights
    assert planner.max_layers(free, info, ctx) == 8
    assert planner.max_layers(free, dict(info, kv_window_layers=[False] * 8), ctx) < 8  # the old charge: full context everywhere
    assert full_charge > window_charge
