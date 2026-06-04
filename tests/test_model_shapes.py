"""Unit shape tests for src/model.py.

CPU-only, small config, < 1 s end-to-end. Catches regressions in the
network building blocks when the architecture is refactored.

Run with either:
    python tests/test_model_shapes.py    # standalone runner, no pytest needed
    pytest tests/                        # also works if pytest is installed
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interfaces import NUM_VARIABLES  # noqa: E402
from src.model import (  # noqa: E402
    CoupledVariableDecoder,
    PatchEmbedding,
    RolloutEmbedding,
    SpatialAttentionBlock,
    TemporalMixtureOfExperts,
    WalkerNet,
)


SMALL_CFG = {
    "data": {"L": 3, "H": 20, "W": 40},
    "model": {
        "patch_size": 4,
        "d_model": 16,
        "nhead": 2,
        "dim_ff": 32,
        "num_layers": 2,
        "patch_fusion_heads": 4,
        "patch_fusion_layers": 1,
        "patch_fusion_dim_ff": 64,
        "num_experts": 4,
        "top_k": 2,
        "decoder_hidden": 8,
        "dropout": 0.0,
        "max_rollout_steps": 8,
    },
}


def _expect_value_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError from {fn!r}")


# --- PatchEmbedding ---------------------------------------------------

def test_patch_embed_shape():
    pe = PatchEmbedding(
        input_length=3,
        num_variables=NUM_VARIABLES,
        patch_size=4,
        d_model=16,
        grid_shape=(20, 40),
        fusion_heads=4,
        dropout=0.0,
    )
    x = torch.randn(2, 3, 4, 20, 40)
    month = torch.tensor([1, 12], dtype=torch.long)
    out = pe(x, month)
    assert out.shape == (2, 5 * 10, 16), out.shape


def test_patch_embed_rejects_non_divisible_grid():
    _expect_value_error(
        PatchEmbedding,
        input_length=3,
        num_variables=NUM_VARIABLES,
        patch_size=4,
        d_model=16,
        grid_shape=(21, 40),
        fusion_heads=4,
    )


def test_patch_embed_rejects_bad_fusion_heads():
    _expect_value_error(
        PatchEmbedding,
        input_length=3,
        num_variables=NUM_VARIABLES,
        patch_size=4,
        d_model=16,
        grid_shape=(20, 40),
        fusion_heads=3,
    )


def test_patch_embed_target_month_changes_output():
    torch.manual_seed(0)
    pe = PatchEmbedding(
        input_length=3,
        num_variables=NUM_VARIABLES,
        patch_size=4,
        d_model=16,
        grid_shape=(20, 40),
        fusion_heads=4,
        dropout=0.0,
    )
    pe.eval()
    x = torch.randn(2, 3, 4, 20, 40)
    out_jan = pe(x, torch.tensor([1, 1], dtype=torch.long))
    out_jul = pe(x, torch.tensor([7, 7], dtype=torch.long))
    max_diff = (out_jan - out_jul).abs().max().item()
    assert max_diff > 0, f"calendar month embedding has no effect (max diff={max_diff})"


def test_patch_embed_time_variable_embeddings_get_grads():
    torch.manual_seed(0)
    pe = PatchEmbedding(
        input_length=3,
        num_variables=NUM_VARIABLES,
        patch_size=4,
        d_model=16,
        grid_shape=(20, 40),
        fusion_heads=4,
        dropout=0.0,
    )
    x = torch.randn(2, 3, 4, 20, 40)
    out = pe(x, torch.tensor([1, 12], dtype=torch.long))
    out.square().mean().backward()

    assert pe.relative_time_embed.grad is not None
    assert torch.isfinite(pe.relative_time_embed.grad).all()
    assert pe.calendar_month_embed.weight.grad is not None
    assert torch.isfinite(pe.calendar_month_embed.weight.grad).all()
    assert pe.variable_embed.weight.grad is not None
    assert torch.isfinite(pe.variable_embed.weight.grad).all()


def test_patch_embed_rejects_bad_target_month():
    pe = PatchEmbedding(
        input_length=3,
        num_variables=NUM_VARIABLES,
        patch_size=4,
        d_model=16,
        grid_shape=(20, 40),
        fusion_heads=4,
        dropout=0.0,
    )
    x = torch.randn(2, 3, 4, 20, 40)
    _expect_value_error(pe, x, torch.tensor([0, 13], dtype=torch.long))


# --- SpatialAttentionBlock --------------------------------------------

def test_spatial_attention_preserves_shape():
    block = SpatialAttentionBlock(d_model=16, nhead=2, dim_ff=32, dropout=0.0)
    x = torch.randn(2, 50, 16)
    out = block(x)
    assert out.shape == (2, 50, 16)


# --- TemporalMixtureOfExperts -----------------------------------------

def test_tmoe_shape():
    tmoe = TemporalMixtureOfExperts(
        d_model=16, dim_ff=32, num_experts=4, top_k=2, dropout=0.0
    )
    x = torch.randn(2, 50, 16)
    month = torch.tensor([1, 12], dtype=torch.long)
    out = tmoe(x, month)
    assert out.shape == (2, 50, 16)


def test_tmoe_rejects_bad_topk():
    for bad in (0, 5):
        _expect_value_error(
            TemporalMixtureOfExperts,
            d_model=16, dim_ff=32, num_experts=4, top_k=bad,
        )


def test_tmoe_routing_depends_on_month():
    """Different target_month values must change the output (routing alive)."""
    torch.manual_seed(0)
    tmoe = TemporalMixtureOfExperts(
        d_model=16, dim_ff=32, num_experts=4, top_k=2, dropout=0.0
    )
    tmoe.eval()
    x = torch.randn(2, 10, 16)
    out_jan = tmoe(x, torch.tensor([1, 1], dtype=torch.long))
    out_jul = tmoe(x, torch.tensor([7, 7], dtype=torch.long))
    max_diff = (out_jan - out_jul).abs().max().item()
    assert max_diff > 0, f"month routing has no effect (max diff={max_diff})"


def test_tmoe_handles_month_boundaries():
    tmoe = TemporalMixtureOfExperts(
        d_model=16, dim_ff=32, num_experts=4, top_k=2, dropout=0.0
    )
    x = torch.randn(2, 5, 16)
    for m in (1, 12):
        out = tmoe(x, torch.tensor([m, m], dtype=torch.long))
        assert torch.isfinite(out).all()


# --- RolloutEmbedding -------------------------------------------------

def test_rollout_embed_none_equals_zero():
    torch.manual_seed(0)
    re = RolloutEmbedding(d_model=16, max_steps=8)
    x = torch.randn(2, 50, 16)
    out_none = re(x, None)
    out_zero = re(x, torch.zeros(2, dtype=torch.long))
    assert out_none.shape == (2, 50, 16)
    assert torch.allclose(out_none, out_zero)


def test_rollout_embed_clamps_out_of_range():
    re = RolloutEmbedding(d_model=16, max_steps=8)
    x = torch.randn(2, 5, 16)
    out = re(x, torch.tensor([100, -5], dtype=torch.long))
    assert out.shape == (2, 5, 16)
    assert torch.isfinite(out).all()


# --- CoupledVariableDecoder -------------------------------------------

def test_decoder_shape_patch4():
    """patch_size=4: two PixelShuffle x2 stages, residual branch is Identity."""
    dec = CoupledVariableDecoder(
        d_model=16, out_channels=4, patch_size=4,
        target_shape=(20, 40), hidden_channels=8,
    )
    x = torch.randn(2, 5 * 10, 16)
    out = dec(x)
    assert out.shape == (2, 1, 4, 20, 40)


def test_decoder_shape_patch8_residual_branch():
    """patch_size=8: residual PixelShuffle x2 stage activates."""
    dec = CoupledVariableDecoder(
        d_model=16, out_channels=4, patch_size=8,
        target_shape=(24, 48), hidden_channels=8,
    )
    x = torch.randn(2, 3 * 6, 16)
    out = dec(x)
    assert out.shape == (2, 1, 4, 24, 48)


def test_decoder_rejects_patch_not_divisible_by_4():
    _expect_value_error(
        CoupledVariableDecoder,
        d_model=16, out_channels=4, patch_size=6,
        target_shape=(24, 48), hidden_channels=8,
    )


# --- WalkerNet end-to-end ---------------------------------------------

def _make_inputs(cfg):
    data = cfg["data"]
    B = 2
    L, V = data["L"], NUM_VARIABLES
    H, W = data["H"], data["W"]
    x = torch.randn(B, L, V, H, W)
    target_month = torch.tensor([1, 12], dtype=torch.long)
    return x, target_month, B, V, H, W


def test_walkernet_forward_shape_no_rollout():
    torch.manual_seed(0)
    model = WalkerNet(SMALL_CFG)
    x, month, B, V, H, W = _make_inputs(SMALL_CFG)
    out = model(x, month)
    assert out.shape == (B, 1, V, H, W)
    assert out.dtype == torch.float32


def test_walkernet_forward_with_rollout():
    torch.manual_seed(0)
    model = WalkerNet(SMALL_CFG)
    x, month, B, V, H, W = _make_inputs(SMALL_CFG)
    rollout = torch.tensor([0, 5], dtype=torch.long)
    out = model(x, month, rollout_step=rollout)
    assert out.shape == (B, 1, V, H, W)


def test_walkernet_backward_grads_all_finite():
    torch.manual_seed(0)
    model = WalkerNet(SMALL_CFG)
    x, month, *_ = _make_inputs(SMALL_CFG)
    out = model(x, month)
    target = torch.randn_like(out)
    ((out - target) ** 2).mean().backward()
    n_total = sum(1 for _ in model.parameters())
    n_finite = sum(
        1 for p in model.parameters()
        if p.grad is not None and torch.isfinite(p.grad).all()
    )
    assert n_finite == n_total, f"non-finite grads: {n_finite}/{n_total}"


def test_walkernet_full_resolution_builds():
    """Smoke: at production resolution 180x360 the model wires up correctly.

    Forward is too heavy for CPU; we only verify __init__ and patch counts.
    """
    cfg = {
        "data": {"L": 3, "H": 180, "W": 360},
        "model": {
            "patch_size": 4, "d_model": 64, "nhead": 4, "dim_ff": 128,
            "num_layers": 2, "patch_fusion_heads": 4, "patch_fusion_layers": 1,
            "patch_fusion_dim_ff": 128, "num_experts": 4, "top_k": 2,
            "decoder_hidden": 32, "dropout": 0.0, "max_rollout_steps": 8,
        },
    }
    model = WalkerNet(cfg)
    assert model.patch_embed.num_patches == (180 // 4) * (360 // 4)


# ---------------------------------------------------------------------
# Standalone runner (no pytest required).
# ---------------------------------------------------------------------

def _run_all():
    import traceback
    tests = sorted(
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    failed: list[str] = []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(name)
        else:
            print(f"PASS  {name}")
    print()
    if failed:
        print(f"{len(failed)}/{len(tests)} FAILED: {failed}")
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
