"""WalkerNet model architecture.

Owner: Ziyi Zhuang.
Implements model(x, target_month, rollout_step=None) -> y_pred per src/interfaces.py.

Step 1 assembly: shapes and gradients flow end-to-end.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .interfaces import GRID_H, GRID_W, NUM_VARIABLES
except ImportError:  # pragma: no cover - allows running this file directly
    from interfaces import GRID_H, GRID_W, NUM_VARIABLES


class PatchEmbedding(nn.Module):
    """(B, L, 4, H, W) -> (B, N, d_model) via Conv2d patch projection + learnable pos embed."""

    def __init__(self, in_channels, patch_size, d_model, grid_shape=(GRID_H, GRID_W), dropout=0.0):
        super().__init__()
        H, W = grid_shape
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(f"grid {(H, W)} not divisible by patch_size {patch_size}")
        self.patch_size = patch_size
        self.grid_h = H // patch_size
        self.grid_w = W // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, V, H, W = x.shape
        x = x.reshape(B, L * V, H, W)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.dropout(x + self.pos_embed)


class SpatialAttentionBlock(nn.Module):
    """Pre-norm ViT block: LN -> MHA -> residual; LN -> FFN -> residual."""

    def __init__(self, d_model, nhead, dim_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalMixtureOfExperts(nn.Module):
    """Top-k soft TMoE routed by target_month.

    Routing is purely temporal: a month embedding feeds a gate that produces
    a soft top-k weight over `num_experts` FFN experts, shared across all tokens
    in the sample. Each expert is an FFN block (LN+residual stay outside).
    """

    def __init__(self, d_model, dim_ff, num_experts=12, top_k=2, dropout=0.1, gate_dim=64):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        self.num_experts = num_experts
        self.top_k = top_k

        self.norm = nn.LayerNorm(d_model)
        self.month_embed = nn.Embedding(12, gate_dim)
        nn.init.trunc_normal_(self.month_embed.weight, std=0.02)
        self.gate = nn.Linear(gate_dim, num_experts)

        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_model, dim_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_ff, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        )

    def forward(self, x, target_month):
        B = x.shape[0]
        month_idx = (target_month - 1).clamp(0, 11)
        logits = self.gate(self.month_embed(month_idx))

        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(-1, topk_idx, topk_vals)
        weights = torch.softmax(masked, dim=-1)

        h = self.norm(x)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out = out + weights[:, i].view(B, 1, 1) * expert(h)
        return x + out


class RolloutEmbedding(nn.Module):
    """Additive embedding for autoregressive step index."""

    def __init__(self, d_model, max_steps=24):
        super().__init__()
        self.max_steps = max_steps
        self.embed = nn.Embedding(max_steps, d_model)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def forward(self, x, rollout_step):
        if rollout_step is None:
            rollout_step = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        rollout_step = rollout_step.clamp(min=0, max=self.max_steps - 1)
        return x + self.embed(rollout_step).unsqueeze(1)


class CoupledVariableDecoder(nn.Module):
    """Decode tokens to (B, 1, 4, H, W) via low-res feature map + PixelShuffle x2 x2.

    Pipeline:
        (B, N, d_model) -> reshape (B, d_model, H/p, W/p)
                        -> Conv refine (variable coupling lives in channel dim)
                        -> PixelShuffle x2 (with conv head producing 4*d_up^2 channels)
                        -> Conv refine
                        -> PixelShuffle x2
                        -> Conv2d(d_up -> 4)
                        -> unsqueeze time dim
    """

    def __init__(self, d_model, out_channels, patch_size, target_shape, hidden_channels=128):
        super().__init__()
        H, W = target_shape
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(f"target_shape {(H, W)} not divisible by patch_size {patch_size}")
        if patch_size % 4 != 0:
            raise ValueError(f"patch_size must be divisible by 4 for two PixelShuffle x2 stages, got {patch_size}")

        self.out_channels = out_channels
        self.patch_size = patch_size
        self.grid_h = H // patch_size
        self.grid_w = W // patch_size
        self.H = H
        self.W = W
        self.residual_patch = patch_size // 4

        c = hidden_channels
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Conv2d(d_model, c, kernel_size=1)

        self.refine1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(c, c * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
        )

        self.refine2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(c, c * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
        )

        # If patch_size > 4, finish lifting to (H, W) with one more PixelShuffle stage.
        if self.residual_patch > 1:
            r = self.residual_patch
            self.up_residual = nn.Sequential(
                nn.Conv2d(c, c * r * r, kernel_size=3, padding=1),
                nn.PixelShuffle(r),
                nn.GELU(),
            )
        else:
            self.up_residual = nn.Identity()

        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        B, N, D = x.shape
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, D, self.grid_h, self.grid_w)
        x = self.proj(x)
        x = self.refine1(x)
        x = self.up1(x)
        x = self.refine2(x)
        x = self.up2(x)
        x = self.up_residual(x)
        x = self.head(x)
        return x.unsqueeze(1)


class WalkerNet(nn.Module):
    """Full pipeline: PatchEmbedding -> N x SpatialAttention -> TMoE -> Rollout -> Decoder."""

    def __init__(self, config):
        super().__init__()
        data_cfg = config.get("data", {})
        model_cfg = config.get("model", config)

        L = int(data_cfg.get("L", 3))
        H = int(data_cfg.get("H", GRID_H))
        W = int(data_cfg.get("W", GRID_W))
        V = NUM_VARIABLES

        patch_size = int(model_cfg["patch_size"])
        d_model = int(model_cfg["d_model"])
        nhead = int(model_cfg["nhead"])
        dim_ff = int(model_cfg["dim_ff"])
        num_layers = int(model_cfg["num_layers"])
        num_experts = int(model_cfg.get("num_experts", 12))
        top_k = int(model_cfg.get("top_k", 2))
        dropout = float(model_cfg.get("dropout", 0.1))
        max_rollout_steps = int(model_cfg.get("max_rollout_steps", 24))
        decoder_hidden = int(model_cfg.get("decoder_hidden", 128))

        self.patch_embed = PatchEmbedding(
            in_channels=L * V,
            patch_size=patch_size,
            d_model=d_model,
            grid_shape=(H, W),
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [SpatialAttentionBlock(d_model, nhead, dim_ff, dropout=dropout) for _ in range(num_layers)]
        )
        self.tmoe = TemporalMixtureOfExperts(
            d_model, dim_ff, num_experts=num_experts, top_k=top_k, dropout=dropout
        )
        self.rollout_embed = RolloutEmbedding(d_model, max_steps=max_rollout_steps)
        self.decoder = CoupledVariableDecoder(
            d_model=d_model,
            out_channels=V,
            patch_size=patch_size,
            target_shape=(H, W),
            hidden_channels=decoder_hidden,
        )

    def forward(self, x, target_month, rollout_step=None):
        z = self.patch_embed(x)
        z = self.rollout_embed(z, rollout_step)
        for block in self.blocks:
            z = block(z)
        z = self.tmoe(z, target_month)
        return self.decoder(z)

