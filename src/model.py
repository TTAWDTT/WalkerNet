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


class FusionAttentionBlock(nn.Module):
    """Patch 内 time-variable token 的轻量 Transformer block。"""

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


class PatchEmbedding(nn.Module):
    """(B, L, 4, H, W) -> (B, N, d_model) with explicit time/variable fusion.

    流程：
        1. 每个变量使用自己的浅层 patch projection，同一变量跨时间共享权重。
        2. 加相对历史位置编码、输入日历月份编码、变量身份编码。
        3. 对每个空间 patch 内的 L*4 个 token 做轻量 self-attention fusion。
        4. 加二维空间位置编码，输出空间 token 序列。
    """

    def __init__(
        self,
        input_length,
        patch_size,
        d_model,
        grid_shape=(GRID_H, GRID_W),
        num_variables=NUM_VARIABLES,
        fusion_heads=4,
        fusion_dim_ff=None,
        fusion_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        H, W = grid_shape
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(f"grid {(H, W)} not divisible by patch_size {patch_size}")
        if d_model % fusion_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by fusion_heads {fusion_heads}")
        if input_length < 1:
            raise ValueError(f"input_length must be >= 1, got {input_length}")
        if num_variables < 1:
            raise ValueError(f"num_variables must be >= 1, got {num_variables}")
        if fusion_layers < 1:
            raise ValueError(f"fusion_layers must be >= 1, got {fusion_layers}")

        self.input_length = input_length
        self.num_variables = num_variables
        self.patch_size = patch_size
        self.d_model = d_model
        self.grid_h = H // patch_size
        self.grid_w = W // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # 每个变量一套 patch projection；同一变量的不同历史月份共享该 projection。
        self.variable_projs = nn.ModuleList(
            nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)
            for _ in range(num_variables)
        )

        self.relative_time_embed = nn.Parameter(torch.zeros(1, input_length, 1, 1, d_model))
        self.calendar_month_embed = nn.Embedding(12, d_model)
        self.variable_embed = nn.Embedding(num_variables, d_model)

        self.fusion_token = nn.Parameter(torch.zeros(1, 1, d_model))
        fusion_dim_ff = int(fusion_dim_ff or d_model * 4)
        self.fusion_blocks = nn.ModuleList(
            FusionAttentionBlock(d_model, fusion_heads, fusion_dim_ff, dropout=dropout)
            for _ in range(fusion_layers)
        )

        # 二维空间位置编码：先保留 patch 网格结构，再展平成 (1, N, d_model)。
        self.lat_pos_embed = nn.Parameter(torch.zeros(1, self.grid_h, 1, d_model))
        self.lon_pos_embed = nn.Parameter(torch.zeros(1, 1, self.grid_w, d_model))
        self.dropout = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.relative_time_embed, std=0.02)
        nn.init.trunc_normal_(self.calendar_month_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.variable_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.fusion_token, std=0.02)
        nn.init.trunc_normal_(self.lat_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.lon_pos_embed, std=0.02)

    def forward(self, x, target_month):
        B, L, V, H, W = x.shape
        if L != self.input_length:
            raise ValueError(f"expected input length {self.input_length}, got {L}")
        if V != self.num_variables:
            raise ValueError(f"expected {self.num_variables} variables, got {V}")
        if target_month.shape != (B,):
            raise ValueError(f"target_month must be shape {(B,)}, got {target_month.shape}")

        target_month = target_month.to(device=x.device, dtype=torch.long)
        if bool(((target_month < 1) | (target_month > 12)).any()):
            raise ValueError("target_month values must be in [1, 12]")

        var_tokens = []
        for var_idx, proj in enumerate(self.variable_projs):
            x_var = x[:, :, var_idx].reshape(B * L, 1, H, W)
            z_var = proj(x_var).flatten(2).transpose(1, 2)
            z_var = z_var.reshape(B, L, self.num_patches, self.d_model)
            var_tokens.append(z_var)
        z = torch.stack(var_tokens, dim=2)

        z = z + self.relative_time_embed
        z = z + self._calendar_month_embedding(target_month)

        variable_ids = torch.arange(V, device=x.device)
        variable_embed = self.variable_embed(variable_ids).view(1, 1, V, 1, self.d_model)
        z = z + variable_embed

        # 每个空间 patch 内部单独融合 L*V 个 token。
        z = z.permute(0, 3, 1, 2, 4).reshape(B * self.num_patches, L * V, self.d_model)
        fusion_token = self.fusion_token.expand(B * self.num_patches, -1, -1)
        z = torch.cat([fusion_token, z], dim=1)
        for block in self.fusion_blocks:
            z = block(z)
        z = z[:, 0].reshape(B, self.num_patches, self.d_model)

        pos = (self.lat_pos_embed + self.lon_pos_embed).reshape(1, self.num_patches, self.d_model)
        return self.dropout(z + pos)

    def _calendar_month_embedding(self, target_month):
        """根据目标月份推出输入窗口每个历史步的日历月份编码。"""
        offsets = torch.arange(self.input_length, device=target_month.device)
        month_idx = (target_month[:, None] - self.input_length + offsets - 1) % 12
        month_embed = self.calendar_month_embed(month_idx)
        return month_embed[:, :, None, None, :]


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

        patch_fusion_heads = int(model_cfg.get("patch_fusion_heads", 4))
        patch_fusion_layers = int(model_cfg.get("patch_fusion_layers", 1))
        patch_fusion_dim_ff = int(model_cfg.get("patch_fusion_dim_ff", d_model * 4))

        self.patch_embed = PatchEmbedding(
            input_length=L,
            patch_size=patch_size,
            d_model=d_model,
            grid_shape=(H, W),
            num_variables=V,
            fusion_heads=patch_fusion_heads,
            fusion_dim_ff=patch_fusion_dim_ff,
            fusion_layers=patch_fusion_layers,
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
        z = self.patch_embed(x, target_month)
        z = self.rollout_embed(z, rollout_step)
        for block in self.blocks:
            z = block(z)
        z = self.tmoe(z, target_month)
        return self.decoder(z)
