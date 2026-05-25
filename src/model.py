"""
WalkerNet model architecture.

Owner: Ziyi Zhuang
Responsibility: All neural network modules

Architecture:
  Input (B, L, 4, H, W)
    -> Joint Time-Variable Patch Embedding
    -> Spatial Attention (ViT)
    -> TMoE (target-month conditioned)
    -> Coupled Variable Decoder
  Output (B, 1, 4, H, W)
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Joint time-variable patch embedding.

    Flattens L time steps x 4 variables into the channel dimension,
    then applies patch projection to produce token sequence.

    Input:  (B, L, 4, H, W)
    Output: (B, N, d_model)  where N = (H/patch_size) * (W/patch_size)
    """

    def __init__(self, in_channels, patch_size, d_model):
        """
        Args:
            in_channels: L * 4 (time steps x variables)
            patch_size: spatial patch size (int or tuple)
            d_model: embedding dimension
        """
        super().__init__()
        # TODO: implement
        pass

    def forward(self, x):
        """
        Args:
            x: (B, L, 4, H, W)
        Returns:
            (B, N, d_model)
        """
        raise NotImplementedError


class SpatialAttentionBlock(nn.Module):
    """Single ViT-style spatial attention block.

    Standard pre-norm Transformer block:
      LayerNorm -> MultiheadAttention -> Residual
      LayerNorm -> FFN -> Residual
    """

    def __init__(self, d_model, nhead, dim_ff, dropout=0.1):
        super().__init__()
        # TODO: implement
        pass

    def forward(self, x):
        """
        Args:
            x: (B, N, d_model)
        Returns:
            (B, N, d_model)
        """
        raise NotImplementedError


class TemporalMixtureOfExperts(nn.Module):
    """TMoE: routes tokens to different expert FFNs based on target month.

    Uses target_month (1-12) to select/expert-weight a set of FFN experts.
    Can be conditioned via month embedding added to gating input.
    """

    def __init__(self, d_model, dim_ff, num_experts=12):
        """
        Args:
            d_model: token dimension
            dim_ff: FFN hidden dimension
            num_experts: number of expert networks (default 12, one per month)
        """
        super().__init__()
        # TODO: implement
        pass

    def forward(self, x, target_month):
        """
        Args:
            x: (B, N, d_model)
            target_month: (B,) int64, values in [1, 12]
        Returns:
            (B, N, d_model)
        """
        raise NotImplementedError


class RolloutEmbedding(nn.Module):
    """Encodes rollout step as a conditioning signal.

    Injects lead-time awareness into the model. Not used for TMoE routing,
    but added to token representations or cross-attended.
    """

    def __init__(self, d_model, max_steps=24):
        """
        Args:
            d_model: embedding dimension
            max_steps: maximum rollout steps to support
        """
        super().__init__()
        # TODO: implement
        pass

    def forward(self, x, rollout_step):
        """
        Args:
            x: (B, N, d_model)
            rollout_step: (B,) int64, 0=single-step, 1/2/3...=rollout
        Returns:
            (B, N, d_model)
        """
        raise NotImplementedError


class CoupledVariableDecoder(nn.Module):
    """Decodes token representations back to 4 coupled physical fields.

    All 4 variables decoded jointly (not independently),
    leveraging physical coupling between SST, HC, taux, tauy.

    Input:  (B, N, d_model)
    Output: (B, 1, 4, H, W)
    """

    def __init__(self, d_model, out_channels, patch_size, target_shape):
        """
        Args:
            d_model: token dimension
            out_channels: 4 (number of output variables)
            patch_size: spatial patch size (must match embedding)
            target_shape: (H, W) of output grid
        """
        super().__init__()
        # TODO: implement
        pass

    def forward(self, x):
        """
        Args:
            x: (B, N, d_model)
        Returns:
            (B, 1, 4, H, W)
        """
        raise NotImplementedError


class WalkerNet(nn.Module):
    """Complete WalkerNet model.

    Assembles all components into the full pipeline.
    See interfaces.py for the exact input/output contract.
    """

    def __init__(self, config):
        """
        Args:
            config: dict or OmegaConf with all hyperparameters.
                    Expected keys: L, patch_size, d_model, nhead, dim_ff,
                    num_layers, num_experts, H, W, max_rollout_steps
        """
        super().__init__()
        # TODO: implement — assemble PatchEmbedding, N x SpatialAttentionBlock,
        #       TMoE, RolloutEmbedding, CoupledVariableDecoder
        pass

    def forward(self, x, target_month, rollout_step=None):
        """
        Args:
            x: (B, L, 4, H, W)
            target_month: (B,) int64, values in [1, 12]
            rollout_step: (B,) int64, 0=single-step (default)
        Returns:
            (B, 1, 4, H, W)
        """
        raise NotImplementedError
