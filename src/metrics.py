"""
Evaluation metrics for ENSO and general field prediction.

Owner: Zhen Luo
Responsibility: Niño3.4 computation, ACC, RMSE, skill scores

All metrics accept torch tensors and return float values.
"""

import torch


def compute_nino34(sst, lat, lon):
    """Compute Niño3.4 index from SST field.

    Niño3.4 region: 5°N-5°S, 170°W-120°W

    Args:
        sst: (B, 1, H, W) or (B, H, W) SST field
        lat: (H,) latitude array
        lon: (W,) longitude array
    Returns:
        (B,) or (B, 1) Niño3.4 index (area-weighted mean SST in region)
    """
    raise NotImplementedError


def anomaly_correlation_coefficient(pred, target):
    """ACC: correlation between predicted and target anomaly fields.

    Args:
        pred: (B, ...) predicted anomaly
        target: (B, ...) target anomaly
    Returns:
        float ACC value
    """
    raise NotImplementedError


def rmse(pred, target):
    """Root mean squared error.

    Args:
        pred: (B, ...) prediction
        target: (B, ...) target
    Returns:
        float RMSE value
    """
    raise NotImplementedError


def nino34_correlation(pred_sst, target_sst, lat, lon, lead_months=None):
    """Correlation of predicted vs observed Niño3.4 index, by lead time.

    Args:
        pred_sst: (B, 1, H, W) predicted SST
        target_sst: (B, 1, H, W) target SST
        lat: (H,) latitude
        lon: (W,) longitude
        lead_months: optional list of lead months to evaluate
    Returns:
        dict of {lead_month: correlation}
    """
    raise NotImplementedError


def nino34_rmse(pred_sst, target_sst, lat, lon, lead_months=None):
    """RMSE of predicted vs observed Niño3.4 index, by lead time.

    Args:
        same as nino34_correlation
    Returns:
        dict of {lead_month: rmse}
    """
    raise NotImplementedError
