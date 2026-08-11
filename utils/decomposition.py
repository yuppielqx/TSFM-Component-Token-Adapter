"""Lightweight decomposition utilities for component-wise time-series tokenization."""

from typing import Dict

import torch
import torch.nn.functional as F


VALID_COMPONENTS = ("trend", "seasonal", "residual", "seasonal_noise")


def moving_average_trend(x: torch.Tensor, kernel_size: int = 25) -> torch.Tensor:
    """Estimate a smooth trend with a centered moving average.

    Args:
        x: Tensor with shape (B, L).
        kernel_size: Odd moving-average window. Even values are rounded up.
    """
    if x.dim() != 2:
        raise ValueError(f"Expected x with shape (B, L), got {tuple(x.shape)}")
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive")
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    x3 = x.unsqueeze(1)
    padded = F.pad(x3, (pad, pad), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=kernel_size, stride=1).squeeze(1)


def seasonal_from_period(detrended: torch.Tensor, period: int = 24) -> torch.Tensor:
    """Estimate seasonal component by averaging values with the same phase."""
    if period <= 1:
        return torch.zeros_like(detrended)
    batch, length = detrended.shape
    seasonal = torch.zeros_like(detrended)
    for phase in range(min(period, length)):
        idx = torch.arange(phase, length, period, device=detrended.device)
        if idx.numel() == 0:
            continue
        phase_mean = detrended.index_select(1, idx).mean(dim=1, keepdim=True)
        seasonal[:, idx] = phase_mean
    # Keep the decomposition centered so residual does not inherit a bias.
    seasonal = seasonal - seasonal.mean(dim=1, keepdim=True)
    return seasonal


def decompose_series(
    x: torch.Tensor,
    period: int = 24,
    trend_kernel: int = 25,
) -> Dict[str, torch.Tensor]:
    """Decompose x into trend, seasonal, residual, and seasonal+noise components."""
    trend = moving_average_trend(x, kernel_size=trend_kernel)
    detrended = x - trend
    seasonal = seasonal_from_period(detrended, period=period)
    residual = x - trend - seasonal
    return {
        "trend": trend,
        "seasonal": seasonal,
        "residual": residual,
        "seasonal_noise": seasonal + residual,
    }


def select_component(
    x: torch.Tensor,
    component: str,
    period: int = 24,
    trend_kernel: int = 25,
) -> torch.Tensor:
    """Return one decomposed component from x."""
    if component not in VALID_COMPONENTS:
        raise ValueError(f"component must be one of {VALID_COMPONENTS}, got {component}")
    return decompose_series(x, period=period, trend_kernel=trend_kernel)[component]
