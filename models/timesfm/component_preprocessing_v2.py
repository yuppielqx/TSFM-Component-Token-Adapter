"""TimesFM-2.5 codebook 与 adapter 共用的成分预处理。

This module follows the released TimesFM-2.5 decode path: values are left
padded to a multiple of the 32-point input patch, masks use ``True`` for
unobserved entries, and each patch is normalized with cumulative running
statistics over that patch and all preceding observed values.

维度约定：
    B: batch size；T: 原始序列长度；P: patch 数；S: patch_size（默认 32）。
    原始序列为 [B, T]，patch 化后为 [B, P, S]。
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .torch import util


PREPROCESSING_NAME = "timesfm25_cumulative_patch_revin_v2"
MASK_POLICY_NAME = "timesfm_prefix_train_only"


@dataclass
class TimesFMComponentPatches:
    """预处理结果；前三个张量的形状均为 [B, P, S]。"""

    # TimesFM 官方累计 RevIN 后的输入，形状 [B, P, S]。
    normalized_patches: torch.Tensor
    # codebook 重建目标，形状 [B, P, S]；当前与 normalized_patches 数值一致。
    target_patches: torch.Tensor
    # True 表示 padding、非有限值或训练时前缀遮挡，形状 [B, P, S]。
    patch_masks: torch.Tensor
    # 每个 patch 对应的累计均值/标准差，形状 [B, P]。
    mus: torch.Tensor
    sigmas: torch.Tensor
    # 截断后的真实时间长度 T，以及为了对齐 patch 所做的左 padding 长度。
    original_length: int
    padding_length: int
    # 每个样本随机遮挡的前缀长度，形状 [B]。
    prefix_lengths: torch.Tensor


def pad_timesfm_context(
    context: torch.Tensor,
    *,
    patch_size: int = 32,
    context_limit: int = 16384,
):
    """按 TimesFM 规则截断并左填充。

    输入：context [B, T]。
    输出：values/masks [B, T_pad]，其中 T_pad 能被 S 整除，mask=True 表示不可见。
    """
    if context.ndim != 2:
        raise ValueError(
            f"context must have shape (batch, time), got {tuple(context.shape)}"
        )
    if context.shape[-1] <= 0:
        raise ValueError("context must contain at least one time point")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if context_limit <= 0:
        raise ValueError(f"context_limit must be positive, got {context_limit}")

    # 只保留最近 context_limit 个点；不会从序列尾部补点或使用未来值。
    context = context.float()[..., -context_limit:]  # [B, T]
    original_length = context.shape[-1]
    finite = torch.isfinite(context)  # [B, T]
    values = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
    masks = ~finite
    padding_length = (-original_length) % patch_size
    if padding_length:
        # TimesFM 使用左 padding：时间轴右端仍对应最新观测。
        values = F.pad(values, (padding_length, 0), value=0.0)
        masks = F.pad(masks, (padding_length, 0), value=True)
    return values, masks, original_length, padding_length


def normalize_timesfm_patches(
    values: torch.Tensor,
    masks: torch.Tensor,
    *,
    patch_size: int = 32,
):
    """执行 TimesFM-2.5 官方累计 patch RevIN。

    输入 values/masks: [B, P*S]；输出 normalized/mask: [B, P, S]，
    mus/sigmas: [B, P]。第 i 个 patch 的统计量只依赖第 0..i 个可见 patch。
    """
    if values.shape != masks.shape:
        raise ValueError(
            f"values and masks must have the same shape, got "
            f"{tuple(values.shape)} and {tuple(masks.shape)}"
        )
    if values.ndim != 2 or values.shape[-1] % patch_size:
        raise ValueError("values must have shape (batch, multiple_of_patch_size)")
    batch_size = values.shape[0]
    patched_values = values.reshape(batch_size, -1, patch_size)  # [B, P, S]
    patched_masks = masks.bool().reshape(batch_size, -1, patch_size)  # [B, P, S]

    # n/mu/sigma 是逐样本累计状态，形状均为 [B]。
    n = torch.zeros(batch_size, device=values.device, dtype=values.dtype)
    mu = torch.zeros_like(n)
    sigma = torch.zeros_like(n)
    patch_mus = []
    patch_sigmas = []
    for patch_index in range(patched_values.shape[1]):
        (n, mu, sigma), _ = util.update_running_stats(
            n,
            mu,
            sigma,
            patched_values[:, patch_index],
            patched_masks[:, patch_index],
        )
        patch_mus.append(mu)
        patch_sigmas.append(sigma)

    mus = torch.stack(patch_mus, dim=1)  # [B, P]
    sigmas = torch.stack(patch_sigmas, dim=1)  # [B, P]
    normalized = util.revin(patched_values, mus, sigmas, reverse=False)  # [B, P, S]
    normalized = torch.where(
        patched_masks, torch.zeros_like(normalized), normalized
    )
    return normalized, patched_masks, mus, sigmas


def prepare_timesfm_component_patches(
    context: torch.Tensor,
    *,
    patch_size: int = 32,
    context_limit: int = 16384,
    apply_prefix_mask: bool = False,
    generator: Optional[torch.Generator] = None,
) -> TimesFMComponentPatches:
    """按照 TimesFM-2.5 decode 输入方式准备成分序列。

    ``apply_prefix_mask`` implements the TimesFM pretraining mask: for each
    sample, a value ``r`` is drawn uniformly from ``[0, patch_size - 1]`` and
    the first ``r`` observed positions are marked unobserved. It is intended
    for codebook training only. Validation, code extraction, and adapter use
    must leave it disabled.
    """
    values, masks, original_length, padding_length = pad_timesfm_context(
        context, patch_size=patch_size, context_limit=context_limit
    )  # values/masks: [B, P*S]
    batch_size = values.shape[0]

    prefix_lengths = torch.zeros(
        batch_size, dtype=torch.long, device=context.device
    )
    if apply_prefix_mask:
        prefix_lengths = torch.randint(
            0,
            patch_size,
            (batch_size,),
            device=context.device,
            generator=generator,
        )
        positions = torch.arange(values.shape[-1], device=context.device)  # [P*S]
        starts = torch.full_like(prefix_lengths, padding_length)
        prefix_masks = (
            (positions.unsqueeze(0) >= starts.unsqueeze(1))
            & (
                positions.unsqueeze(0)
                < (starts + prefix_lengths).unsqueeze(1)
            )
        )  # [B, P*S]
        masks = masks | prefix_masks

    normalized, patched_masks, mus, sigmas = normalize_timesfm_patches(
        values, masks, patch_size=patch_size
    )

    # 重建目标与 backbone 输入采用完全相同的官方归一化。
    # 随机前缀、padding 与非有限位置会在重建损失中排除。
    target = normalized.detach().clone()
    return TimesFMComponentPatches(
        normalized_patches=normalized,
        target_patches=target,
        patch_masks=patched_masks,
        mus=mus,
        sigmas=sigmas,
        original_length=original_length,
        padding_length=padding_length,
        prefix_lengths=prefix_lengths,
    )
