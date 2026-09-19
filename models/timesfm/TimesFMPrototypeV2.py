"""TimesFM-2.5 成分 RQ-VAE / codebook 训练实现（V2）。

This module mirrors the ChronosBolt codebook stage while using the frozen
PyTorch TimesFM 2.5 backbone. It learns separate RQ codebooks for trend,
seasonal, and optional residual components in TimesFM's 1280-dimensional
patch representation space.

维度约定：B 为 batch size，P 为输入 patch 数，S=32，D=1280，R 为 RQ stage 数。
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
from .component_preprocessing_v2 import (
    MASK_POLICY_NAME,
    PREPROCESSING_NAME,
    prepare_timesfm_component_patches,
)




class ResidualQuantizer(nn.Module):
    """带 straight-through 梯度的残差向量量化器。

    输入/量化向量为 [B, P, D]，离散 code 为 [B, P, R]。
    """

    def __init__(self, n_stages=2, n_proto=256, d_model=1280, beta=0.25):
        super().__init__()
        self.beta = beta
        self.codebooks = nn.ModuleList([nn.Embedding(n_proto, d_model) for _ in range(n_stages)])
        for codebook in self.codebooks:
            nn.init.trunc_normal_(codebook.weight, std=0.02)

    def forward(self, z):
        # z/residual/quantized_sum: [B, P, D]。
        residual, quantized_sum, commit_loss, quantized_per_stage = z, torch.zeros_like(z), 0.0, []
        for codebook in self.codebooks:
            flat = residual.reshape(-1, residual.shape[-1])  # [B*P, D]
            # codebook.weight: [K, D]；distances: [B*P, K]。
            distances = flat.pow(2).sum(dim=1, keepdim=True) + codebook.weight.pow(2).sum(dim=1) - 2 * flat @ codebook.weight.T
            indices = distances.argmin(dim=-1)  # [B*P]
            quantized = codebook(indices).reshape_as(residual)  # [B, P, D]
            commit_loss = commit_loss + F.mse_loss(residual.detach(), quantized)
            quantized_sum = quantized_sum + residual + (quantized - residual).detach()
            quantized_per_stage.append(quantized.detach())
            residual = residual - quantized.detach()
        return quantized_sum, commit_loss * self.beta, quantized_per_stage

    @torch.no_grad()
    def get_codes(self, z):
        # 推理时仅返回每个 patch、每个 RQ stage 的 code ID。
        residual, all_indices = z, []
        for codebook in self.codebooks:
            flat = residual.reshape(-1, residual.shape[-1])  # [B*P, D]
            distances = flat.pow(2).sum(dim=1, keepdim=True) + codebook.weight.pow(2).sum(dim=1) - 2 * flat @ codebook.weight.T
            indices = distances.argmin(dim=-1)
            all_indices.append(indices.reshape(z.shape[0], z.shape[1]))  # [B, P]
            residual = residual - codebook(indices).reshape_as(residual)
        return torch.stack(all_indices, dim=-1)  # [B, P, R]


class TimesFMWithRQVAEPrototypesV2(nn.Module):
    """冻结 TimesFM 主干，仅训练成分 codebook 与 patch 重建头。"""

    input_patch_size = 32
    d_model = 1280
    context_limit = 16384

    def __init__(
        self,
        n_proto=256,
        n_rq_stages=2,
        beta=0.25,
        train_prefix_mask=True,
    ):
        super().__init__()
        self.train_prefix_mask = train_prefix_mask
        self.n_rq_stages = n_rq_stages
        self.backbone = TimesFM_2p5_200M_torch_module()
        self.residual_quantizer = ResidualQuantizer(
            n_stages=n_rq_stages, n_proto=n_proto, d_model=self.d_model, beta=beta
        )
        self.reconstruction_heads = nn.ModuleList(
            # 每个 RQ stage 将 [B, P, D] 解码成一个 [B, P, S] 的重建增量。
            [nn.Linear(self.d_model, self.input_patch_size) for _ in range(n_rq_stages)]
        )
        self.freeze_backbone()

    @classmethod
    def from_pretrained_for_rqvae(
        cls, pretrained_model_path, n_proto=256, n_rq_stages=2,
        beta=0.25, train_prefix_mask=True, device=None,
    ):
        model = cls(
            n_proto=n_proto,
            n_rq_stages=n_rq_stages,
            beta=beta,
            train_prefix_mask=train_prefix_mask,
        )
        weights_path = os.path.join(pretrained_model_path, "model.safetensors")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"TimesFM weights not found: {weights_path}")
        state = load_file(weights_path, device="cpu")
        model.backbone.load_state_dict(state, strict=True)
        model.freeze_backbone()
        if device is not None:
            model.to(device)
        print(f"TimesFM backbone loaded from {pretrained_model_path}")
        return model

    def freeze_backbone(self):
        # codebook 阶段绝不更新 TimesFM-2.5 的任何参数。
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def print_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameters: total={total:,}, trainable={trainable:,}")

    def _prepare_inputs(self, context, apply_prefix_mask=False, generator=None):
        # context: [B, T] -> normalized_patches/patch_masks: [B, P, 32]。
        return prepare_timesfm_component_patches(
            context,
            patch_size=self.input_patch_size,
            context_limit=self.context_limit,
            apply_prefix_mask=apply_prefix_mask,
            generator=generator,
        )

    def _extract_hidden(self, normalized_patches, patch_masks):
        # normalized_patches/patch_masks: [B, P, 32]。
        with torch.no_grad():
            outputs, _ = self.backbone(normalized_patches, patch_masks)
            _, hidden_states, _, _ = outputs
        return hidden_states.detach()  # [B, P, 1280]

    def forward(
        self,
        context,
        apply_prefix_mask=None,
        return_reconstruction=False,
        generator=None,
    ):
        """计算 RQ-VAE 损失。

        context: [B, T]；重建 patch: [B, P, 32]；返回标量 loss。
        随机前缀 mask 只在训练模式且 train_prefix_mask=True 时启用。
        """
        if apply_prefix_mask is None:
            apply_prefix_mask = self.training and self.train_prefix_mask
        prepared = self._prepare_inputs(
            context,
            apply_prefix_mask=apply_prefix_mask,
            generator=generator,
        )
        hidden_states = self._extract_hidden(
            prepared.normalized_patches, prepared.patch_masks
        )

        _, commit_loss, quantized_per_stage = self.residual_quantizer(hidden_states)
        # quantized_per_stage 是 R 个 [B, P, 1280] 张量。
        pred_patches = torch.zeros(
            hidden_states.shape[0], hidden_states.shape[1], self.input_patch_size,
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
        for head, quantized in zip(self.reconstruction_heads, quantized_per_stage):
            pred_patches = pred_patches + head(quantized)
        valid_target = ~prepared.patch_masks  # [B, P, 32]
        recon_loss = (
            (pred_patches - prepared.target_patches).pow(2) * valid_target
        ).sum()
        recon_loss = recon_loss / valid_target.sum().clamp_min(1)
        loss = recon_loss + commit_loss
        result = {
            "loss": loss,
            "reconstruction_loss": recon_loss.detach(),
            "commitment_loss": commit_loss.detach(),
        }
        if return_reconstruction:
            # 展平为 [B, P*32] 后裁掉左 padding，最终形状 [B, T]。
            result["reconstruction"] = pred_patches.reshape(
                pred_patches.shape[0], -1
            )[..., -prepared.original_length:]
            result["prefix_lengths"] = prepared.prefix_lengths
        return result

    @torch.no_grad()
    def encode_codes(self, context):
        """将 component context [B, T] 编码为离散 codes [B, P, R]。"""
        prepared = self._prepare_inputs(context, apply_prefix_mask=False)
        hidden_states = self._extract_hidden(
            prepared.normalized_patches, prepared.patch_masks
        )
        return self.residual_quantizer.get_codes(hidden_states)

    def save_codebooks(self, output_path):
        # checkpoint 同时保存 R 个 [K, D] codebook 及 R 个重建头。
        state = {
            "format_version": 2,
            "backbone": "timesfm-2.5-200m-pytorch",
            "preprocessing": PREPROCESSING_NAME,
            "mask_policy": (
                MASK_POLICY_NAME if self.train_prefix_mask else "structural_only"
            ),
            "outer_normalization": False,
            "input_patch_size": self.input_patch_size,
            "d_model": self.d_model,
            "n_rq_stages": self.n_rq_stages,
            **{f"codebook_{i}": cb.weight.detach().cpu().float()
               for i, cb in enumerate(self.residual_quantizer.codebooks)},
            "reconstruction_heads": {f"head_{i}": head.state_dict()
                                     for i, head in enumerate(self.reconstruction_heads)},
        }
        torch.save(state, output_path)
        print(f"Saved {self.n_rq_stages} TimesFM codebooks + heads to {output_path}")

    def load_trained_codebook(self, codebook_path):
        state = torch.load(codebook_path, map_location="cpu")
        if state.get("backbone") not in (None, "timesfm-2.5-200m-pytorch"):
            raise ValueError(f"Not a TimesFM codebook checkpoint: {codebook_path}")
        if state.get("format_version") != 2:
            raise ValueError(f"Expected TimesFM codebook format_version=2: {codebook_path}")
        if state.get("preprocessing") != PREPROCESSING_NAME:
            raise ValueError(
                f"Unexpected TimesFM preprocessing in codebook: {codebook_path}"
            )
        for i, codebook in enumerate(self.residual_quantizer.codebooks):
            codebook.weight.data.copy_(state[f"codebook_{i}"].to(codebook.weight.dtype))
        for i, head in enumerate(self.reconstruction_heads):
            head.load_state_dict(state["reconstruction_heads"][f"head_{i}"])
