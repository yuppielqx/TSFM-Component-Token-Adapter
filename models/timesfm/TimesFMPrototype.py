"""TimesFM 2.5 + component-wise RQ-VAE prototype learning.

This module mirrors the ChronosBolt codebook stage while using the frozen
PyTorch TimesFM 2.5 backbone. It learns separate RQ codebooks for trend,
seasonal, and optional residual components in TimesFM's 1280-dimensional
patch representation space.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module




class ResidualQuantizer(nn.Module):
    """Residual vector quantizer with straight-through gradients."""

    def __init__(self, n_stages=2, n_proto=256, d_model=1280, beta=0.25):
        super().__init__()
        self.beta = beta
        self.codebooks = nn.ModuleList([nn.Embedding(n_proto, d_model) for _ in range(n_stages)])
        for codebook in self.codebooks:
            nn.init.trunc_normal_(codebook.weight, std=0.02)

    def forward(self, z):
        residual, quantized_sum, commit_loss, quantized_per_stage = z, torch.zeros_like(z), 0.0, []
        for codebook in self.codebooks:
            flat = residual.reshape(-1, residual.shape[-1])
            distances = flat.pow(2).sum(dim=1, keepdim=True) + codebook.weight.pow(2).sum(dim=1) - 2 * flat @ codebook.weight.T
            indices = distances.argmin(dim=-1)
            quantized = codebook(indices).reshape_as(residual)
            commit_loss = commit_loss + F.mse_loss(residual.detach(), quantized)
            quantized_sum = quantized_sum + residual + (quantized - residual).detach()
            quantized_per_stage.append(quantized.detach())
            residual = residual - quantized.detach()
        return quantized_sum, commit_loss * self.beta, quantized_per_stage

    @torch.no_grad()
    def get_codes(self, z):
        residual, all_indices = z, []
        for codebook in self.codebooks:
            flat = residual.reshape(-1, residual.shape[-1])
            distances = flat.pow(2).sum(dim=1, keepdim=True) + codebook.weight.pow(2).sum(dim=1) - 2 * flat @ codebook.weight.T
            indices = distances.argmin(dim=-1)
            all_indices.append(indices.reshape(z.shape[0], z.shape[1]))
            residual = residual - codebook(indices).reshape_as(residual)
        return torch.stack(all_indices, dim=-1)
class TimesFMWithRQVAEPrototypes(nn.Module):
    """Frozen TimesFM backbone with trainable RQ-VAE codebooks."""

    input_patch_size = 32
    d_model = 1280
    context_limit = 16384

    def __init__(self, n_proto=256, n_rq_stages=2, beta=0.25, mask_ratio=0.1):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.n_rq_stages = n_rq_stages
        self.backbone = TimesFM_2p5_200M_torch_module()
        self.residual_quantizer = ResidualQuantizer(
            n_stages=n_rq_stages, n_proto=n_proto, d_model=self.d_model, beta=beta
        )
        self.reconstruction_heads = nn.ModuleList(
            [nn.Linear(self.d_model, self.input_patch_size) for _ in range(n_rq_stages)]
        )
        self.freeze_backbone()

    @classmethod
    def from_pretrained_for_rqvae(
        cls, pretrained_model_path, n_proto=256, n_rq_stages=2,
        beta=0.25, mask_ratio=0.1, device=None,
    ):
        model = cls(n_proto=n_proto, n_rq_stages=n_rq_stages, beta=beta, mask_ratio=mask_ratio)
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

    def _prepare_inputs(self, context):
        if context.ndim != 2:
            raise ValueError(f"context must have shape (batch, time), got {tuple(context.shape)}")
        context = context.float()
        if context.shape[-1] > self.context_limit:
            context = context[..., -self.context_limit:]
        batch_size, original_len = context.shape
        pad_len = (-original_len) % self.input_patch_size
        if pad_len:
            context = F.pad(context, (pad_len, 0))
        valid = torch.ones_like(context, dtype=torch.bool)
        if pad_len:
            valid[:, :pad_len] = False
        safe_context = torch.nan_to_num(context, nan=0.0)
        valid = valid & torch.isfinite(context)
        denom = valid.sum(dim=-1, keepdim=True).clamp_min(1)
        mean = (safe_context * valid).sum(dim=-1, keepdim=True) / denom
        variance = (((safe_context - mean) * valid) ** 2).sum(dim=-1, keepdim=True) / denom
        scale = variance.sqrt().clamp_min(1e-6)
        normalized = (safe_context - mean) / scale
        normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
        return normalized, valid, original_len

    def _extract_hidden(self, normalized, observed_mask):
        patched_values = normalized.reshape(normalized.shape[0], -1, self.input_patch_size)
        # TimesFM convention: True marks masked/unobserved positions.
        patched_masks = (~observed_mask).reshape(
            observed_mask.shape[0], -1, self.input_patch_size
        )
        with torch.no_grad():
            outputs, _ = self.backbone(patched_values, patched_masks)
            _, hidden_states, _, _ = outputs
        return hidden_states.detach()

    def forward(self, context, mask_ratio=None, return_reconstruction=False):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        normalized, valid, original_len = self._prepare_inputs(context)
        target = normalized.detach()
        recovery_mask = (torch.rand_like(normalized) < mask_ratio) & valid
        masked = normalized.masked_fill(recovery_mask, 0.0)
        observed_mask = valid & ~recovery_mask
        hidden_states = self._extract_hidden(masked, observed_mask)

        _, commit_loss, quantized_per_stage = self.residual_quantizer(hidden_states)
        pred_patches = torch.zeros(
            hidden_states.shape[0], hidden_states.shape[1], self.input_patch_size,
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
        for head, quantized in zip(self.reconstruction_heads, quantized_per_stage):
            pred_patches = pred_patches + head(quantized)
        reconstruction = pred_patches.reshape(pred_patches.shape[0], -1)[..., -original_len:]
        target = target[..., -original_len:]
        valid_target = valid[..., -original_len:]
        recon_loss = ((reconstruction - target).pow(2) * valid_target).sum()
        recon_loss = recon_loss / valid_target.sum().clamp_min(1)
        loss = recon_loss + commit_loss
        result = {"loss": loss}
        if return_reconstruction:
            result["reconstruction"] = reconstruction
        return result

    @torch.no_grad()
    def encode_codes(self, context):
        normalized, valid, _ = self._prepare_inputs(context)
        return self.residual_quantizer.get_codes(self._extract_hidden(normalized, valid))

    def save_codebooks(self, output_path):
        state = {
            "format_version": 1,
            "backbone": "timesfm-2.5-200m-pytorch",
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
        for i, codebook in enumerate(self.residual_quantizer.codebooks):
            codebook.weight.data.copy_(state[f"codebook_{i}"].to(codebook.weight.dtype))
        for i, head in enumerate(self.reconstruction_heads):
            head.load_state_dict(state["reconstruction_heads"][f"head_{i}"])
