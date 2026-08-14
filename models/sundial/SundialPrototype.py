"""Sundial + component-wise RQ-VAE prototype learning."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_sundial import SundialConfig
from .modeling_sundial import SundialForPrediction


class ResidualQuantizer(nn.Module):
    """Residual vector quantizer with straight-through gradients."""

    def __init__(self, n_stages=2, n_proto=256, d_model=768, beta=0.25):
        super().__init__()
        self.beta = beta
        self.codebooks = nn.ModuleList(
            [nn.Embedding(n_proto, d_model) for _ in range(n_stages)]
        )
        for codebook in self.codebooks:
            nn.init.trunc_normal_(codebook.weight, std=0.02)

    def forward(self, z):
        residual = z
        quantized_sum = torch.zeros_like(z)
        commit_loss = z.new_zeros(())
        quantized_per_stage = []
        for codebook in self.codebooks:
            flat = residual.reshape(-1, residual.shape[-1])
            distances = (
                flat.pow(2).sum(dim=1, keepdim=True)
                + codebook.weight.pow(2).sum(dim=1)
                - 2 * flat @ codebook.weight.T
            )
            indices = distances.argmin(dim=-1)
            quantized = codebook(indices).reshape_as(residual)
            commit_loss = commit_loss + F.mse_loss(residual.detach(), quantized)
            quantized_sum = quantized_sum + residual + (quantized - residual).detach()
            quantized_per_stage.append(quantized.detach())
            residual = residual - quantized.detach()
        return quantized_sum, self.beta * commit_loss, quantized_per_stage

    @torch.no_grad()
    def get_codes(self, z):
        residual = z
        all_indices = []
        for codebook in self.codebooks:
            flat = residual.reshape(-1, residual.shape[-1])
            distances = (
                flat.pow(2).sum(dim=1, keepdim=True)
                + codebook.weight.pow(2).sum(dim=1)
                - 2 * flat @ codebook.weight.T
            )
            indices = distances.argmin(dim=-1)
            all_indices.append(indices.reshape(z.shape[0], z.shape[1]))
            residual = residual - codebook(indices).reshape_as(residual)
        return torch.stack(all_indices, dim=-1)


class SundialWithRQVAEPrototypes(nn.Module):
    """Frozen Sundial backbone with trainable component codebooks and heads."""

    backbone_name = "thuml/sundial-base-128m"

    def __init__(self, config, n_proto=256, n_rq_stages=2, beta=0.25, mask_ratio=0.1):
        super().__init__()
        self.config = config
        self.input_patch_size = config.input_token_len
        self.d_model = config.hidden_size
        self.context_limit = config.max_position_embeddings * config.input_token_len
        self.mask_ratio = mask_ratio
        self.n_rq_stages = n_rq_stages
        self.backbone = SundialForPrediction(config)
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
        config = SundialConfig.from_pretrained(pretrained_model_path, local_files_only=True)
        model = cls(config, n_proto, n_rq_stages, beta, mask_ratio)
        model.backbone = SundialForPrediction.from_pretrained(
            pretrained_model_path,
            config=config,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        model.freeze_backbone()
        if device is not None:
            model.to(device)
        print(f"Sundial backbone loaded from {pretrained_model_path}")
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
        original_len = context.shape[-1]
        pad_len = (-original_len) % self.input_patch_size
        valid = torch.isfinite(context)
        safe = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        denom = valid.sum(dim=-1, keepdim=True).clamp_min(1)
        mean = (safe * valid).sum(dim=-1, keepdim=True) / denom
        variance = (((safe - mean) * valid) ** 2).sum(dim=-1, keepdim=True) / denom
        scale = variance.sqrt().clamp_min(1e-6)
        normalized = torch.where(valid, (safe - mean) / scale, torch.zeros_like(safe))
        if pad_len:
            normalized = F.pad(normalized, (pad_len, 0))
            valid = F.pad(valid, (pad_len, 0), value=False)
        return normalized, valid, original_len

    def _extract_hidden(self, normalized):
        with torch.no_grad():
            outputs = self.backbone.model(
                input_ids=normalized,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        return outputs.last_hidden_state.detach()

    def forward(self, context, mask_ratio=None, return_reconstruction=False):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        normalized, valid, original_len = self._prepare_inputs(context)
        target = normalized.detach()
        recovery_mask = (torch.rand_like(normalized) < mask_ratio) & valid
        masked = normalized.masked_fill(recovery_mask, 0.0)
        hidden_states = self._extract_hidden(masked)

        _, commit_loss, quantized_per_stage = self.residual_quantizer(hidden_states)
        pred_patches = hidden_states.new_zeros(
            hidden_states.shape[0], hidden_states.shape[1], self.input_patch_size
        )
        for head, quantized in zip(self.reconstruction_heads, quantized_per_stage):
            pred_patches = pred_patches + head(quantized)
        reconstruction = pred_patches.reshape(pred_patches.shape[0], -1)[..., -original_len:]
        target = target[..., -original_len:]
        valid_target = valid[..., -original_len:]
        recon_loss = ((reconstruction - target).pow(2) * valid_target).sum()
        recon_loss = recon_loss / valid_target.sum().clamp_min(1)
        result = {"loss": recon_loss + commit_loss, "reconstruction_loss": recon_loss,
                  "commitment_loss": commit_loss}
        if return_reconstruction:
            result["reconstruction"] = reconstruction
        return result

    @torch.no_grad()
    def encode_codes(self, context):
        normalized, _, _ = self._prepare_inputs(context)
        return self.residual_quantizer.get_codes(self._extract_hidden(normalized))

    def save_codebooks(self, output_path):
        state = {
            "format_version": 1,
            "backbone": self.backbone_name,
            "input_patch_size": self.input_patch_size,
            "d_model": self.d_model,
            "n_rq_stages": self.n_rq_stages,
            **{f"codebook_{i}": cb.weight.detach().cpu().float()
               for i, cb in enumerate(self.residual_quantizer.codebooks)},
            "reconstruction_heads": {
                f"head_{i}": head.state_dict() for i, head in enumerate(self.reconstruction_heads)
            },
        }
        torch.save(state, output_path)
        print(f"Saved {self.n_rq_stages} Sundial codebooks + heads to {output_path}")

    def load_trained_codebook(self, codebook_path):
        state = torch.load(codebook_path, map_location="cpu")
        if state.get("backbone") not in (None, self.backbone_name):
            raise ValueError(f"Not a Sundial codebook checkpoint: {codebook_path}")
        if state.get("d_model") not in (None, self.d_model):
            raise ValueError(f"Codebook hidden size does not match Sundial: {state.get('d_model')}")
        for index, codebook in enumerate(self.residual_quantizer.codebooks):
            codebook.weight.data.copy_(state[f"codebook_{index}"].to(codebook.weight.dtype))
        for index, head in enumerate(self.reconstruction_heads):
            head.load_state_dict(state["reconstruction_heads"][f"head_{index}"])
