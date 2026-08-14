"""Chronos-2 component-wise RQ-VAE codebook learning."""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from chronos.chronos2 import Chronos2Pipeline


class ResidualQuantizer(nn.Module):
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
        commit_loss = 0.0
        per_stage = []
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
            per_stage.append(quantized.detach())
            residual = residual - quantized.detach()
        return quantized_sum, commit_loss * self.beta, per_stage

    @torch.no_grad()
    def get_codes(self, z):
        residual, all_indices = z, []
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


class Chronos2WithRQVAEPrototypes(nn.Module):
    """Frozen Chronos-2 encoder plus trainable component RQ-VAE codebooks.

    Codebook training always supplies a 512-point context. Chronos-2 retains
    its pretrained maximum context length (8192), but processes the 32 context
    patches generated from these 512 points.
    """

    context_length = 512
    input_patch_size = 16
    d_model = 768

    def __init__(self, backbone, n_proto=256, n_rq_stages=2, beta=0.25, mask_ratio=0.1):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio
        self.n_rq_stages = n_rq_stages
        self.input_patch_size = backbone.chronos_config.input_patch_size
        self.d_model = backbone.config.d_model
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
        pipeline = Chronos2Pipeline.from_pretrained(pretrained_model_path, device_map="cpu")
        model = cls(
            pipeline.model, n_proto=n_proto, n_rq_stages=n_rq_stages,
            beta=beta, mask_ratio=mask_ratio,
        )
        if device is not None:
            model.to(device)
        print(f"Chronos-2 backbone loaded from {pretrained_model_path}")
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

    def _prepare_context(self, context):
        if context.ndim != 2:
            raise ValueError(f"context must have shape (batch, time), got {tuple(context.shape)}")
        if context.shape[-1] != self.context_length:
            raise ValueError(
                f"Chronos-2 codebook training requires {self.context_length} points, got {context.shape[-1]}"
            )
        context = context.float()
        valid = torch.isfinite(context)
        clean = torch.nan_to_num(context, nan=0.0)
        return clean, valid

    def _target_patches(self, context, valid):
        patched, _, _ = self.backbone._prepare_patched_context(context, valid)
        patch_size = self.input_patch_size
        return patched[..., patch_size : 2 * patch_size]

    def _extract_hidden(self, context, valid):
        with torch.no_grad():
            encoded, _, _, num_context_patches = self.backbone.encode(
                context=context,
                context_mask=valid,
                num_output_patches=1,
            )
        return encoded.last_hidden_state[:, :num_context_patches].detach()

    def forward(self, context, mask_ratio=None, return_reconstruction=False):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        clean, valid = self._prepare_context(context)
        target_patches = self._target_patches(clean, valid).detach()
        recovery_mask = (torch.rand_like(clean) < mask_ratio) & valid
        masked_context = clean.masked_fill(recovery_mask, 0.0)
        hidden_states = self._extract_hidden(masked_context, valid)

        _, commit_loss, per_stage = self.residual_quantizer(hidden_states)
        reconstruction = torch.zeros_like(target_patches)
        for head, quantized in zip(self.reconstruction_heads, per_stage):
            reconstruction = reconstruction + head(quantized)

        patch_valid = valid.reshape(valid.shape[0], -1, self.input_patch_size).to(reconstruction.dtype)
        recon_loss = ((reconstruction - target_patches).pow(2) * patch_valid).sum()
        recon_loss = recon_loss / patch_valid.sum().clamp_min(1)
        result = {"loss": recon_loss + commit_loss}
        if return_reconstruction:
            result["reconstruction"] = reconstruction.reshape(reconstruction.shape[0], -1)
        return result

    @torch.no_grad()
    def encode_codes(self, context):
        clean, valid = self._prepare_context(context)
        return self.residual_quantizer.get_codes(self._extract_hidden(clean, valid))

    def save_codebooks(self, output_path):
        state = {
            "format_version": 1,
            "backbone": "amazon/chronos-2",
            "context_length": self.context_length,
            "input_patch_size": self.input_patch_size,
            "d_model": self.d_model,
            "n_rq_stages": self.n_rq_stages,
            **{f"codebook_{i}": codebook.weight.detach().cpu().float()
               for i, codebook in enumerate(self.residual_quantizer.codebooks)},
            "reconstruction_heads": {f"head_{i}": head.state_dict()
                                     for i, head in enumerate(self.reconstruction_heads)},
        }
        torch.save(state, output_path)
        print(f"Saved {self.n_rq_stages} Chronos-2 codebooks + heads to {output_path}")

    def load_trained_codebook(self, codebook_path):
        state = torch.load(codebook_path, map_location="cpu")
        if state.get("backbone") not in (None, "amazon/chronos-2"):
            raise ValueError(f"Not a Chronos-2 codebook checkpoint: {codebook_path}")
        for i, codebook in enumerate(self.residual_quantizer.codebooks):
            codebook.weight.data.copy_(state[f"codebook_{i}"].to(codebook.weight.dtype))
        for i, head in enumerate(self.reconstruction_heads):
            head.load_state_dict(state["reconstruction_heads"][f"head_{i}"])
