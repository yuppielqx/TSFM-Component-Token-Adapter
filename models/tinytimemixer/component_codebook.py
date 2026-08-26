"""RQ-VAE codebooks trained on frozen Tiny Time Mixer patch embeddings."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQuantizer(nn.Module):
    """Straight-through residual vector quantizer.

    Each stage quantizes the residual left by preceding stages.  Codebook
    embeddings receive gradients through the commitment loss while the frozen
    TTM encoder is never part of the optimization graph.
    """

    def __init__(self, d_model: int, n_proto: int = 256, n_stages: int = 2, beta: float = 0.25):
        super().__init__()
        if min(d_model, n_proto, n_stages) <= 0:
            raise ValueError("d_model, n_proto, and n_stages must be positive")
        self.beta = beta
        self.codebooks = nn.ModuleList([nn.Embedding(n_proto, d_model) for _ in range(n_stages)])
        for codebook in self.codebooks:
            nn.init.trunc_normal_(codebook.weight, std=0.02)

    @staticmethod
    def _nearest(residual: torch.Tensor, codebook: nn.Embedding) -> Tuple[torch.Tensor, torch.Tensor]:
        flat = residual.flatten(0, -2)
        weight = codebook.weight
        distance = flat.square().sum(-1, keepdim=True) + weight.square().sum(-1) - 2 * flat @ weight.T
        codes = distance.argmin(dim=-1)
        return codebook(codes).view_as(residual), codes.view(residual.shape[:-1])

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        residual = embeddings
        quantized_sum = torch.zeros_like(embeddings)
        commitment = embeddings.new_zeros(())
        stage_quantized: List[torch.Tensor] = []
        for codebook in self.codebooks:
            quantized, indices = self._nearest(residual, codebook)
            # First term trains codebook entries; second term is the encoder
            # commitment objective (zero-gradient here because encoder is frozen).
            commitment = commitment + F.mse_loss(quantized, residual.detach()) + F.mse_loss(quantized.detach(), residual)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized.detach()
            stage_quantized.append(quantized)
        quantized_st = embeddings + (quantized_sum - embeddings).detach()
        return quantized_st, self.beta * commitment, stage_quantized

    @torch.no_grad()
    def get_codes(self, embeddings: torch.Tensor) -> torch.Tensor:
        residual, stages = embeddings, []
        for codebook in self.codebooks:
            quantized, indices = self._nearest(residual, codebook)
            stages.append(indices)
            residual = residual - quantized
        return torch.stack(stages, dim=-1)


class TTMComponentCodebook(nn.Module):
    """Learn an RQ-VAE dictionary for one decomposed component.

    ``backbone_hidden`` must be the output of a frozen TTM backbone with shape
    ``[batch, channels, patches, d_model]``.  Reconstruction is performed in
    TTM's normalized patch space, which keeps scale changes from dominating the
    code assignments.
    """

    FORMAT_VERSION = 1

    def __init__(self, d_model: int, patch_length: int, n_proto: int = 256, n_stages: int = 2, beta: float = 0.25):
        super().__init__()
        self.d_model, self.patch_length = d_model, patch_length
        self.rq = ResidualQuantizer(d_model, n_proto=n_proto, n_stages=n_stages, beta=beta)
        self.reconstruction_heads = nn.ModuleList([nn.Linear(d_model, patch_length) for _ in range(n_stages)])

    def forward(self, backbone_hidden: torch.Tensor, target_patches: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, commitment_loss, stage_quantized = self.rq(backbone_hidden)
        reconstruction = sum(head(value) for head, value in zip(self.reconstruction_heads, stage_quantized))
        squared_error = (reconstruction - target_patches).square() * valid_mask.to(reconstruction.dtype)
        reconstruction_loss = squared_error.sum() / valid_mask.sum().clamp_min(1)
        return {"loss": reconstruction_loss + commitment_loss, "reconstruction_loss": reconstruction_loss,
                "commitment_loss": commitment_loss, "reconstruction": reconstruction}

    @torch.no_grad()
    def checkpoint(self, *, model_id: str, context_length: int, component: str, **metadata) -> Dict[str, object]:
        return {
            "format_version": self.FORMAT_VERSION,
            "backbone": "tinytimemixer",
            "model_id": model_id,
            "component": component,
            "context_length": context_length,
            "d_model": self.d_model,
            "patch_length": self.patch_length,
            "n_proto": self.rq.codebooks[0].num_embeddings,
            "n_rq_stages": len(self.rq.codebooks),
            **{f"codebook_{stage}": table.weight.detach().cpu().float() for stage, table in enumerate(self.rq.codebooks)},
            "reconstruction_heads": {f"head_{stage}": head.state_dict() for stage, head in enumerate(self.reconstruction_heads)},
            **metadata,
        }
