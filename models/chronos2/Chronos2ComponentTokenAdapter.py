"""Chronos-2 adapter conditioned on frozen component RQ tokens."""

import math
import os
import sys
from typing import Dict, Sequence

import torch
import torch.nn as nn
from einops import rearrange

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from chronos.chronos2 import Chronos2Pipeline
from .Chronos2Prototype import ResidualQuantizer
from utils.decomposition import select_component


class Chronos2ComponentTokenAdapter(nn.Module):
    """Fuse component tokens into Chronos-2 future-patch hidden states.

    Chronos-2 predicts from dedicated future tokens, so the adapter uses
    cross-attention from those future hidden states to component patch tokens.
    The backbone and RQ codebooks remain frozen.
    """

    context_length = 512

    def __init__(
        self,
        backbone,
        component_codebook_paths: Dict[str, str],
        components: Sequence[str] = ("trend", "seasonal"),
        n_proto: int = 256,
        n_rq_stages: int = 2,
        period: int = 24,
        trend_kernel: int = 25,
        token_dropout: float = 0.1,
        gate_bias_init: float = -2.0,
        cross_attention_heads: int = 8,
    ):
        super().__init__()
        self.backbone = backbone
        self.components = tuple(components)
        self.n_proto = n_proto
        self.n_rq_stages = n_rq_stages
        self.period = period
        self.trend_kernel = trend_kernel
        self.input_patch_size = backbone.chronos_config.input_patch_size
        self.output_patch_size = backbone.chronos_config.output_patch_size
        self.max_output_patches = backbone.chronos_config.max_output_patches
        self.d_model = backbone.config.d_model
        if self.d_model % cross_attention_heads:
            raise ValueError("d_model must be divisible by cross_attention_heads")

        self.component_to_id = {name: index for index, name in enumerate(self.components)}
        self.component_quantizers = nn.ModuleDict()
        self.code_embedding_tables = nn.ModuleDict()
        for component in self.components:
            checkpoint = torch.load(component_codebook_paths[component], map_location="cpu")
            if checkpoint.get("backbone") not in (None, "amazon/chronos-2"):
                raise ValueError(f"Not a Chronos-2 codebook: {component_codebook_paths[component]}")
            if checkpoint.get("context_length", self.context_length) != self.context_length:
                raise ValueError(f"Codebook context length must be {self.context_length}")
            quantizer = ResidualQuantizer(
                n_stages=n_rq_stages, n_proto=n_proto, d_model=self.d_model
            )
            stage_tables = nn.ModuleList()
            for stage in range(n_rq_stages):
                weight = checkpoint[f"codebook_{stage}"].float()
                if tuple(weight.shape) != (n_proto, self.d_model):
                    raise ValueError(
                        f"Unexpected {component} stage {stage} shape {tuple(weight.shape)}"
                    )
                quantizer.codebooks[stage].weight.data.copy_(weight)
                embedding = nn.Embedding(n_proto, self.d_model)
                embedding.weight.data.copy_(weight)
                stage_tables.append(embedding)
            for parameter in quantizer.parameters():
                parameter.requires_grad = False
            self.component_quantizers[component] = quantizer
            self.code_embedding_tables[component] = stage_tables

        num_context_patches = self.context_length // self.input_patch_size
        self.component_embedding = nn.Embedding(len(self.components), self.d_model)
        self.stage_embedding = nn.Embedding(n_rq_stages, self.d_model)
        self.patch_position_embedding = nn.Embedding(num_context_patches, self.d_model)
        tokens_per_patch = len(self.components) * n_rq_stages
        self.token_dropout = nn.Dropout(token_dropout)
        self.patch_token_aggregator = nn.Sequential(
            nn.Linear(tokens_per_patch * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(token_dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.future_to_component_attention = nn.MultiheadAttention(
            self.d_model, cross_attention_heads, dropout=token_dropout, batch_first=True
        )
        self.token_projection = nn.Linear(self.d_model, self.d_model)
        self.fusion_gate = nn.Linear(2 * self.d_model, self.d_model)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, gate_bias_init)
        self.freeze_backbone()

    @classmethod
    def from_pretrained_component_adapter(
        cls, pretrained_model_path: str, component_codebook_paths: Dict[str, str], **kwargs
    ):
        pipeline = Chronos2Pipeline.from_pretrained(pretrained_model_path, device_map="cpu")
        model = cls(
            pipeline.model,
            component_codebook_paths=component_codebook_paths,
            **kwargs,
        )
        print(f"Chronos-2 backbone loaded from {pretrained_model_path}")
        return model

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.component_quantizers.eval()
        return self

    def print_trainable_params(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        print(f"Parameters: total={total:,}, trainable={trainable:,}")

    def _prepare_context(self, context):
        if context.ndim != 2 or context.shape[-1] != self.context_length:
            raise ValueError(
                f"context must have shape (batch, {self.context_length}), got {tuple(context.shape)}"
            )
        context = context.float()
        valid = torch.isfinite(context)
        return torch.nan_to_num(context, nan=0.0), valid

    @torch.no_grad()
    def _encode_component_codes(self, values, component):
        clean, valid = self._prepare_context(values)
        encoded, _, _, num_context_patches = self.backbone.encode(
            context=clean,
            context_mask=valid,
            num_output_patches=1,
        )
        hidden = encoded.last_hidden_state[:, :num_context_patches].detach()
        return self.component_quantizers[component].get_codes(hidden)

    def _component_condition(self, raw_context):
        num_patches = self.context_length // self.input_patch_size
        device = raw_context.device
        patch_ids = torch.arange(num_patches, device=device)
        patch_positions = self.patch_position_embedding(patch_ids).unsqueeze(0)
        token_embeddings = []
        for component in self.components:
            values = select_component(
                raw_context,
                component,
                period=self.period,
                trend_kernel=self.trend_kernel,
            )
            codes = self._encode_component_codes(values, component)
            component_ids = torch.full(
                (num_patches,), self.component_to_id[component], device=device, dtype=torch.long
            )
            component_positions = self.component_embedding(component_ids).unsqueeze(0)
            for stage in range(self.n_rq_stages):
                stage_ids = torch.full((num_patches,), stage, device=device, dtype=torch.long)
                stage_positions = self.stage_embedding(stage_ids).unsqueeze(0)
                code_embeddings = self.code_embedding_tables[component][stage](codes[..., stage])
                token_embeddings.append(
                    code_embeddings + component_positions + stage_positions + patch_positions
                )
        condition = self.token_dropout(torch.cat(token_embeddings, dim=-1))
        return self.patch_token_aggregator(condition)

    def forward(self, context, target=None, prediction_length=64):
        max_prediction_length = self.max_output_patches * self.output_patch_size
        if prediction_length <= 0 or prediction_length > max_prediction_length:
            raise ValueError(
                f"prediction_length must be in [1, {max_prediction_length}], got {prediction_length}"
            )
        clean, valid = self._prepare_context(context)
        num_output_patches = math.ceil(prediction_length / self.output_patch_size)
        with torch.no_grad():
            encoded, loc_scale, patched_future_mask, _ = self.backbone.encode(
                context=clean,
                context_mask=valid,
                num_output_patches=num_output_patches,
                future_target=target,
            )
            hidden_states = encoded.last_hidden_state.detach()
        future_hidden = hidden_states[:, -num_output_patches:]
        condition = self._component_condition(clean)
        attended, _ = self.future_to_component_attention(
            query=future_hidden, key=condition, value=condition, need_weights=False
        )
        projected = self.token_projection(attended)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([future_hidden, attended], dim=-1)))
        fused_future = future_hidden + gate * projected

        normalized_quantiles = self.backbone.output_patch_embedding(fused_future)
        normalized_quantiles = rearrange(
            normalized_quantiles,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.backbone.num_quantiles,
            p=self.output_patch_size,
        )
        loss = None
        if target is not None:
            loss = self.backbone._compute_loss(
                quantile_preds=normalized_quantiles,
                future_target=target,
                future_target_mask=None,
                patched_future_covariates_mask=patched_future_mask,
                loc_scale=loc_scale,
                num_output_patches=num_output_patches,
            )
        flat = rearrange(normalized_quantiles, "b q h -> b (q h)")
        flat = self.backbone.instance_norm.inverse(flat, loc_scale)
        quantile_predictions = rearrange(
            flat,
            "b (q h) -> b q h",
            q=self.backbone.num_quantiles,
            h=num_output_patches * self.output_patch_size,
        )[..., :prediction_length]
        return {"loss": loss, "quantile_predictions": quantile_predictions}

    def central_prediction(self, output):
        central_index = int(torch.abs(self.backbone.quantiles - 0.5).argmin().item())
        return output["quantile_predictions"][:, central_index]

    def adapter_state_dict(self):
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if not name.startswith(("backbone.", "component_quantizers."))
        }

    def load_adapter_state_dict(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad_missing = [
            name for name in missing
            if not name.startswith(("backbone.", "component_quantizers."))
        ]
        if bad_missing or unexpected:
            raise RuntimeError(f"Adapter load mismatch: missing={bad_missing}, unexpected={unexpected}")
