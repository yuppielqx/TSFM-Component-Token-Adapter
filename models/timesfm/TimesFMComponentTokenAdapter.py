"""TimesFM 2.5 forecasting adapter conditioned on component RQ tokens.

This is intentionally separate from the ChronosBolt adapter implementation.
The TimesFM backbone and trained component codebooks stay frozen; downstream
training updates only the token embeddings, aggregator, and gated fusion.
"""

import os
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .TimesFMPrototype import ResidualQuantizer
from .timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
from .torch import util
from utils.decomposition import select_component


class TimesFMComponentTokenAdapter(nn.Module):
    input_patch_size = 32
    output_patch_size = 128
    d_model = 1280
    context_limit = 16384

    def __init__(
        self,
        component_codebook_paths: Dict[str, str],
        components: Sequence[str] = ("trend", "seasonal", "residual"),
        n_proto: int = 256,
        n_rq_stages: int = 2,
        period: int = 24,
        trend_kernel: int = 25,
        token_dropout: float = 0.1,
        gate_bias_init: float = -2.0,
    ):
        super().__init__()
        self.components = tuple(components)
        self.n_proto = n_proto
        self.n_rq_stages = n_rq_stages
        self.period = period
        self.trend_kernel = trend_kernel
        self.backbone = TimesFM_2p5_200M_torch_module()

        self.component_to_id = {name: index for index, name in enumerate(self.components)}
        self.component_quantizers = nn.ModuleDict()
        self.code_embedding_tables = nn.ModuleDict()
        for component in self.components:
            checkpoint = torch.load(component_codebook_paths[component], map_location="cpu")
            if checkpoint.get("backbone") not in (None, "timesfm-2.5-200m-pytorch"):
                raise ValueError(f"Not a TimesFM codebook: {component_codebook_paths[component]}")
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

        max_patch_positions = self.context_limit // self.input_patch_size + 8
        self.component_embedding = nn.Embedding(len(self.components), self.d_model)
        self.stage_embedding = nn.Embedding(n_rq_stages, self.d_model)
        self.patch_position_embedding = nn.Embedding(max_patch_positions, self.d_model)
        tokens_per_patch = len(self.components) * n_rq_stages
        self.token_dropout = nn.Dropout(token_dropout)
        self.patch_token_aggregator = nn.Sequential(
            nn.Linear(tokens_per_patch * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(token_dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
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
        model = cls(component_codebook_paths=component_codebook_paths, **kwargs)
        weights_path = os.path.join(pretrained_model_path, "model.safetensors")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"TimesFM weights not found: {weights_path}")
        model.backbone.load_state_dict(load_file(weights_path, device="cpu"), strict=True)
        model.freeze_backbone()
        print(f"TimesFM backbone loaded from {pretrained_model_path}")
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

    def _pad_context(self, context):
        if context.ndim != 2:
            raise ValueError(f"context must have shape (batch, time), got {tuple(context.shape)}")
        context = context.float()[..., -self.context_limit:]
        pad_len = (-context.shape[-1]) % self.input_patch_size
        if pad_len:
            context = F.pad(context, (pad_len, 0))
        mask = ~torch.isfinite(context)
        if pad_len:
            mask[:, :pad_len] = True
        return torch.nan_to_num(context, nan=0.0), mask

    def _normalize_patches(self, values, masks):
        patched_values = values.reshape(values.shape[0], -1, self.input_patch_size)
        patched_masks = masks.reshape(masks.shape[0], -1, self.input_patch_size)
        n = torch.zeros(values.shape[0], device=values.device)
        mu = torch.zeros_like(n)
        sigma = torch.zeros_like(n)
        patch_mu, patch_sigma = [], []
        for patch_index in range(patched_values.shape[1]):
            (n, mu, sigma), _ = util.update_running_stats(
                n, mu, sigma, patched_values[:, patch_index], patched_masks[:, patch_index]
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)
        mus = torch.stack(patch_mu, dim=1)
        sigmas = torch.stack(patch_sigma, dim=1)
        normalized = util.revin(patched_values, mus, sigmas, reverse=False)
        normalized = torch.where(patched_masks, 0.0, normalized)
        return normalized, patched_masks, mus, sigmas

    @torch.no_grad()
    def _backbone_hidden(self, normalized_patches, patch_masks):
        (_, hidden_states, _, _), _ = self.backbone(normalized_patches, patch_masks)
        return hidden_states.detach()

    @torch.no_grad()
    def _encode_component_codes(self, values, component):
        values, masks = self._pad_context(values)
        normalized, patch_masks, _, _ = self._normalize_patches(values, masks)
        hidden_states = self._backbone_hidden(normalized, patch_masks)
        return self.component_quantizers[component].get_codes(hidden_states)

    def _component_condition(self, raw_context, num_patches):
        token_embeddings = []
        device = raw_context.device
        patch_ids = torch.arange(num_patches, device=device)
        patch_positions = self.patch_position_embedding(patch_ids).unsqueeze(0)
        for component in self.components:
            component_values = select_component(
                raw_context, component, period=self.period, trend_kernel=self.trend_kernel
            )
            codes = self._encode_component_codes(component_values, component)
            if codes.shape[1] != num_patches:
                raise RuntimeError(
                    f"Component {component} produced {codes.shape[1]} patches; expected {num_patches}"
                )
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

    def _fuse(self, hidden_states, condition):
        projected = self.token_projection(condition)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([hidden_states, condition], dim=-1)))
        return hidden_states + gate * projected

    def forward(self, context, target=None, prediction_length=64):
        if prediction_length <= 0 or prediction_length > self.output_patch_size:
            raise ValueError(
                f"prediction_length must be in [1, {self.output_patch_size}], got {prediction_length}"
            )
        raw_context = context[..., -self.context_limit:]
        values, masks = self._pad_context(raw_context)
        normalized, patch_masks, mus, sigmas = self._normalize_patches(values, masks)
        hidden_states = self._backbone_hidden(normalized, patch_masks)
        condition = self._component_condition(raw_context, hidden_states.shape[1])
        fused_hidden = self._fuse(hidden_states, condition)
        normalized_outputs = self.backbone.output_projection_point(fused_hidden)
        normalized_outputs = normalized_outputs.reshape(
            normalized_outputs.shape[0], normalized_outputs.shape[1],
            self.output_patch_size, self.backbone.q,
        )
        normalized_prediction = normalized_outputs[:, -1, :prediction_length, self.backbone.aridx]
        prediction = util.revin(
            normalized_prediction, mus[:, -1], sigmas[:, -1], reverse=True
        )
        loss = None
        if target is not None:
            target = target[..., :prediction_length]
            valid = torch.isfinite(target)
            safe_target = torch.nan_to_num(target, nan=0.0)
            loss = (((prediction - safe_target) ** 2) * valid).sum() / valid.sum().clamp_min(1)
        return {"loss": loss, "prediction": prediction}

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
