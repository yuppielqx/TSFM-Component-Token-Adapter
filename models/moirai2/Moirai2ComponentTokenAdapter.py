"""Discrete trend/seasonal token adapter for a frozen Moirai-2 backbone."""

import math
import json
import os
import sys
import warnings
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "models", "moirai2"))

from uni2ts.common.torch_util import packed_causal_attention_mask
from uni2ts.model.moirai2.forecast import Moirai2Forecast
from uni2ts.model.moirai2.module import Moirai2Module
from utils.decomposition import select_component


class Moirai2ComponentTokenAdapter(nn.Module):
    """Inject frozen RQ-VAE component codes into Moirai-2 future patch states."""

    def __init__(
        self,
        backbone: Moirai2Module,
        component_codebook_paths: Dict[str, str],
        components: Sequence[str] = ("trend", "seasonal"),
        context_length: int = 512,
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
        self.context_length = context_length
        self.n_proto = n_proto
        self.n_rq_stages = n_rq_stages
        self.period = period
        self.trend_kernel = trend_kernel
        self.patch_size = backbone.patch_size
        self.d_model = backbone.d_model
        self.quantile_levels = tuple(backbone.quantile_levels)
        if context_length % self.patch_size:
            raise ValueError("context_length must be divisible by Moirai-2 patch_size")
        if self.d_model % cross_attention_heads:
            raise ValueError("d_model must be divisible by cross_attention_heads")

        self.component_to_id = {name: index for index, name in enumerate(self.components)}
        self.codebooks = nn.ModuleDict()
        self.code_embedding_tables = nn.ModuleDict()
        for component in self.components:
            codebook_path = component_codebook_paths[component]
            metadata_path = codebook_path[:-3] + ".meta.json" if codebook_path.endswith(".pt") else codebook_path + ".meta.json"
            if os.path.isfile(metadata_path):
                with open(metadata_path) as handle:
                    metadata = json.load(handle)
                trained_period = metadata.get("period")
                if trained_period is not None and int(trained_period) != self.period:
                    warnings.warn(
                        f"Using shared pretrained {component} codebook across periods: "
                        f"adapter period={self.period}, codebook training period={trained_period} "
                        f"({metadata_path})",
                        RuntimeWarning,
                    )
            checkpoint = torch.load(codebook_path, map_location="cpu")
            if checkpoint.get("backbone") not in (None, "moirai-2.0-R-small"):
                raise ValueError(f"Not a Moirai-2 codebook: {component_codebook_paths[component]}")
            if checkpoint.get("d_model", self.d_model) != self.d_model:
                raise ValueError("Codebook d_model does not match backbone")
            if checkpoint.get("patch_size", self.patch_size) != self.patch_size:
                raise ValueError("Codebook patch_size does not match backbone")
            stages, tables = [], []
            for stage in range(n_rq_stages):
                weight = checkpoint[f"codebook_{stage}"].float()
                if tuple(weight.shape) != (n_proto, self.d_model):
                    raise ValueError(f"Unexpected codebook shape {tuple(weight.shape)}")
                embedding = nn.Embedding(n_proto, self.d_model)
                embedding.weight.data.copy_(weight)
                embedding.weight.requires_grad_(False)
                stages.append(embedding)
                # Preserve the RQ-VAE prototype geometry at initialization.
                # The previous implementation used a random lookup table, so
                # a discrete code no longer represented its learned prototype.
                token_embedding = nn.Embedding(n_proto, self.d_model)
                token_embedding.weight.data.copy_(weight)
                tables.append(token_embedding)
            self.codebooks[component] = nn.ModuleList(stages)
            self.code_embedding_tables[component] = nn.ModuleList(tables)

        patches = context_length // self.patch_size
        self.component_embedding = nn.Embedding(len(self.components), self.d_model)
        self.stage_embedding = nn.Embedding(n_rq_stages, self.d_model)
        self.patch_position_embedding = nn.Embedding(patches, self.d_model)
        self.token_dropout = nn.Dropout(token_dropout)
        self.patch_token_aggregator = nn.Sequential(
            nn.Linear(len(self.components) * n_rq_stages * self.d_model, self.d_model),
            nn.GELU(), nn.Dropout(token_dropout), nn.Linear(self.d_model, self.d_model), nn.LayerNorm(self.d_model),
        )
        self.future_to_component_attention = nn.MultiheadAttention(
            self.d_model, cross_attention_heads, dropout=token_dropout, batch_first=True
        )
        self.token_projection = nn.Linear(self.d_model, self.d_model)
        # Start as an exact identity adapter. This keeps the frozen backbone's
        # forecast unchanged before downstream optimization learns an update.
        nn.init.zeros_(self.token_projection.weight)
        nn.init.zeros_(self.token_projection.bias)
        self.fusion_gate = nn.Linear(2 * self.d_model, self.d_model)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, gate_bias_init)
        self.adapter_enabled = True
        self._skip_encoder_injection = False
        self._pending_injection = None
        self.freeze_backbone()

    @classmethod
    def from_pretrained_component_adapter(cls, model_id, component_codebook_paths, **kwargs):
        backbone = Moirai2Module.from_pretrained(model_id)
        print(f"Moirai-2 backbone loaded: {model_id}")
        return cls(backbone, component_codebook_paths, **kwargs)

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.codebooks.eval()
        return self

    def print_trainable_params(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        print(f"Parameters: total={total:,}, trainable={trainable:,}")

    def set_adapter_enabled(self, enabled: bool):
        """Enable or bypass adapter injection without changing any weights."""
        self.adapter_enabled = bool(enabled)

    def _prepare_context(self, context):
        if context.ndim != 2 or context.shape[-1] < self.context_length:
            raise ValueError(f"context must be (batch, length >= {self.context_length}), got {tuple(context.shape)}")
        if context.shape[-1] % self.patch_size:
            raise ValueError("context length must be divisible by Moirai-2 patch_size")
        return torch.nan_to_num(context.float(), nan=0.0), torch.isfinite(context)

    def _metadata(self, batch_size, tokens, device):
        sample_id = torch.zeros((batch_size, tokens), dtype=torch.long, device=device)
        time_id = torch.arange(tokens, device=device).unsqueeze(0).expand(batch_size, -1)
        var_id = torch.zeros((batch_size, tokens), dtype=torch.long, device=device)
        return sample_id, time_id, var_id

    @torch.no_grad()
    def _encode_component_hidden(self, component_values):
        values, observed = self._prepare_context(component_values)
        batch_size = values.shape[0]
        tokens = self.context_length // self.patch_size
        target = values.unsqueeze(-1).reshape(batch_size, tokens, self.patch_size)
        mask = observed.unsqueeze(-1).reshape_as(target)
        sample_id, time_id, var_id = self._metadata(batch_size, tokens, values.device)
        loc, scale = self.backbone.scaler(target, mask, sample_id, var_id)
        inputs = self.backbone.in_proj(torch.cat([(target - loc) / scale, mask.float()], dim=-1))
        return self.backbone.encoder(
            inputs, packed_causal_attention_mask(sample_id, time_id), time_id=time_id, var_id=var_id
        ).detach()

    @torch.no_grad()
    def _component_codes(self, raw_context, component):
        # Codebooks were trained on fixed 512-step windows. During official
        # autoregressive inference, the backbone context grows as future
        # patches are assigned, while component conditioning uses the latest
        # fixed-length window.
        raw_context = raw_context[..., -self.context_length:]
        hidden = self._encode_component_hidden(select_component(
            raw_context, component, period=self.period, trend_kernel=self.trend_kernel
        ))
        residual, codes = hidden, []
        for codebook in self.codebooks[component]:
            flat = residual.reshape(-1, self.d_model)
            distance = flat.pow(2).sum(1, keepdim=True) + codebook.weight.pow(2).sum(1) - 2 * flat @ codebook.weight.T
            indices = distance.argmin(-1).reshape(residual.shape[:-1])
            quantized = codebook(indices)
            codes.append(indices)
            residual = residual - quantized
        return codes

    def _component_condition(self, raw_context):
        patches = self.context_length // self.patch_size
        device = raw_context.device
        position = self.patch_position_embedding(torch.arange(patches, device=device)).unsqueeze(0)
        tokens = []
        for component in self.components:
            codes = self._component_codes(raw_context, component)
            component_embed = self.component_embedding.weight[self.component_to_id[component]].view(1, 1, -1)
            for stage, indices in enumerate(codes):
                stage_embed = self.stage_embedding.weight[stage].view(1, 1, -1)
                tokens.append(self.code_embedding_tables[component][stage](indices) + component_embed + stage_embed + position)
        return self.patch_token_aggregator(self.token_dropout(torch.cat(tokens, dim=-1)))

    def _prepare_official_injection(self, module, args):
        """Prepare component conditioning from an official module invocation.

        ``Moirai2Forecast`` owns future-token construction and recursive
        assignment.  This pre-hook only identifies the latest known source
        token and reconstructs its most recent fixed-length raw context.
        """
        target, prediction_mask = args[0], args[5]
        if target.shape[-1] != self.patch_size:
            raise RuntimeError(f"Unexpected official target shape: {tuple(target.shape)}")
        token_length = target.shape[-2]
        flat_target = target.reshape(-1, token_length, self.patch_size)
        flat_prediction_mask = prediction_mask.reshape(-1, token_length)
        known_tokens = (~flat_prediction_mask).sum(dim=-1)
        context_tokens = self.context_length // self.patch_size
        if torch.any(known_tokens < context_tokens):
            raise RuntimeError("Official Moirai2 call has fewer known tokens than adapter context")

        flat_values = flat_target.reshape(flat_target.shape[0], -1)
        context_end = known_tokens * self.patch_size
        offsets = torch.arange(self.context_length, device=target.device).unsqueeze(0)
        gather_index = context_end.unsqueeze(1) - self.context_length + offsets
        raw_context = flat_values.gather(1, gather_index)

        self._pending_injection = None
        self._skip_encoder_injection = True
        try:
            condition = self._component_condition(raw_context)
        finally:
            self._skip_encoder_injection = False
        self._pending_injection = (condition, known_tokens - 1)

    def _inject_official_hidden(self, module, args, hidden):
        """Inject into exactly the source states decoded by official Moirai2."""
        if self._skip_encoder_injection or self._pending_injection is None:
            return hidden
        condition, source_index = self._pending_injection
        self._pending_injection = None
        token_length = hidden.shape[-2]
        flat_hidden = hidden.reshape(-1, token_length, self.d_model)
        if flat_hidden.shape[0] != condition.shape[0]:
            raise RuntimeError(
                f"Adapter batch mismatch: hidden={flat_hidden.shape[0]} condition={condition.shape[0]}"
            )
        rows = torch.arange(flat_hidden.shape[0], device=hidden.device)
        source_hidden = flat_hidden[rows, source_index].unsqueeze(1)
        attended, _ = self.future_to_component_attention(
            source_hidden, condition, condition, need_weights=False
        )
        gate = torch.sigmoid(self.fusion_gate(torch.cat([source_hidden, attended], dim=-1)))
        fused = source_hidden + gate * self.token_projection(attended)
        updated = flat_hidden.clone()
        updated[rows, source_index] = fused.squeeze(1)
        return updated.reshape_as(hidden)

    def _official_predictions(self, context, prediction_length):
        if context.ndim != 2 or context.shape[-1] != self.context_length:
            raise ValueError(
                f"official context must be (batch, {self.context_length}), got {tuple(context.shape)}"
            )
        observed = torch.isfinite(context)
        clean_context = torch.nan_to_num(context.float(), nan=0.0)
        forecast = Moirai2Forecast(
            module=self.backbone,
            prediction_length=prediction_length,
            context_length=self.context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(context.device)
        pre_handle = encoder_handle = None
        try:
            if self.adapter_enabled:
                pre_handle = self.backbone.register_forward_pre_hook(self._prepare_official_injection)
                encoder_handle = self.backbone.encoder.register_forward_hook(self._inject_official_hidden)
            return forecast(
                past_target=clean_context.unsqueeze(-1),
                past_observed_target=observed.unsqueeze(-1),
                past_is_pad=torch.zeros_like(clean_context, dtype=torch.bool),
            )
        finally:
            self._pending_injection = None
            self._skip_encoder_injection = False
            if encoder_handle is not None:
                encoder_handle.remove()
            if pre_handle is not None:
                pre_handle.remove()

    def forward(self, context, target=None, prediction_length=64):
        if prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        prediction = self._official_predictions(context, prediction_length)
        loss = None
        if target is not None:
            target = target[..., :prediction_length]
            valid = torch.isfinite(target)
            error = torch.nan_to_num(target, nan=0.0).unsqueeze(1) - prediction
            levels = prediction.new_tensor(self.quantile_levels).view(1, -1, 1)
            pinball = torch.maximum(levels * error, (levels - 1.0) * error)
            loss = (pinball * valid.unsqueeze(1)).sum() / (valid.sum().clamp_min(1) * len(self.quantile_levels))
        return {"loss": loss, "quantile_predictions": prediction}

    def central_prediction(self, output):
        index = min(range(len(self.quantile_levels)), key=lambda value: abs(self.quantile_levels[value] - 0.5))
        return output["quantile_predictions"][:, index]

    def adapter_state_dict(self):
        return {name: tensor.detach().cpu() for name, tensor in self.state_dict().items()
                if not name.startswith(("backbone.", "codebooks."))}

    def load_adapter_state_dict(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [name for name in missing if not name.startswith(("backbone.", "codebooks."))]
        if missing or unexpected:
            raise RuntimeError(f"Adapter load mismatch: missing={missing}, unexpected={unexpected}")
