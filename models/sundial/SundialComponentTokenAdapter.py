"""Sundial adapter conditioned on frozen trend/seasonal RQ tokens."""

from typing import Dict, Sequence

import torch
import torch.nn as nn

from .SundialPrototype import ResidualQuantizer
from .configuration_sundial import SundialConfig
from .modeling_sundial import SundialForPrediction
from utils.decomposition import select_component


class SundialComponentTokenAdapter(nn.Module):
    """Fuse component-code tokens into Sundial's final forecasting condition."""

    backbone_name = "thuml/sundial-base-128m"
    context_length = 512
    max_prediction_length = 720

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
        if "residual" in components:
            raise ValueError("Sundial adapter intentionally excludes residual component tokens")
        self.backbone = backbone
        self.components = tuple(components)
        self.n_proto = n_proto
        self.n_rq_stages = n_rq_stages
        self.period = period
        self.trend_kernel = trend_kernel
        self.input_patch_size = backbone.config.input_token_len
        self.d_model = backbone.config.hidden_size
        self.max_prediction_length = backbone.config.output_token_lens[-1]
        if self.context_length % self.input_patch_size:
            raise ValueError("context_length must be divisible by Sundial input patch size")
        if self.d_model % cross_attention_heads:
            raise ValueError("d_model must be divisible by cross_attention_heads")

        self.component_to_id = {name: index for index, name in enumerate(self.components)}
        self.component_quantizers = nn.ModuleDict()
        self.code_embedding_tables = nn.ModuleDict()
        for component in self.components:
            checkpoint = torch.load(component_codebook_paths[component], map_location="cpu")
            if checkpoint.get("backbone") not in (None, self.backbone_name):
                raise ValueError(f"Not a Sundial codebook: {component_codebook_paths[component]}")
            if checkpoint.get("input_patch_size", self.input_patch_size) != self.input_patch_size:
                raise ValueError(f"Unexpected Sundial patch size for {component}")
            if checkpoint.get("d_model", self.d_model) != self.d_model:
                raise ValueError(f"Unexpected Sundial hidden size for {component}")
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

        num_patches = self.context_length // self.input_patch_size
        self.component_embedding = nn.Embedding(len(self.components), self.d_model)
        self.stage_embedding = nn.Embedding(n_rq_stages, self.d_model)
        self.patch_position_embedding = nn.Embedding(num_patches, self.d_model)
        tokens_per_patch = len(self.components) * n_rq_stages
        self.token_dropout = nn.Dropout(token_dropout)
        self.patch_token_aggregator = nn.Sequential(
            nn.Linear(tokens_per_patch * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(token_dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.final_to_component_attention = nn.MultiheadAttention(
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
        config = SundialConfig.from_pretrained(pretrained_model_path, local_files_only=True)
        backbone = SundialForPrediction.from_pretrained(
            pretrained_model_path, config=config, local_files_only=True, low_cpu_mem_usage=True
        )
        model = cls(backbone, component_codebook_paths, **kwargs)
        print(f"Sundial backbone loaded from {pretrained_model_path}")
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
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameters: total={total:,}, trainable={trainable:,}")

    def _normalize(self, values):
        if values.ndim != 2 or values.shape[-1] != self.context_length:
            raise ValueError(
                f"context must have shape (batch, {self.context_length}), got {tuple(values.shape)}"
            )
        values = values.float()
        valid = torch.isfinite(values)
        safe = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
        mean = (safe * valid).sum(dim=-1, keepdim=True) / count
        variance = (((safe - mean) * valid) ** 2).sum(dim=-1, keepdim=True) / count
        scale = variance.sqrt().clamp_min(1e-5)
        normalized = torch.where(valid, (safe - mean) / scale, torch.zeros_like(safe))
        return normalized, mean.detach(), scale.detach()

    @torch.no_grad()
    def _hidden(self, normalized):
        return self.backbone.model(
            input_ids=normalized, use_cache=False, output_hidden_states=False, return_dict=True
        ).last_hidden_state.detach()

    @torch.no_grad()
    def _encode_component_codes(self, values, component):
        normalized, _, _ = self._normalize(values)
        return self.component_quantizers[component].get_codes(self._hidden(normalized))

    def _component_condition(self, raw_context):
        num_patches = self.context_length // self.input_patch_size
        device = raw_context.device
        patch_ids = torch.arange(num_patches, device=device)
        patch_positions = self.patch_position_embedding(patch_ids).unsqueeze(0)
        token_embeddings = []
        for component in self.components:
            values = select_component(
                raw_context, component, period=self.period, trend_kernel=self.trend_kernel
            )
            codes = self._encode_component_codes(values, component)
            if codes.shape[1] != num_patches:
                raise RuntimeError(f"{component} produced {codes.shape[1]} patches, expected {num_patches}")
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

    def _fused_condition(self, context):
        normalized, mean, scale = self._normalize(context)
        hidden = self._hidden(normalized)
        component_condition = self._component_condition(context)
        final_hidden = hidden[:, -1:, :]
        attended, _ = self.final_to_component_attention(
            query=final_hidden, key=component_condition, value=component_condition,
            need_weights=False,
        )
        projected = self.token_projection(attended)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([final_hidden, attended], dim=-1)))
        return (final_hidden + gate * projected).squeeze(1), mean, scale

    def forward(self, context, target=None, prediction_length=96, num_samples=1,
                return_prediction=True):
        if prediction_length <= 0 or prediction_length > self.max_prediction_length:
            raise ValueError(
                f"prediction_length must be in [1, {self.max_prediction_length}], got {prediction_length}"
            )
        fused, mean, scale = self._fused_condition(context)
        loss = None
        if target is not None:
            target = target[..., :prediction_length].float()
            valid = torch.isfinite(target)
            normalized_target = (
                torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0) - mean
            ) / scale
            padded_target = normalized_target.new_zeros(
                normalized_target.shape[0], self.max_prediction_length
            )
            target_mask = torch.zeros_like(padded_target)
            padded_target[:, :prediction_length] = normalized_target
            target_mask[:, :prediction_length] = valid
            loss = self.backbone.flow_loss(padded_target, fused, mask_y=target_mask)

        prediction = None
        samples = None
        if return_prediction:
            with torch.no_grad():
                normalized_samples = self.backbone.flow_loss.sample(fused.detach(), num_samples)
                samples = normalized_samples[..., :prediction_length] * scale.unsqueeze(1) + mean.unsqueeze(1)
                prediction = samples.median(dim=1).values
        return {"loss": loss, "prediction": prediction, "samples": samples}

    def adapter_state_dict(self):
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if not name.startswith(("backbone.", "component_quantizers."))
        }

    def load_adapter_state_dict(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad_missing = [name for name in missing
                       if not name.startswith(("backbone.", "component_quantizers."))]
        if bad_missing or unexpected:
            raise RuntimeError(f"Adapter load mismatch: missing={bad_missing}, unexpected={unexpected}")
