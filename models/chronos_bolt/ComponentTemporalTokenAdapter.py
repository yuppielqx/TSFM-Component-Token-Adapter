"""ChronosBolt adapter conditioned on component-wise temporal tokens."""

import os
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.models.t5.modeling_t5 import T5Config

from .ChronosBolt import ChronosBoltModelForForecasting, ChronosBoltOutput
from .ChronosBoltPrototype import ResidualQuantizer
from utils.decomposition import select_component


class ComponentTemporalTokenAdapter(ChronosBoltModelForForecasting):
    """Inject frozen component RQ-VAE temporal tokens into a frozen ChronosBolt.

    The ChronosBolt backbone and component RQ codebooks are frozen. Downstream
    training updates only token embeddings, a patch token aggregator, and a
    gated residual fusion adapter.
    """

    def __init__(
        self,
        config: T5Config,
        component_codebook_paths: Dict[str, str],
        components: Sequence[str] = ("trend", "seasonal", "residual"),
        n_proto: int = 256,
        n_rq_stages: int = 2,
        period: int = 24,
        trend_kernel: int = 25,
        token_dropout: float = 0.1,
        gate_bias_init: float = -2.0,
    ):
        super().__init__(config)
        self.components = tuple(components)
        self.n_proto = n_proto
        self.n_rq_stages = n_rq_stages
        self.period = period
        self.trend_kernel = trend_kernel

        d_model = config.d_model
        self.component_to_id = {name: idx for idx, name in enumerate(self.components)}

        self.component_quantizers = nn.ModuleDict()
        self.code_embedding_tables = nn.ModuleDict()
        for component in self.components:
            ckpt_path = component_codebook_paths[component]
            ckpt = torch.load(ckpt_path, map_location="cpu")
            quantizer = ResidualQuantizer(n_stages=n_rq_stages, n_proto=n_proto, d_model=d_model)
            stage_tables = nn.ModuleList()
            for stage in range(n_rq_stages):
                weight = ckpt[f"codebook_{stage}"].to(dtype=quantizer.codebooks[stage].weight.dtype)
                quantizer.codebooks[stage].weight.data.copy_(weight)
                quantizer.codebooks[stage].weight.requires_grad = False
                emb = nn.Embedding(n_proto, d_model)
                emb.weight.data.copy_(weight.to(dtype=emb.weight.dtype))
                stage_tables.append(emb)
            self.component_quantizers[component] = quantizer
            self.code_embedding_tables[component] = stage_tables

        max_patch_positions = self.chronos_config.context_length // self.chronos_config.input_patch_stride + 8
        self.component_embedding = nn.Embedding(len(self.components), d_model)
        self.stage_embedding = nn.Embedding(n_rq_stages, d_model)
        self.patch_position_embedding = nn.Embedding(max_patch_positions, d_model)

        tokens_per_patch = len(self.components) * n_rq_stages
        self.token_dropout = nn.Dropout(token_dropout)
        self.patch_token_aggregator = nn.Sequential(
            nn.Linear(tokens_per_patch * d_model, d_model),
            nn.GELU(),
            nn.Dropout(token_dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.token_projection = nn.Linear(d_model, d_model)
        self.fusion_gate = nn.Linear(2 * d_model, d_model)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, gate_bias_init)

        self.freeze_backbone()

    def freeze_backbone(self):
        for mod in [
            self.encoder,
            self.decoder,
            self.shared,
            self.input_patch_embedding,
            self.output_patch_embedding,
        ]:
            for param in mod.parameters():
                param.requires_grad = False

    def print_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameters: total={total:,}, trainable={trainable:,}")

    @torch.no_grad()
    def _encode_component_codes(self, component_values: torch.Tensor, component: str) -> torch.Tensor:
        batch_size = component_values.shape[0]
        if component_values.shape[-1] > self.chronos_config.context_length:
            component_values = component_values[..., -self.chronos_config.context_length:]

        context_norm, _loc_scale = self.instance_norm(component_values)
        context_norm = context_norm.to(self.dtype)
        seq_len = context_norm.shape[-1]
        mask = torch.ones(batch_size, seq_len, device=component_values.device, dtype=self.dtype)

        patched_context = self.patch(context_norm)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        attention_mask = patched_mask.sum(dim=-1) > 0

        input_embeds = self.input_patch_embedding(patched_context)
        encoder_outputs = self.encoder(attention_mask=attention_mask, inputs_embeds=input_embeds)
        hidden_states = encoder_outputs[0]
        return self.component_quantizers[component].get_codes(hidden_states)

    def _component_token_condition(self, raw_context: torch.Tensor, num_patches: int) -> torch.Tensor:
        token_embeds = []
        device = raw_context.device
        patch_ids = torch.arange(num_patches, device=device)
        patch_pos = self.patch_position_embedding(patch_ids).unsqueeze(0)

        for component in self.components:
            component_values = select_component(
                raw_context,
                component,
                period=self.period,
                trend_kernel=self.trend_kernel,
            )
            codes = self._encode_component_codes(component_values, component)
            if codes.shape[1] != num_patches:
                raise RuntimeError(
                    f"Component {component} produced {codes.shape[1]} patches, expected {num_patches}"
                )
            comp_id = torch.full((num_patches,), self.component_to_id[component], device=device, dtype=torch.long)
            comp_pos = self.component_embedding(comp_id).unsqueeze(0)
            for stage in range(self.n_rq_stages):
                stage_id = torch.full((num_patches,), stage, device=device, dtype=torch.long)
                stage_pos = self.stage_embedding(stage_id).unsqueeze(0)
                emb = self.code_embedding_tables[component][stage](codes[..., stage])
                token_embeds.append(emb + comp_pos + stage_pos + patch_pos)

        condition = torch.cat(token_embeds, dim=-1)
        condition = self.token_dropout(condition)
        return self.patch_token_aggregator(condition)

    def _fuse_patch_hidden(self, hidden_states: torch.Tensor, token_condition: torch.Tensor) -> torch.Tensor:
        projected = self.token_projection(token_condition)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([hidden_states, token_condition], dim=-1)))
        return hidden_states + gate * projected

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        excluded_prefixes = (
            "encoder.",
            "decoder.",
            "shared.",
            "input_patch_embedding.",
            "output_patch_embedding.",
            "component_quantizers.",
        )
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if not name.startswith(excluded_prefixes)
        }

    def load_adapter_state_dict(self, state: Dict[str, torch.Tensor]):
        missing, unexpected = self.load_state_dict(state, strict=False)
        expected_missing = [name for name in missing if not any(
            name.startswith(prefix)
            for prefix in (
                "encoder.",
                "decoder.",
                "shared.",
                "input_patch_embedding.",
                "output_patch_embedding.",
                "component_quantizers.",
            )
        )]
        if expected_missing or unexpected:
            raise RuntimeError(f"Adapter load mismatch: missing={expected_missing}, unexpected={unexpected}")

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> ChronosBoltOutput:
        raw_context = context
        mask = mask.to(context.dtype) if mask is not None else torch.isnan(context).logical_not().to(context.dtype)

        batch_size, _ = context.shape
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length:]
            raw_context = raw_context[..., -self.chronos_config.context_length:]
            mask = mask[..., -self.chronos_config.context_length:]

        context, loc_scale = self.instance_norm(context)
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)

        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        attention_mask = patched_mask.sum(dim=-1) > 0
        input_embeds = self.input_patch_embedding(patched_context)

        patch_token_count = input_embeds.shape[1]
        if self.chronos_config.use_reg_token:
            reg_input_ids = torch.full((batch_size, 1), self.config.reg_token_id, device=input_embeds.device)
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat([attention_mask, torch.ones_like(reg_input_ids)], dim=-1)

        with torch.no_grad():
            encoder_outputs = self.encoder(attention_mask=attention_mask, inputs_embeds=input_embeds)
            hidden_states = encoder_outputs[0]

        token_condition = self._component_token_condition(raw_context, patch_token_count)
        patch_hidden = hidden_states[:, :patch_token_count]
        fused_patch_hidden = self._fuse_patch_hidden(patch_hidden, token_condition)
        if hidden_states.shape[1] > patch_token_count:
            hidden_states = torch.cat([fused_patch_hidden, hidden_states[:, patch_token_count:]], dim=1)
        else:
            hidden_states = fused_patch_hidden

        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)
        quantile_preds_shape = (
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length,
        )
        quantile_preds = self.output_patch_embedding(sequence_output).view(*quantile_preds_shape)

        loss = None
        if target is not None:
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)
            assert self.chronos_config.prediction_length >= target.shape[-1]
            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device) if target_mask is not None else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (*target.shape[:-1], self.chronos_config.prediction_length - target.shape[-1])
                target = torch.cat([target, torch.zeros(padding_shape).to(target)], dim=-1)
                target_mask = torch.cat([target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1)
            loss = (
                2
                * torch.abs(
                    (target - quantile_preds)
                    * ((target <= quantile_preds).float() - self.quantiles.view(1, self.num_quantiles, 1))
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2)
            loss = loss.sum(dim=-1)
            loss = loss.mean()

        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*quantile_preds_shape)

        return ChronosBoltOutput(loss=loss, quantile_preds=quantile_preds)

    @classmethod
    def from_pretrained_component_adapter(
        cls,
        pretrained_model_path: str,
        component_codebook_paths: Dict[str, str],
        components: Sequence[str] = ("trend", "seasonal", "residual"),
        n_proto: int = 256,
        n_rq_stages: int = 2,
        period: int = 24,
        trend_kernel: int = 25,
        token_dropout: float = 0.1,
        gate_bias_init: float = -2.0,
    ) -> "ComponentTemporalTokenAdapter":
        config = AutoConfig.from_pretrained(pretrained_model_path)
        model = cls(
            config,
            component_codebook_paths=component_codebook_paths,
            components=components,
            n_proto=n_proto,
            n_rq_stages=n_rq_stages,
            period=period,
            trend_kernel=trend_kernel,
            token_dropout=token_dropout,
            gate_bias_init=gate_bias_init,
        )

        backbone_ckpt = os.path.join(pretrained_model_path, "autogluon_model.pth")
        safetensors_ckpt = os.path.join(pretrained_model_path, "model.safetensors")
        if os.path.exists(backbone_ckpt):
            state = torch.load(backbone_ckpt, map_location="cpu")
        elif os.path.exists(safetensors_ckpt):
            state = load_file(safetensors_ckpt, device="cpu")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {pretrained_model_path}. "
                "Expected autogluon_model.pth or model.safetensors."
            )

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"Backbone loaded from {pretrained_model_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        model.freeze_backbone()
        return model
