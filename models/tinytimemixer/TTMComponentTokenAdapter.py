"""Discrete component-token adapter following TinyTimeMixer prediction flow.

TTM predicts by passing context-patch states through its decoder and flattened
forecast head; unlike autoregressive/future-token models, it has no future
query states.  This adapter therefore fuses component tokens into the temporal
backbone states immediately before the official TTM decoder while preserving
the resolution-prefix state.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling_tinytimemixer import TinyTimeMixerForPrediction, TinyTimeMixerModelOutput


VALID_COMPONENTS = {"trend", "seasonal", "residual"}


def _select_component(
    values: torch.Tensor,
    component: str,
    period: int,
    trend_kernel: int,
) -> torch.Tensor:
    """Decompose ``[batch, time, channels]`` independently per channel."""
    if component not in VALID_COMPONENTS:
        raise ValueError(f"Unknown component: {component}")
    if trend_kernel < 1 or trend_kernel % 2 == 0:
        raise ValueError("trend_kernel must be a positive odd number")
    if period < 1:
        raise ValueError("period must be positive")

    channel_first = values.transpose(1, 2)
    padding = trend_kernel // 2
    padded = F.pad(channel_first, (padding, padding), mode="replicate")
    trend = F.avg_pool1d(padded, kernel_size=trend_kernel, stride=1)
    if component == "trend":
        return trend.transpose(1, 2)

    detrended = channel_first - trend
    seasonal = torch.zeros_like(detrended)
    for phase in range(period):
        phase_values = detrended[..., phase::period]
        if phase_values.shape[-1]:
            seasonal[..., phase::period] = phase_values.mean(dim=-1, keepdim=True)
    selected = seasonal if component == "seasonal" else detrended - seasonal
    return selected.transpose(1, 2)


class TTMComponentTokenAdapter(nn.Module):
    """Fuse frozen component RQ codes into TTM context-patch states.

    The official predictor, codebooks, scaler, decoder, and forecast head stay
    frozen.  Gradients flow through the frozen decoder/head into only the
    trainable token embeddings, attention, projection, and fusion gate.
    """

    def __init__(
        self,
        forecast_model: TinyTimeMixerForPrediction,
        component_codebook_paths: Dict[str, str],
        components: Sequence[str] = ("trend", "seasonal"),
        n_proto: int = 256,
        n_rq_stages: int = 2,
        period: int = 24,
        trend_kernel: int = 25,
        frequency_token: int = 7,
        token_dropout: float = 0.1,
        gate_bias_init: float = -2.0,
        cross_attention_heads: int = 8,
    ):
        super().__init__()
        self.forecast_model = forecast_model
        self.config = forecast_model.config
        self.components = tuple(components)
        self.n_proto = n_proto
        self.n_rq_stages = n_rq_stages
        self.period = period
        self.trend_kernel = trend_kernel
        self.frequency_token = frequency_token
        self.context_length = self.config.context_length
        self.patch_length = self.config.patch_length
        self.d_model = self.config.d_model

        if not self.components or set(self.components) - VALID_COMPONENTS:
            raise ValueError("components must be a nonempty subset of trend/seasonal/residual")
        if self.context_length % self.patch_length:
            raise ValueError("TTM context_length must be divisible by patch_length")
        if self.d_model % cross_attention_heads:
            raise ValueError("d_model must be divisible by cross_attention_heads")
        if self.config.resolution_prefix_tuning and not (
            0 <= frequency_token < self.config.frequency_token_vocab_size
        ):
            raise ValueError(
                f"frequency_token must be in [0, {self.config.frequency_token_vocab_size})"
            )

        self.component_to_id = {name: index for index, name in enumerate(self.components)}
        self.codebooks = nn.ModuleDict()
        self.code_embedding_tables = nn.ModuleDict()
        for component in self.components:
            if component not in component_codebook_paths:
                raise KeyError(f"Missing codebook path for {component}")
            checkpoint = torch.load(
                component_codebook_paths[component], map_location="cpu", weights_only=False
            )
            if checkpoint.get("backbone") not in (None, "tinytimemixer"):
                raise ValueError(f"Not a TTM codebook: {component_codebook_paths[component]}")
            expected = {
                "context_length": self.context_length,
                "d_model": self.d_model,
                "patch_length": self.patch_length,
                "n_proto": n_proto,
                "n_rq_stages": n_rq_stages,
            }
            for key, value in expected.items():
                if checkpoint.get(key, value) != value:
                    raise ValueError(
                        f"{component} codebook {key}={checkpoint.get(key)} does not match TTM {value}"
                    )

            frozen_stages, trainable_tables = [], []
            for stage in range(n_rq_stages):
                weight = checkpoint[f"codebook_{stage}"].float()
                if tuple(weight.shape) != (n_proto, self.d_model):
                    raise ValueError(
                        f"Unexpected {component} stage {stage} shape {tuple(weight.shape)}"
                    )
                frozen = nn.Embedding(n_proto, self.d_model)
                frozen.weight.data.copy_(weight)
                frozen.weight.requires_grad_(False)
                frozen_stages.append(frozen)
                # Code IDs are discrete inputs; their downstream embeddings are
                # adapter parameters rather than copies of VQ centroids.
                trainable_tables.append(nn.Embedding(n_proto, self.d_model))
            self.codebooks[component] = nn.ModuleList(frozen_stages)
            self.code_embedding_tables[component] = nn.ModuleList(trainable_tables)

        num_patches = self.context_length // self.patch_length
        self.component_embedding = nn.Embedding(len(self.components), self.d_model)
        self.stage_embedding = nn.Embedding(n_rq_stages, self.d_model)
        self.patch_position_embedding = nn.Embedding(num_patches, self.d_model)
        self.token_dropout = nn.Dropout(token_dropout)
        self.patch_token_aggregator = nn.Sequential(
            nn.Linear(len(self.components) * n_rq_stages * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(token_dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.context_to_component_attention = nn.MultiheadAttention(
            self.d_model,
            cross_attention_heads,
            dropout=token_dropout,
            batch_first=True,
        )
        self.token_projection = nn.Linear(self.d_model, self.d_model)
        self.fusion_gate = nn.Linear(2 * self.d_model, self.d_model)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, gate_bias_init)
        self.freeze_forecast_model()

    @classmethod
    def from_pretrained_component_adapter(
        cls,
        pretrained_model_path: str,
        component_codebook_paths: Dict[str, str],
        **kwargs,
    ) -> "TTMComponentTokenAdapter":
        prediction_filter_length = kwargs.pop("prediction_filter_length", None)
        load_kwargs = {}
        if prediction_filter_length is not None:
            load_kwargs["prediction_filter_length"] = prediction_filter_length
        forecast_model = TinyTimeMixerForPrediction.from_pretrained(
            pretrained_model_path, **load_kwargs
        )
        print(f"TTM backbone loaded from {pretrained_model_path}")
        return cls(forecast_model, component_codebook_paths, **kwargs)

    def freeze_forecast_model(self) -> None:
        for parameter in self.forecast_model.parameters():
            parameter.requires_grad = False
        self.forecast_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.forecast_model.eval()
        self.codebooks.eval()
        return self

    def print_trainable_params(self) -> None:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        print(f"Parameters: total={total:,}, trainable={trainable:,}")

    def _prepare_inputs(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if past_values.ndim == 2:
            past_values = past_values.unsqueeze(-1)
        if past_values.ndim != 3:
            raise ValueError("past_values must be [batch,time] or [batch,time,channels]")
        past_values = past_values.float()
        if past_observed_mask is None:
            past_observed_mask = torch.isfinite(past_values)
        else:
            if past_observed_mask.ndim == 2:
                past_observed_mask = past_observed_mask.unsqueeze(-1)
            past_observed_mask = past_observed_mask.bool() & torch.isfinite(past_values)

        if past_values.shape[1] > self.context_length:
            past_values = past_values[:, -self.context_length :, :]
            past_observed_mask = past_observed_mask[:, -self.context_length :, :]
        elif past_values.shape[1] < self.context_length:
            pad = self.context_length - past_values.shape[1]
            past_values = F.pad(past_values, (0, 0, pad, 0), value=0.0)
            past_observed_mask = F.pad(past_observed_mask, (0, 0, pad, 0), value=False)
        return torch.nan_to_num(past_values, nan=0.0), past_observed_mask

    def _frequency_token(
        self,
        batch_size: int,
        device: torch.device,
        freq_token: Optional[Union[int, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        if not self.config.resolution_prefix_tuning:
            return None
        token = self.frequency_token if freq_token is None else freq_token
        if isinstance(token, int):
            token = torch.full((batch_size,), token, dtype=torch.long, device=device)
        else:
            token = token.to(device=device, dtype=torch.long).reshape(-1)
            if token.numel() == 1:
                token = token.expand(batch_size)
            elif token.numel() != batch_size:
                raise ValueError("freq_token must be scalar or have one value per batch item")
        if token.min().item() < 0 or token.max().item() >= self.config.frequency_token_vocab_size:
            raise ValueError("freq_token is outside TTM frequency vocabulary")
        return token

    @torch.no_grad()
    def _encode_component_hidden(
        self,
        component_values: torch.Tensor,
        observed_mask: torch.Tensor,
        freq_token: Optional[torch.Tensor],
    ) -> torch.Tensor:
        output = self.forecast_model.backbone(
            past_values=component_values,
            past_observed_mask=observed_mask,
            return_dict=True,
            freq_token=freq_token,
        )
        hidden = output.last_hidden_state
        if self.config.resolution_prefix_tuning:
            hidden = hidden[:, :, 1:, :]
        expected_patches = self.context_length // self.patch_length
        if hidden.shape[-2] != expected_patches:
            raise RuntimeError(
                f"TTM component encoder returned {hidden.shape[-2]} temporal patches; "
                f"expected {expected_patches}"
            )
        return hidden.detach()

    @torch.no_grad()
    def _component_codes(
        self,
        past_values: torch.Tensor,
        observed_mask: torch.Tensor,
        component: str,
        freq_token: Optional[torch.Tensor],
    ):
        component_values = _select_component(
            past_values,
            component,
            period=self.period,
            trend_kernel=self.trend_kernel,
        )
        residual = self._encode_component_hidden(component_values, observed_mask, freq_token)
        codes = []
        for codebook in self.codebooks[component]:
            flat = residual.reshape(-1, self.d_model)
            weight = codebook.weight
            distance = (
                flat.square().sum(dim=-1, keepdim=True)
                + weight.square().sum(dim=-1)
                - 2 * flat @ weight.T
            )
            indices = distance.argmin(dim=-1).view(residual.shape[:-1])
            quantized = codebook(indices)
            codes.append(indices)
            residual = residual - quantized
        return codes

    def _component_condition(
        self,
        past_values: torch.Tensor,
        observed_mask: torch.Tensor,
        freq_token: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_patches = self.context_length // self.patch_length
        position = self.patch_position_embedding(
            torch.arange(num_patches, device=past_values.device)
        ).view(1, 1, num_patches, self.d_model)
        tokens = []
        for component in self.components:
            codes = self._component_codes(
                past_values, observed_mask, component, freq_token
            )
            component_embedding = self.component_embedding.weight[
                self.component_to_id[component]
            ].view(1, 1, 1, self.d_model)
            for stage, indices in enumerate(codes):
                stage_embedding = self.stage_embedding.weight[stage].view(
                    1, 1, 1, self.d_model
                )
                tokens.append(
                    self.code_embedding_tables[component][stage](indices)
                    + component_embedding
                    + stage_embedding
                    + position
                )
        concatenated = torch.cat(tokens, dim=-1)
        return self.patch_token_aggregator(self.token_dropout(concatenated))

    def _fuse_backbone_states(
        self,
        backbone_hidden: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.resolution_prefix_tuning:
            prefix = backbone_hidden[:, :, :1, :]
            temporal_hidden = backbone_hidden[:, :, 1:, :]
        else:
            prefix = None
            temporal_hidden = backbone_hidden
        if temporal_hidden.shape[:-1] != condition.shape[:-1]:
            raise RuntimeError(
                f"TTM/condition shape mismatch: {temporal_hidden.shape} vs {condition.shape}"
            )

        batch, channels, patches, width = temporal_hidden.shape
        queries = temporal_hidden.reshape(batch * channels, patches, width)
        keys = condition.reshape(batch * channels, patches, width)
        attended, _ = self.context_to_component_attention(
            query=queries, key=keys, value=keys, need_weights=False
        )
        gate = torch.sigmoid(self.fusion_gate(torch.cat([queries, attended], dim=-1)))
        fused = queries + gate * self.token_projection(attended)
        fused = fused.view(batch, channels, patches, width)
        return torch.cat([prefix, fused], dim=2) if prefix is not None else fused

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        freq_token: Optional[Union[int, torch.Tensor]] = None,
        static_categorical_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        past_values, past_observed_mask = self._prepare_inputs(
            past_values, past_observed_mask
        )
        frequency = self._frequency_token(
            past_values.shape[0], past_values.device, freq_token
        )

        with torch.no_grad():
            model_output = self.forecast_model.backbone(
                past_values=past_values,
                past_observed_mask=past_observed_mask,
                return_dict=True,
                freq_token=frequency,
            )
        if isinstance(model_output, tuple):
            model_output = TinyTimeMixerModelOutput(*model_output)

        condition = self._component_condition(
            past_values, past_observed_mask, frequency
        )
        decoder_input = self._fuse_backbone_states(
            model_output.last_hidden_state.detach(), condition
        )

        if self.forecast_model.use_decoder:
            decoder_output, _ = self.forecast_model.decoder(
                hidden_state=decoder_input,
                patch_input=model_output.patch_input,
                output_hidden_states=False,
                static_categorical_values=static_categorical_values,
            )
        else:
            decoder_output = decoder_input

        normalized_prediction = self.forecast_model.head(
            decoder_output,
            past_values=past_values,
            future_values=future_values,
        )
        if self.forecast_model.prediction_channel_indices is not None:
            indices = self.forecast_model.prediction_channel_indices
            loc, scale = model_output.loc[..., indices], model_output.scale[..., indices]
        else:
            loc, scale = model_output.loc, model_output.scale
        prediction = self.forecast_model.backbone.scaler.inverse(
            data=normalized_prediction, loc=loc, scale=scale
        )

        target = future_values
        if target is not None and self.forecast_model.prediction_filter_length is not None:
            target = target[:, : self.forecast_model.prediction_filter_length, :]
            if future_observed_mask is not None:
                future_observed_mask = future_observed_mask[
                    :, : self.forecast_model.prediction_filter_length, :
                ]
        if (
            target is not None
            and self.forecast_model.prediction_channel_indices is not None
            and target.shape[-1] == self.forecast_model.num_input_channels
            and target.shape[-1] != len(self.forecast_model.prediction_channel_indices)
        ):
            target = target[..., self.forecast_model.prediction_channel_indices]
            if future_observed_mask is not None:
                future_observed_mask = future_observed_mask[
                    ..., self.forecast_model.prediction_channel_indices
                ]

        loss = None
        if target is not None and return_loss and self.forecast_model.loss is not None:
            valid = torch.isfinite(target)
            if future_observed_mask is not None:
                valid = valid & future_observed_mask.bool()
            clean_target = torch.nan_to_num(target, nan=0.0)
            if self.forecast_model.loss == "mse":
                element_loss = (prediction - clean_target).square()
            elif self.forecast_model.loss == "mae":
                element_loss = (prediction - clean_target).abs()
            elif self.forecast_model.loss == "huber":
                element_loss = F.huber_loss(
                    prediction,
                    clean_target,
                    reduction="none",
                    delta=self.config.huber_delta,
                )
            elif self.forecast_model.loss == "pinball":
                error = clean_target - prediction
                quantile = self.config.quantile
                element_loss = torch.maximum(quantile * error, (quantile - 1) * error)
            else:
                raise ValueError(
                    f"TTM adapter does not support loss={self.forecast_model.loss}"
                )
            loss = (element_loss * valid).sum() / valid.sum().clamp_min(1)

        return {
            "loss": loss,
            "prediction_outputs": prediction,
            "backbone_hidden_state": model_output.last_hidden_state,
            "fused_backbone_hidden_state": decoder_input,
            "decoder_hidden_state": decoder_output,
            "component_condition": condition,
            "loc": loc,
            "scale": scale,
        }

    @staticmethod
    def central_prediction(output):
        return output["prediction_outputs"]

    def adapter_state_dict(self):
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if not name.startswith(("forecast_model.", "codebooks."))
        }

    def load_adapter_state_dict(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad_missing = [
            name
            for name in missing
            if not name.startswith(("forecast_model.", "codebooks."))
        ]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"Adapter load mismatch: missing={bad_missing}, unexpected={unexpected}"
            )
