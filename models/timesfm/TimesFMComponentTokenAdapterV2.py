"""由 V2 成分 RQ tokens 条件化的 TimesFM-2.5 预测 Adapter。

This is intentionally separate from the ChronosBolt adapter implementation.
The TimesFM backbone and trained component codebooks stay frozen; downstream
training updates only the token embeddings, aggregator, and gated fusion.

维度约定：B 为 batch size，T 为时间长度，P 为 patch 数，S=32，D=1280，
O=128 为 TimesFM 单次输出长度，Q 为 point/quantile 输出通道数，R 为 RQ stage 数。
"""

import os
from typing import Dict, Sequence

import torch
import torch.nn as nn
from safetensors.torch import load_file

from .TimesFMPrototypeV2 import ResidualQuantizer
from .component_preprocessing_v2 import (
    PREPROCESSING_NAME,
    normalize_timesfm_patches,
    pad_timesfm_context,
    prepare_timesfm_component_patches,
)
from .timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
from .torch import util
from utils.decomposition import select_component


class TimesFMComponentTokenAdapterV2(nn.Module):
    """在冻结的 TimesFM hidden 与官方预测头之间注入成分 token 条件。"""

    input_patch_size = 32
    output_patch_size = 128
    d_model = 1280
    context_limit = 16384
    component_context_length = 512

    def __init__(
        self,
        component_codebook_paths: Dict[str, str],
        components: Sequence[str] = ("trend", "seasonal"),
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
            # 每个成分加载 R 个冻结 codebook；权重形状均为 [K, D]。
            checkpoint = torch.load(component_codebook_paths[component], map_location="cpu")
            if checkpoint.get("backbone") not in (None, "timesfm-2.5-200m-pytorch"):
                raise ValueError(f"Not a TimesFM codebook: {component_codebook_paths[component]}")
            if checkpoint.get("format_version") != 2:
                raise ValueError(
                    "TimesFMComponentTokenAdapterV2 requires codebook "
                    f"format_version=2: {component_codebook_paths[component]}"
                )
            if checkpoint.get("preprocessing") != PREPROCESSING_NAME:
                raise ValueError(
                    "TimesFMComponentTokenAdapterV2 requires preprocessing="
                    f"{PREPROCESSING_NAME}: {component_codebook_paths[component]}"
                )
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
                # Adapter 内的 code embedding 从训练好的 codebook 初始化，但保持可训练。
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
            # [B, P, C*R*D] -> [B, P, D]，其中 C 为成分数量。
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
        # TimesFM 主干与官方 output projection 始终冻结。
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

    def _prepare_component_patches(self, values):
        # values [B, T] -> normalized/mask [B, P, 32]；Adapter 不使用随机 prefix mask。
        return prepare_timesfm_component_patches(
            values,
            patch_size=self.input_patch_size,
            context_limit=self.context_limit,
            apply_prefix_mask=False,
        )

    def _pad_context(self, context):
        # context [B, T] -> values/masks [B, P*32]。
        values, masks, _, _ = pad_timesfm_context(
            context,
            patch_size=self.input_patch_size,
            context_limit=self.context_limit,
        )
        return values, masks

    def _normalize_patches(self, values, masks):
        # values/masks [B, P*32] -> normalized/mask [B, P, 32]，mu/sigma [B, P]。
        return normalize_timesfm_patches(
            values, masks, patch_size=self.input_patch_size
        )

    @torch.no_grad()
    def _backbone_hidden(self, normalized_patches, patch_masks):
        # 输入 [B, P, 32]；冻结主干输出 hidden_states [B, P, 1280]。
        (_, hidden_states, _, _), _ = self.backbone(normalized_patches, patch_masks)
        return hidden_states.detach()

    @torch.no_grad()
    def _encode_component_codes(self, values, component):
        # component values [B, T] -> codes [B, P, R]。
        prepared = self._prepare_component_patches(values)
        hidden_states = self._backbone_hidden(
            prepared.normalized_patches, prepared.patch_masks
        )
        return self.component_quantizers[component].get_codes(hidden_states)

    def _component_condition(self, raw_context, num_patches=None, tail_patches=None):
        """从原始数值历史构造成分条件，返回 [B, P, D]。"""
        # Adapter 用 512 点（16 个 patch）训练；rollout 时始终滑动保留最近 512 点，
        # 防止条件位置超出训练见过的 0..15。
        raw_context = raw_context[..., -self.component_context_length:]  # [B, <=512]
        token_embeddings = []
        device = raw_context.device
        condition_patches = None
        for component in self.components:
            component_values = select_component(
                raw_context, component, period=self.period, trend_kernel=self.trend_kernel
            )  # [B, <=512]
            codes = self._encode_component_codes(component_values, component)  # [B, P, R]
            if condition_patches is None:
                condition_patches = codes.shape[1]
            elif codes.shape[1] != condition_patches:
                raise RuntimeError(
                    f"Component {component} produced {codes.shape[1]} patches; "
                    f"expected {condition_patches}"
                )
            if num_patches is not None and codes.shape[1] != num_patches:
                raise RuntimeError(
                    f"Component {component} produced {codes.shape[1]} patches; expected {num_patches}"
                )
            patch_ids = torch.arange(codes.shape[1], device=device)  # [P]
            patch_positions = self.patch_position_embedding(patch_ids).unsqueeze(0)  # [1,P,D]
            component_ids = torch.full(
                (codes.shape[1],), self.component_to_id[component], device=device, dtype=torch.long
            )
            component_positions = self.component_embedding(component_ids).unsqueeze(0)  # [1,P,D]
            for stage in range(self.n_rq_stages):
                stage_ids = torch.full((codes.shape[1],), stage, device=device, dtype=torch.long)
                stage_positions = self.stage_embedding(stage_ids).unsqueeze(0)  # [1,P,D]
                code_embeddings = self.code_embedding_tables[component][stage](codes[..., stage])  # [B,P,D]
                token_embeddings.append(
                    code_embeddings + component_positions + stage_positions + patch_positions
                )
        # 拼接后 [B,P,C*R*D]，聚合后 [B,P,D]。
        condition = self.token_dropout(torch.cat(token_embeddings, dim=-1))
        condition = self.patch_token_aggregator(condition)
        if tail_patches is not None:
            if tail_patches <= 0 or tail_patches > condition.shape[1]:
                raise ValueError(
                    f"tail_patches must be in [1, {condition.shape[1]}], got {tail_patches}"
                )
            condition = condition[:, -tail_patches:]
        return condition

    def _fuse(self, hidden_states, condition):
        """门控残差融合；输入和输出均为 [B, P, D]。"""
        projected = self.token_projection(condition)  # [B,P,D]
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat([hidden_states, condition], dim=-1))
        )  # [B,P,D]
        return hidden_states + gate * projected  # [B,P,D]

    def _adapted_backbone_outputs(
        self,
        normalized_patches,
        patch_masks,
        raw_history,
        decode_caches=None,
        tail_condition=False,
        adapter_weight=1.0,
        correction_clip=0.0,
    ):
        """运行官方 backbone，并在官方 head 前注入 Adapter 修正。

        ``adapter_weight`` and ``correction_clip`` are inference-only rollout
        controls. They operate on the difference between the adapted and
        official heads, so existing adapter checkpoints remain fully
        compatible and ``adapter_weight=1, correction_clip=0`` reproduces the
        original behavior exactly.
        """
        # normalized_patches [B,P,32] -> hidden_states [B,P,1280]。
        (_, hidden_states, _, _), new_decode_caches = self.backbone(
            normalized_patches, patch_masks, decode_caches
        )
        hidden_states = hidden_states.detach()
        official_point = self.backbone.output_projection_point(hidden_states)  # [B,P,O*Q]
        official_quantiles = self.backbone.output_projection_quantiles(hidden_states)  # [B,P,OS*Q]
        if adapter_weight == 0.0:
            # adapter 权重为 0 时直接走官方 head，避免无效的成分分解与编码。
            return official_point, official_quantiles, new_decode_caches
        if tail_condition:
            condition = self._component_condition(
                raw_history, tail_patches=hidden_states.shape[1]
            )
        else:
            condition = self._component_condition(
                raw_history, num_patches=hidden_states.shape[1]
            )
        fused_hidden = self._fuse(hidden_states, condition)  # [B,P,D]
        adapted_point = self.backbone.output_projection_point(fused_hidden)
        adapted_quantiles = self.backbone.output_projection_quantiles(fused_hidden)

        if adapter_weight == 1.0 and correction_clip <= 0.0:
            # 保持旧调用在 weight=1、clip=0 时的精确数值路径。
            point_outputs = adapted_point
            quantile_outputs = adapted_quantiles
        else:
            point_correction = adapted_point - official_point
            quantile_correction = adapted_quantiles - official_quantiles
            if correction_clip > 0:
                point_correction = point_correction.clamp(
                    -correction_clip, correction_clip
                )
                quantile_correction = quantile_correction.clamp(
                    -correction_clip, correction_clip
                )
            point_outputs = official_point + adapter_weight * point_correction
            quantile_outputs = official_quantiles + adapter_weight * quantile_correction
        return point_outputs, quantile_outputs, new_decode_caches

    def decode(
        self,
        horizon,
        inputs,
        masks,
        rollout_adapter_decay=1.0,
        rollout_adapter_min_weight=0.0,
        rollout_correction_clip=0.0,
        rollout_adapter_max_steps=-1,
        rollout_targets=None,
        scheduled_sampling_probability=1.0,
        detach_rollout_feedback=False,
    ):
        """保留官方 KV-cache decode，仅增加成分 token 融合。

        This follows TimesFM 2.5's official ``decode`` implementation. The
        only model-path change is fusion between Transformer hidden states and
        the official point/quantile prediction heads.
        """
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if inputs.ndim != 2 or masks.shape != inputs.shape:
            raise ValueError(
                "inputs and masks must have the same shape (batch, time), got "
                f"{tuple(inputs.shape)} and {tuple(masks.shape)}"
            )
        if inputs.shape[1] % self.backbone.p:
            raise ValueError(
                f"context length must be divisible by {self.backbone.p}, "
                f"got {inputs.shape[1]}"
            )
        if not 0.0 <= rollout_adapter_decay <= 1.0:
            raise ValueError("rollout_adapter_decay must be in [0, 1]")
        if not 0.0 <= rollout_adapter_min_weight <= 1.0:
            raise ValueError("rollout_adapter_min_weight must be in [0, 1]")
        if rollout_correction_clip < 0.0:
            raise ValueError("rollout_correction_clip must be non-negative")
        if rollout_adapter_max_steps < -1:
            raise ValueError("rollout_adapter_max_steps must be -1 or non-negative")
        if not 0.0 <= scheduled_sampling_probability <= 1.0:
            raise ValueError("scheduled_sampling_probability must be in [0, 1]")
        if rollout_targets is not None:
            if rollout_targets.ndim != 2 or rollout_targets.shape[0] != inputs.shape[0]:
                raise ValueError(
                    "rollout_targets must have shape (batch, horizon), got "
                    f"{tuple(rollout_targets.shape)}"
                )
            if rollout_targets.shape[1] < horizon:
                raise ValueError(
                    f"rollout_targets must cover horizon={horizon}, got "
                    f"{rollout_targets.shape[1]}"
                )

        batch_size, context = inputs.shape  # inputs/masks: [B,T]
        num_decode_steps = (horizon - 1) // self.backbone.o
        num_input_patches = context // self.backbone.p
        decode_cache_size = num_input_patches + num_decode_steps * self.backbone.m

        # Official prefill preparation and running statistics.
        patched_inputs = inputs.reshape(batch_size, -1, self.backbone.p)  # [B,P,32]
        patched_masks = masks.reshape(batch_size, -1, self.backbone.p)  # [B,P,32]
        n = torch.zeros(batch_size, device=inputs.device)  # [B]
        mu = torch.zeros(batch_size, device=inputs.device)
        sigma = torch.zeros(batch_size, device=inputs.device)
        patch_mu, patch_sigma = [], []
        for patch_index in range(num_input_patches):
            (n, mu, sigma), _ = util.update_running_stats(
                n, mu, sigma, patched_inputs[:, patch_index], patched_masks[:, patch_index]
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)
        last_n, last_mu, last_sigma = n, mu, sigma
        context_mu = torch.stack(patch_mu, dim=1)  # [B,P]
        context_sigma = torch.stack(patch_sigma, dim=1)  # [B,P]

        decode_caches = [
            util.DecodeCache(
                next_index=torch.zeros(
                    batch_size, dtype=torch.int32, device=inputs.device
                ),
                num_masked=torch.zeros(
                    batch_size, dtype=torch.int32, device=inputs.device
                ),
                # 每层 KV cache: [B, cache_tokens, num_heads, head_dim]。
                key=torch.zeros(
                    batch_size,
                    decode_cache_size,
                    self.backbone.h,
                    self.backbone.hd,
                    device=inputs.device,
                ),
                value=torch.zeros(
                    batch_size,
                    decode_cache_size,
                    self.backbone.h,
                    self.backbone.hd,
                    device=inputs.device,
                ),
            )
            for _ in range(self.backbone.x)
        ]

        # 成分分支使用原始数值尺度的最近 512 点；mask 位置置 0。
        component_history = torch.where(masks, 0.0, inputs)[
            ..., -self.component_context_length:
        ]  # [B, <=512]
        normed_inputs = util.revin(
            patched_inputs, context_mu, context_sigma, reverse=False
        )  # [B,P,32]
        normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
        normed_outputs, normed_quantile_spread, decode_caches = (
            self._adapted_backbone_outputs(
                normed_inputs,
                patched_masks,
                component_history,
                decode_caches=decode_caches,
            )
        )
        renormed_outputs = util.revin(
            normed_outputs, context_mu, context_sigma, reverse=True
        ).reshape(batch_size, -1, self.backbone.o, self.backbone.q)  # [B,P,O,Q]
        renormed_quantile_spread = util.revin(
            normed_quantile_spread, context_mu, context_sigma, reverse=True
        ).reshape(batch_size, -1, self.backbone.os, self.backbone.q)[:, -1, ...]

        # 官方 cache 自回归路径保持不变；Adapter 条件独立维护固定 512 点滑窗。
        ar_outputs = []
        last_renormed_output = renormed_outputs[
            :, -1, :, self.backbone.aridx
        ]  # [B,O]
        for decode_step in range(1, num_decode_steps + 1):
            feedback = last_renormed_output
            if detach_rollout_feedback:
                feedback = feedback.detach()
            if rollout_targets is not None and scheduled_sampling_probability < 1.0:
                target_start = (decode_step - 1) * self.backbone.o
                teacher = rollout_targets[
                    :, target_start:target_start + self.backbone.o
                ]
                teacher = torch.where(torch.isfinite(teacher), teacher, feedback)
                use_prediction = (
                    torch.rand(
                        feedback.shape[0], 1, device=feedback.device
                    ) < scheduled_sampling_probability
                )
                feedback = torch.where(use_prediction, feedback, teacher)
            # O=128 可拆成 m=4 个输入 patch，每个 patch 长度 p=32。
            new_patched_input = feedback.reshape(
                batch_size, self.backbone.m, self.backbone.p
            )  # [B,m,32]
            new_mask = torch.zeros_like(new_patched_input, dtype=torch.bool)

            n, mu, sigma = last_n, last_mu, last_sigma
            new_mus, new_sigmas = [], []
            for patch_index in range(self.backbone.m):
                (n, mu, sigma), _ = util.update_running_stats(
                    n,
                    mu,
                    sigma,
                    new_patched_input[:, patch_index],
                    new_mask[:, patch_index],
                )
                new_mus.append(mu)
                new_sigmas.append(sigma)
            last_n, last_mu, last_sigma = n, mu, sigma
            new_mu = torch.stack(new_mus, dim=1)  # [B,m]
            new_sigma = torch.stack(new_sigmas, dim=1)  # [B,m]

            component_history = torch.cat(
                [component_history, feedback], dim=-1
            )[..., -self.component_context_length:]  # [B,512]
            new_normed_input = util.revin(
                new_patched_input, new_mu, new_sigma, reverse=False
            )
            adapter_weight = max(
                rollout_adapter_min_weight,
                rollout_adapter_decay ** decode_step,
            )
            if (
                rollout_adapter_max_steps >= 0
                and decode_step > rollout_adapter_max_steps
            ):
                adapter_weight = 0.0
            new_normed_output, _, decode_caches = self._adapted_backbone_outputs(
                new_normed_input,
                new_mask,
                component_history,
                decode_caches=decode_caches,
                tail_condition=True,
                adapter_weight=adapter_weight,
                correction_clip=rollout_correction_clip,
            )
            new_renormed_output = util.revin(
                new_normed_output, new_mu, new_sigma, reverse=True
            ).reshape(batch_size, self.backbone.m, self.backbone.o, self.backbone.q)  # [B,m,O,Q]
            ar_outputs.append(new_renormed_output[:, -1, ...])
            last_renormed_output = new_renormed_output[
                :, -1, :, self.backbone.aridx
            ]

        # ar_renormed_outputs: [B, decode_steps, O, Q]；无需 rollout 时为 None。
        ar_renormed_outputs = (
            torch.stack(ar_outputs, dim=1) if ar_outputs else None
        )
        return renormed_outputs, renormed_quantile_spread, ar_renormed_outputs

    def multi_horizon_forward(
        self,
        context,
        target,
        horizon,
        scheduled_sampling_probability=1.0,
        rollout_adapter_decay=0.75,
        rollout_adapter_min_weight=0.0,
        rollout_correction_clip=0.0,
        rollout_adapter_max_steps=-1,
        mae_loss_weight=0.0,
    ):
        """使用截断梯度的多 block rollout 训练 Adapter。

        TimesFM still emits 128-point blocks. Generated feedback is detached
        between blocks, while every adapter invocation receives a local loss
        through the concatenated forecast. Scheduled sampling mixes detached
        predictions and ground-truth blocks without changing the architecture.
        """
        if target is None or target.ndim != 2:
            raise ValueError("target must have shape (batch, horizon)")
        if horizon <= 0 or horizon > target.shape[1]:
            raise ValueError(
                f"horizon must be in [1, {target.shape[1]}], got {horizon}"
            )
        raw_context = context[..., -self.component_context_length:]  # [B,<=512]
        inputs, masks = self._pad_context(raw_context)
        prefill, _, autoregressive = self.decode(
            horizon,
            inputs,
            masks,
            rollout_adapter_decay=rollout_adapter_decay,
            rollout_adapter_min_weight=rollout_adapter_min_weight,
            rollout_correction_clip=rollout_correction_clip,
            rollout_adapter_max_steps=rollout_adapter_max_steps,
            rollout_targets=target,
            scheduled_sampling_probability=scheduled_sampling_probability,
            detach_rollout_feedback=True,
        )
        chunks = [prefill[:, -1, :, self.backbone.aridx]]  # 首块 [B,O]
        if autoregressive is not None:
            chunks.append(
                autoregressive[..., self.backbone.aridx].reshape(
                    context.shape[0], -1
                )
            )
        prediction = torch.cat(chunks, dim=-1)[:, :horizon]  # [B,H]
        selected_target = target[:, :horizon]  # [B,H]
        valid = torch.isfinite(selected_target)
        safe_target = torch.nan_to_num(selected_target, nan=0.0)
        difference = prediction - safe_target
        denominator = valid.sum().clamp_min(1)
        mse = ((difference.square()) * valid).sum() / denominator
        mae = ((difference.abs()) * valid).sum() / denominator
        loss = mse + mae_loss_weight * mae
        return {
            "loss": loss,
            "mse_loss": mse,
            "mae_loss": mae,
            "prediction": prediction,
            "horizon": horizon,
        }

    @torch.no_grad()
    def forecast(
        self,
        context,
        horizon,
        rollout_adapter_decay=1.0,
        rollout_adapter_min_weight=0.0,
        rollout_correction_clip=0.0,
        rollout_adapter_max_steps=-1,
    ):
        """对 raw context [B,T] 返回官方 cached-decode 点预测 [B,H]。"""
        raw_context = context[..., -self.context_limit:]
        inputs, masks = self._pad_context(raw_context)
        prefill, _, autoregressive = self.decode(
            horizon,
            inputs,
            masks,
            rollout_adapter_decay=rollout_adapter_decay,
            rollout_adapter_min_weight=rollout_adapter_min_weight,
            rollout_correction_clip=rollout_correction_clip,
            rollout_adapter_max_steps=rollout_adapter_max_steps,
        )
        forecast_chunks = [prefill[:, -1, :, self.backbone.aridx]]  # [B,O]
        if autoregressive is not None:
            forecast_chunks.append(
                autoregressive[..., self.backbone.aridx].reshape(context.shape[0], -1)
            )
        return torch.cat(forecast_chunks, dim=-1)[:, :horizon]  # [B,H]

    def forward(self, context, target=None, prediction_length=64):
        """单个输出 patch 的直接训练路径：context [B,T] -> prediction [B,L]。"""
        if prediction_length <= 0 or prediction_length > self.output_patch_size:
            raise ValueError(
                f"prediction_length must be in [1, {self.output_patch_size}], got {prediction_length}"
            )
        raw_context = context[..., -self.context_limit:]
        values, masks = self._pad_context(raw_context)
        normalized, patch_masks, mus, sigmas = self._normalize_patches(values, masks)
        hidden_states = self._backbone_hidden(normalized, patch_masks)  # [B,P,D]
        condition = self._component_condition(raw_context, hidden_states.shape[1])  # [B,P,D]
        fused_hidden = self._fuse(hidden_states, condition)
        normalized_outputs = self.backbone.output_projection_point(fused_hidden)
        normalized_outputs = normalized_outputs.reshape(
            normalized_outputs.shape[0], normalized_outputs.shape[1],
            self.output_patch_size, self.backbone.q,
        )  # [B,P,O,Q]
        normalized_prediction = normalized_outputs[:, -1, :prediction_length, self.backbone.aridx]  # [B,L]
        prediction = util.revin(
            normalized_prediction, mus[:, -1], sigmas[:, -1], reverse=True
        )
        loss = None
        if target is not None:
            target = target[..., :prediction_length]
            valid = torch.isfinite(target)
            safe_target = torch.nan_to_num(target, nan=0.0)
            loss = (((prediction - safe_target) ** 2) * valid).sum() / valid.sum().clamp_min(1)
        return {"loss": loss, "prediction": prediction}  # prediction: [B,L]

    def adapter_state_dict(self):
        # 仅保存可迁移的 Adapter 参数，不重复保存冻结 backbone/codebook。
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
