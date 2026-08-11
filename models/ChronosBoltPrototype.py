"""
ChronosBolt + RQ-VAE 原型学习：包含两个核心模型类。

模型 1 — ChronosBoltWithRQVAEPrototypes（阶段一：码本训练）
    用途：在预训练数据上通过掩码重建任务学习 RQ-VAE 离散码本。
    流程：
      context → 随机掩码 10% → 冻结 Encoder → ResidualQuantizer(4层)
        → 每层 quantized 向量 → 独立 linear head → 求和重建 → MSE + 承诺损失
    可训练参数：ResidualQuantizer 码本 + reconstruction_heads
    冻结参数：T5 Encoder/Decoder、patch embedding

模型 2 — ChronosBoltWithDiscreteCodes（阶段二：few-shot 下游微调）
    用途：在小样本下游数据上，用冻结码本将 encoder 输出离散化，
          通过可学习的 code embedding + 融合层增强 decoder 预测。
    流程：
      context → 冻结 Encoder → 冻结 RQ 码本查离散索引
        → 可学习 Embedding 查表 → 融合层(gate/concat/cross_attn)
        → 冻结 Decoder → 分位数预测 → Quantile Loss
    可训练参数：code_embedding_tables + 融合层参数
    冻结参数：T5 Encoder/Decoder、patch embedding、RQ 码本（默认）
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers.models.t5.modeling_t5 import T5Config

from .ChronosBolt import ChronosBoltModelForForecasting, ChronosBoltOutput


# ---------------------------------------------------------------------------
# 残差量化器 (Residual Quantizer)
# ---------------------------------------------------------------------------

class ResidualQuantizer(nn.Module):
    """
    残差量化 (RQ-VAE)：使用直通估计器 (STE) 实现端到端训练。

    算法流程（n_stages 轮）：
        z_q_1 = codebook_1[argmin ||z - e||]     # 第一轮：量化原始向量
        r_1   = z - z_q_1                         # 计算残差
        z_q_2 = codebook_2[argmin ||r_1 - e||]    # 第二轮：量化残差
        ...
        quantized = z_q_1 + z_q_2 + ... + z_q_K  # 最终量化结果 = 各阶段之和

    STE 机制：前向使用量化值，反向传播时梯度直接穿过量化操作回到输入 z，
             使码本可以通过梯度下降学习。

    参数：
        n_stages: 残差量化阶段数（码本深度），默认 4
        n_proto:  每个码本的条目数（码本大小），默认 256
        d_model:  每个向量的维度，默认 768（ChronosBolt-base 的 hidden dim）
        beta:     承诺损失权重（拉近编码器输出和码本向量）
    """

    def __init__(
        self,
        n_stages: int = 4,
        n_proto: int = 256,
        d_model: int = 768,
        beta: float = 0.25,
    ):
        super().__init__()
        self.n_stages = n_stages
        self.beta = beta
        # 创建 n_stages 个码本，每个码本有 n_proto 个 d_model 维向量
        self.codebooks = nn.ModuleList([
            nn.Embedding(n_proto, d_model) for _ in range(n_stages)
        ])
        # 用截断正态分布初始化码本权重
        for cb in self.codebooks:
            nn.init.trunc_normal_(cb.weight, std=0.02)

    def forward(self, z: torch.Tensor):
        """
        前向传播：对输入进行残差量化。

        参数：
            z: encoder 隐藏状态 (B, n_patches, d_model)

        返回：
            quantized_sum:     通过 STE 的全量化和向量，与 z 同形状
            commit_loss:       承诺损失（标量），鼓励码本向编码器输出靠拢
            quantized_per_stage: 每阶段实际量化向量的列表（已 detach）
        """
        residual = z  # 初始残差 = 原始输入
        quantized_sum = torch.zeros_like(z)  # 累积量化结果
        commit_loss = 0.0
        quantized_per_stage = []

        for cb in self.codebooks:
            # 展平为 (B*n_patches, d) 便于计算距离
            flat_residual = residual.view(-1, residual.shape[-1])

            # 计算欧氏距离：||flat_residual - cb.weight||^2
            # 展开公式：||a||^2 + ||b||^2 - 2*a·b^T
            dists = (
                flat_residual.pow(2).sum(dim=1, keepdim=True)      # (N, 1)
                + cb.weight.pow(2).sum(dim=1)                      # (K,)
                - 2 * flat_residual @ cb.weight.T                  # (N, K)
            )
            # 找距离最近的码本索引
            indices = dists.argmin(dim=-1)                          # (N,)
            quantized_flat = cb(indices)                            # (N, d) 查表获取量化向量
            quantized = quantized_flat.view_as(residual)

            # 承诺损失：让码本向量向编码器输出靠拢（detach 阻止梯度回传到 encoder）
            commit_loss = commit_loss + F.mse_loss(residual.detach(), quantized)

            # STE：前向是 quantized，反向梯度直接复制到 residual
            quantized_ste = residual + (quantized - residual).detach()

            # 累积量化结果
            quantized_sum = quantized_sum + quantized_ste
            quantized_per_stage.append(quantized.detach())
            # 更新残差：减去实际量化值，供下一阶段使用
            residual = residual - quantized.detach()

        commit_loss = commit_loss * self.beta
        return quantized_sum, commit_loss, quantized_per_stage

    def get_codes(self, z: torch.Tensor) -> torch.Tensor:
        """
        获取每阶段的离散码本索引（推理用，无需 STE 和承诺损失）。

        参数：
            z: encoder 隐藏状态 (B, n_patches, d_model)

        返回：
            codes: (B, n_patches, n_stages) LongTensor，每个值 ∈ [0, n_proto-1]
        """
        residual = z
        all_indices = []
        for cb in self.codebooks:
            flat = residual.view(-1, residual.shape[-1])
            dists = (
                flat.pow(2).sum(dim=1, keepdim=True)
                + cb.weight.pow(2).sum(dim=1)
                - 2 * flat @ cb.weight.T
            )
            indices = dists.argmin(dim=-1)
            all_indices.append(indices.view(z.shape[0], z.shape[1]))
            quantized = cb(indices).view_as(residual)
            residual = residual - quantized
        return torch.stack(all_indices, dim=-1)  # (B, n_patches, n_stages)


# ---------------------------------------------------------------------------
# 模型 1：RQ-VAE 原型学习（阶段一：码本训练）
# ---------------------------------------------------------------------------

class ChronosBoltWithRQVAEPrototypes(ChronosBoltModelForForecasting):
    """
    通过掩码重建任务学习 RQ-VAE 离散码本。

    训练目标（三个阶段）：
      1. 随机掩码 mask_ratio 比例的时间点（逐点掩码，默认 10%）
      2. 被掩码的 context 通过【冻结的】T5 Encoder 得到 hidden states
      3. 对每个 patch 表示做残差量化（4 层码本）
      4. 每层量化向量通过独立的线性头解码 → 重建原始 512 个点
      5. Loss = MSE(重建序列, 归一化原始序列) + β × 承诺损失

    可训练参数：
      - ResidualQuantizer 的码本权重（n_stages × n_proto × d_model）
      - reconstruction_heads（n_stages 个 Linear(d_model → patch_size)）

    冻结参数：
      - T5 encoder、decoder、input_patch_embedding、output_patch_embedding

    参数：
        config:       T5Config（从预训练 checkpoint 加载）
        n_proto:      每层码本大小，默认 256
        n_rq_stages:  残差量化阶段数，默认 4
        beta:         承诺损失权重，默认 0.25
        mask_ratio:   掩码比例（0.1 = 随机掩码 10% 的时间点）
    """

    def __init__(
        self,
        config: T5Config,
        n_proto: int = 256,
        n_rq_stages: int = 4,
        beta: float = 0.25,
        mask_ratio: float = 0.1,
    ):
        super().__init__(config)

        d_model = config.d_model  # 768 for ChronosBolt-base
        patch_size = self.chronos_config.input_patch_size  # 16

        self.mask_ratio = mask_ratio
        self.n_rq_stages = n_rq_stages

        # 残差量化器：4 层码本，每层 256 个 768 维向量
        self.residual_quantizer = ResidualQuantizer(
            n_stages=n_rq_stages, n_proto=n_proto, d_model=d_model, beta=beta
        )

        # 每阶段独立的线性重建头：将量化后的 768 维向量映射回 16 维 patch
        self.reconstruction_heads = nn.ModuleList([
            nn.Linear(d_model, patch_size) for _ in range(n_rq_stages)
        ])

        # 冻结 T5 backbone：encoder、decoder、embedding 层均不参与训练
        self.freeze_backbone()

    def freeze_backbone(self):
        """冻结 ChronosBolt 主干网络所有参数，仅训练码本和重建头"""
        for mod in [
            self.encoder,
            self.decoder,
            self.shared,
            self.input_patch_embedding,
            self.output_patch_embedding,
        ]:
            for p in mod.parameters():
                p.requires_grad = False

    def print_trainable_params(self):
        """打印总参数量和可训练参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameters: total={total:,}, trainable={trainable:,}")

    def forward(
        self,
        context: torch.Tensor,
        mask_ratio: Optional[float] = None,
        return_reconstruction: bool = False,
    ) -> ChronosBoltOutput:
        """
        前向传播：掩码 → 编码 → 量化 → 重建 → 计算损失。

        参数：
            context:    输入时间序列 (B, seq_len)，可以是任意尺度
            mask_ratio: 可选，覆盖实例默认的 mask_ratio
            return_reconstruction: 若为 True，在输出中返回逆归一化后的重建序列

        返回：
            ChronosBoltOutput，包含 .loss（重建损失 + 承诺损失）
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        batch_size = context.shape[0]

        # 截断到模型的 context_length（如果输入过长）
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length:]

        # ── 步骤 1：实例归一化 ──
        # loc_scale 保存均值和标准差，后续不需要逆变换（损失在归一化空间计算）
        context_norm, loc_scale = self.instance_norm(context)
        context_norm = context_norm.to(self.dtype)

        # ── 步骤 2：目标 = 归一化后的原始序列（用于损失计算） ──
        target = context_norm.detach().clone()

        # ── 步骤 3：逐点随机掩码 ──
        # 随机选择 mask_ratio 比例的时间点，将其值设为 0（归一化空间中 0 ≈ 均值）
        seq_len = context_norm.shape[-1]
        recovery_mask = torch.rand(batch_size, seq_len, device=context.device) < mask_ratio
        context_masked = context_norm.clone()
        context_masked[recovery_mask] = 0.0

        # 所有位置都是有效的（parquet 数据无 NaN）
        attn_mask_1d = torch.ones(batch_size, seq_len, device=context.device, dtype=self.dtype)

        # ── 步骤 4：Patch 分块 ──
        # context: (B, seq_len) → (B, n_patches, patch_size)
        patched_context = self.patch(context_masked)
        patched_mask = torch.nan_to_num(self.patch(attn_mask_1d), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        # 拼接 context 和 mask 信息：每个 patch 维度 = patch_size（数据） + patch_size（掩码）
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        attention_mask = patched_mask.sum(dim=-1) > 0  # (B, n_patches)，至少有一个有效点的 patch 才参与注意力

        # ── 步骤 5：冻结 Encoder 前向传播 ──
        # 使用 torch.no_grad() 节省显存
        with torch.no_grad():
            input_embeds = self.input_patch_embedding(patched_context)
            encoder_outputs = self.encoder(
                attention_mask=attention_mask,
                inputs_embeds=input_embeds,
            )
            hidden_states = encoder_outputs[0]  # (B, n_patches, d_model)

        # 从 encoder 计算图中 detach，只让量化器和重建头接收梯度
        hidden_states = hidden_states.detach()

        # ── 步骤 6：残差量化 ──
        # 对每个 patch 的 768 维表示进行 4 层残差量化
        quantized, commit_loss, quantized_per_stage = self.residual_quantizer(hidden_states)

        # ── 步骤 7：逐阶段解码 + 求和 ──
        # 每阶段的量化向量通过各自的线性头 → 加和得到重建 patch
        pred_patches = torch.zeros(
            batch_size, quantized.shape[1], self.chronos_config.input_patch_size,
            device=quantized.device, dtype=quantized.dtype,
        )
        for i in range(self.n_rq_stages):
            pred_patches = pred_patches + self.reconstruction_heads[i](quantized_per_stage[i])

        # ── 步骤 8：展平为时间序列 ──
        # (B, n_patches, patch_size) → (B, n_patches * patch_size)
        pred_x = pred_patches.reshape(batch_size, -1)

        # 对齐长度（处理 padding 情况）
        target_len = target.shape[-1]
        pred_x = pred_x[..., :target_len]

        # ── 步骤 9：计算损失 ──
        # 重建损失：在所有时间点上计算 MSE（归一化空间）
        recon_loss = F.mse_loss(pred_x, target)
        # 总损失 = 重建损失 + 承诺损失（鼓励码本向编码器输出靠拢）
        total_loss = recon_loss + commit_loss

        reconstruction = None
        if return_reconstruction:
            reconstruction = self.instance_norm.inverse(pred_x, loc_scale)

        return ChronosBoltOutput(loss=total_loss, reconstruction=reconstruction)

    def load_trained_codebook(self, codebook_path: str) -> None:
        """加载阶段一训练好的码本权重和重建头权重"""
        ckpt = torch.load(codebook_path, map_location="cpu")
        for i, cb in enumerate(self.residual_quantizer.codebooks):
            cb.weight.data.copy_(ckpt[f"codebook_{i}"].to(cb.weight.dtype))
        for i, head in enumerate(self.reconstruction_heads):
            head.load_state_dict(ckpt["reconstruction_heads"][f"head_{i}"])
        print(f"Loaded {self.n_rq_stages} codebooks + {self.n_rq_stages} heads from {codebook_path}")

    @torch.no_grad()
    def encode_codes(self, context: torch.Tensor) -> torch.Tensor:
        """Encode a time-series batch into hierarchical RQ code indices.

        Parameters:
            context: Input tensor with shape (B, seq_len).

        Returns:
            codes: LongTensor with shape (B, n_patches, n_rq_stages).
        """
        batch_size = context.shape[0]

        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length:]

        context_norm, _ = self.instance_norm(context)
        context_norm = context_norm.to(self.dtype)

        seq_len = context_norm.shape[-1]
        attn_mask_1d = torch.ones(batch_size, seq_len, device=context.device, dtype=self.dtype)

        patched_context = self.patch(context_norm)
        patched_mask = torch.nan_to_num(self.patch(attn_mask_1d), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        attention_mask = patched_mask.sum(dim=-1) > 0

        input_embeds = self.input_patch_embedding(patched_context)
        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
        )
        hidden_states = encoder_outputs[0]
        return self.residual_quantizer.get_codes(hidden_states)

    @torch.no_grad()
    def reconstruct(self, context: torch.Tensor) -> torch.Tensor:
        """
        通过 RQ-VAE 码本重建输入序列（无掩码，纯推理）。

        参数：
            context: 输入时间序列 (B, seq_len)，任意尺度

        返回：
            reconstruction: 重建序列 (B, seq_len)，已逆归一化到原始空间
        """
        batch_size = context.shape[0]

        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length:]

        context_norm, loc_scale = self.instance_norm(context)
        context_norm = context_norm.to(self.dtype)

        seq_len = context_norm.shape[-1]
        attn_mask_1d = torch.ones(batch_size, seq_len, device=context.device, dtype=self.dtype)

        patched_context = self.patch(context_norm)
        patched_mask = torch.nan_to_num(self.patch(attn_mask_1d), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        attention_mask = patched_mask.sum(dim=-1) > 0

        with torch.no_grad():
            input_embeds = self.input_patch_embedding(patched_context)
            encoder_outputs = self.encoder(
                attention_mask=attention_mask,
                inputs_embeds=input_embeds,
            )
            hidden_states = encoder_outputs[0]

        quantized, _commit_loss, quantized_per_stage = self.residual_quantizer(hidden_states)

        pred_patches = torch.zeros(
            batch_size, quantized.shape[1], self.chronos_config.input_patch_size,
            device=quantized.device, dtype=quantized.dtype,
        )
        for i in range(self.n_rq_stages):
            pred_patches = pred_patches + self.reconstruction_heads[i](quantized_per_stage[i])

        pred_x = pred_patches.reshape(batch_size, -1)
        pred_x = pred_x[..., :seq_len]

        return self.instance_norm.inverse(pred_x, loc_scale)

    @torch.no_grad()
    def impute(self, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        填补输入序列中的缺失值。

        参数：
            context: 输入时间序列 (B, seq_len)，缺失位置的值被忽略
            mask:    布尔张量 (B, seq_len)，True=已观测，False=缺失

        返回：
            imputed: 完整序列 (B, seq_len)，缺失位置已被填补，在原始空间
        """
        batch_size = context.shape[0]

        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length:]
            mask = mask[..., -self.chronos_config.context_length:]

        context_norm, loc_scale = self.instance_norm(context)
        context_norm = context_norm.to(self.dtype)
        mask = mask.to(self.dtype)

        # 将缺失位置置零（归一化空间中 0 ≈ 均值）
        context_norm = context_norm * mask

        seq_len = context_norm.shape[-1]

        patched_context = self.patch(context_norm)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        attention_mask = patched_mask.sum(dim=-1) > 0

        with torch.no_grad():
            input_embeds = self.input_patch_embedding(patched_context)
            encoder_outputs = self.encoder(
                attention_mask=attention_mask,
                inputs_embeds=input_embeds,
            )
            hidden_states = encoder_outputs[0]

        quantized, _commit_loss, quantized_per_stage = self.residual_quantizer(hidden_states)

        pred_patches = torch.zeros(
            batch_size, quantized.shape[1], self.chronos_config.input_patch_size,
            device=quantized.device, dtype=quantized.dtype,
        )
        for i in range(self.n_rq_stages):
            pred_patches = pred_patches + self.reconstruction_heads[i](quantized_per_stage[i])

        pred_x = pred_patches.reshape(batch_size, -1)
        pred_x = pred_x[..., :seq_len]

        return self.instance_norm.inverse(pred_x, loc_scale)

    @torch.no_grad()
    def anomaly_score(self, context: torch.Tensor, criterion: str = "mse") -> torch.Tensor:
        """
        通过重建误差计算逐点异常分数。

        参数：
            context:   输入时间序列 (B, seq_len)
            criterion: "mse" 或 "mae"

        返回：
            scores: 逐点异常分数 (B, seq_len)，值越大越异常
        """
        reconstruction = self.reconstruct(context)
        t = min(context.shape[-1], reconstruction.shape[-1])
        context_aligned = context[..., :t].float()
        reconstruction_aligned = reconstruction[..., :t]

        context_norm, loc_scale = self.instance_norm(context_aligned)
        reconstruction_norm, _ = self.instance_norm(reconstruction_aligned, loc_scale)

        if criterion == "mse":
            return (reconstruction_norm - context_norm) ** 2
        elif criterion == "mae":
            return torch.abs(reconstruction_norm - context_norm)
        else:
            raise ValueError(f"Unknown anomaly criterion: {criterion}")

    def save_codebooks(self, output_path: str) -> None:
        """Save trained RQ-VAE codebooks and reconstruction heads for downstream use."""
        state = {
            f"codebook_{i}": cb.weight.detach().cpu().float()
            for i, cb in enumerate(self.residual_quantizer.codebooks)
        }
        state["reconstruction_heads"] = {
            f"head_{i}": h.state_dict() for i, h in enumerate(self.reconstruction_heads)
        }
        torch.save(state, output_path)
        print(f"Saved {len(state) - 1} codebooks + {len(self.reconstruction_heads)} heads to {output_path}")

    @classmethod
    def from_pretrained_for_inference(
        cls,
        pretrained_model_path: str,
        codebook_path: str,
        n_proto: int = 256,
        n_rq_stages: int = 4,
    ) -> "ChronosBoltWithRQVAEPrototypes":
        """
        加载预训练 backbone 和阶段一码本，设置为 eval 模式用于下游推理。

        用法示例：
            model = ChronosBoltWithRQVAEPrototypes.from_pretrained_for_inference(
                pretrained_model_path='./checkpoints/base/',
                codebook_path='./checkpoints/prototypes_rqvae.pt',
            )
        """
        model = cls.from_pretrained_for_rqvae(
            pretrained_model_path=pretrained_model_path,
            n_proto=n_proto,
            n_rq_stages=n_rq_stages,
        )
        model.load_trained_codebook(codebook_path)
        model.eval()
        return model

    @classmethod
    def from_pretrained_for_rqvae(
        cls,
        pretrained_model_path: str,
        n_proto: int = 256,
        n_rq_stages: int = 4,
        beta: float = 0.25,
        mask_ratio: float = 0.1,
    ) -> "ChronosBoltWithRQVAEPrototypes":
        """
        加载预训练 ChronosBolt backbone 权重，冻结之，然后初始化 RQ-VAE 模块。

        用法示例：
            model = ChronosBoltWithRQVAEPrototypes.from_pretrained_for_rqvae(
                pretrained_model_path='./checkpoints/base/',
                n_proto=256,
            )
        """
        from transformers import AutoConfig
        from safetensors.torch import load_file

        # 加载模型配置
        config = AutoConfig.from_pretrained(pretrained_model_path)
        model = cls(
            config,
            n_proto=n_proto,
            n_rq_stages=n_rq_stages,
            beta=beta,
            mask_ratio=mask_ratio,
        )

        # 加载 backbone 权重（支持 .pth 和 .safetensors 两种格式）
        backbone_ckpt = os.path.join(pretrained_model_path, "autogluon_model.pth")
        safetensors_ckpt = os.path.join(pretrained_model_path, "model.safetensors")

        if os.path.exists(backbone_ckpt):
            state = torch.load(backbone_ckpt, map_location="cpu")
        elif os.path.exists(safetensors_ckpt):
            state = load_file(safetensors_ckpt, device="cpu")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {pretrained_model_path}. "
                f"Expected autogluon_model.pth or model.safetensors."
            )

        # strict=False：仅加载 backbone 中存在的权重，忽略 RQ-VAE 新增的参数
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"Backbone loaded from {pretrained_model_path}  "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

        # 冻结 backbone，仅码本和重建头可训练
        model.freeze_backbone()
        return model


# ---------------------------------------------------------------------------
# 模型 2：Few-shot 离散码条件预测（阶段二：下游微调）
# ---------------------------------------------------------------------------

class ChronosBoltWithDiscreteCodes(ChronosBoltModelForForecasting):
    """
    使用 RQ-VAE 离散码作为条件信号的 Few-shot 预测模型。

    架构概览：
      - 【冻结】T5 encoder + decoder + input/output embeddings
      - 【冻结】RQ-VAE 码本（仅用于离散码查找，不参与梯度更新）
      - 【可训练】code_embedding_tables（4 个 Embedding 层，将离散码映射为连续向量）
      - 【可训练】fusion_layer（将码条件与 encoder 输出融合）

    融合模式：
      - 'gate':       每阶段一个可学习标量 gate_i，enhanced = hidden + Σ σ(gate_i) × code_embed_i
      - 'concat':     拼接后线性投影，[hidden; code_embed] → Linear → enhanced
      - 'cross_attn': 交叉注意力 + FFN，hidden 做 query，code_embed 做 key/value

    参数：
        config:           T5Config（从预训练 checkpoint 加载）
        codebook_path:    prototypes_rqvae.pt 路径（包含阶段一训练的码本权重）
        n_proto:          每层码本大小，默认 256
        n_rq_stages:      残差量化阶段数，默认 4
        fusion_mode:      融合模式，可选 'gate' / 'concat' / 'cross_attn'
        freeze_codebooks: 是否冻结码本权重（默认 True）
    """

    def __init__(
        self,
        config: T5Config,
        codebook_path: str,
        n_proto: int = 256,
        n_rq_stages: int = 4,
        fusion_mode: str = "gate",
        freeze_codebooks: bool = True,
    ):
        super().__init__(config)

        d_model = config.d_model  # 768
        self.n_rq_stages = n_rq_stages
        self.fusion_mode = fusion_mode

        # ── 从磁盘加载阶段一训练好的 RQ-VAE 码本权重 ──
        ckpt = torch.load(codebook_path, map_location="cpu")
        codebook_weights = [ckpt[f"codebook_{i}"] for i in range(n_rq_stages)]

        # ── 冻结的 RQ 量化器（仅用于离散码查找，不做训练） ──
        self.rq_quantizer = ResidualQuantizer(
            n_stages=n_rq_stages, n_proto=n_proto, d_model=d_model
        )
        for i, cb in enumerate(self.rq_quantizer.codebooks):
            cb.weight.data.copy_(codebook_weights[i].to(cb.weight.dtype))
            if freeze_codebooks:
                cb.weight.requires_grad = False

        # ── 可学习的 Code Embedding 表（从码本权重初始化） ──
        # 每阶段一个独立的 Embedding 表，将离散码索引映射为连续向量
        self.code_embedding_tables = nn.ModuleList([
            nn.Embedding(n_proto, d_model) for _ in range(n_rq_stages)
        ])
        for i, emb in enumerate(self.code_embedding_tables):
            emb.weight.data.copy_(codebook_weights[i].to(emb.weight.dtype))

        # ── 融合层：将 encoder hidden states 与 code condition 融合 ──
        if fusion_mode == "gate":
            # 门控融合：每阶段一个可学习标量，通过 sigmoid 控制信息流入量
            self.gate_scalars = nn.Parameter(torch.zeros(n_rq_stages))
        elif fusion_mode == "concat":
            # 拼接融合：将 hidden 和 code_embed 拼接后线性投影回 d_model
            self.fusion_proj = nn.Linear(2 * d_model, d_model)
        elif fusion_mode == "cross_attn":
            # 交叉注意力融合：hidden 做 Q，code_embed 做 K/V，再加 FFN
            self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
            self.attn_norm = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
            self.ffn_norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        # 冻结 backbone
        self.freeze_backbone()

    def freeze_backbone(self):
        """冻结 ChronosBolt 主干网络所有参数"""
        for mod in [
            self.encoder,
            self.decoder,
            self.shared,
            self.input_patch_embedding,
            self.output_patch_embedding,
        ]:
            for p in mod.parameters():
                p.requires_grad = False

    def print_trainable_params(self):
        """打印总参数量和可训练参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameters: total={total:,}, trainable={trainable:,}")

    def _fuse(self, hidden_states: torch.Tensor, code_condition: torch.Tensor) -> torch.Tensor:
        """
        将 encoder 隐藏状态与离散码条件融合。

        参数：
            hidden_states:  encoder 输出 (B, n_patches, d_model)
            code_condition: 离散码嵌入，形状取决于 fusion_mode：
                            - gate: (B, n_patches, n_stages, d_model)
                            - concat/cross_attn: (B, n_patches, d_model)

        返回：
            enhanced: 融合后的隐藏状态 (B, n_patches, d_model)
        """
        if self.fusion_mode == "gate":
            # 门控融合：每阶段独立加权加和
            # enhanced = hidden + Σ σ(gate_i) × code_embed_i
            enhanced = hidden_states.clone()
            for i in range(self.n_rq_stages):
                gate = torch.sigmoid(self.gate_scalars[i])  # 标量 gate ∈ (0, 1)
                enhanced = enhanced + gate * code_condition[..., i, :]
            return enhanced

        elif self.fusion_mode == "concat":
            # 拼接融合：将 hidden 和 code_embed 在最后一维拼接后线性投影
            fused = torch.cat([hidden_states, code_condition], dim=-1)  # (B, n_patches, 2*d_model)
            return self.fusion_proj(fused)                               # (B, n_patches, d_model)

        elif self.fusion_mode == "cross_attn":
            # 交叉注意力融合：hidden 做 query，code_embed 做 key/value
            # 相当于让 encoder 表示"关注"离散码信息
            attn_out, _ = self.cross_attn(hidden_states, code_condition, code_condition)
            hidden_states = self.attn_norm(hidden_states + attn_out)     # 残差连接 + LayerNorm
            ffn_out = self.ffn(hidden_states)
            return self.ffn_norm(hidden_states + ffn_out)                # FFN + 残差连接 + LayerNorm

        else:
            return hidden_states  # fallback：不做融合

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> ChronosBoltOutput:
        """
        Few-shot 预测的前向传播。

        流程概览：
          1. context → InstanceNorm → Patch → input_patch_embedding
          2. 冻结 Encoder → hidden_states
          3. 冻结 RQ 码本 → 离散码索引 → 可学习 Embedding 查表 → code_condition
          4. 融合层：hidden_states + code_condition → enhanced_hidden_states
          5. 冻结 Decoder → output_patch_embedding → 分位数预测
          6. 若提供 target，计算 Quantile Loss（Pinball Loss）

        参数：
            context:     输入时间序列 (B, seq_len)
            mask:        可选的掩码张量
            target:      真实未来值 (B, label_len+pred_len)，训练时提供
            target_mask: 目标掩码

        返回：
            ChronosBoltOutput，包含 .loss 和 .quantile_preds
        """
        # 自动生成 mask（将 NaN 视为无效位置）
        mask = mask.to(context.dtype) if mask is not None else torch.isnan(context).logical_not().to(context.dtype)

        batch_size, _ = context.shape
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length:]
            mask = mask[..., -self.chronos_config.context_length:]

        # ── 步骤 1：实例归一化 ──
        context, loc_scale = self.instance_norm(context)
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)

        # ── 步骤 2：Patch 分块 ──
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)

        attention_mask = patched_mask.sum(dim=-1) > 0  # (B, n_patches)

        input_embeds = self.input_patch_embedding(patched_context)

        # 可选的 [REG] token（ChronosBolt 原生的回归 token）
        if self.chronos_config.use_reg_token:
            reg_input_ids = torch.full(
                (batch_size, 1),
                self.config.reg_token_id,
                device=input_embeds.device,
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat([attention_mask, torch.ones_like(reg_input_ids)], dim=-1)

        # ── 步骤 3：冻结 Encoder ──
        # 使用 no_grad 节省显存，因为 backbone 不参与训练
        with torch.no_grad():
            encoder_outputs = self.encoder(
                attention_mask=attention_mask,
                inputs_embeds=input_embeds,
            )
            hidden_states = encoder_outputs[0]  # (B, n_patches, d_model)

        # ── 步骤 4：离散码查找（冻结码本） ──
        # 对每个 patch 的隐藏状态，在 4 层码本中找最近邻 → 得到离散索引
        codes = self.rq_quantizer.get_codes(hidden_states)  # (B, n_patches, n_stages)，LongTensor

        # ── 步骤 5：可学习 Code Embedding 查表 ──
        # 将离散码索引通过可学习的 Embedding 表映射为连续向量
        # 4 个阶段的嵌入加和得到最终的 code_condition
        code_condition = torch.zeros_like(hidden_states)
        for i, emb_table in enumerate(self.code_embedding_tables):
            code_condition = code_condition + emb_table(codes[..., i])

        # ── 步骤 6：融合层 ──
        # gate 模式需要每阶段独立的嵌入，其它模式使用加和后的
        if self.fusion_mode == "gate":
            per_stage_embeds = torch.stack(
                [self.code_embedding_tables[i](codes[..., i]) for i in range(self.n_rq_stages)],
                dim=-2,
            )  # (B, n_patches, n_stages, d_model)
            enhanced_hidden_states = self._fuse(hidden_states, per_stage_embeds)
        else:
            enhanced_hidden_states = self._fuse(hidden_states, code_condition)

        # ── 步骤 7：冻结 Decoder ──
        # 注意：这里不用 no_grad！梯度需要通过冻结的 decoder 回传到
        # code_embedding_tables 和融合层。虽然 decoder 参数不更新，
        # 但梯度可以"穿过"它们到达前面的可训练参数。
        sequence_output = self.decode(input_embeds, attention_mask, enhanced_hidden_states)

        # ── 步骤 8：输出投影 → 分位数预测 ──
        quantile_preds_shape = (
            batch_size,
            self.num_quantiles,                # 分位数个数，例如 9
            self.chronos_config.prediction_length,  # 预测长度，例如 64
        )
        quantile_preds = self.output_patch_embedding(sequence_output).view(*quantile_preds_shape)

        # ── 步骤 9：计算 Quantile Loss（Pinball Loss） ──
        loss = None
        if target is not None:
            # 使用与 context 相同的 loc_scale 归一化 target
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)  # (B, 1, pred_len)
            assert self.chronos_config.prediction_length >= target.shape[-1]

            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device) if target_mask is not None else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0

            # 如果 target 长度不足，用 0 填充
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (*target.shape[:-1], self.chronos_config.prediction_length - target.shape[-1])
                target = torch.cat([target, torch.zeros(padding_shape).to(target)], dim=-1)
                target_mask = torch.cat([target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1)

            # Pinball Loss（分位数损失）：
            #   loss = 2 * |target - pred| * (I(target <= pred) - quantile_level)
            # 当 pred < target 时惩罚权重 = 1 - quantile（低估惩罚大，因为是上分位数）
            # 当 pred > target 时惩罚权重 = quantile（高估惩罚大，因为是下分位数）
            loss = (
                2
                * torch.abs(
                    (target - quantile_preds)
                    * ((target <= quantile_preds).float() - self.quantiles.view(1, self.num_quantiles, 1))
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2)   # 在预测长度维度取均值
            loss = loss.sum(dim=-1)    # 在分位数维度求和
            loss = loss.mean()         # 在 batch 维度取均值

        # ── 步骤 10：逆归一化预测结果 ──
        # 将归一化空间的预测还原到原始数值空间
        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*quantile_preds_shape)

        return ChronosBoltOutput(loss=loss, quantile_preds=quantile_preds)

    # -------------------------------------------------------------------
    # Imputation 下游任务方法
    # -------------------------------------------------------------------

    def add_imputation_heads(self, init_from_stage1_ckpt: Optional[str] = None):
        """添加可训练的重建头，用于 imputation 下游任务。

        不在 __init__ 中创建，由训练脚本显式调用，
        使得同一模型实例可按需在 forecasting / imputation 模式间切换。
        """
        d_model = self.config.d_model
        patch_size = self.chronos_config.input_patch_size
        self.imputation_heads = nn.ModuleList([
            nn.Linear(d_model, patch_size) for _ in range(self.n_rq_stages)
        ])
        for head in self.imputation_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

        if init_from_stage1_ckpt is not None:
            ckpt = torch.load(init_from_stage1_ckpt, map_location="cpu")
            for i, head in enumerate(self.imputation_heads):
                head.load_state_dict(ckpt["reconstruction_heads"][f"head_{i}"])
            print(f"Initialized imputation_heads from {init_from_stage1_ckpt}")

    def forward_imputation(
        self,
        context: torch.Tensor,
        mask_ratio: float = 0.1,
    ) -> ChronosBoltOutput:
        """Imputation 训练的前向传播：随机掩码 → 编码 → 码条件融合 → 重建 → MSE。

        context (B, seq_len) → random mask → frozen Encoder → RQ codes
          → code_embedding_tables → _fuse() → enhanced_hidden_states
          → imputation_heads → sum → unflatten → MSE(target, pred)
        """
        batch_size = context.shape[0]

        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length:]

        context_norm, loc_scale = self.instance_norm(context)
        context_norm = context_norm.to(self.dtype)
        target = context_norm.detach().clone()

        # 随机掩码
        seq_len = context_norm.shape[-1]
        recovery_mask = torch.rand(batch_size, seq_len, device=context.device) < mask_ratio
        context_masked = context_norm.clone()
        context_masked[recovery_mask] = 0.0

        attn_mask_1d = torch.ones(batch_size, seq_len, device=context.device, dtype=self.dtype)

        # Patch 分块
        patched_context = self.patch(context_masked)
        patched_mask = torch.nan_to_num(self.patch(attn_mask_1d), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        attention_mask = patched_mask.sum(dim=-1) > 0

        # 冻结 Encoder
        with torch.no_grad():
            input_embeds = self.input_patch_embedding(patched_context)
            encoder_outputs = self.encoder(
                attention_mask=attention_mask,
                inputs_embeds=input_embeds,
            )
            hidden_states = encoder_outputs[0]

        # 离散码查找 (冻结码本)
        codes = self.rq_quantizer.get_codes(hidden_states)  # (B, n_patches, n_stages)

        # Code Embedding 查表 + 融合
        per_stage_embeds = torch.stack(
            [self.code_embedding_tables[i](codes[..., i]) for i in range(self.n_rq_stages)],
            dim=-2,
        )  # (B, n_patches, n_stages, d_model)

        if self.fusion_mode == "gate":
            enhanced_hidden_states = self._fuse(hidden_states, per_stage_embeds)
        else:
            code_condition = per_stage_embeds.sum(dim=-2)  # (B, n_patches, d_model)
            enhanced_hidden_states = self._fuse(hidden_states, code_condition)

        # 重建头解码
        pred_patches = torch.zeros(
            batch_size, enhanced_hidden_states.shape[1], self.chronos_config.input_patch_size,
            device=enhanced_hidden_states.device, dtype=enhanced_hidden_states.dtype,
        )
        for i in range(self.n_rq_stages):
            pred_patches = pred_patches + self.imputation_heads[i](enhanced_hidden_states)

        pred_x = pred_patches.reshape(batch_size, -1)
        pred_x = pred_x[..., :target.shape[-1]]

        # MSE 重建损失 (所有位置)
        recon_loss = F.mse_loss(pred_x, target)
        total_loss = recon_loss

        return ChronosBoltOutput(loss=total_loss, reconstruction=pred_x)

    @torch.no_grad()
    def impute(self, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """用训练好的 code_embedding_tables + imputation_heads 填补缺失值。

        context: (B, seq_len) 含缺失值的输入序列
        mask:    (B, seq_len) 布尔/浮点，True=已观测，False=缺失

        返回: (B, seq_len) 完整序列(原始空间)，缺失位置已被填补
        """
        batch_size = context.shape[0]

        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length:]
            mask = mask[..., -self.chronos_config.context_length:]

        context_norm, loc_scale = self.instance_norm(context)
        context_norm = context_norm.to(self.dtype)
        mask = mask.to(self.dtype)
        context_norm = context_norm * mask  # 缺失位置置零

        seq_len = context_norm.shape[-1]

        patched_context = self.patch(context_norm)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        attention_mask = patched_mask.sum(dim=-1) > 0

        with torch.no_grad():
            input_embeds = self.input_patch_embedding(patched_context)
            encoder_outputs = self.encoder(
                attention_mask=attention_mask,
                inputs_embeds=input_embeds,
            )
            hidden_states = encoder_outputs[0]

        codes = self.rq_quantizer.get_codes(hidden_states)

        per_stage_embeds = torch.stack(
            [self.code_embedding_tables[i](codes[..., i]) for i in range(self.n_rq_stages)],
            dim=-2,
        )

        if self.fusion_mode == "gate":
            enhanced_hidden_states = self._fuse(hidden_states, per_stage_embeds)
        else:
            code_condition = per_stage_embeds.sum(dim=-2)
            enhanced_hidden_states = self._fuse(hidden_states, code_condition)

        pred_patches = torch.zeros(
            batch_size, enhanced_hidden_states.shape[1], self.chronos_config.input_patch_size,
            device=enhanced_hidden_states.device, dtype=enhanced_hidden_states.dtype,
        )
        for i in range(self.n_rq_stages):
            pred_patches = pred_patches + self.imputation_heads[i](enhanced_hidden_states)

        pred_x = pred_patches.reshape(batch_size, -1)
        pred_x = pred_x[..., :seq_len]

        return self.instance_norm.inverse(pred_x, loc_scale)

    @classmethod
    def from_pretrained_with_codes(
        cls,
        pretrained_model_path: str,
        codebook_path: str,
        n_proto: int = 256,
        n_rq_stages: int = 4,
        fusion_mode: str = "gate",
        freeze_codebooks: bool = True,
    ) -> "ChronosBoltWithDiscreteCodes":
        """
        加载 ChronosBolt backbone、RQ-VAE 码本，并初始化离散码条件模块。

        步骤：
          1. 加载预训练 backbone 权重 → 冻结
          2. 加载阶段一训练的码本权重 → 冻结（可选）
          3. 初始化可学习的 code_embedding_tables + 融合层

        用法示例：
            model = ChronosBoltWithDiscreteCodes.from_pretrained_with_codes(
                pretrained_model_path='./checkpoints/base/',
                codebook_path='./checkpoints/prototypes_rqvae.pt',
                fusion_mode='gate',
            )
        """
        from transformers import AutoConfig
        from safetensors.torch import load_file

        config = AutoConfig.from_pretrained(pretrained_model_path)
        model = cls(
            config,
            codebook_path=codebook_path,
            n_proto=n_proto,
            n_rq_stages=n_rq_stages,
            fusion_mode=fusion_mode,
            freeze_codebooks=freeze_codebooks,
        )

        # 加载 backbone 权重
        backbone_ckpt = os.path.join(pretrained_model_path, "autogluon_model.pth")
        safetensors_ckpt = os.path.join(pretrained_model_path, "model.safetensors")

        if os.path.exists(backbone_ckpt):
            state = torch.load(backbone_ckpt, map_location="cpu")
        elif os.path.exists(safetensors_ckpt):
            state = load_file(safetensors_ckpt, device="cpu")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {pretrained_model_path}. "
                f"Expected autogluon_model.pth or model.safetensors."
            )

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"Backbone loaded from {pretrained_model_path}  "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

        model.freeze_backbone()
        return model
