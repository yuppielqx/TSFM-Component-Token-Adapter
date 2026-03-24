import os
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from transformers.models.t5.modeling_t5 import T5Config, T5LayerNorm

from .ChronosBolt import ChronosBoltModelForForecasting, ChronosBoltOutput


class PrototypeCrossAttentionLayer(nn.Module):
    """
    Single prototype cross-attention layer inserted between T5 Encoder and Decoder.

    Structure (pre-norm):
        x_norm = LayerNorm(x)
        attn_out = MultiheadAttention(Q=x_norm, K=prototypes, V=prototypes)
        x = x + dropout(attn_out)
        x_norm2 = LayerNorm(x)
        ffn_out = FFN(x_norm2)
        x = x + ffn_out

    Args:
        d_model: Feature dimension (768 for ChronosBolt-base)
        num_heads: Number of attention heads
        d_ff: FFN hidden dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        # Use T5LayerNorm to match backbone convention
        self.norm1 = T5LayerNorm(d_model)
        self.norm2 = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: encoder hidden states, shape (B, n_patches, d_model)
            prototypes: prototype matrix, shape (n_clusters, d_model)

        Returns:
            augmented: shape (B, n_patches, d_model)
        """
        batch_size = x.shape[0]
        # broadcast: (n_clusters, d_model) -> (B, n_clusters, d_model)
        proto = prototypes.unsqueeze(0).expand(batch_size, -1, -1)

        # Pre-norm cross-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.cross_attn(
            query=x_norm,
            key=proto,
            value=proto,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class ChronosBoltWithPrototypes(ChronosBoltModelForForecasting):
    """
    ChronosBolt with a frozen prototype matrix and a learnable cross-attention layer.

    Architecture:
        context -> InstanceNorm -> Patch -> input_patch_embedding
        -> T5Encoder -> hidden_states
        -> PrototypeCrossAttentionLayer  [prototype_matrix is frozen]
        -> T5Decoder -> output_patch_embedding -> quantile_preds

    Training strategy:
        - prototype_matrix: register_buffer (not in parameters(), never updated)
        - prototype_cross_attn: trainable
        - T5 encoder/decoder + patch embeddings: trainable by default;
          call freeze_backbone() to freeze them for extreme few-shot scenarios.

    Usage:
        model = ChronosBoltWithPrototypes.from_pretrained_with_prototypes(
            pretrained_model_path='./checkpoints/base/',
            prototype_path='./checkpoints/prototypes_k512.pt',
        )
    """

    _keys_to_ignore_on_load_missing = [
        r"input_patch_embedding\.",
        r"output_patch_embedding\.",
        r"prototype_cross_attn\.",
        r"prototype_matrix",
    ]

    def __init__(
        self,
        config: T5Config,
        prototype_path: Optional[str] = None,
        num_heads: int = 8,
        proto_dropout: float = 0.1,
    ):
        super().__init__(config)

        d_model = config.d_model
        d_ff = config.d_ff

        self.prototype_cross_attn = PrototypeCrossAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=proto_dropout,
        )

        # Placeholder buffer; replaced by load_prototypes()
        self.register_buffer(
            "prototype_matrix",
            torch.zeros(512, d_model),
            persistent=True,
        )

        if prototype_path is not None:
            self.load_prototypes(prototype_path)

    def load_prototypes(self, prototype_path: str):
        """Load prototype matrix from a .npy or .pt file and store as frozen buffer."""
        if not os.path.exists(prototype_path):
            raise FileNotFoundError(f"Prototype file not found: {prototype_path}")

        if prototype_path.endswith(".npy"):
            proto_tensor = torch.from_numpy(np.load(prototype_path)).float()
        elif prototype_path.endswith(".pt"):
            proto_tensor = torch.load(prototype_path, map_location="cpu").float()
        else:
            raise ValueError(f"Unsupported format: {prototype_path}. Use .npy or .pt")

        assert proto_tensor.ndim == 2, f"Prototype must be 2-D, got shape {proto_tensor.shape}"
        assert proto_tensor.shape[1] == self.model_dim, (
            f"Prototype dim {proto_tensor.shape[1]} != model dim {self.model_dim}"
        )

        # Re-register with correct shape; register_buffer keeps requires_grad=False
        self.register_buffer("prototype_matrix", proto_tensor, persistent=True)
        print(f"Loaded prototypes: shape={proto_tensor.shape}, path={prototype_path}")

    def freeze_backbone(self):
        """
        Freeze the T5 encoder/decoder and patch embeddings.
        Only prototype_cross_attn will be trained.
        Useful for extreme few-shot scenarios.
        """
        backbone_modules = [
            self.encoder,
            self.decoder,
            self.shared,
            self.input_patch_embedding,
            self.output_patch_embedding,
        ]
        for mod in backbone_modules:
            for param in mod.parameters():
                param.requires_grad = False

    def print_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        proto_numel = self.prototype_matrix.numel()
        print(
            f"Parameters: total={total:,}, trainable={trainable:,}, "
            f"frozen prototype buffer={proto_numel:,}"
        )

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> ChronosBoltOutput:
        """
        Same as ChronosBoltModelForForecasting.forward(), with prototype cross-attention
        inserted after the encoder and before self.decode().
        """
        mask = (
            mask.to(context.dtype)
            if mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )

        batch_size, _ = context.shape
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length:]
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

        if self.chronos_config.use_reg_token:
            reg_input_ids = torch.full(
                (batch_size, 1),
                self.config.reg_token_id,
                device=input_embeds.device,
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(reg_input_ids, dtype=attention_mask.dtype)],
                dim=-1,
            )

        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
        )
        hidden_states = encoder_outputs[0]  # (B, n_patches, d_model)

        # --- Prototype cross-attention ---
        proto = self.prototype_matrix.to(hidden_states.dtype)
        hidden_states = self.prototype_cross_attn(hidden_states, proto)
        # ---------------------------------

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
                target_mask.unsqueeze(1).to(quantile_preds.device)
                if target_mask is not None
                else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0

            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (
                    *target.shape[:-1],
                    self.chronos_config.prediction_length - target.shape[-1],
                )
                target = torch.cat([target, torch.zeros(padding_shape).to(target)], dim=-1)
                target_mask = torch.cat(
                    [target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1
                )

            loss = (
                2
                * torch.abs(
                    (target - quantile_preds)
                    * (
                        (target <= quantile_preds).float()
                        - self.quantiles.view(1, self.num_quantiles, 1)
                    )
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2).sum(dim=-1).mean()

        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1), loc_scale
        ).view(*quantile_preds_shape)

        return ChronosBoltOutput(loss=loss, quantile_preds=quantile_preds)

    @classmethod
    def from_pretrained_with_prototypes(
        cls,
        pretrained_model_path: str,
        prototype_path: str,
        num_heads: int = 8,
        proto_dropout: float = 0.1,
        **kwargs,
    ) -> "ChronosBoltWithPrototypes":
        """
        Load backbone weights from pretrained_model_path, then load prototypes.

        Example:
            model = ChronosBoltWithPrototypes.from_pretrained_with_prototypes(
                pretrained_model_path='./checkpoints/base/',
                prototype_path='./checkpoints/prototypes_k512.pt',
            )
        """
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained_model_path)
        model = cls.from_pretrained(
            pretrained_model_path,
            config=config,
            prototype_path=None,
            num_heads=num_heads,
            proto_dropout=proto_dropout,
            **kwargs,
        )

        # Load backbone checkpoint if available
        backbone_ckpt = os.path.join(pretrained_model_path, "autogluon_model.pth")
        if os.path.exists(backbone_ckpt):
            state = torch.load(backbone_ckpt, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(
                f"Backbone loaded from {backbone_ckpt}  "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )

        # Load prototypes
        model.load_prototypes(prototype_path)

        # Initialize cross-attention weights
        _init_cross_attn_weights(model.prototype_cross_attn)

        return model


def _init_cross_attn_weights(layer: PrototypeCrossAttentionLayer):
    nn.init.xavier_uniform_(layer.cross_attn.in_proj_weight)
    nn.init.xavier_uniform_(layer.cross_attn.out_proj.weight)
    if layer.cross_attn.in_proj_bias is not None:
        layer.cross_attn.in_proj_bias.data.zero_()
    if layer.cross_attn.out_proj.bias is not None:
        layer.cross_attn.out_proj.bias.data.zero_()
    for m in layer.ffn.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()


class ChronosBoltWithLearnablePrototypes(ChronosBoltWithPrototypes):
    """
    ChronosBolt with a **learnable** prototype matrix.

    Differences from ChronosBoltWithPrototypes:
      - prototype_matrix is an nn.Parameter (trainable) instead of a register_buffer (frozen)
      - Backbone (encoder, decoder, patch embeddings) is frozen in __init__
      - Only prototype_matrix and prototype_cross_attn are trained

    Intended workflow:
      1. Instantiate via from_pretrained_for_proto_learning()
      2. Train on retrieval_database parquet (x=context, y=forecast target)
         using the built-in quantile loss (pass target= to forward())
      3. Call save_prototypes(path) to export prototype_matrix as .npy
      4. Load the saved .npy in PatchTSTWithPrototypes or ChronosBoltWithPrototypes
         for few-shot fine-tuning
    """

    def __init__(
        self,
        config: T5Config,
        n_proto: int = 512,
        num_heads: int = 8,
        proto_dropout: float = 0.1,
    ):
        # Parent creates register_buffer("prototype_matrix", zeros(512, d_model))
        super().__init__(config, prototype_path=None, num_heads=num_heads, proto_dropout=proto_dropout)

        d_model = config.d_model

        # Replace the frozen buffer with a trainable Parameter.
        # PyTorch __setattr__ automatically removes it from _buffers when
        # a Parameter with the same name is assigned.
        self.prototype_matrix = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(n_proto, d_model), std=0.02)
        )

        # Freeze backbone; only prototype_matrix + prototype_cross_attn remain trainable
        self.freeze_backbone()

    # ------------------------------------------------------------------
    def save_prototypes(self, output_path: str):
        """Save trained prototype_matrix to a .npy file."""
        proto = self.prototype_matrix.detach().cpu().float().numpy()
        np.save(output_path, proto)
        print(f"Saved prototypes: shape={proto.shape}, path={output_path}")

    # ------------------------------------------------------------------
    def print_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        proto_numel = self.prototype_matrix.numel()
        print(
            f"Parameters: total={total:,}, trainable={trainable:,} "
            f"(prototype={proto_numel:,} + cross_attn={trainable - proto_numel:,})"
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained_for_proto_learning(
        cls,
        pretrained_model_path: str,
        n_proto: int = 512,
        num_heads: int = 8,
        proto_dropout: float = 0.1,
    ) -> "ChronosBoltWithLearnablePrototypes":
        """
        Load backbone weights from pretrained_model_path, freeze them,
        and initialise a random learnable prototype matrix.

        Supports both model.safetensors (HuggingFace) and autogluon_model.pth formats.

        Example::
            model = ChronosBoltWithLearnablePrototypes.from_pretrained_for_proto_learning(
                pretrained_model_path='./checkpoints/base/',
                n_proto=512,
            )
        """
        from transformers import AutoConfig
        from safetensors.torch import load_file

        config = AutoConfig.from_pretrained(pretrained_model_path)

        # Instantiate model first (prototype_matrix initialised as nn.Parameter)
        model = cls(config, n_proto=n_proto, num_heads=num_heads, proto_dropout=proto_dropout)

        # Load backbone weights directly to avoid HuggingFace's from_pretrained
        # overwriting our prototype_matrix with an empty tensor
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

        # Re-apply freeze and init cross-attention
        model.freeze_backbone()
        _init_cross_attn_weights(model.prototype_cross_attn)

        return model
