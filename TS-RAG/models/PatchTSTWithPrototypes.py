"""
PatchTSTWithPrototypes

Channel-independent PatchTST with frozen prototype retrieval via cosine
similarity inserted between the Transformer encoder and the linear forecast head.

Architecture:
    Input (B, seq_len, C)
        ↓  RevIN normalization (per series)
        ↓  Reshape → (B*C, seq_len)
        ↓  PatchEmbedding: unfold → linear → (B*C, n_patches, d_model)
        ↓  PatchTSTEncoder: Transformer encoder → (B*C, n_patches, d_model)
        ↓  Prototype retrieval: cosine similarity → best prototype per patch
           → (B*C, n_patches, d_model)  [frozen prototype embeddings]
        ↓  PatchTSTDecoder: flatten → linear → (B*C, pred_len)
        ↓  Reshape → (B, pred_len, C)

For each patch embedding, the most similar prototype (by cosine similarity)
is retrieved and passed directly to the decoder instead of the encoder output.
Prototypes are stored as a frozen register_buffer.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder


# ---------------------------------------------------------------------------
# RevIN (instance normalization with optional affine transform)
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    """
    Reversible instance normalisation (per time-series, over the time axis).
    Applied before patching and reversed after decoding.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Args:
            x:    (B, T, C)
            mode: 'norm' or 'denorm'
        """
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()          # (B, 1, C)
            self.stdev = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + self.eps
            ).detach()                                                   # (B, 1, C)
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
        return x


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Splits each univariate series into overlapping patches and projects to d_model.

    Input:  (N, seq_len)          N = B * C (channel-independent)
    Output: (N, n_patches, d_model)
    """

    def __init__(self, seq_len: int, patch_len: int, stride: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1

        self.projection = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, seq_len)
        x = x.unsqueeze(1)                                   # (N, 1, seq_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: (N, 1, n_patches, patch_len)
        x = x.squeeze(1)                                     # (N, n_patches, patch_len)
        x = self.projection(x)                               # (N, n_patches, d_model)
        x = x + self.pos_embed
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer encoder
# ---------------------------------------------------------------------------

class PatchTSTEncoder(nn.Module):
    """
    Stack of Transformer encoder layers (standard PyTorch implementation).

    Input / output: (N, n_patches, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(x))


# ---------------------------------------------------------------------------
# Flatten head (decoder)
# ---------------------------------------------------------------------------

class PatchTSTDecoder(nn.Module):
    """
    Linear forecast head: flattens patch representations and projects to pred_len.

    Input:  (N, n_patches, d_model)
    Output: (N, pred_len)
    """

    def __init__(self, n_patches: int, d_model: int, pred_len: int, dropout: float = 0.1):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(n_patches * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(self.flatten(x)))    # (N, pred_len)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class PatchTSTWithPrototypes(nn.Module):
    """
    Channel-independent PatchTST with frozen prototype retrieval.

    For each patch embedding produced by the encoder, the most similar
    prototype (cosine similarity) is retrieved and passed to the decoder.
    The encoder output is used only for retrieval — it is NOT passed to
    the decoder directly.

    Frozen:  prototype_matrix (register_buffer, not in parameters())
    Trained: RevIN, PatchEmbedding, PatchTSTEncoder, PatchTSTDecoder

    Args:
        c_in:      Number of input channels (variables)
        seq_len:   Input sequence length
        pred_len:  Forecast horizon
        patch_len: Length of each patch token (default 16)
        stride:    Stride between patches (default 8)
        d_model:   Hidden dimension — MUST match prototype dimension (default 768)
        n_heads:   Attention heads in encoder
        n_layers:  Number of Transformer encoder layers
        d_ff:      Feed-forward hidden dimension
        dropout:   Dropout probability
    """

    def __init__(
        self,
        c_in: int = 7,
        seq_len: int = 336,
        pred_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.c_in = c_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        n_patches = (seq_len - patch_len) // stride + 1
        self.n_patches = n_patches

        # Sub-modules
        self.revin = RevIN(num_features=c_in, affine=True)
        self.patch_embed = PatchEmbedding(seq_len, patch_len, stride, d_model, dropout)
        self.encoder = PatchTSTEncoder(d_model, n_heads, d_ff, n_layers, dropout)
        self.decoder = PatchTSTDecoder(n_patches, d_model, pred_len, dropout)

        # Prototype matrix: frozen register_buffer (not in parameters())
        self.register_buffer(
            "prototype_matrix",
            torch.zeros(512, d_model),
            persistent=True,
        )

    # ------------------------------------------------------------------
    def load_prototypes(self, prototype_path: str):
        """Load prototype centroids from .npy or .pt file."""
        if prototype_path.endswith(".npy"):
            proto = torch.from_numpy(np.load(prototype_path)).float()
        elif prototype_path.endswith(".pt"):
            proto = torch.load(prototype_path, map_location="cpu").float()
        else:
            raise ValueError(f"Unsupported prototype format: {prototype_path}")

        assert proto.ndim == 2 and proto.shape[1] == self.d_model, (
            f"Expected prototype shape (n_clusters, {self.d_model}), got {proto.shape}"
        )
        self.register_buffer("prototype_matrix", proto, persistent=True)
        print(f"Loaded {proto.shape[0]} prototypes from {prototype_path}")

    # ------------------------------------------------------------------
    def retrieve_prototypes(self, encoder_out: torch.Tensor):
        """
        VQ-VAE style prototype retrieval:
          1. Find nearest prototype per patch via cosine similarity (argmax)
          2. Apply straight-through estimator so gradients flow to encoder
          3. Compute commitment loss (encoder commits to nearest prototype)

        Args:
            encoder_out: (N, n_patches, d_model)

        Returns:
            proto_st:      (N, n_patches, d_model)  straight-through output
                           = retrieved prototype in forward,
                           = encoder_out in backward (gradient bypass)
            commit_loss:   scalar — beta-weighted commitment loss
        """
        proto = self.prototype_matrix.to(encoder_out.dtype)   # (n_clusters, d_model)

        # L2-normalise for cosine similarity
        enc_norm   = F.normalize(encoder_out, dim=-1)         # (N, n_patches, d_model)
        proto_norm = F.normalize(proto, dim=-1)               # (n_clusters, d_model)

        # Cosine similarity → (N, n_patches, n_clusters)
        sim = torch.einsum('npd,kd->npk', enc_norm, proto_norm)

        # Nearest prototype index per patch (argmax, non-differentiable)
        best_idx = sim.argmax(dim=-1)                         # (N, n_patches)
        retrieved = proto[best_idx]                           # (N, n_patches, d_model)

        # Straight-through estimator:
        # forward pass  : proto_st == retrieved  (discrete prototype values)
        # backward pass : gradient flows to encoder_out (bypasses argmax)
        proto_st = encoder_out + (retrieved - encoder_out).detach()

        # Commitment loss: pull encoder output toward nearest prototype
        # stop_gradient on prototype side (frozen anyway, but explicit for clarity)
        commit_loss = F.mse_loss(encoder_out, retrieved.detach())

        return proto_st, commit_loss

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, seq_len, C)

        Returns:
            out:         (B, pred_len, C)
            commit_loss: scalar — use in training loss as beta * commit_loss
        """
        B, T, C = x.shape

        # 1. Instance normalisation
        x = self.revin(x, 'norm')                             # (B, T, C)

        # 2. Channel independence
        x = x.permute(0, 2, 1).reshape(B * C, T)             # (B*C, T)

        # 3. Patch embedding
        x = self.patch_embed(x)                              # (B*C, n_patches, d_model)

        # 4. Transformer encoder
        enc_out = self.encoder(x)                            # (B*C, n_patches, d_model)

        # 5. VQ-VAE prototype retrieval (straight-through + commitment loss)
        proto_st, commit_loss = self.retrieve_prototypes(enc_out)
        # proto_st: (B*C, n_patches, d_model)

        # 6. Forecast head (receives straight-through prototype)
        out = self.decoder(proto_st)                         # (B*C, pred_len)

        # 7. Reshape back and reverse normalisation
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)  # (B, pred_len, C)
        out = self.revin(out, 'denorm')                      # (B, pred_len, C)

        return out, commit_loss

    # ------------------------------------------------------------------
    def print_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"Parameters: total={total:,}  trainable={trainable:,}  "
            f"frozen_prototypes={self.prototype_matrix.numel():,}"
        )
