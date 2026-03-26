"""
train_prototypes.py

Learn prototypes using a frozen ChronosBolt backbone.

Two training tasks are supported via --task:

  forecasting (default / legacy)
  --------------------------------
    context (B, context_length)
        ↓  frozen ChronosBolt encoder + preprocessing
        ↓  learnable PrototypeCrossAttentionLayer(Q=enc_out, K=V=prototypes)
        ↓  frozen ChronosBolt decoder
        →  quantile forecast  →  built-in quantile loss vs. target y

  recovery (new)
  ---------------
    context (B, context_length)
        ↓  random point-level masking (mask_ratio=0.1 → ~10% of points)
        ↓  frozen ChronosBolt encoder
        ↓  ResidualQuantizer (4 codebooks × 256 entries × 768 dim)
        ↓  linear reconstruction head
        →  reconstruct all 512 points  →  MSE + codebook/commitment losses
    Data split: 70% train / 10% val / 20% test

Usage
-----
    # forecasting task (legacy)
    python train_prototypes.py \\
        --task forecasting \\
        --pretrained_model_path ./checkpoints/base/ \\
        --parquet_path ./datasets/pretrain/retrieval_database_512.parquet \\
        --n_clusters 512 \\
        --train_epochs 5 \\
        --batch_size 256 \\
        --output_path ./checkpoints/prototypes_learned.npy

    # recovery task (RQ-VAE)
    python train_prototypes.py \\
        --task recovery \\
        --pretrained_model_path ./checkpoints/base/ \\
        --parquet_path ./datasets/pretrain/retrieval_database_512.parquet \\
        --n_proto 256 \\
        --n_rq_stages 4 \\
        --mask_ratio 0.1 \\
        --train_epochs 20 \\
        --patience 3 \\
        --batch_size 128 \\
        --output_path ./checkpoints/prototypes_rqvae.pt
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from models.ChronosBoltPrototype import (
    ChronosBoltWithLearnablePrototypes,
    ChronosBoltWithRQVAEPrototypes,
)

# utils/tools.py is in a sibling directory; add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.tools import EarlyStopping

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ParquetXYDataset(Dataset):
    """Load (x, y) pairs from a parquet file (for forecasting task)."""

    def __init__(self, parquet_path: str):
        df = pd.read_parquet(parquet_path, columns=["x", "y"])
        print(f"  Loaded parquet: {len(df):,} rows from {parquet_path}")
        self.xs = np.vstack(df["x"].to_numpy()).astype(np.float32)
        self.ys = np.vstack(df["y"].to_numpy()).astype(np.float32)
        print(f"  x shape: {self.xs.shape}  |  y shape: {self.ys.shape}")

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.xs[idx]), torch.from_numpy(self.ys[idx])


class ParquetRecoveryDataset(Dataset):
    """
    Load x-only sequences from a parquet file (for recovery task).

    Splits are index-based (no shuffle):
        train: [0,       0.7 * N)
        val:   [0.7 * N, 0.8 * N)
        test:  [0.8 * N, N)
    """

    SPLIT_RATIOS = {"train": (0.0, 0.7), "val": (0.7, 0.8), "test": (0.8, 1.0)}

    def __init__(self, parquet_path: str, split: str = "train"):
        assert split in self.SPLIT_RATIOS, f"split must be one of {list(self.SPLIT_RATIOS)}"
        df = pd.read_parquet(parquet_path, columns=["x"])
        print(f"  Loaded parquet: {len(df):,} rows from {parquet_path}")

        xs = np.vstack(df["x"].to_numpy()).astype(np.float32)

        n = len(xs)
        lo_ratio, hi_ratio = self.SPLIT_RATIOS[split]
        lo, hi = int(lo_ratio * n), int(hi_ratio * n)
        self.xs = xs[lo:hi]
        print(f"  [{split}] x shape: {self.xs.shape}  (rows {lo:,} – {hi:,})")

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.xs[idx])


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Learn prototypes via frozen ChronosBolt")

    # Task
    parser.add_argument("--task", type=str, default="forecasting",
                        choices=["forecasting", "recovery"],
                        help="Training objective: 'forecasting' (legacy) or 'recovery' (RQ-VAE)")

    # Paths
    parser.add_argument("--pretrained_model_path", type=str, default="./checkpoints/base/")
    parser.add_argument("--parquet_path", type=str,
                        default="./datasets/pretrain/retrieval_database_512.parquet")
    parser.add_argument("--output_path", type=str,
                        default="./checkpoints/prototypes_learned.npy")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/proto_ckpt/",
                        help="Directory for EarlyStopping checkpoints (recovery task)")

    # Forecasting task hyperparameters
    parser.add_argument("--n_clusters", type=int, default=512,
                        help="[forecasting] Number of learnable prototypes")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="[forecasting] Attention heads in cross-attention layer")
    parser.add_argument("--proto_dropout", type=float, default=0.1)

    # Recovery task hyperparameters
    parser.add_argument("--n_proto", type=int, default=256,
                        help="[recovery] Number of codebook entries per RQ stage")
    parser.add_argument("--n_rq_stages", type=int, default=4,
                        help="[recovery] Number of residual quantisation stages")
    parser.add_argument("--beta", type=float, default=0.25,
                        help="[recovery] Commitment loss weight")
    parser.add_argument("--mask_ratio", type=float, default=0.1,
                        help="[recovery] Fraction of time-points to mask (0.1 = 10%%)")

    # Common training hyperparameters
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=200,
                        help="Print training loss every N batches")
    parser.add_argument("--patience", type=int, default=3,
                        help="[recovery] EarlyStopping patience (epochs)")

    # GPU
    parser.add_argument("--gpu_loc", type=int, default=2)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_forecasting(args, device):
    """Legacy forecasting task (cross-attention prototypes)."""
    print("Loading dataset ...")
    dataset = ParquetXYDataset(args.parquet_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"  batches per epoch: {len(loader):,}")

    print("Loading model ...")
    model = ChronosBoltWithLearnablePrototypes.from_pretrained_for_proto_learning(
        pretrained_model_path=args.pretrained_model_path,
        n_proto=args.n_clusters,
        num_heads=args.n_heads,
        proto_dropout=args.proto_dropout,
    )
    model.to(device)
    model.print_trainable_params()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    print(f"\nStarting training for {args.train_epochs} epoch(s) ...")
    global_step = 0
    for epoch in range(1, args.train_epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()

        for batch_x, batch_y in tqdm(loader, desc=f"Epoch {epoch}/{args.train_epochs}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(context=batch_x, target=batch_y)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            epoch_loss += output.loss.item()
            n_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                print(
                    f"  step={global_step}  "
                    f"avg_loss={epoch_loss / n_batches:.6f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        print(
            f"Epoch [{epoch}/{args.train_epochs}]  "
            f"loss={epoch_loss / n_batches:.6f}  "
            f"time={time.time() - t0:.1f}s"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    model.save_prototypes(args.output_path)
    print(f"\nDone. Prototypes saved to: {args.output_path}")


def validate_recovery(model, loader, device):
    """Compute average total loss on a validation/test set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            output = model(context=batch_x)
            total_loss += output.loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def train_recovery(args, device):
    """RQ-VAE recovery task with 7:1:2 train/val/test split and early stopping."""
    print("Loading datasets ...")
    train_set = ParquetRecoveryDataset(args.parquet_path, split="train")
    val_set   = ParquetRecoveryDataset(args.parquet_path, split="val")
    test_set  = ParquetRecoveryDataset(args.parquet_path, split="test")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    train_loader = DataLoader(train_set, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_set,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_set,  shuffle=False, **loader_kwargs)
    print(f"  train batches: {len(train_loader):,}  |  val: {len(val_loader):,}  |  test: {len(test_loader):,}")

    print("Loading model ...")
    model = ChronosBoltWithRQVAEPrototypes.from_pretrained_for_rqvae(
        pretrained_model_path=args.pretrained_model_path,
        n_proto=args.n_proto,
        n_rq_stages=args.n_rq_stages,
        beta=args.beta,
        mask_ratio=args.mask_ratio,
    )
    model.to(device)
    model.print_trainable_params()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    print(f"\nStarting recovery training for up to {args.train_epochs} epoch(s) ...")
    global_step = 0
    for epoch in range(1, args.train_epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()

        for batch_x in tqdm(train_loader, desc=f"Epoch {epoch}/{args.train_epochs}"):
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            output = model(context=batch_x)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            epoch_loss += output.loss.item()
            n_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                print(
                    f"  step={global_step}  "
                    f"avg_loss={epoch_loss / n_batches:.6f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        val_loss = validate_recovery(model, val_loader, device)
        print(
            f"Epoch [{epoch}/{args.train_epochs}]  "
            f"train_loss={epoch_loss / n_batches:.6f}  "
            f"val_loss={val_loss:.6f}  "
            f"time={time.time() - t0:.1f}s"
        )

        early_stopping(val_loss, model, args.checkpoint_dir)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load best model and evaluate on test set
    best_ckpt = os.path.join(args.checkpoint_dir, "checkpoint.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"Loaded best checkpoint from {best_ckpt}")

    test_loss = validate_recovery(model, test_loader, device)
    print(f"\nTest reconstruction MSE: {test_loss:.6f}")

    # Save codebooks
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    model.save_codebooks(args.output_path)
    print(f"Done. Codebooks saved to: {args.output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Task: {args.task}")

    if args.task == "forecasting":
        train_forecasting(args, device)
    else:
        train_recovery(args, device)


if __name__ == "__main__":
    main()
