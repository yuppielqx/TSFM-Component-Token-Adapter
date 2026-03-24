"""
train_prototypes.py

Learn a prototype matrix end-to-end using a frozen ChronosBolt backbone.

Architecture
------------
    context (B, context_length)
        ↓  frozen ChronosBolt encoder + preprocessing
        ↓  learnable PrototypeCrossAttentionLayer(Q=enc_out, K=V=prototypes)
        ↓  frozen ChronosBolt decoder
        →  quantile forecast  →  built-in quantile loss vs. target y

Trainable parameters
--------------------
    - prototype_matrix  (n_proto × d_model  =  512 × 768 = 393 216)
    - prototype_cross_attn  (~2.4 M)

Data
----
    retrieval_database_512.parquet
      columns: x (float32 array of length context_length)
               y (float32 array of length pred_length)

Usage
-----
    python train_prototypes.py \\
        --pretrained_model_path ./checkpoints/base/ \\
        --parquet_path ./datasets/pretrain/retrieval_database_512.parquet \\
        --n_clusters 512 \\
        --train_epochs 5 \\
        --batch_size 256 \\
        --learning_rate 1e-4 \\
        --output_path ./checkpoints/prototypes_learned.npy \\
        --gpu_loc 0
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.ChronosBoltPrototype import ChronosBoltWithLearnablePrototypes

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ParquetXYDataset(Dataset):
    """
    Load (x, y) pairs from a parquet file.

    x: context window  (context_length,)  float32
    y: forecast target (pred_length,)     float32
    """

    def __init__(self, parquet_path: str):
        df = pd.read_parquet(parquet_path, columns=["x", "y"])
        print(f"  Loaded parquet: {len(df):,} rows from {parquet_path}")

        xs = df["x"].to_numpy()
        ys = df["y"].to_numpy()

        self.xs = np.vstack(xs).astype(np.float32)   # (N, context_length)
        self.ys = np.vstack(ys).astype(np.float32)   # (N, pred_length)

        print(f"  x shape: {self.xs.shape}  |  y shape: {self.ys.shape}")

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.xs[idx]), torch.from_numpy(self.ys[idx])


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Learn prototypes via frozen ChronosBolt")

    # Paths
    parser.add_argument("--pretrained_model_path", type=str, default="./checkpoints/base/")
    parser.add_argument("--parquet_path", type=str,
                        default="./datasets/pretrain/retrieval_database_512.parquet")
    parser.add_argument("--output_path", type=str,
                        default="./checkpoints/prototypes_learned.npy")

    # Prototype hyperparameters
    parser.add_argument("--n_clusters", type=int, default=512,
                        help="Number of learnable prototypes")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Attention heads in prototype cross-attention layer")
    parser.add_argument("--proto_dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=200,
                        help="Print training loss every N batches")

    # GPU
    parser.add_argument("--gpu_loc", type=int, default=0)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
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

    # --- Model ---
    print("Loading model ...")
    model = ChronosBoltWithLearnablePrototypes.from_pretrained_for_proto_learning(
        pretrained_model_path=args.pretrained_model_path,
        n_proto=args.n_clusters,
        num_heads=args.n_heads,
        proto_dropout=args.proto_dropout,
    )
    model.to(device)
    model.print_trainable_params()

    # --- Optimizer (only trainable params: prototype_matrix + cross_attn) ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # --- Training loop ---
    print(f"\nStarting training for {args.train_epochs} epoch(s) ...")
    global_step = 0
    for epoch in range(1, args.train_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_x, batch_y in tqdm(loader, desc=f"Epoch {epoch}/{args.train_epochs}", leave=True):
            batch_x = batch_x.to(device)   # (B, context_length)
            batch_y = batch_y.to(device)   # (B, pred_length)

            optimizer.zero_grad()

            # ChronosBolt forward with target → returns built-in quantile loss
            output = model(context=batch_x, target=batch_y)
            loss = output.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                avg = epoch_loss / n_batches
                print(
                    f"  step={global_step}  "
                    f"avg_loss={avg:.6f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch}/{args.train_epochs}]  "
            f"loss={epoch_loss / n_batches:.6f}  "
            f"time={elapsed:.1f}s"
        )

    # --- Save prototypes ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    model.save_prototypes(args.output_path)
    print(f"\nDone. Prototypes saved to: {args.output_path}")
    print("Use --prototype_path", args.output_path,
          "in fewshot_proto.py or ChronosBoltWithPrototypes.")


if __name__ == "__main__":
    main()
