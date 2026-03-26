"""
fewshot_proto.py

Few-shot fine-tuning of prototype-augmented models on ETT/custom datasets.
Currently supports: PatchTSTWithPrototypes
(extend by adding model choices to get_args and build_model)

Frozen:  prototype_matrix (register_buffer)
Trained: all other model parameters

Usage:
  python fewshot_proto.py \
      --model_id ETTh1_patchtst_proto512_p10_pred96 \
      --prototype_path ./checkpoints/prototypes_k512.npy \
      --root_path ./datasets/ETT-small/ \
      --data_path ETTh1.csv \
      --data ett_h \
      --seq_len 512 \
      --pred_len 96 \
      --percent 10 \
      --train_epochs 20 \
      --batch_size 64 \
      --learning_rate 1e-4 \
      --gpu_loc 0
"""

import os
import time
import random
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

from utils.metrics import metric
from utils.tools import EarlyStopping
from data_provider.data_factory import data_provider
from models.PatchTSTWithPrototypes import PatchTSTWithPrototypes

warnings.filterwarnings("ignore")

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Prototype-augmented model few-shot fine-tuning")

    # Experiment
    parser.add_argument("--model_id", type=str, default="ETTh1_patchtst_proto512_p10_pred96")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--model", type=str, default="patchtst",
                        choices=["patchtst"],
                        help="Model architecture to use")

    # Prototype
    parser.add_argument("--prototype_path", type=str, default="./checkpoints/prototypes_k512.npy")

    # PatchTST hyperparameters
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=768,
                        help="Must match prototype dimension (768 for ChronosBolt prototypes)")
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Data
    parser.add_argument("--root_path", type=str, default="./datasets/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--data", type=str, default="ett_h",
                        choices=["ett_h", "ett_m", "custom"])
    parser.add_argument("--features", type=str, default="M",
                        help="M: multivariate, S: univariate (target only)")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--percent", type=int, default=100,
                        help="Percentage of training set to use")
    parser.add_argument("--max_len", type=int, default=-1)

    # Sequence lengths
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)

    # Training
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip_value", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--tmax", type=int, default=10)

    # GPU
    parser.add_argument("--gpu_loc", type=int, default=0)

    # Output
    parser.add_argument("--save_file_name", type=str, default=None)

    # Required by data_provider
    parser.add_argument("--return_feature_id", action="store_true", default=False)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(args, c_in: int) -> nn.Module:
    if args.model == "patchtst":
        model = PatchTSTWithPrototypes(
            c_in=c_in,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            patch_len=args.patch_len,
            stride=args.stride,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.load_prototypes(args.prototype_path)
    return model


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip, pred_len):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="  train", leave=False):
        batch_x, batch_y, *_ = batch
        batch_x = batch_x.float().to(device)   # (B, seq_len, C)
        batch_y = batch_y.float().to(device)   # (B, label_len + pred_len, C)

        target = batch_y[:, -pred_len:, :]     # (B, pred_len, C)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, target)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, pred_len):
    model.eval()
    preds, trues = [], []

    for batch in tqdm(loader, desc="  eval ", leave=False):
        batch_x, batch_y, *_ = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        pred = model(batch_x)
        true = batch_y[:, -pred_len:, :]       # (B, pred_len, C)

        preds.append(pred.cpu())
        trues.append(true.cpu())

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()

    mae, mse, *_ = metric(preds, trues)
    return mse, mae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {args.model}  |  Model ID: {args.model_id}")

    # --- Data ---
    print(f"Loading data: {args.data_path}  features={args.features}  percent={args.percent}%")
    train_data, train_loader = data_provider(args, "train")
    val_data,   val_loader   = data_provider(args, "val")
    test_data,  test_loader  = data_provider(args, "test")
    print(f"  train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    c_in = train_data[0][0].shape[-1]
    print(f"  c_in={c_in}")

    # --- Model ---
    model = build_model(args, c_in)
    model.to(device)
    model.print_trainable_params()

    # --- Optimizer / scheduler ---
    criterion = nn.MSELoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.tmax, eta_min=1e-8
    )

    save_dir = os.path.join(args.checkpoints, args.model_id)
    os.makedirs(save_dir, exist_ok=True)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # --- Training loop ---
    for epoch in range(1, args.train_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            args.grad_clip_value, args.pred_len
        )
        val_mse, val_mae = evaluate(model, val_loader, device, args.pred_len)
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:>2}/{args.train_epochs}]  "
            f"train_loss={train_loss:.6f}  val_mse={val_mse:.4f}  val_mae={val_mae:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  time={elapsed:.1f}s"
        )

        early_stopping(val_mse, model, save_dir)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        scheduler.step()

    # --- Test with best checkpoint ---
    print("Loading best checkpoint for testing ...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "checkpoint.pth")))
    test_mse, test_mae = evaluate(model, test_loader, device, args.pred_len)
    print(f"Test  MSE={test_mse:.4f}  MAE={test_mae:.4f}")

    # --- Save results ---
    result_dir = "results/fewshot_evaluation"
    os.makedirs(result_dir, exist_ok=True)
    log_file = args.save_file_name or f"{args.model_id}.txt"
    with open(os.path.join(result_dir, log_file), "a") as f:
        f.write(f"{args.model_id}\n")
        f.write(f"model={args.model}, percent={args.percent}%, pred_len={args.pred_len}, features={args.features}\n")
        f.write(f"mse:{test_mse:.4f}, mae:{test_mae:.4f}\n\n")

    return test_mse, test_mae


if __name__ == "__main__":
    main()
