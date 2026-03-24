"""
fewshot.py

Few-shot fine-tuning of ChronosBoltWithPrototypes.

Frozen: prototype_matrix (register_buffer, never in optimizer)
Trained by default: T5 encoder, T5 decoder, patch embeddings, prototype_cross_attn
Option --freeze_backbone: additionally freeze the T5 backbone,
       training only the prototype_cross_attn layer (extreme few-shot)

Usage:
  python fewshot.py \
      --model_id ETTh1_fewshot_proto512 \
      --pretrained_model_path ./checkpoints/base/ \
      --prototype_path ./checkpoints/prototypes_k512.pt \
      --root_path ./datasets/ETT-small/ \
      --data_path ETTh1.csv \
      --data ett_h \
      --seq_len 512 \
      --pred_len 96 \
      --percent 10 \
      --train_epochs 10 \
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
from models.ChronosBoltPrototype import ChronosBoltWithPrototypes

warnings.filterwarnings("ignore")

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="ChronosBoltWithPrototypes few-shot fine-tuning")

    # Experiment
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

    # Model paths
    parser.add_argument("--pretrained_model_path", type=str, default="./checkpoints/base/")
    parser.add_argument("--prototype_path", type=str, required=True)
    parser.add_argument("--num_proto_heads", type=int, default=8)
    parser.add_argument("--proto_dropout", type=float, default=0.1)
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=False,
        help="Freeze T5 encoder/decoder; only train prototype_cross_attn (extreme few-shot)",
    )

    # Data
    parser.add_argument("--root_path", type=str, default="./datasets/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument(
        "--data",
        type=str,
        default="ett_h",
        choices=["ett_h", "ett_m", "custom"],
        help="Dataset type (non-retrieval). Maps to data_factory.data_dict.",
    )
    parser.add_argument("--features", type=str, default="S")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument(
        "--percent",
        type=int,
        default=10,
        help="Percentage of training set to use (e.g. 10 = 10%%)",
    )
    parser.add_argument("--max_len", type=int, default=-1)

    # Sequence lengths
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)

    # Training
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip_value", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--tmax", type=int, default=10)

    # GPU
    parser.add_argument("--gpu_loc", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1")

    # Output
    parser.add_argument("--save_file_name", type=str, default=None)

    # Required by data_provider (non-retrieve branch)
    parser.add_argument("--return_feature_id", action="store_true", default=False)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, grad_clip, pred_len):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="  train", leave=False):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        # batch_x: (B, seq_len, n_vars) or (B, seq_len, 1) for univariate
        # squeeze last dim for ChronosBolt which expects (B, seq_len)
        batch_x = batch_x.float().to(device).squeeze(-1)
        batch_y = batch_y.float().to(device).squeeze(-1)

        # batch_y shape: (B, label_len + pred_len); we need only the last pred_len steps
        target = batch_y[:, -pred_len:]

        optimizer.zero_grad()
        outputs = model(context=batch_x, target=target)
        loss = outputs.loss
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, pred_len, quantiles):
    model.eval()
    preds, trues = [], []

    # Index of the median quantile (q closest to 0.5)
    median_idx = int(np.argmin(np.abs(np.array(quantiles) - 0.5)))

    for batch in tqdm(loader, desc="  eval ", leave=False):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float().to(device).squeeze(-1)
        batch_y = batch_y.float().to(device).squeeze(-1)

        outputs = model(context=batch_x)
        # quantile_preds: (B, n_quantiles, pred_len)
        pred = outputs.quantile_preds[:, median_idx, :]  # (B, pred_len)
        true = batch_y[:, -pred_len:]  # (B, pred_len)

        preds.append(pred.cpu())
        trues.append(true.cpu())

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()

    # metric expects (..., pred_len) arrays
    mae, mse, *_ = metric(preds, trues)
    return mse, mae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model ID: {args.model_id}")

    # --- Load model ---
    print("Loading ChronosBoltWithPrototypes ...")
    model = ChronosBoltWithPrototypes.from_pretrained_with_prototypes(
        pretrained_model_path=args.pretrained_model_path,
        prototype_path=args.prototype_path,
        num_heads=args.num_proto_heads,
        proto_dropout=args.proto_dropout,
    )

    if args.freeze_backbone:
        model.freeze_backbone()
        print("Backbone frozen: only prototype_cross_attn will be trained.")

    model.to(device)
    model.print_trainable_params()

    if args.use_multi_gpu:
        device_ids = [int(i) for i in args.devices.split(",")]
        model = nn.DataParallel(model, device_ids=device_ids)

    inner_model = model.module if hasattr(model, "module") else model
    quantiles = inner_model.chronos_config.quantiles
    pred_len_model = inner_model.chronos_config.prediction_length
    # Use model's prediction_length for target; pred_len arg drives evaluation slicing
    pred_len = args.pred_len

    # --- Data ---
    # data_provider branches on 'retrieve' in args.model_id;
    # since our model_id doesn't contain 'retrieve', it uses the standard (non-retrieval) path.
    print(f"Loading data: {args.data_path}  percent={args.percent}%")
    train_data, train_loader = data_provider(args, "train")
    val_data, val_loader = data_provider(args, "val")
    test_data, test_loader = data_provider(args, "test")
    print(f"  train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # --- Optimizer / scheduler ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
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
            model, train_loader, optimizer, device, args.grad_clip_value, pred_len_model
        )
        val_mse, val_mae = evaluate(inner_model, val_loader, device, pred_len, quantiles)
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:>2}/{args.train_epochs}]  "
            f"train_loss={train_loss:.6f}  val_mse={val_mse:.4f}  val_mae={val_mae:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  time={elapsed:.1f}s"
        )

        early_stopping(val_mse, inner_model, save_dir)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        scheduler.step()

    # --- Test with best checkpoint ---
    print("Loading best checkpoint for testing ...")
    inner_model.load_state_dict(torch.load(os.path.join(save_dir, "checkpoint.pth")))
    test_mse, test_mae = evaluate(inner_model, test_loader, device, pred_len, quantiles)
    print(f"Test  MSE={test_mse:.4f}  MAE={test_mae:.4f}")

    # --- Save results ---
    result_dir = "results/fewshot_evaluation"
    os.makedirs(result_dir, exist_ok=True)
    log_file = args.save_file_name or f"{args.model_id}.txt"
    with open(os.path.join(result_dir, log_file), "a") as f:
        f.write(f"{args.model_id}\n")
        f.write(f"percent={args.percent}%, pred_len={pred_len}\n")
        f.write(f"mse:{test_mse:.4f}, mae:{test_mae:.4f}\n\n")

    return test_mse, test_mae


if __name__ == "__main__":
    main()
