"""Train a dataset-specific Component-Token Adapter on frozen Moirai-2."""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_factory import data_provider
from Moirai2ComponentTokenAdapter import Moirai2ComponentTokenAdapter


def get_args():
    parser = argparse.ArgumentParser(description="Moirai-2 Component-Token Adapter")
    parser.add_argument("--model_id", default="ETTh1_moirai2_component_adapter")
    parser.add_argument("--data", default="ett_h")
    parser.add_argument("--root_path", default="./datasets/ETT-small/")
    parser.add_argument("--data_path", default="ETTh1.csv")
    parser.add_argument("--features", default="M")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--freq", default="h")
    parser.add_argument("--embed", default="timeF")
    parser.add_argument("--percent", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=-1)
    parser.add_argument("--return_feature_id", action="store_true", default=False)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=64)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--moirai_model_id", default="Salesforce/moirai-2.0-R-small")
    parser.add_argument("--component_codebook_dir", default="./checkpoints/moirai2_component_rqvae")
    parser.add_argument("--components", default="trend seasonal")
    parser.add_argument("--n_proto", type=int, default=256)
    parser.add_argument("--n_rq_stages", type=int, default=2)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--trend_kernel", type=int, default=25)
    parser.add_argument("--token_dropout", type=float, default=0.1)
    parser.add_argument("--gate_bias_init", type=float, default=-2.0)
    parser.add_argument("--cross_attention_heads", type=int, default=8)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--max_val_batches", type=int, default=-1)
    parser.add_argument("--max_test_batches", type=int, default=-1)
    parser.add_argument("--gpu_loc", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--checkpoint_dir", default="./checkpoints/moirai2_component_token_adapter_ETTh1")
    parser.add_argument("--results_dir", default="./results/moirai2_component_token_adapter_ETTh1")
    return parser.parse_args()


def codebook_paths(args, components):
    paths = {}
    for component in components:
        path = os.path.join(args.component_codebook_dir, f"{component}_prototypes_rqvae.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing Moirai-2 component codebook: {path}")
        paths[component] = path
    return paths


def build_model(args, device):
    components = tuple(args.components.split())
    model = Moirai2ComponentTokenAdapter.from_pretrained_component_adapter(
        args.moirai_model_id, codebook_paths(args, components), components=components,
        context_length=args.seq_len, n_proto=args.n_proto, n_rq_stages=args.n_rq_stages,
        period=args.period, trend_kernel=args.trend_kernel, token_dropout=args.token_dropout,
        gate_bias_init=args.gate_bias_init, cross_attention_heads=args.cross_attention_heads,
    ).to(device)
    model.print_trainable_params()
    return model


def unpack_batch(batch, args, device):
    return (batch[0].float().to(device).squeeze(-1),
            batch[1].float().to(device).squeeze(-1)[:, -args.pred_len:])


def run_epoch(model, loader, args, device, optimizer=None, max_batches=-1, desc="Train"):
    training = optimizer is not None
    model.train(training)
    totals = dict(loss=0.0, se=0.0, ae=0.0, n=0, batches=0)
    for index, batch in enumerate(tqdm(loader, desc=desc)):
        if max_batches > 0 and index >= max_batches:
            break
        context, target = unpack_batch(batch, args, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad() if training else torch.no_grad():
            output = model(context, target=target, prediction_length=args.pred_len)
            if training:
                output["loss"].backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
                optimizer.step()
        prediction, valid = model.central_prediction(output).detach(), torch.isfinite(target)
        difference = prediction - torch.nan_to_num(target, nan=0.0)
        totals["loss"] += output["loss"].item()
        totals["se"] += ((difference ** 2) * valid).sum().item()
        totals["ae"] += (difference.abs() * valid).sum().item()
        totals["n"] += valid.sum().item(); totals["batches"] += 1
    if not totals["batches"]:
        raise RuntimeError(f"No batches processed for {desc}")
    return {"loss": totals["loss"] / totals["batches"], "mse": totals["se"] / totals["n"], "mae": totals["ae"] / totals["n"]}


def save_checkpoint(model, path, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"adapter": model.adapter_state_dict(), "components": tuple(args.components.split()),
                "n_proto": args.n_proto, "n_rq_stages": args.n_rq_stages, "period": args.period,
                "trend_kernel": args.trend_kernel, "cross_attention_heads": args.cross_attention_heads,
                "context_length": args.seq_len, "prediction_length": args.pred_len,
                "moirai_model_id": args.moirai_model_id}, path)


def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    train_set, train_loader = data_provider(args, "train")
    val_set, val_loader = data_provider(args, "val")
    test_set, test_loader = data_provider(args, "test")
    print(f"Device={device}; samples train={len(train_set):,} val={len(val_set):,} test={len(test_set):,}")
    model = build_model(args, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
    checkpoint_path, best, best_epoch, bad = os.path.join(args.checkpoint_dir, "adapter_checkpoint.pt"), float("inf"), 0, 0
    for epoch in range(1, args.train_epochs + 1):
        started = time.time()
        train_metrics = run_epoch(model, train_loader, args, device, optimizer, args.max_train_batches, f"Train {epoch}")
        val_metrics = run_epoch(model, val_loader, args, device, None, args.max_val_batches, "Val")
        print(f"Epoch {epoch}: train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f} val_mse={val_metrics['mse']:.6f} val_mae={val_metrics['mae']:.6f} time={time.time()-started:.1f}s")
        if val_metrics["loss"] < best:
            best, best_epoch, bad = val_metrics["loss"], epoch, 0
            save_checkpoint(model, checkpoint_path, args); print(f"Saved best checkpoint to {checkpoint_path}")
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping triggered"); break
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_adapter_state_dict(checkpoint["adapter"])
    test_metrics = run_epoch(model, test_loader, args, device, None, args.max_test_batches, "Test")
    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, f"{os.path.splitext(os.path.basename(args.data_path))[0]}_moirai2_component_adapter_pred{args.pred_len}.txt")
    with open(path, "w") as handle:
        handle.write(f"data_path={args.data_path}\ncomponents={args.components}\nperiod={args.period}\n")
        handle.write(f"best_epoch={best_epoch}\nloss={test_metrics['loss']:.6f}\nMSE={test_metrics['mse']:.6f}\nMAE={test_metrics['mae']:.6f}\n")
    print(f"Test: MSE={test_metrics['mse']:.6f} MAE={test_metrics['mae']:.6f}; results={path}")


if __name__ == "__main__":
    main()
