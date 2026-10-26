"""Train a downstream forecasting adapter conditioned on component temporal tokens."""

import argparse
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_factory import data_provider
from models.chronos_bolt.ComponentTemporalTokenAdapter import ComponentTemporalTokenAdapter

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Downstream component temporal-token adapter tuning")
    parser.add_argument("--model_id", type=str, default="ETTh1_component_token_adapter")
    parser.add_argument("--data", type=str, default="ett_h")
    parser.add_argument("--root_path", type=str, default="./datasets/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--percent", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=-1)
    parser.add_argument("--return_feature_id", action="store_true", default=False)

    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=64)
    parser.add_argument("--label_len", type=int, default=48)

    parser.add_argument("--pretrained_model_path", type=str, default="./checkpoints/base/")
    parser.add_argument("--component_codebook_dir", type=str, default="./checkpoints/component_rqvae/")
    parser.add_argument("--components", type=str, default="trend seasonal")
    parser.add_argument("--n_proto", type=int, default=256)
    parser.add_argument("--n_rq_stages", type=int, default=2)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--trend_kernel", type=int, default=25)
    parser.add_argument("--token_dropout", type=float, default=0.1)
    parser.add_argument("--gate_bias_init", type=float, default=-2.0)

    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
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
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/component_token_adapter_ckpt/")
    parser.add_argument("--results_dir", type=str, default="./results/component_token_adapter/")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unpack_batch(batch, args, device):
    batch_x, batch_y = batch[0], batch[1]
    batch_x = batch_x.float().to(device).squeeze(-1)
    batch_y = batch_y.float().to(device).squeeze(-1)
    target = batch_y[:, -args.pred_len:]
    return batch_x, target


def central_prediction(model, output, pred_len):
    quantiles = model.quantiles.detach().float().cpu()
    central_idx = int(torch.abs(quantiles - 0.5).argmin().item())
    return output.quantile_preds[:, central_idx, :pred_len]


def run_epoch(model, loader, args, device, optimizer=None, max_batches=-1, desc="Train"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    preds, trues = [], []
    n_batches = 0

    iterator = tqdm(loader, desc=desc)
    for batch_idx, batch in enumerate(iterator):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch_x, target = unpack_batch(batch, args, device)
        if is_train:
            optimizer.zero_grad()
        grad_context = torch.enable_grad() if is_train else torch.no_grad()
        with grad_context:
            output = model(context=batch_x, target=target)
            if is_train:
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
                optimizer.step()
        with torch.no_grad():
            n_batches += 1
            total_loss += float(output.loss.item())
            pred = central_prediction(model, output, args.pred_len)
            preds.append(pred.detach().cpu())
            trues.append(target.detach().cpu())
            iterator.set_postfix(loss=total_loss / n_batches)

    if n_batches == 0:
        raise RuntimeError(f"No batches processed for {desc}")
    preds = torch.cat(preds, dim=0).float()
    trues = torch.cat(trues, dim=0).float()
    mse = torch.mean((preds - trues) ** 2).item()
    mae = torch.mean(torch.abs(preds - trues)).item()
    return {
        "loss": total_loss / n_batches,
        "mse": mse,
        "mae": mae,
        "rmse": mse ** 0.5,
        "batches": n_batches,
    }


def component_codebook_paths(args, components):
    paths = {}
    for component in components:
        path = os.path.join(args.component_codebook_dir, f"{component}_prototypes_rqvae.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing component codebook: {path}")
        paths[component] = path
    return paths


def build_model(args, device):
    components = tuple(args.components.split())
    model = ComponentTemporalTokenAdapter.from_pretrained_component_adapter(
        pretrained_model_path=args.pretrained_model_path,
        component_codebook_paths=component_codebook_paths(args, components),
        components=components,
        n_proto=args.n_proto,
        n_rq_stages=args.n_rq_stages,
        period=args.period,
        trend_kernel=args.trend_kernel,
        token_dropout=args.token_dropout,
        gate_bias_init=args.gate_bias_init,
    ).to(device)
    model.print_trainable_params()
    return model


def save_adapter(model, path, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "adapter": model.adapter_state_dict(),
            "components": tuple(args.components.split()),
            "n_proto": args.n_proto,
            "n_rq_stages": args.n_rq_stages,
            "period": args.period,
            "trend_kernel": args.trend_kernel,
        },
        path,
    )


def load_adapter(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_adapter_state_dict(ckpt["adapter"])


def write_results(args, metrics):
    os.makedirs(args.results_dir, exist_ok=True)
    tag = f"{args.data_path.split('.')[0]}_component_token_adapter_p{args.percent}_pred{args.pred_len}"
    path = os.path.join(args.results_dir, f"{tag}.txt")
    with open(path, "w") as f:
        f.write(f"model_id={args.model_id}\n")
        f.write(f"data_path={args.data_path}\n")
        f.write(f"percent={args.percent}\n")
        f.write(f"seq_len={args.seq_len}  pred_len={args.pred_len}\n")
        f.write(f"components={args.components}\n")
        f.write(f"period={args.period}  trend_kernel={args.trend_kernel}\n")
        f.write(f"n_proto={args.n_proto}  n_rq_stages={args.n_rq_stages}\n")
        f.write(f"token_dropout={args.token_dropout}  gate_bias_init={args.gate_bias_init}\n")
        f.write(f"loss={metrics['loss']:.6f}\n")
        f.write(f"MSE={metrics['mse']:.6f}\n")
        f.write(f"MAE={metrics['mae']:.6f}\n")
        f.write(f"RMSE={metrics['rmse']:.6f}\n")
    print(f"Results saved to {path}")


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.data_path} | downstream percent={args.percent} | pred_len={args.pred_len}")

    train_set, train_loader = data_provider(args, "train")
    val_set, val_loader = data_provider(args, "val")
    test_set, test_loader = data_provider(args, "test")
    print(f"Samples: train={len(train_set):,} val={len(val_set):,} test={len(test_set):,}")

    model = build_model(args, device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    ckpt_path = os.path.join(args.checkpoint_dir, "adapter_checkpoint.pt")
    best_val = float("inf")
    bad_epochs = 0
    best_epoch = 0

    for epoch in range(1, args.train_epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            args,
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            desc=f"Epoch {epoch}/{args.train_epochs}",
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            args,
            device,
            optimizer=None,
            max_batches=args.max_val_batches,
            desc="Val",
        )
        print(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_mse={val_metrics['mse']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} time={time.time() - t0:.1f}s"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            bad_epochs = 0
            best_epoch = epoch
            save_adapter(model, ckpt_path, args)
            print(f"Saved best adapter checkpoint to {ckpt_path}")
        else:
            bad_epochs += 1
            print(f"EarlyStopping counter: {bad_epochs} out of {args.patience}")
            if bad_epochs >= args.patience:
                print("Early stopping triggered.")
                break

    load_adapter(model, ckpt_path, device)
    print(f"Loaded best adapter from epoch {best_epoch}: {ckpt_path}")
    test_metrics = run_epoch(
        model,
        test_loader,
        args,
        device,
        optimizer=None,
        max_batches=args.max_test_batches,
        desc="Test",
    )
    print(
        f"Test: loss={test_metrics['loss']:.6f} "
        f"MSE={test_metrics['mse']:.6f} MAE={test_metrics['mae']:.6f}"
    )
    write_results(args, test_metrics)


if __name__ == "__main__":
    main()
