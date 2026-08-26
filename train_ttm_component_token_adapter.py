"""Train the trend/seasonal component-token adapter on an ETT dataset."""

import argparse
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from data_provider.data_factory import data_provider
from models.tinytimemixer.TTMComponentTokenAdapter import TTMComponentTokenAdapter


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="ETTh1_ttm_component_adapter")
    p.add_argument("--data", default="ett_h")
    p.add_argument("--root_path", default="./datasets/ETT-small/")
    p.add_argument("--data_path", default="ETTh1.csv")
    p.add_argument("--features", default="M")
    p.add_argument("--target", default="OT")
    p.add_argument("--freq", default="h")
    p.add_argument("--embed", default="timeF")
    p.add_argument("--percent", type=int, default=100)
    p.add_argument("--max_len", type=int, default=-1)
    p.add_argument("--return_feature_id", action="store_true", default=False)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--pred_len", type=int, choices=(96, 192, 336, 720), required=True)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--pretrained_model_path", default="./checkpoints/ttm-research-r2")
    p.add_argument("--component_codebook_dir", default="./checkpoints/ttm_component_rqvae")
    p.add_argument("--components", default="trend seasonal")
    p.add_argument("--n_proto", type=int, default=256)
    p.add_argument("--n_rq_stages", type=int, default=2)
    p.add_argument("--period", type=int, required=True)
    p.add_argument("--frequency_token", type=int, required=True)
    p.add_argument("--trend_kernel", type=int, default=25)
    p.add_argument("--token_dropout", type=float, default=0.1)
    p.add_argument("--gate_bias_init", type=float, default=-2.0)
    p.add_argument("--cross_attention_heads", type=int, default=8)
    p.add_argument("--train_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max_train_batches", type=int, default=-1)
    p.add_argument("--max_val_batches", type=int, default=-1)
    p.add_argument("--max_test_batches", type=int, default=-1)
    p.add_argument("--gpu_loc", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--results_dir", required=True)
    return p.parse_args()


def codebook_paths(args):
    if tuple(args.components.split()) != ("trend", "seasonal"):
        raise ValueError("TTM ETT training uses exactly: trend seasonal")
    paths = {}
    for name in ("trend", "seasonal"):
        path = os.path.join(args.component_codebook_dir, f"{name}_prototypes_rqvae.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        paths[name] = path
    return paths


def build_model(args, device):
    model = TTMComponentTokenAdapter.from_pretrained_component_adapter(
        args.pretrained_model_path,
        codebook_paths(args),
        components=("trend", "seasonal"),
        n_proto=args.n_proto,
        n_rq_stages=args.n_rq_stages,
        period=args.period,
        trend_kernel=args.trend_kernel,
        frequency_token=args.frequency_token,
        token_dropout=args.token_dropout,
        gate_bias_init=args.gate_bias_init,
        cross_attention_heads=args.cross_attention_heads,
        prediction_filter_length=args.pred_len,
    ).to(device)
    if model.config.prediction_length != args.pred_len:
        raise ValueError(
            f"Official TTM checkpoint horizon={model.config.prediction_length}, "
            f"but requested pred_len={args.pred_len}"
        )
    model.print_trainable_params()
    return model


def unpack_batch(batch, args, device):
    context = batch[0].float().to(device)
    target = batch[1].float().to(device)[:, -args.pred_len :]
    return context, target


def run_epoch(model, loader, args, device, optimizer=None, limit=-1, desc="Train"):
    training = optimizer is not None
    model.train(training)
    loss_sum = se = ae = count = batches = 0
    for index, batch in enumerate(tqdm(loader, desc=desc)):
        if limit > 0 and index >= limit:
            break
        context, target = unpack_batch(batch, args, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad() if training else torch.no_grad():
            output = model(context, future_values=target, freq_token=args.frequency_token)
            loss = output["loss"]
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [x for x in model.parameters() if x.requires_grad], args.grad_clip
                )
                optimizer.step()
        prediction = output["prediction_outputs"].detach()
        valid = torch.isfinite(target)
        diff = prediction - torch.nan_to_num(target, nan=0.0)
        loss_sum += loss.item()
        se += ((diff * diff) * valid).sum().item()
        ae += (diff.abs() * valid).sum().item()
        count += valid.sum().item()
        batches += 1
    if not batches:
        raise RuntimeError(f"No batches processed for {desc}")
    return {"loss": loss_sum / batches, "mse": se / count, "mae": ae / count}


def save_checkpoint(model, path, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"adapter": model.adapter_state_dict(), "components": ("trend", "seasonal"),
         "period": args.period, "frequency_token": args.frequency_token,
         "prediction_length": args.pred_len, "n_proto": args.n_proto,
         "n_rq_stages": args.n_rq_stages, "trend_kernel": args.trend_kernel,
         "cross_attention_heads": args.cross_attention_heads}, path
    )


def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    train_set, train_loader = data_provider(args, "train")
    val_set, val_loader = data_provider(args, "val")
    test_set, test_loader = data_provider(args, "test")
    print(f"Device={device}; period={args.period}; frequency_token={args.frequency_token}; "
          f"samples train={len(train_set):,} val={len(val_set):,} test={len(test_set):,}")
    model = build_model(args, device)
    optimizer = torch.optim.AdamW(
        [x for x in model.parameters() if x.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay)
    checkpoint = os.path.join(args.checkpoint_dir, "adapter_checkpoint.pt")
    best, best_epoch, bad = float("inf"), 0, 0
    for epoch in range(1, args.train_epochs + 1):
        started = time.time()
        train = run_epoch(model, train_loader, args, device, optimizer, args.max_train_batches, f"Train {epoch}")
        val = run_epoch(model, val_loader, args, device, None, args.max_val_batches, "Val")
        print(f"Epoch {epoch}: train={train['loss']:.6f} val={val['loss']:.6f} "
              f"val_mse={val['mse']:.6f} val_mae={val['mae']:.6f} time={time.time()-started:.1f}s")
        if val["loss"] < best:
            best, best_epoch, bad = val["loss"], epoch, 0
            save_checkpoint(model, checkpoint, args)
            print(f"Saved best checkpoint to {checkpoint}")
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping triggered"); break
    model.load_adapter_state_dict(torch.load(checkpoint, map_location=device)["adapter"])
    print(f"Loaded best epoch {best_epoch}")
    test = run_epoch(model, test_loader, args, device, None, args.max_test_batches, "Test")
    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, f"{os.path.splitext(args.data_path)[0]}_ttm_pred{args.pred_len}.txt")
    with open(path, "w") as handle:
        handle.write(f"components=trend seasonal\nperiod={args.period}\nfrequency_token={args.frequency_token}\n")
        handle.write(f"loss={test['loss']:.6f}\nMSE={test['mse']:.6f}\nMAE={test['mae']:.6f}\n")
    print(f"Test: MSE={test['mse']:.6f} MAE={test['mae']:.6f}; results={path}")


if __name__ == "__main__":
    main()
