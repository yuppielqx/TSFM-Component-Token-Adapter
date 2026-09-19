"""训练数据集专属的 TimesFM-2.5 V2 Component Token Adapter。

维度约定：context [B,T]，target/prediction [B,H]；直接训练时 H<=128，
multi-horizon 模式通过官方 128 点 block 自回归得到更长预测。
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_factory import data_provider
from models.timesfm.TimesFMComponentTokenAdapterV2 import TimesFMComponentTokenAdapterV2
from models.timesfm.component_preprocessing_v2 import PREPROCESSING_NAME


def get_args():
    parser = argparse.ArgumentParser(description="Dataset-specific TimesFM v2 component adapter")
    parser.add_argument("--model_id", default="ETTh1_timesfm_component_adapter")
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
    parser.add_argument("--pred_len", type=int, default=128)
    parser.add_argument("--multi_horizon_training", action="store_true")
    parser.add_argument("--train_horizons", default="96 192 336 720")
    parser.add_argument("--scheduled_sampling_start", type=float, default=0.0)
    parser.add_argument("--scheduled_sampling_end", type=float, default=0.75)
    parser.add_argument("--scheduled_sampling_warmup_epochs", type=int, default=-1)
    parser.add_argument("--train_rollout_adapter_decay", type=float, default=0.75)
    parser.add_argument("--train_rollout_adapter_min_weight", type=float, default=0.0)
    parser.add_argument("--train_rollout_correction_clip", type=float, default=0.0)
    parser.add_argument("--train_rollout_adapter_max_steps", type=int, default=-1)
    parser.add_argument("--mae_loss_weight", type=float, default=0.0)
    parser.add_argument("--init_adapter_checkpoint", default="")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pretrained_model_path", default="./checkpoints/timesfm-2.5-200m-pytorch")
    parser.add_argument("--component_codebook_dir", default="./checkpoints/timesfm_component_rqvae_v2")
    parser.add_argument("--components", default="trend seasonal")
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
    parser.add_argument("--checkpoint_dir", default="./checkpoints/timesfm_component_token_adapter_v2_ETTh1")
    parser.add_argument("--results_dir", default="./results/timesfm_component_token_adapter_v2_ETTh1")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unpack_batch(batch, args, device):
    # data_provider 原始形状通常为 [B,T,1]，多变量被展为独立序列后去掉末维。
    context = batch[0].float().to(device).squeeze(-1)  # [B,seq_len]
    target = batch[1].float().to(device).squeeze(-1)[:, -args.pred_len:]  # [B,H]
    return context, target


def codebook_paths(args, components):
    paths = {}
    for component in components:
        path = os.path.join(args.component_codebook_dir, f"{component}_prototypes_rqvae.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing TimesFM component codebook: {path}")
        paths[component] = path
    return paths


def build_model(args, device):
    # 仅 Adapter 相关模块可训练；backbone 与 RQ codebooks 在模型内部被冻结。
    components = tuple(args.components.split())
    model = TimesFMComponentTokenAdapterV2.from_pretrained_component_adapter(
        pretrained_model_path=args.pretrained_model_path,
        component_codebook_paths=codebook_paths(args, components),
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


def horizon_loss(prediction, target, horizons, mae_loss_weight):
    """对同一 [B,Hmax] 预测计算多个前缀 horizon 的宏平均训练损失。"""
    losses = []
    for horizon in horizons:
        selected_target = target[:, :horizon]  # [B,h]
        difference = prediction[:, :horizon] - torch.nan_to_num(
            selected_target, nan=0.0
        )
        valid = torch.isfinite(selected_target)
        denominator = valid.sum().clamp_min(1)
        mse = (difference.square() * valid).sum() / denominator
        mae = (difference.abs() * valid).sum() / denominator
        losses.append(mse + mae_loss_weight * mae)
    return torch.stack(losses).mean()


def scheduled_sampling_probability(args, epoch):
    # 返回当前 epoch 使用模型预测作为下一 block 输入的概率。
    warmup = (
        args.train_epochs
        if args.scheduled_sampling_warmup_epochs < 0
        else max(1, args.scheduled_sampling_warmup_epochs)
    )
    progress = min(max(epoch - 1, 0) / max(warmup - 1, 1), 1.0)
    return (
        args.scheduled_sampling_start
        + progress
        * (args.scheduled_sampling_end - args.scheduled_sampling_start)
    )


def run_epoch(
    model, loader, args, device, optimizer=None, max_batches=-1,
    desc="Train", epoch=1, horizons=(128,),
):
    """执行一个 train/val/test epoch，并按所有有效标量累计 MSE/MAE。"""
    training = optimizer is not None
    model.train(training)
    loss_sum = squared_error = absolute_error = count = batches = 0
    for batch_index, batch in enumerate(tqdm(loader, desc=desc)):
        if max_batches > 0 and batch_index >= max_batches:
            break
        context, target = unpack_batch(batch, args, device)  # [B,T], [B,Hmax]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad() if training else torch.no_grad():
            if args.multi_horizon_training:
                if training:
                    selected_horizon = random.choice(horizons)
                    sampling_probability = scheduled_sampling_probability(args, epoch)
                else:
                    selected_horizon = max(horizons)
                    sampling_probability = 1.0
                output = model.multi_horizon_forward(
                    context,
                    target,
                    selected_horizon,
                    scheduled_sampling_probability=sampling_probability,
                    rollout_adapter_decay=args.train_rollout_adapter_decay,
                    rollout_adapter_min_weight=args.train_rollout_adapter_min_weight,
                    rollout_correction_clip=args.train_rollout_correction_clip,
                    rollout_adapter_max_steps=args.train_rollout_adapter_max_steps,
                    mae_loss_weight=args.mae_loss_weight,
                )
                if not training:
                    output["loss"] = horizon_loss(
                        output["prediction"], target, horizons, args.mae_loss_weight
                    )
            else:
                # 单 patch 路径最多输出 TimesFM 官方 head 的 128 个点。
                output = model(
                    context, target=target, prediction_length=args.pred_len
                )
            if training:
                output["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    args.grad_clip,
                )
                optimizer.step()
        prediction = output["prediction"].detach()  # [B,H_selected]
        metric_target = target[:, :prediction.shape[1]]  # [B,H_selected]
        valid = torch.isfinite(metric_target)
        difference = prediction - torch.nan_to_num(metric_target, nan=0.0)
        loss_sum += float(output["loss"].item())
        squared_error += ((difference ** 2) * valid).sum().item()
        absolute_error += (difference.abs() * valid).sum().item()
        count += valid.sum().item()
        batches += 1
    if not batches:
        raise RuntimeError(f"No batches processed for {desc}")
    return {"loss": loss_sum / batches, "mse": squared_error / count, "mae": absolute_error / count}


def save_checkpoint(model, path, args):
    # checkpoint 不包含冻结的 TimesFM backbone 和冻结 RQ quantizer。
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "adapter": model.adapter_state_dict(),
        "components": tuple(args.components.split()),
        "n_proto": args.n_proto,
        "n_rq_stages": args.n_rq_stages,
        "period": args.period,
        "trend_kernel": args.trend_kernel,
        "prediction_length": (
            TimesFMComponentTokenAdapterV2.output_patch_size
            if args.multi_horizon_training else args.pred_len
        ),
        "target_length": args.pred_len,
        "multi_horizon_training": args.multi_horizon_training,
        "train_horizons": tuple(
            int(value) for value in args.train_horizons.split()
        ),
        "scheduled_sampling_start": args.scheduled_sampling_start,
        "scheduled_sampling_end": args.scheduled_sampling_end,
        "train_rollout_adapter_decay": args.train_rollout_adapter_decay,
        "train_rollout_adapter_min_weight": args.train_rollout_adapter_min_weight,
        "train_rollout_correction_clip": args.train_rollout_correction_clip,
        "train_rollout_adapter_max_steps": args.train_rollout_adapter_max_steps,
        "mae_loss_weight": args.mae_loss_weight,
        "implementation_version": 2,
        "component_preprocessing": PREPROCESSING_NAME,
    }, path)


def write_results(args, metrics):
    os.makedirs(args.results_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.data_path))[0]
    path = os.path.join(
        args.results_dir,
        f"{stem}_timesfm_component_adapter_v2_pred{args.pred_len}.txt",
    )
    with open(path, "w") as handle:
        handle.write(f"data_path={args.data_path}\npercent={args.percent}\n")
        handle.write(f"seq_len={args.seq_len} pred_len={args.pred_len}\n")
        handle.write(f"components={args.components}\nperiod={args.period}\n")
        handle.write(
            f"multi_horizon_training={args.multi_horizon_training}\n"
            f"train_horizons={args.train_horizons}\n"
            f"rollout_adapter_decay={args.train_rollout_adapter_decay}\n"
        )
        handle.write(f"loss={metrics['loss']:.6f}\nMSE={metrics['mse']:.6f}\nMAE={metrics['mae']:.6f}\n")
    print(f"Results saved to {path}")


def main():
    args = get_args()
    horizons = tuple(sorted(set(
        int(value) for value in args.train_horizons.split()
    )))
    if not horizons or min(horizons) <= 0:
        raise ValueError("train_horizons must contain positive integers")
    if args.multi_horizon_training:
        if args.pred_len < max(horizons):
            print(
                f"Expanding data target length from pred_len={args.pred_len} "
                f"to max train horizon={max(horizons)}"
            )
            args.pred_len = max(horizons)
        if not 0.0 <= args.scheduled_sampling_start <= 1.0:
            raise ValueError("scheduled_sampling_start must be in [0, 1]")
        if not 0.0 <= args.scheduled_sampling_end <= 1.0:
            raise ValueError("scheduled_sampling_end must be in [0, 1]")
    elif args.pred_len > TimesFMComponentTokenAdapterV2.output_patch_size:
        raise ValueError("Direct adapter training supports pred_len <= 128")
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    # 三个 loader 的 forecast origins 由统一 data_provider 划分产生。
    train_set, train_loader = data_provider(args, "train")
    val_set, val_loader = data_provider(args, "val")
    test_set, test_loader = data_provider(args, "test")
    print(f"Device={device}; samples train={len(train_set):,} val={len(val_set):,} test={len(test_set):,}")
    model = build_model(args, device)
    if args.init_adapter_checkpoint:
        initial = torch.load(args.init_adapter_checkpoint, map_location=device)
        model.load_adapter_state_dict(initial["adapter"])
        print(f"Initialized adapter from {args.init_adapter_checkpoint}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    checkpoint_path = os.path.join(args.checkpoint_dir, "adapter_checkpoint.pt")
    best_val, best_epoch, bad_epochs = float("inf"), 0, 0
    for epoch in range(1, args.train_epochs + 1):
        started = time.time()
        train_metrics = run_epoch(
            model, train_loader, args, device, optimizer,
            args.max_train_batches, f"Train {epoch}", epoch, horizons,
        )
        val_metrics = run_epoch(
            model, val_loader, args, device, None,
            args.max_val_batches, "Val", epoch, horizons,
        )
        print(f"Epoch {epoch}: train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f} "
              f"val_mse={val_metrics['mse']:.6f} val_mae={val_metrics['mae']:.6f} "
              f"time={time.time() - started:.1f}s")
        if val_metrics["loss"] < best_val:
            best_val, best_epoch, bad_epochs = val_metrics["loss"], epoch, 0
            save_checkpoint(model, checkpoint_path, args)
            print(f"Saved best checkpoint to {checkpoint_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping triggered")
                break
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_adapter_state_dict(checkpoint["adapter"])
    print(f"Loaded best epoch {best_epoch}")
    if args.skip_test:
        print("Skipping test evaluation; use eval_timesfm_component_token_adapter_v2.py")
        return
    test_metrics = run_epoch(
        model, test_loader, args, device, None,
        args.max_test_batches, "Test", best_epoch, horizons,
    )
    print(f"Test: MSE={test_metrics['mse']:.6f} MAE={test_metrics['mae']:.6f}")
    write_results(args, test_metrics)


if __name__ == "__main__":
    main()
