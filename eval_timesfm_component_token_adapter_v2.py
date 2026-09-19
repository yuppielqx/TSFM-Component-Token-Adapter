"""使用 TimesFM 官方 cached decode 评测 V2 Component Token Adapter。

维度约定：context [B,T]，target/prediction [B,Hmax]。每个 horizon 的指标
都从同一组 Hmax 预测 origin 上累计 SE、AE 和有效标量数 n。
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_factory import data_provider
from models.timesfm.TimesFMComponentTokenAdapterV2 import TimesFMComponentTokenAdapterV2
from models.timesfm.component_preprocessing_v2 import PREPROCESSING_NAME


def get_args():
    parser = argparse.ArgumentParser(description="TimesFM v2 component adapter rollout evaluation")
    parser.add_argument("--model_id", default="ETTh1_timesfm_component_adapter_rollout")
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
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--horizons", default="96 192 336 720")
    parser.add_argument("--rollout_chunk", type=int, default=64)
    parser.add_argument("--rollout_adapter_decay", type=float, default=1.0)
    parser.add_argument("--rollout_adapter_min_weight", type=float, default=0.0)
    parser.add_argument("--rollout_correction_clip", type=float, default=0.0)
    parser.add_argument("--rollout_adapter_max_steps", type=int, default=-1)
    parser.add_argument("--pretrained_model_path", default="./checkpoints/timesfm-2.5-200m-pytorch")
    parser.add_argument("--component_codebook_dir", default="./checkpoints/timesfm_component_rqvae_v2")
    parser.add_argument("--adapter_checkpoint", required=True)
    parser.add_argument("--components", default="trend seasonal")
    parser.add_argument("--n_proto", type=int, default=256)
    parser.add_argument("--n_rq_stages", type=int, default=2)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--trend_kernel", type=int, default=25)
    parser.add_argument("--token_dropout", type=float, default=0.1)
    parser.add_argument("--gate_bias_init", type=float, default=-2.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_test_batches", type=int, default=-1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--gpu_loc", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--results_dir", default="./results/timesfm_component_token_adapter_v2_rollout")
    return parser.parse_args()


def codebook_paths(args, components):
    # 每个成分对应一个 V2 RQ codebook checkpoint。
    return {component: os.path.join(args.component_codebook_dir, f"{component}_prototypes_rqvae.pt")
            for component in components}


def build_model(args, device):
    # 先校验 checkpoint 协议，防止旧预处理训练出的 codebook/adapter 被混用。
    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu")
    if checkpoint.get("implementation_version") != 2:
        raise ValueError("Expected a TimesFM component adapter v2 checkpoint")
    if checkpoint.get("component_preprocessing") != PREPROCESSING_NAME:
        raise ValueError(
            f"Expected component_preprocessing={PREPROCESSING_NAME}"
        )
    components = tuple(checkpoint.get("components", args.components.split()))
    model = TimesFMComponentTokenAdapterV2.from_pretrained_component_adapter(
        pretrained_model_path=args.pretrained_model_path,
        component_codebook_paths=codebook_paths(args, components),
        components=components,
        n_proto=checkpoint.get("n_proto", args.n_proto),
        n_rq_stages=checkpoint.get("n_rq_stages", args.n_rq_stages),
        period=args.period,
        trend_kernel=checkpoint.get("trend_kernel", args.trend_kernel),
        token_dropout=args.token_dropout,
        gate_bias_init=args.gate_bias_init,
    ).to(device)
    model.load_adapter_state_dict(checkpoint["adapter"])
    model.eval()
    return model


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.num_shards <= 0 or args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("shard_id must be in [0, num_shards)")
    horizons = tuple(sorted(set(int(value) for value in args.horizons.split())))
    if args.pred_len < max(horizons):
        raise ValueError("pred_len must cover the largest horizon")
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    _, test_loader = data_provider(args, "test")
    model = build_model(args, device)
    # 每个 horizon 分别累计总平方误差、总绝对误差和有效元素数。
    sums = {horizon: {"se": 0.0, "ae": 0.0, "n": 0} for horizon in horizons}
    for batch_index, batch in enumerate(tqdm(test_loader, desc="TimesFM official decode")):
        if args.max_test_batches > 0 and batch_index >= args.max_test_batches:
            break
        if batch_index % args.num_shards != args.shard_id:
            continue
        context = batch[0].float().to(device).squeeze(-1)  # [B,seq_len]
        target = batch[1].float().to(device).squeeze(-1)[:, -args.pred_len:]  # [B,Hmax]
        prediction = model.forecast(
            context,
            max(horizons),
            rollout_adapter_decay=args.rollout_adapter_decay,
            rollout_adapter_min_weight=args.rollout_adapter_min_weight,
            rollout_correction_clip=args.rollout_correction_clip,
            rollout_adapter_max_steps=args.rollout_adapter_max_steps,
        )  # [B,max(horizons)]
        for horizon in horizons:
            # 所有 horizon 都是同一次公共-origin 预测的前缀 [B,h]。
            difference = prediction[:, :horizon] - target[:, :horizon]
            valid = torch.isfinite(target[:, :horizon])
            sums[horizon]["se"] += ((difference ** 2) * valid).sum().item()
            sums[horizon]["ae"] += (difference.abs() * valid).sum().item()
            sums[horizon]["n"] += valid.sum().item()
    # micro 指标：先跨 batch/channel/time 求和，再除以有效元素总数。
    metrics = {horizon: {"mse": values["se"] / values["n"], "mae": values["ae"] / values["n"]}
               for horizon, values in sums.items()}
    os.makedirs(args.results_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.data_path))[0]
    path = os.path.join(
        args.results_dir,
        f"{stem}_timesfm_component_adapter_v2_rollout_"
        f"shard{args.shard_id}of{args.num_shards}.txt",
    )
    with open(path, "w") as handle:
        handle.write(f"data_path={args.data_path}\nseq_len={args.seq_len}\n")
        handle.write("inference=official_cached_decode_with_component_adapter\n")
        handle.write(
            f"rollout_adapter_decay={args.rollout_adapter_decay}\n"
            f"rollout_adapter_min_weight={args.rollout_adapter_min_weight}\n"
            f"rollout_correction_clip={args.rollout_correction_clip}\n"
            f"rollout_adapter_max_steps={args.rollout_adapter_max_steps}\n"
        )
        handle.write(f"legacy_rollout_chunk_arg={args.rollout_chunk} (unused)\n")
        handle.write(f"shard_id={args.shard_id} num_shards={args.num_shards}\n")
        for horizon in horizons:
            handle.write("horizon={} se={:.10f} ae={:.10f} n={}\n".format(horizon, sums[horizon]["se"], sums[horizon]["ae"], sums[horizon]["n"]))
            handle.write(f"horizon={horizon} MSE={metrics[horizon]['mse']:.6f} MAE={metrics[horizon]['mae']:.6f}\n")
    for horizon in horizons:
        print(f"H={horizon}: MSE={metrics[horizon]['mse']:.6f} MAE={metrics[horizon]['mae']:.6f}")
    print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
