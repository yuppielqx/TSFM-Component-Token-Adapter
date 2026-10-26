"""Evaluate a trained 64-step component-token adapter with rolling forecasts."""

import argparse
import math
import os
import random
import sys
import warnings

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_factory import data_provider
from models.chronos_bolt.ComponentTemporalTokenAdapter import ComponentTemporalTokenAdapter

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Rolling evaluation for component temporal-token adapter")
    parser.add_argument("--model_id", type=str, default="ETTh1_component_token_adapter_rollout")
    parser.add_argument("--data", type=str, default="ett_h")
    parser.add_argument("--root_path", type=str, default="./datasets/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--percent", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=-1)
    parser.add_argument("--return_feature_id", action="store_true", default=False)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=720,
                        help="Dataset target length; must cover the largest requested horizon.")
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--horizons", type=str, default="96 192 336 720")
    parser.add_argument("--pretrained_model_path", type=str, default="./checkpoints/base/")
    parser.add_argument("--component_codebook_dir", type=str, default="./checkpoints/component_rqvae/")
    parser.add_argument("--adapter_checkpoint", type=str, required=True)
    parser.add_argument("--components", type=str, default="trend seasonal")
    parser.add_argument("--n_proto", type=int, default=256)
    parser.add_argument("--n_rq_stages", type=int, default=2)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--trend_kernel", type=int, default=25)
    parser.add_argument("--token_dropout", type=float, default=0.1)
    parser.add_argument("--gate_bias_init", type=float, default=-2.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_test_batches", type=int, default=-1)
    parser.add_argument("--gpu_loc", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--results_dir", type=str, default="./results/component_token_adapter_rollout/")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def central_prediction(model, output, out_len):
    quantiles = model.quantiles.detach().float().cpu()
    central_idx = int(torch.abs(quantiles - 0.5).argmin().item())
    return output.quantile_preds[:, central_idx, :out_len].float()


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
    checkpoint = torch.load(args.adapter_checkpoint, map_location=device)
    model.load_adapter_state_dict(checkpoint["adapter"])
    model.eval()
    return model


@torch.no_grad()
def rolling_forecast(model, context, horizon):
    chunk_len = int(model.chronos_config.prediction_length)
    if chunk_len <= 0:
        raise ValueError(f"Invalid model prediction length: {chunk_len}")
    rolling_context = context
    chunks = []
    for _ in range(math.ceil(horizon / chunk_len)):
        output = model(context=rolling_context)
        chunk = central_prediction(model, output, chunk_len)
        chunks.append(chunk)
        rolling_context = torch.cat([rolling_context, chunk], dim=-1)[..., -context.shape[-1]:]
    return torch.cat(chunks, dim=-1)[..., :horizon]


def main():
    args = get_args()
    set_seed(args.seed)
    horizons = tuple(sorted(set(int(x) for x in args.horizons.split())))
    if not horizons or horizons[0] <= 0:
        raise ValueError("Horizons must be positive integers")
    if args.pred_len < max(horizons):
        raise ValueError(f"pred_len={args.pred_len} must cover max horizon={max(horizons)}")

    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    _test_set, test_loader = data_provider(args, "test")
    model = build_model(args, device)
    sums = {h: {"se": 0.0, "ae": 0.0, "n": 0} for h in horizons}

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Rolling test")):
        if args.max_test_batches > 0 and batch_idx >= args.max_test_batches:
            break
        context = batch[0].float().to(device).squeeze(-1)
        full_target = batch[1].float().to(device).squeeze(-1)[:, -args.pred_len:]
        max_pred = rolling_forecast(model, context, max(horizons))
        for horizon in horizons:
            pred = max_pred[..., :horizon]
            target = full_target[..., :horizon]
            sums[horizon]["se"] += torch.sum((pred - target) ** 2).item()
            sums[horizon]["ae"] += torch.sum(torch.abs(pred - target)).item()
            sums[horizon]["n"] += target.numel()

    if not sums[horizons[0]]["n"]:
        raise RuntimeError("No test samples were processed")
    metrics = {
        h: {"mse": sums[h]["se"] / sums[h]["n"], "mae": sums[h]["ae"] / sums[h]["n"]}
        for h in horizons
    }
    avg_mse = sum(v["mse"] for v in metrics.values()) / len(metrics)
    avg_mae = sum(v["mae"] for v in metrics.values()) / len(metrics)

    os.makedirs(args.results_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.data_path))[0]
    path = os.path.join(args.results_dir, f"{stem}_component_token_adapter_rollout.txt")
    with open(path, "w") as handle:
        handle.write(f"data_path={args.data_path}\npercent={args.percent}\nseq_len={args.seq_len}\n")
        handle.write(f"base_prediction_length={model.chronos_config.prediction_length}\n")
        for horizon in horizons:
            handle.write(f"horizon={horizon} MSE={metrics[horizon]['mse']:.6f} MAE={metrics[horizon]['mae']:.6f}\n")
        handle.write(f"average MSE={avg_mse:.6f} MAE={avg_mae:.6f}\n")
    for horizon in horizons:
        print(f"H={horizon}: MSE={metrics[horizon]['mse']:.6f} MAE={metrics[horizon]['mae']:.6f}")
    print(f"Average: MSE={avg_mse:.6f} MAE={avg_mae:.6f}")
    print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
