"""Direct long-horizon evaluation for a Chronos-2 component adapter."""

import argparse
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_factory import data_provider
from models.chronos2.Chronos2ComponentTokenAdapter import Chronos2ComponentTokenAdapter


def get_args():
    parser = argparse.ArgumentParser(description="Chronos-2 component adapter evaluation")
    parser.add_argument("--model_id", default="ETTh1_chronos2_component_adapter_eval")
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
    parser.add_argument("--pretrained_model_path", default="./checkpoints/chronos-2")
    parser.add_argument("--component_codebook_dir", default="./checkpoints/chronos2_component_rqvae")
    parser.add_argument("--adapter_checkpoint", required=True)
    parser.add_argument("--components", default="trend seasonal")
    parser.add_argument("--n_proto", type=int, default=256)
    parser.add_argument("--n_rq_stages", type=int, default=2)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--trend_kernel", type=int, default=25)
    parser.add_argument("--token_dropout", type=float, default=0.1)
    parser.add_argument("--gate_bias_init", type=float, default=-2.0)
    parser.add_argument("--cross_attention_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_test_batches", type=int, default=-1)
    parser.add_argument("--gpu_loc", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--results_dir", default="./results/chronos2_component_token_adapter_eval")
    return parser.parse_args()


def codebook_paths(args, components):
    return {component: os.path.join(args.component_codebook_dir, f"{component}_prototypes_rqvae.pt")
            for component in components}


def build_model(args, device):
    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu")
    components = tuple(checkpoint.get("components", args.components.split()))
    model = Chronos2ComponentTokenAdapter.from_pretrained_component_adapter(
        pretrained_model_path=args.pretrained_model_path,
        component_codebook_paths=codebook_paths(args, components),
        components=components,
        n_proto=checkpoint.get("n_proto", args.n_proto),
        n_rq_stages=checkpoint.get("n_rq_stages", args.n_rq_stages),
        period=checkpoint.get("period", args.period),
        trend_kernel=checkpoint.get("trend_kernel", args.trend_kernel),
        token_dropout=args.token_dropout,
        gate_bias_init=args.gate_bias_init,
        cross_attention_heads=checkpoint.get("cross_attention_heads", args.cross_attention_heads),
    ).to(device)
    model.load_adapter_state_dict(checkpoint["adapter"])
    model.eval()
    return model


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    horizons = tuple(sorted(set(int(value) for value in args.horizons.split())))
    if args.pred_len < max(horizons):
        raise ValueError("pred_len must cover the largest horizon")
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    _, test_loader = data_provider(args, "test")
    model = build_model(args, device)
    sums = {horizon: {"se": 0.0, "ae": 0.0, "n": 0} for horizon in horizons}
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(test_loader, desc="Chronos-2 direct eval")):
            if args.max_test_batches > 0 and batch_index >= args.max_test_batches:
                break
            context = batch[0].float().to(device).squeeze(-1)
            target = batch[1].float().to(device).squeeze(-1)[:, -args.pred_len:]
            output = model(context, prediction_length=max(horizons))
            prediction = model.central_prediction(output)
            for horizon in horizons:
                difference = prediction[:, :horizon] - target[:, :horizon]
                valid = torch.isfinite(target[:, :horizon])
                sums[horizon]["se"] += ((difference ** 2) * valid).sum().item()
                sums[horizon]["ae"] += (difference.abs() * valid).sum().item()
                sums[horizon]["n"] += valid.sum().item()
    metrics = {horizon: {"mse": values["se"] / values["n"], "mae": values["ae"] / values["n"]}
               for horizon, values in sums.items()}
    os.makedirs(args.results_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.data_path))[0]
    path = os.path.join(args.results_dir, f"{stem}_chronos2_component_adapter_long_horizon.txt")
    with open(path, "w") as handle:
        handle.write(f"data_path={args.data_path}\nseq_len={args.seq_len}\nmode=direct\n")
        for horizon in horizons:
            handle.write(f"horizon={horizon} MSE={metrics[horizon]['mse']:.6f} MAE={metrics[horizon]['mae']:.6f}\n")
    for horizon in horizons:
        print(f"H={horizon}: MSE={metrics[horizon]['mse']:.6f} MAE={metrics[horizon]['mae']:.6f}")
    print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
