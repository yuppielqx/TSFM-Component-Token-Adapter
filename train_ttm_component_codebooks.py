"""Train frozen-TTM component RQ-VAE codebooks from parquet pretraining windows.

Only the codebook and reconstruction heads are optimized.  TTM is loaded from
the public checkpoint, then kept in eval mode with gradients disabled.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from models.tinytimemixer import TTMComponentCodebook, TinyTimeMixerForPrediction


class ParquetWindows(IterableDataset):
    """Stream one parquet ``x`` column without materializing the full corpus."""

    SPLITS = {"train": (0.0, 0.7), "val": (0.7, 0.8)}

    def __init__(self, path: str, split: str, context_length: int, arrow_batch_size: int = 4096, shuffle_buffer: int = 0, seed: int = 0):
        self.path, self.split = path, split
        self.context_length, self.arrow_batch_size = context_length, arrow_batch_size
        self.shuffle_buffer, self.seed = shuffle_buffer, seed
        total_rows = pq.ParquetFile(path).metadata.num_rows
        lo, hi = self.SPLITS[split]
        self.start, self.end = int(total_rows * lo), int(total_rows * hi)
        if self.end <= self.start:
            raise ValueError(f"Empty {split} split in {path}")

    def __iter__(self) -> Iterator[torch.Tensor]:
        file = pq.ParquetFile(self.path)
        rng = random.Random(self.seed)
        buffer = []
        row = 0
        for batch in file.iter_batches(columns=["x"], batch_size=self.arrow_batch_size):
            values = batch.column(0).to_pylist()
            for value in values:
                if self.start <= row < self.end:
                    array = np.asarray(value, dtype=np.float32)
                    if array.ndim != 1 or array.size < self.context_length:
                        raise ValueError("Every x value must be a 1-D window at least context_length long")
                    item = torch.from_numpy(array[-self.context_length:].copy())
                    if self.shuffle_buffer <= 1:
                        yield item
                    elif len(buffer) < self.shuffle_buffer:
                        buffer.append(item)
                    else:
                        index = rng.randrange(len(buffer))
                        yield buffer[index]
                        buffer[index] = item
                row += 1
                if row >= self.end:
                    break
            if row >= self.end:
                break
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))


def decompose(values: torch.Tensor, component: str, period: int, trend_kernel: int) -> torch.Tensor:
    """Additive moving-average trend / periodic seasonal decomposition."""
    if component not in {"trend", "seasonal", "residual"}:
        raise ValueError(f"Unknown component: {component}")
    if trend_kernel < 1 or trend_kernel % 2 == 0:
        raise ValueError("trend_kernel must be a positive odd number")
    padding = trend_kernel // 2
    padded = F.pad(values.unsqueeze(1), (padding, padding), mode="replicate")
    trend = F.avg_pool1d(padded, kernel_size=trend_kernel, stride=1).squeeze(1)
    detrended = values - trend
    if component == "trend":
        return trend
    if period < 1:
        raise ValueError("period must be positive")
    phase = torch.arange(values.shape[-1], device=values.device) % period
    sums = torch.zeros(values.shape[0], period, device=values.device, dtype=values.dtype)
    counts = torch.zeros(period, device=values.device, dtype=values.dtype)
    sums.scatter_add_(1, phase.unsqueeze(0).expand_as(detrended), detrended)
    counts.scatter_add_(0, phase, torch.ones_like(phase, dtype=values.dtype))
    seasonal = (sums / counts.clamp_min(1)).gather(1, phase.unsqueeze(0).expand_as(detrended))
    return seasonal if component == "seasonal" else detrended - seasonal


@torch.no_grad()
def extract_ttm_features(backbone, values: torch.Tensor, frequency_token: int):
    valid = torch.isfinite(values)
    clean = torch.nan_to_num(values, nan=0.0).unsqueeze(-1)
    observed = valid.unsqueeze(-1)
    freq_token = None
    if backbone.config.resolution_prefix_tuning:
        if not 0 <= frequency_token < backbone.config.frequency_token_vocab_size:
            raise ValueError(
                f"frequency_token must be in [0, {backbone.config.frequency_token_vocab_size}), got {frequency_token}"
            )
        freq_token = torch.full((values.shape[0],), frequency_token, dtype=torch.long, device=values.device)
    output = backbone(
        past_values=clean, past_observed_mask=observed, freq_token=freq_token, return_dict=True
    )
    # channel independent TTM yields [B, 1, N, D]. Keep this general for
    # fine-tuned channel-mixing checkpoints too.
    hidden = output.last_hidden_state
    # R2's resolution prefix is not a temporal patch and has no reconstruction target.
    if backbone.config.resolution_prefix_tuning:
        hidden = hidden[:, :, 1:, :]
    return hidden.detach(), output.patch_input.detach(), output.loc.detach(), output.scale.detach(), valid


def save_checkpoint(path, model, args, component, best_val):
    payload = model.checkpoint(
        model_id=args.model_id, context_length=args.context_length, component=component,
        period=args.period, frequency_token=args.frequency_token,
        trend_kernel=args.trend_kernel, best_val_loss=best_val,
    )
    torch.save(payload, path)
    with open(Path(path).with_suffix(".meta.json"), "w") as handle:
        json.dump({key: value for key, value in payload.items() if not isinstance(value, (torch.Tensor, dict))}, handle, indent=2)


def run_epoch(model, backbone, loader, component, args, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, steps = 0.0, 0
    limit = args.max_train_batches if training else args.max_val_batches
    iterator = tqdm(loader, desc=f"{component} {'train' if training else 'val'}", leave=False)
    for step, batch in enumerate(iterator):
        if limit > 0 and step >= limit:
            break
        values = decompose(batch.to(device, non_blocking=True), component, args.period, args.trend_kernel)
        hidden, patches, _, _, valid = extract_ttm_features(backbone, values, args.frequency_token)
        patch_length = backbone.config.patch_length
        patch_mask = valid.unfold(-1, patch_length, patch_length).unsqueeze(1)
        result = model(hidden, patches, patch_mask)
        if training:
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        total += result["loss"].item()
        steps += 1
        iterator.set_postfix(loss=f"{total / steps:.5f}")
    return total / max(steps, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="ibm-research/ttm-research-r2")
    parser.add_argument("--parquet_path", default="./datasets/pretrain/retrieval_database_512.parquet")
    parser.add_argument("--output_dir", default="./checkpoints/ttm_component_rqvae")
    parser.add_argument("--components", default="trend seasonal residual")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument(
        "--frequency_token", type=int, default=7,
        help="TTM-R2 resolution token; 7 is hourly and 0 is out-of-vocabulary.",
    )
    parser.add_argument("--trend_kernel", type=int, default=25)
    parser.add_argument("--n_proto", type=int, default=256)
    parser.add_argument("--n_rq_stages", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--arrow_batch_size", type=int, default=4096)
    parser.add_argument("--shuffle_buffer", type=int, default=8192)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--max_val_batches", type=int, default=-1)
    parser.add_argument("--gpu_loc", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    components = tuple(filter(None, args.components.replace(",", " ").split()))
    if not components or set(components) - {"trend", "seasonal", "residual"}:
        raise ValueError("components must be a nonempty subset of: trend seasonal residual")

    print(f"Loading frozen TTM checkpoint: {args.model_id}")
    ttm = TinyTimeMixerForPrediction.from_pretrained(args.model_id).to(device).eval()
    backbone = ttm.backbone
    config = ttm.config
    if config.context_length != args.context_length:
        raise ValueError(f"Checkpoint context_length={config.context_length}, expected {args.context_length}")
    if config.num_input_channels != 1:
        raise ValueError("Codebook pretraining expects a univariate/channel-independent TTM checkpoint")
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    print(f"TTM: d_model={config.d_model}, patch_length={config.patch_length}, device={device}")

    loaders = {split: DataLoader(ParquetWindows(args.parquet_path, split, args.context_length, args.arrow_batch_size,
                                                args.shuffle_buffer if split == "train" else 0, args.seed),
                                 batch_size=args.batch_size, num_workers=0, pin_memory=device.type == "cuda")
               for split in ("train", "val")}
    os.makedirs(args.output_dir, exist_ok=True)
    for component in components:
        model = TTMComponentCodebook(config.d_model, config.patch_length, args.n_proto, args.n_rq_stages, args.beta).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        best_state, best_val, stale = None, float("inf"), 0
        for epoch in range(1, args.train_epochs + 1):
            began = time.time()
            train_loss = run_epoch(model, backbone, loaders["train"], component, args, device, optimizer)
            with torch.no_grad():
                val_loss = run_epoch(model, backbone, loaders["val"], component, args, device)
            print(f"{component} epoch={epoch} train={train_loss:.6f} val={val_loss:.6f} seconds={time.time()-began:.1f}")
            if val_loss < best_val:
                best_val, stale = val_loss, 0
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(best_state)
        target = Path(args.output_dir) / f"{component}_prototypes_rqvae.pt"
        save_checkpoint(target, model, args, component, best_val)
        print(f"saved {target}")


if __name__ == "__main__":
    main()
