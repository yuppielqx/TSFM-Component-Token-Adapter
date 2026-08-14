"""Train Sundial component-wise RQ-VAE codebooks on 512-point windows."""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.sundial.SundialPrototype import SundialWithRQVAEPrototypes
from utils.decomposition import VALID_COMPONENTS, select_component


class ParquetWindowDataset(Dataset):
    SPLITS = {"train": (0.0, 0.7), "val": (0.7, 0.8), "test": (0.8, 1.0)}

    def __init__(self, values, split):
        lo_ratio, hi_ratio = self.SPLITS[split]
        lo, hi = int(lo_ratio * len(values)), int(hi_ratio * len(values))
        self.values = values[lo:hi]
        if self.values.shape[1] != 512:
            raise ValueError(f"Sundial codebook training requires 512-point windows, got {self.values.shape[1]}")
        print(f"  [{split}] {self.values.shape} rows {lo:,}-{hi:,}")

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return torch.from_numpy(self.values[index])


def get_args():
    parser = argparse.ArgumentParser(description="Train Sundial component RQ-VAE codebooks")
    parser.add_argument("--components", default="trend seasonal residual")
    parser.add_argument("--pretrained_model_path", default="./checkpoints/sundial-base-128m")
    parser.add_argument("--parquet_path", default="./datasets/pretrain/retrieval_database_512.parquet")
    parser.add_argument("--output_dir", default="./checkpoints/sundial_component_rqvae")
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--trend_kernel", type=int, default=25)
    parser.add_argument("--n_proto", type=int, default=256)
    parser.add_argument("--n_rq_stages", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--max_val_batches", type=int, default=-1)
    parser.add_argument("--max_test_batches", type=int, default=-1)
    parser.add_argument("--gpu_loc", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def parse_components(text):
    result = [x.strip() for x in text.replace(",", " ").split() if x.strip()]
    invalid = set(result) - set(VALID_COMPONENTS)
    if not result or invalid:
        raise ValueError(f"components must be selected from {VALID_COMPONENTS}")
    return result


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_windows(parquet_path):
    frame = pd.read_parquet(parquet_path, columns=["x"])
    values = np.vstack(frame["x"].to_numpy()).astype(np.float32)
    del frame
    if values.ndim != 2 or values.shape[1] != 512:
        raise ValueError(f"Sundial codebook training requires 512-point windows, got {values.shape}")
    print(f"Loaded shared windows once: {values.shape}")
    return values


def make_loaders(args):
    values = load_windows(args.parquet_path)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=True, drop_last=True)
    return (
        DataLoader(ParquetWindowDataset(values, "train"), shuffle=True, **kwargs),
        DataLoader(ParquetWindowDataset(values, "val"), shuffle=False, **kwargs),
        DataLoader(ParquetWindowDataset(values, "test"), shuffle=False, **kwargs),
    )


def component_batch(batch, component, args, device):
    return select_component(
        batch.float().to(device, non_blocking=True), component,
        period=args.period, trend_kernel=args.trend_kernel,
    )


def snapshot(model):
    return {
        "codebooks": [cb.weight.detach().cpu().clone() for cb in model.residual_quantizer.codebooks],
        "heads": [{k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                  for head in model.reconstruction_heads],
    }


def restore(model, state, device):
    for codebook, weight in zip(model.residual_quantizer.codebooks, state["codebooks"]):
        codebook.weight.data.copy_(weight.to(device=device, dtype=codebook.weight.dtype))
    for head, state_dict in zip(model.reconstruction_heads, state["heads"]):
        head.load_state_dict({k: v.to(device) for k, v in state_dict.items()})


@torch.no_grad()
def evaluate(model, loader, component, args, device, limit):
    model.eval()
    total, count = 0.0, 0
    for index, batch in enumerate(loader):
        if limit > 0 and index >= limit:
            break
        total += model(component_batch(batch, component, args, device))["loss"].item()
        count += 1
    if not count:
        raise RuntimeError(f"No evaluation batches for component={component}")
    return total / count


def train_component(component, all_loaders, args, device):
    print(f"\n=== Sundial component: {component} ===")
    model = SundialWithRQVAEPrototypes.from_pretrained_for_rqvae(
        args.pretrained_model_path, args.n_proto, args.n_rq_stages,
        args.beta, args.mask_ratio, device,
    )
    model.print_trainable_params()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    train_loader, val_loader, test_loader = all_loaders
    best_state, best_loss, stale, step = None, float("inf"), 0, 0

    for epoch in range(1, args.train_epochs + 1):
        model.train()
        total, count, started = 0.0, 0, time.time()
        for index, batch in enumerate(tqdm(train_loader, desc=f"{component} {epoch}/{args.train_epochs}")):
            if args.max_train_batches > 0 and index >= args.max_train_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = model(component_batch(batch, component, args, device))["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            total += loss.item()
            count += 1
            step += 1
            if step % args.log_every == 0:
                print(f"  step={step} avg_loss={total / count:.6f}")
        if not count:
            raise RuntimeError(f"No training batches for component={component}")
        val_loss = evaluate(model, val_loader, component, args, device, args.max_val_batches)
        print(f"{component} epoch {epoch}: train_loss={total / count:.6f} "
              f"val_loss={val_loss:.6f} time={time.time() - started:.1f}s")
        if val_loss < best_loss:
            best_state, best_loss, stale = snapshot(model), val_loss, 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping for {component}.")
                break

    if best_state is None:
        raise RuntimeError(f"No best checkpoint captured for component={component}")
    restore(model, best_state, device)
    test_loss = evaluate(model, test_loader, component, args, device, args.max_test_batches)
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"{component}_prototypes_rqvae.pt")
    model.save_codebooks(path)
    metadata = {
        "component": component,
        "backbone": model.backbone_name,
        "pretrained_model_path": args.pretrained_model_path,
        "parquet_path": args.parquet_path,
        "context_length": 512,
        "input_patch_size": model.input_patch_size,
        "d_model": model.d_model,
        "period": args.period,
        "trend_kernel": args.trend_kernel,
        "n_proto": args.n_proto,
        "n_rq_stages": args.n_rq_stages,
        "beta": args.beta,
        "mask_ratio": args.mask_ratio,
        "best_val_loss": best_loss,
        "test_loss": test_loss,
    }
    with open(os.path.join(args.output_dir, f"{component}_prototypes_rqvae.meta.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Component {component}: test_loss={test_loss:.6f}; saved={path}")
    return test_loss


def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    components = parse_components(args.components)
    print(f"Device: {device}; components={components}; Sundial context=512, patch_size=16")
    all_loaders = make_loaders(args)
    results = {component: train_component(component, all_loaders, args, device)
               for component in components}
    print("\n=== Sundial component RQ-VAE summary ===")
    for component, loss in results.items():
        print(f"{component}: test_loss={loss:.6f}")


if __name__ == "__main__":
    main()
