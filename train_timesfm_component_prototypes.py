"""Train TimesFM component-wise RQ-VAE codebooks on pretraining windows."""

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

from models.timesfm.TimesFMPrototype import TimesFMWithRQVAEPrototypes
from utils.decomposition import VALID_COMPONENTS, select_component


class ParquetWindowDataset(Dataset):
    SPLITS = {"train": (0.0, 0.7), "val": (0.7, 0.8), "test": (0.8, 1.0)}

    def __init__(self, parquet_path, split):
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {tuple(self.SPLITS)}")
        frame = pd.read_parquet(parquet_path, columns=["x"])
        values = np.vstack(frame["x"].to_numpy()).astype(np.float32)
        lo_ratio, hi_ratio = self.SPLITS[split]
        lo, hi = int(lo_ratio * len(values)), int(hi_ratio * len(values))
        self.values = values[lo:hi]
        print(f"  [{split}] {self.values.shape} rows {lo:,}-{hi:,}")

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return torch.from_numpy(self.values[index])


def get_args():
    parser = argparse.ArgumentParser(description="Train TimesFM component RQ-VAE codebooks")
    parser.add_argument("--components", default="trend seasonal residual")
    parser.add_argument("--pretrained_model_path", default="./checkpoints/timesfm-2.5-200m-pytorch")
    parser.add_argument("--parquet_path", default="./datasets/pretrain/retrieval_database_512.parquet")
    parser.add_argument("--output_dir", default="./checkpoints/timesfm_component_rqvae")
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
    components = [item.strip() for item in text.replace(",", " ").split() if item.strip()]
    if not components:
        raise ValueError("At least one component is required")
    invalid = set(components) - set(VALID_COMPONENTS)
    if invalid:
        raise ValueError(f"Unknown components: {sorted(invalid)}; valid={VALID_COMPONENTS}")
    return components


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(args):
    kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=True, drop_last=True)
    return (
        DataLoader(ParquetWindowDataset(args.parquet_path, "train"), shuffle=True, **kwargs),
        DataLoader(ParquetWindowDataset(args.parquet_path, "val"), shuffle=False, **kwargs),
        DataLoader(ParquetWindowDataset(args.parquet_path, "test"), shuffle=False, **kwargs),
    )


def component_batch(batch_x, component, args, device):
    return select_component(
        batch_x.float().to(device, non_blocking=True),
        component, period=args.period, trend_kernel=args.trend_kernel,
    )


def capture_codebook_state(model):
    return {
        "codebooks": [cb.weight.detach().cpu().clone() for cb in model.residual_quantizer.codebooks],
        "reconstruction_heads": [
            {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            for head in model.reconstruction_heads
        ],
    }


def restore_codebook_state(model, state, device):
    for codebook, weight in zip(model.residual_quantizer.codebooks, state["codebooks"]):
        codebook.weight.data.copy_(weight.to(device=device, dtype=codebook.weight.dtype))
    for head, head_state in zip(model.reconstruction_heads, state["reconstruction_heads"]):
        head.load_state_dict({key: value.to(device) for key, value in head_state.items()})


@torch.no_grad()
def validate(model, loader, component, args, device, max_batches):
    model.eval()
    total, count = 0.0, 0
    for batch_index, batch_x in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        total += model(component_batch(batch_x, component, args, device))["loss"].item()
        count += 1
    if not count:
        raise RuntimeError(f"No validation batches for component={component}")
    return total / count


def train_one_component(component, loaders, args, device):
    print(f"\n=== TimesFM component: {component} ===")
    model = TimesFMWithRQVAEPrototypes.from_pretrained_for_rqvae(
        pretrained_model_path=args.pretrained_model_path,
        n_proto=args.n_proto,
        n_rq_stages=args.n_rq_stages,
        beta=args.beta,
        mask_ratio=args.mask_ratio,
        device=device,
    )
    model.print_trainable_params()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader, val_loader, test_loader = loaders
    best_state, best_val, patience_counter, step = None, float("inf"), 0, 0

    for epoch in range(1, args.train_epochs + 1):
        model.train()
        epoch_loss, count, started = 0.0, 0, time.time()
        for batch_index, batch_x in enumerate(tqdm(train_loader, desc=f"{component} {epoch}/{args.train_epochs}")):
            if args.max_train_batches > 0 and batch_index >= args.max_train_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = model(component_batch(batch_x, component, args, device))["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            count += 1
            step += 1
            if step % args.log_every == 0:
                print(f"  step={step} avg_loss={epoch_loss / count:.6f}")

        val_loss = validate(model, val_loader, component, args, device, args.max_val_batches)
        print(f"{component} epoch {epoch}: train_loss={epoch_loss / max(count, 1):.6f} "
              f"val_loss={val_loss:.6f} time={time.time() - started:.1f}s")
        if val_loss < best_val:
            best_val, best_state, patience_counter = val_loss, capture_codebook_state(model), 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping for {component}.")
                break

    restore_codebook_state(model, best_state, device)
    test_loss = validate(model, test_loader, component, args, device, args.max_test_batches)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{component}_prototypes_rqvae.pt")
    model.save_codebooks(output_path)
    with open(os.path.join(args.output_dir, f"{component}_prototypes_rqvae.meta.json"), "w") as handle:
        json.dump({
            "component": component,
            "backbone": "timesfm-2.5-200m-pytorch",
            "pretrained_model_path": args.pretrained_model_path,
            "parquet_path": args.parquet_path,
            "period": args.period,
            "trend_kernel": args.trend_kernel,
            "n_proto": args.n_proto,
            "n_rq_stages": args.n_rq_stages,
            "beta": args.beta,
            "mask_ratio": args.mask_ratio,
            "test_loss": test_loss,
        }, handle, indent=2)
    print(f"Component {component}: test_loss={test_loss:.6f}; saved={output_path}")
    return test_loss


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    components = parse_components(args.components)
    print(f"Device: {device}; components={components}; TimesFM patch_size=32")
    loaders = make_loaders(args)
    results = {component: train_one_component(component, loaders, args, device)
               for component in components}
    print("\n=== TimesFM component RQ-VAE summary ===")
    for component, loss in results.items():
        print(f"{component}: test_loss={loss:.6f}")


if __name__ == "__main__":
    main()
