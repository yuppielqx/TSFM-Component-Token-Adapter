"""Train component-wise RQ-VAE codebooks after time-series decomposition."""

import argparse
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.chronos_bolt.ChronosBoltPrototype import ChronosBoltWithRQVAEPrototypes
from utils.decomposition import VALID_COMPONENTS, select_component

warnings.filterwarnings("ignore")


class ParquetWindowDataset(Dataset):
    SPLITS = {"train": (0.0, 0.7), "val": (0.7, 0.8), "test": (0.8, 1.0)}

    def __init__(self, parquet_path: str, split: str = "train"):
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {tuple(self.SPLITS)}")
        df = pd.read_parquet(parquet_path, columns=["x"])
        xs = np.vstack(df["x"].to_numpy()).astype(np.float32)
        lo_r, hi_r = self.SPLITS[split]
        lo, hi = int(lo_r * len(xs)), int(hi_r * len(xs))
        self.xs = xs[lo:hi]
        print(f"  [{split}] {self.xs.shape} rows {lo:,}-{hi:,}")

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.xs[idx])


def parse_components(text: str):
    components = [x.strip() for x in text.replace(",", " ").split() if x.strip()]
    if not components:
        raise ValueError("At least one component is required")
    for component in components:
        if component not in VALID_COMPONENTS:
            raise ValueError(f"Unknown component {component}; valid={VALID_COMPONENTS}")
    return components


def get_args():
    parser = argparse.ArgumentParser(description="Train component-wise RQ-VAE prototypes")
    parser.add_argument("--components", type=str, default="trend seasonal residual")
    parser.add_argument("--pretrained_model_path", type=str, default="./checkpoints/chronosbolt-base")
    parser.add_argument("--parquet_path", type=str, default="./datasets/pretrain/retrieval_database_512.parquet")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/component_rqvae/")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/component_proto_ckpt/")
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--trend_kernel", type=int, default=25)
    parser.add_argument("--n_proto", type=int, default=256)
    parser.add_argument("--n_rq_stages", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(args):
    train_set = ParquetWindowDataset(args.parquet_path, "train")
    val_set = ParquetWindowDataset(args.parquet_path, "val")
    test_set = ParquetWindowDataset(args.parquet_path, "test")
    kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    return (
        DataLoader(train_set, shuffle=True, **kwargs),
        DataLoader(val_set, shuffle=False, **kwargs),
        DataLoader(test_set, shuffle=False, **kwargs),
    )


def component_batch(batch_x, component, args, device):
    batch_x = batch_x.float().to(device)
    return select_component(batch_x, component, period=args.period, trend_kernel=args.trend_kernel)


def capture_codebook_state(model):
    return {
        "codebooks": [cb.weight.detach().cpu().clone() for cb in model.residual_quantizer.codebooks],
        "reconstruction_heads": [
            {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            for head in model.reconstruction_heads
        ],
    }


def load_codebook_state(model, state, device):
    for cb, weight in zip(model.residual_quantizer.codebooks, state["codebooks"]):
        cb.weight.data.copy_(weight.to(device=device, dtype=cb.weight.dtype))
    for head, head_state in zip(model.reconstruction_heads, state["reconstruction_heads"]):
        head.load_state_dict({k: v.to(device) for k, v in head_state.items()})


def validate(model, loader, component, args, device, max_batches=-1):
    model.eval()
    total, n_batches = 0.0, 0
    with torch.no_grad():
        for batch_idx, batch_x in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            comp_x = component_batch(batch_x, component, args, device)
            output = model(context=comp_x)
            total += output.loss.item()
            n_batches += 1
    if n_batches == 0:
        raise RuntimeError(f"No validation batches for component={component}")
    return total / n_batches


def train_one_component(component, loaders, args, device):
    print(f"\n=== Component: {component} ===")
    train_loader, val_loader, test_loader = loaders
    model = ChronosBoltWithRQVAEPrototypes.from_pretrained_for_rqvae(
        pretrained_model_path=args.pretrained_model_path,
        n_proto=args.n_proto,
        n_rq_stages=args.n_rq_stages,
        beta=args.beta,
        mask_ratio=args.mask_ratio,
    )
    model.to(device)
    model.print_trainable_params()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(1, args.train_epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()
        iterator = tqdm(train_loader, desc=f"{component} epoch {epoch}/{args.train_epochs}")
        for batch_idx, batch_x in enumerate(iterator):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            comp_x = component_batch(batch_x, component, args, device)
            optimizer.zero_grad()
            output = model(context=comp_x)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            epoch_loss += output.loss.item()
            n_batches += 1
            global_step += 1
            if global_step % args.log_every == 0:
                print(f"  step={global_step} avg_loss={epoch_loss / n_batches:.6f}")
            iterator.set_postfix(loss=epoch_loss / max(n_batches, 1))
        val_loss = validate(model, val_loader, component, args, device, args.max_val_batches)
        print(
            f"{component} epoch {epoch}: train_loss={epoch_loss / max(n_batches, 1):.6f} "
            f"val_loss={val_loss:.6f} time={time.time() - t0:.1f}s"
        )
        if val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.6f} --> {val_loss:.6f}). Keeping lightweight codebook state in memory ...")
            best_val_loss = val_loss
            best_state = capture_codebook_state(model)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {args.patience}")
            if patience_counter >= args.patience:
                print(f"Early stopping for {component}.")
                break

    if best_state is not None:
        load_codebook_state(model, best_state, device)
        print(f"Loaded best lightweight codebook state for {component}")
    test_loss = validate(model, test_loader, component, args, device, args.max_test_batches)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{component}_prototypes_rqvae.pt")
    model.save_codebooks(output_path)
    metadata_path = os.path.join(args.output_dir, f"{component}_prototypes_rqvae.meta.json")
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "component": component,
                "tokenizer_train_source": "retrieval_database",
                "parquet_path": args.parquet_path,
                "pretrained_model_path": args.pretrained_model_path,
                "period": args.period,
                "trend_kernel": args.trend_kernel,
                "n_proto": args.n_proto,
                "n_rq_stages": args.n_rq_stages,
                "beta": args.beta,
                "mask_ratio": args.mask_ratio,
                "test_loss": test_loss,
            },
            f,
            indent=2,
        )
    print(f"Component {component}: test_loss={test_loss:.6f}; saved={output_path}")
    return test_loss


def main():
    args = get_args()
    set_seed(args.seed)
    components = parse_components(args.components)
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; components={components}; period={args.period}; trend_kernel={args.trend_kernel}")
    loaders = make_loaders(args)
    results = {}
    for component in components:
        results[component] = train_one_component(component, loaders, args, device)
    print("\n=== Component RQ-VAE Summary ===")
    for component, loss in results.items():
        print(f"{component}: test_loss={loss:.6f}")


if __name__ == "__main__":
    main()
