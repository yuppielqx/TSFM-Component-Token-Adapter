"""
create_prototypes.py

Extract encoder embeddings from ChronosBolt backbone on pre-training data,
then run K-means clustering to obtain prototype centroids.

Two data-source modes (--data_source):
  pretrain  - reads parquet files under datasets/pretrain/pretrain_pairs_ctx{N}/
              (requires the pre-training data to be downloaded)
  pkl       - reads existing retrieval_database/*.pkl files
              (already on disk; useful as a fallback)

Usage examples:

  # Primary mode (pretrain parquet)
  python create_prototypes.py \
      --data_source pretrain \
      --data_path ../datasets/pretrain/pretrain_pairs_ctx512 \
      --pretrained_model_path ./checkpoints/base/ \
      --n_clusters 512 \
      --max_samples 500000 \
      --output_path ./checkpoints/prototypes_k512.npy \
      --gpu_loc 0

  # Fallback mode (existing pkl databases)
  python create_prototypes.py \
      --data_source pkl \
      --pkl_dir ../retrieval_database/ \
      --pretrained_model_path ./checkpoints/base/ \
      --n_clusters 512 \
      --output_path ./checkpoints/prototypes_k512.npy \
      --gpu_loc 0
"""

import os
import pickle
import argparse
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset
from transformers import AutoConfig

from models.ChronosBolt import ChronosBoltModelForForecasting

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class SimplePretrainDataset(IterableDataset):
    """
    Reads pretrain parquet files (same directory format used by CustomPretrainDataset)
    but only returns context windows from the 'target' field.
    Does NOT require a retriever object.
    """

    def __init__(self, dataset_path: str, context_length: int = 512, max_samples: int = None):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.context_length = context_length
        self.max_samples = max_samples

        if not self.dataset_path.is_dir():
            raise ValueError(f"dataset_path is not a directory: {dataset_path}")

        parquet_files = list(self.dataset_path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {dataset_path}")

        from gluonts.dataset.common import FileDataset
        self.dataset = FileDataset(self.dataset_path, freq="1H")

    def __iter__(self):
        count = 0
        for entry in self.dataset:
            if self.max_samples is not None and count >= self.max_samples:
                return
            target = np.asarray(entry["target"], dtype=np.float32)
            x = target[: self.context_length]
            if len(x) < self.context_length:
                pad = np.full(self.context_length - len(x), np.nan, dtype=np.float32)
                x = np.concatenate([pad, x])
            yield torch.from_numpy(x)
            count += 1


class PklWindowDataset(Dataset):
    """
    Extracts sliding windows of length context_length from the 'raw_data' field
    in retrieval_database/*.pkl files.

    Each pkl file stores a dict of {variable_name: {raw_data, timestamps, embeddings, metadata}}.
    """

    def __init__(self, pkl_dir: str, context_length: int = 512, stride: int = 64):
        self.context_length = context_length
        self.stride = stride
        self.windows = []  # list of np.ndarray, each shape (context_length,)

        pkl_dir = Path(pkl_dir)
        pkl_files = sorted(pkl_dir.glob("*.pkl"))
        if not pkl_files:
            raise ValueError(f"No pkl files found in {pkl_dir}")

        print(f"Loading pkl files from {pkl_dir} ...")
        for pkl_path in pkl_files:
            print(f"  Reading {pkl_path.name}")
            with open(pkl_path, "rb") as f:
                databases = pickle.load(f)
            # databases: {variable_name: {raw_data, timestamps, embeddings, metadata}}
            for var_name, db in databases.items():
                raw_data = np.asarray(db["raw_data"], dtype=np.float32)
                n = len(raw_data)
                starts = range(0, n - context_length + 1, stride)
                for s in starts:
                    self.windows.append(raw_data[s : s + context_length])

        print(f"Total windows extracted from pkl: {len(self.windows):,}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx])


class ParquetXDataset(Dataset):
    """
    Reads the 'x' column from a single retrieval_database parquet file.
    Each row's x is a (context_length,) float32 array (pre-extracted context window).
    """

    def __init__(self, parquet_path: str):
        import pandas as pd
        print(f"Loading parquet from {parquet_path} ...")
        df = pd.read_parquet(parquet_path, columns=["x"])
        self.xs = np.vstack(df["x"].to_numpy()).astype(np.float32)  # (N, context_length)
        print(f"Loaded {len(self.xs):,} samples, shape: {self.xs.shape}")

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.xs[idx])  # (context_length,)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def iter_patch_embeddings(
    model: ChronosBoltModelForForecasting,
    dataloader: DataLoader,
    device: torch.device,
):
    """
    Generator: run ChronosBolt encoder batch by batch, yield valid patch
    hidden states as numpy arrays of shape (n_valid_patches, d_model).
    Each valid patch is an independent clustering point.
    """
    model.eval()

    for batch in dataloader:
        context = batch.float().to(device)  # (B, context_length)
        B = context.shape[0]

        mask = torch.isnan(context).logical_not().to(context.dtype)

        if context.shape[-1] > model.chronos_config.context_length:
            context = context[..., -model.chronos_config.context_length:]
            mask = mask[..., -model.chronos_config.context_length:]

        context_norm, _ = model.instance_norm(context)
        context_norm = context_norm.to(model.dtype)
        mask = mask.to(model.dtype)

        patched_context = model.patch(context_norm)
        patched_mask = torch.nan_to_num(model.patch(mask), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        attention_mask = patched_mask.sum(dim=-1) > 0  # (B, n_patches)

        input_embeds = model.input_patch_embedding(patched_context)  # (B, n_patches, d_model)

        if model.chronos_config.use_reg_token:
            reg_input_ids = torch.full(
                (B, 1), model.config.reg_token_id, device=device
            )
            reg_embeds = model.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(reg_input_ids, dtype=attention_mask.dtype)],
                dim=-1,
            )

        encoder_outputs = model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
        )
        hidden_states = encoder_outputs[0].float()  # (B, n_patches, d_model)

        # Yield each batch's valid patches — exclude reg_token if present
        batch_patches = []
        for b in range(B):
            valid_mask = attention_mask[b].bool()
            if model.chronos_config.use_reg_token:
                valid_mask = valid_mask[:-1]
                patch_hidden = hidden_states[b, :-1]
            else:
                patch_hidden = hidden_states[b]
            batch_patches.append(patch_hidden[valid_mask].cpu().numpy())

        yield np.concatenate(batch_patches, axis=0)  # (n_valid_patches_in_batch, d_model)


# ---------------------------------------------------------------------------
# K-means clustering
# ---------------------------------------------------------------------------

def kmeans_faiss(
    embeddings: np.ndarray,
    n_clusters: int,
    n_iter: int = 100,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    K-means using faiss.Kmeans (GPU-accelerated when available).
    Returns centroids of shape (n_clusters, d_model).
    """
    import faiss

    d = embeddings.shape[1]
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

    kmeans = faiss.Kmeans(
        d=d,
        k=n_clusters,
        niter=n_iter,
        nredo=3,
        gpu=use_gpu,
        verbose=True,
        seed=2021,
    )
    kmeans.train(embeddings)
    return kmeans.centroids  # (n_clusters, d_model)


def kmeans_streaming(
    patch_iter,
    n_clusters: int,
    n_iter: int = 100,
    kmeans_batch_size: int = 10000,
) -> np.ndarray:
    """
    Streaming Mini-batch K-means using sklearn.partial_fit().
    Consumes patch embeddings from a generator without storing all in memory.
    Returns centroids of shape (n_clusters, d_model).
    """
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=kmeans_batch_size,
        max_iter=n_iter,
        random_state=2021,
        verbose=1,
        n_init=3,
    )

    buffer = []
    buffer_size = 0
    total_patches = 0

    for patch_batch in tqdm(patch_iter, desc="Streaming K-means"):
        buffer.append(patch_batch)
        buffer_size += len(patch_batch)
        total_patches += len(patch_batch)

        if buffer_size >= kmeans_batch_size:
            data = np.concatenate(buffer, axis=0)
            km.partial_fit(data)
            buffer = []
            buffer_size = 0

    # flush remaining buffer
    if buffer:
        km.partial_fit(np.concatenate(buffer, axis=0))

    print(f"Streaming K-means done. Total patches processed: {total_patches:,}")
    return km.cluster_centers_.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Create prototype centroids from backbone embeddings")

    # Data source
    parser.add_argument(
        "--data_source",
        type=str,
        default="parquet",
        choices=["pretrain", "pkl", "parquet"],
        help="pretrain: parquet files under --data_path; pkl: existing retrieval_database pkl files; parquet: single retrieval_database parquet file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../datasets/pretrain/pretrain_pairs_ctx512",
        help="[pretrain mode] Directory containing pretrain parquet files",
    )
    parser.add_argument(
        "--pkl_dir",
        type=str,
        default="../retrieval_database/",
        help="[pkl mode] Directory containing *.pkl retrieval database files",
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="../datasets/pretrain/retrieval_database_512.parquet",
        help="[parquet mode] Path to single retrieval_database parquet file",
    )
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument(
        "--pkl_stride",
        type=int,
        default=64,
        help="[pkl mode] Stride for sliding window extraction",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500_000,
        help="Maximum number of embeddings to collect for clustering",
    )

    # Model
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="./checkpoints/base/",
        help="Path to ChronosBolt pretrained model directory",
    )

    # Clustering
    parser.add_argument("--n_clusters", type=int, default=512, help="Number of prototype clusters k")
    parser.add_argument(
        "--kmeans_backend",
        type=str,
        default="sklearn",
        choices=["faiss", "sklearn"],
    )
    parser.add_argument("--n_iter", type=int, default=100, help="K-means iterations")
    parser.add_argument("--kmeans_batch_size", type=int, default=100000, help="[streaming sklearn] partial_fit batch size")

    # Output
    parser.add_argument(
        "--output_path",
        type=str,
        default="./checkpoints/prototypes_k512.npy",
        help="Output path for prototype matrix (.npy); a .pt file is also saved alongside",
    )

    # GPU / dataloader
    parser.add_argument("--gpu_loc", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Step 1: Load backbone (frozen) ---
    print(f"Loading backbone from {args.pretrained_model_path} ...")
    config = AutoConfig.from_pretrained(args.pretrained_model_path)
    model = ChronosBoltModelForForecasting.from_pretrained(
        args.pretrained_model_path, config=config
    )

    backbone_ckpt = os.path.join(args.pretrained_model_path, "autogluon_model.pth")
    if os.path.exists(backbone_ckpt):
        state = torch.load(backbone_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"Loaded backbone weights from {backbone_ckpt}")

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # --- Step 2: Build dataset and dataloader ---
    if args.data_source == "pretrain":
        print(f"Building SimplePretrainDataset from {args.data_path} ...")
        dataset = SimplePretrainDataset(
            dataset_path=args.data_path,
            context_length=args.context_length,
            max_samples=args.max_samples * 2,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    elif args.data_source == "pkl":
        print(f"Building PklWindowDataset from {args.pkl_dir} ...")
        dataset = PklWindowDataset(
            pkl_dir=args.pkl_dir,
            context_length=args.context_length,
            stride=args.pkl_stride,
        )
        # Limit dataset size to max_samples
        if len(dataset) > args.max_samples:
            import torch.utils.data as tud
            indices = torch.randperm(len(dataset))[: args.max_samples].tolist()
            dataset = tud.Subset(dataset, indices)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:  # parquet
        print(f"Building ParquetXDataset from {args.parquet_path} ...")
        dataset = ParquetXDataset(parquet_path=args.parquet_path)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # --- Step 3 + 4: Stream patch embeddings directly into K-means ---
    print(f"Running streaming K-means: k={args.n_clusters}, backend={args.kmeans_backend} ...")
    if args.kmeans_backend == "faiss":
        # faiss.Kmeans requires all data in memory — collect with max_samples limit
        print(f"  (faiss mode) collecting up to {args.max_samples:,} patches ...")
        all_patches = []
        total = 0
        for patch_batch in tqdm(iter_patch_embeddings(model, dataloader, device), desc="Extracting"):
            all_patches.append(patch_batch)
            total += len(patch_batch)
            if total >= args.max_samples:
                break
        embeddings = np.concatenate(all_patches, axis=0)[:args.max_samples]
        print(f"Collected {embeddings.shape[0]:,} patch embeddings, shape: {embeddings.shape}")
        use_gpu = torch.cuda.is_available()
        prototypes = kmeans_faiss(embeddings, args.n_clusters, args.n_iter, use_gpu)
    else:
        # sklearn streaming: no memory limit needed
        patch_iter = iter_patch_embeddings(model, dataloader, device)
        prototypes = kmeans_streaming(patch_iter, args.n_clusters, args.n_iter, args.kmeans_batch_size)

    print(f"Prototypes shape: {prototypes.shape}")

    # --- Step 5: Save ---
    out_path = args.output_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, prototypes)
    torch_path = out_path.replace(".npy", ".pt")
    torch.save(torch.from_numpy(prototypes), torch_path)
    print(f"Saved: {out_path}  and  {torch_path}")

    # Stats
    print(f"Prototype stats: mean={prototypes.mean():.4f}, std={prototypes.std():.4f}")
    dists = np.linalg.norm(prototypes[:, None] - prototypes[None, :], axis=-1)
    np.fill_diagonal(dists, np.nan)
    print(f"Inter-prototype mean L2 distance: {np.nanmean(dists):.4f}")


if __name__ == "__main__":
    main()
