"""Train component RQ-VAE codebooks from frozen Uni2TS Moirai-2 states."""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "models" / "moirai2"))
from uni2ts.common.torch_util import packed_causal_attention_mask
from uni2ts.model.moirai2.module import Moirai2Module
from utils.decomposition import VALID_COMPONENTS, select_component


class Windows(Dataset):
    SPLITS = {"train": (0.0, 0.7), "val": (0.7, 0.8), "test": (0.8, 1.0)}
    def __init__(self, path, split):
        frame = pd.read_parquet(path, columns=["x"])
        values = np.vstack(frame["x"].to_numpy()).astype(np.float32)
        lo, hi = self.SPLITS[split]
        self.values = values[int(lo * len(values)):int(hi * len(values))]
        print(split, self.values.shape)
    def __len__(self): return len(self.values)
    def __getitem__(self, i): return torch.from_numpy(self.values[i])


class RQ(nn.Module):
    def __init__(self, d, n_proto, n_stages, beta):
        super().__init__(); self.beta = beta
        self.codebooks = nn.ModuleList([nn.Embedding(n_proto, d) for _ in range(n_stages)])
        for cb in self.codebooks: nn.init.trunc_normal_(cb.weight, std=0.02)
    def forward(self, z):
        residual, total, commit, qs = z, torch.zeros_like(z), 0., []
        for cb in self.codebooks:
            flat = residual.reshape(-1, residual.shape[-1])
            dist = flat.pow(2).sum(1, keepdim=True) + cb.weight.pow(2).sum(1) - 2 * flat @ cb.weight.T
            q = cb(dist.argmin(-1)).reshape_as(residual)
            commit += F.mse_loss(residual.detach(), q)
            total += residual + (q - residual).detach(); qs.append(q.detach()); residual = residual - q.detach()
        return total, self.beta * commit, qs


class Moirai2Codebook(nn.Module):
    def __init__(self, module, context_length, prediction_length, n_proto, n_stages, beta):
        super().__init__(); self.module = module.eval(); self.context_length = context_length
        self.prediction_length = prediction_length; self.patch_size = module.patch_size
        self.rq = RQ(module.d_model, n_proto, n_stages, beta)
        self.heads = nn.ModuleList([nn.Linear(module.d_model, self.patch_size) for _ in range(n_stages)])
        for p in self.module.parameters(): p.requires_grad = False

    @torch.no_grad()
    def hidden(self, x):
        b = x.shape[0]; past = torch.nan_to_num(x, nan=0.).unsqueeze(-1)
        observed = torch.isfinite(x).unsqueeze(-1)
        future_len = self.prediction_length
        padded_future = ((future_len + self.patch_size - 1) // self.patch_size) * self.patch_size
        total_tokens = self.context_length // self.patch_size + padded_future // self.patch_size
        target = torch.cat([past, torch.zeros((b, padded_future, 1), device=x.device)], dim=1)
        mask = torch.cat([observed, torch.ones((b, padded_future, 1), dtype=torch.bool, device=x.device)], dim=1)
        target = target.reshape(b, total_tokens, self.patch_size)
        mask = mask.reshape(b, total_tokens, self.patch_size)
        sample_id = torch.zeros((b, total_tokens), dtype=torch.long, device=x.device)
        time_id = torch.arange(total_tokens, device=x.device).unsqueeze(0).expand(b, -1)
        var_id = torch.zeros((b, total_tokens), dtype=torch.long, device=x.device)
        pred_mask = torch.zeros((b, total_tokens), dtype=torch.bool, device=x.device)
        pred_mask[:, self.context_length // self.patch_size:] = True
        loc, scale = self.module.scaler(target, mask * ~pred_mask.unsqueeze(-1), sample_id, var_id)
        tokens = self.module.in_proj(torch.cat([(target - loc) / scale, mask.float()], dim=-1))
        hidden = self.module.encoder(tokens, packed_causal_attention_mask(sample_id, time_id),
                                     time_id=time_id, var_id=var_id)
        return hidden[:, :int(np.ceil(self.context_length / self.patch_size))].detach()

    def forward(self, x):
        z = self.hidden(x); length = z.shape[1] * self.patch_size
        target = torch.nan_to_num(x[..., -length:], nan=0.).reshape(x.shape[0], z.shape[1], self.patch_size)
        valid = torch.isfinite(x[..., -length:]).reshape_as(target)
        _, commit, qs = self.rq(z)
        recon = sum(head(q) for head, q in zip(self.heads, qs))
        return ((recon - target).pow(2) * valid).sum() / valid.sum().clamp_min(1) + commit

    @torch.no_grad()
    def save(self, path):
        torch.save({"format_version": 1, "backbone": "moirai-2.0-R-small",
                    "d_model": self.module.d_model, "patch_size": self.patch_size,
                    "n_rq_stages": len(self.rq.codebooks),
                    **{f"codebook_{i}": cb.weight.detach().cpu().float() for i, cb in enumerate(self.rq.codebooks)},
                    "reconstruction_heads": {f"head_{i}": h.state_dict() for i, h in enumerate(self.heads)}}, path)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Salesforce/moirai-2.0-R-small")
    p.add_argument("--parquet_path", default="./datasets/pretrain/retrieval_database_512.parquet")
    p.add_argument("--output_dir", default="./checkpoints/moirai2_component_rqvae")
    p.add_argument("--components", default="trend seasonal residual")
    p.add_argument("--period", type=int, default=24); p.add_argument("--trend_kernel", type=int, default=25)
    p.add_argument("--context_length", type=int, default=512); p.add_argument("--prediction_length", type=int, default=1)
    p.add_argument("--n_proto", type=int, default=256); p.add_argument("--n_rq_stages", type=int, default=2)
    p.add_argument("--beta", type=float, default=.25); p.add_argument("--train_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32); p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4); p.add_argument("--weight_decay", type=float, default=.01)
    p.add_argument("--grad_clip", type=float, default=1.); p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max_train_batches", type=int, default=-1); p.add_argument("--max_val_batches", type=int, default=-1)
    p.add_argument("--gpu_loc", type=int, default=0); p.add_argument("--seed", type=int, default=2021)
    return p.parse_args()


def main():
    a = get_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(f"cuda:{a.gpu_loc}" if torch.cuda.is_available() else "cpu")
    components = [x for x in a.components.replace(",", " ").split() if x]
    if set(components) - set(VALID_COMPONENTS): raise ValueError(components)
    print(f"Loading {a.model_id}; Hugging Face weights download on first run")
    module = Moirai2Module.from_pretrained(a.model_id).to(device).eval()
    print(f"Moirai-2: d_model={module.d_model}, patch_size={module.patch_size}, device={device}")
    loaders = {s: DataLoader(Windows(a.parquet_path, s), batch_size=a.batch_size,
        shuffle=s == "train", num_workers=a.num_workers, pin_memory=True, drop_last=True) for s in ("train", "val")}
    os.makedirs(a.output_dir, exist_ok=True)
    for component in components:
        model = Moirai2Codebook(module, a.context_length, a.prediction_length, a.n_proto, a.n_rq_stages, a.beta).to(device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.learning_rate, weight_decay=a.weight_decay)
        best, best_val, bad = None, float("inf"), 0
        for epoch in range(1, a.train_epochs + 1):
            model.train(); total = 0.; count = 0; start = time.time()
            for i, batch in enumerate(tqdm(loaders["train"], desc=f"{component} {epoch}/{a.train_epochs}")):
                if a.max_train_batches > 0 and i >= a.max_train_batches: break
                x = select_component(batch.float().to(device), component, period=a.period, trend_kernel=a.trend_kernel)
                opt.zero_grad(set_to_none=True); loss = model(x); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip); opt.step()
                total += loss.item(); count += 1
            model.eval(); val = 0.; vn = 0
            with torch.no_grad():
                for i, batch in enumerate(loaders["val"]):
                    if a.max_val_batches > 0 and i >= a.max_val_batches: break
                    val += model(select_component(batch.float().to(device), component, period=a.period, trend_kernel=a.trend_kernel)).item(); vn += 1
            val /= max(vn, 1); print(f"{component} epoch={epoch} train={total/max(count,1):.6f} val={val:.6f} time={time.time()-start:.1f}s")
            if val < best_val: best_val = val; best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; bad = 0
            else: bad += 1
            if bad >= a.patience: break
        model.load_state_dict(best); out = os.path.join(a.output_dir, f"{component}_prototypes_rqvae.pt"); model.save(out)
        with open(os.path.join(a.output_dir, f"{component}_prototypes_rqvae.meta.json"), "w") as f:
            json.dump({"component": component, "backbone": "moirai-2.0-R-small", "model_id": a.model_id,
                       "parquet_path": a.parquet_path, "period": a.period, "trend_kernel": a.trend_kernel,
                       "context_length": a.context_length, "patch_size": module.patch_size, "d_model": module.d_model,
                       "n_proto": a.n_proto, "n_rq_stages": a.n_rq_stages, "best_val_loss": best_val}, f, indent=2)
        print(f"saved {out}")


if __name__ == "__main__": main()
