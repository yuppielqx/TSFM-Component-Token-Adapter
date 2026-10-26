"""Minimal pipeline helpers; optional S3 imports are loaded only on demand."""

import os
import re
from pathlib import Path
from typing import List

import torch


def left_pad_and_stack_1D(tensors: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(len(c) for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 1
        padding = torch.full(size=(max_len - len(c),), fill_value=torch.nan, device=c.device)
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)


def cache_model_from_s3(s3_uri: str, force=False):
    from autogluon.common.loaders.load_s3 import download, list_bucket_prefix_suffix_contains_s3

    if re.match("^s3://([^/]+)/(.*?([^/]+)/?)$", s3_uri) is None:
        raise ValueError(f"Not a valid S3 URI: {s3_uri}")

    # we expect the prefix to point to a "directory" on S3
    if not s3_uri.endswith("/"):
        s3_uri += "/"

    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    bucket, prefix = s3_uri.replace("s3://", "").split("/", 1)
    bucket_cache_path = cache_home / "autogluon" / "timeseries" / bucket

    for obj_path in list_bucket_prefix_suffix_contains_s3(bucket=bucket, prefix=prefix):
        destination_path = bucket_cache_path / obj_path
        if not force and destination_path.exists():
            continue
        download(bucket, obj_path, local_path=str(destination_path))

    return str(bucket_cache_path / prefix)
