"""Download the public Chronos-Bolt-base checkpoint for local loading."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="amazon/chronos-bolt-base")
    parser.add_argument("--revision", default="main", help="Use a commit SHA to pin an experiment.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/base"))
    args = parser.parse_args()
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(args.output_dir),
        allow_patterns=["config.json", "model.safetensors"],
    )
    for name in ("config.json", "model.safetensors"):
        if not (args.output_dir / name).is_file():
            raise FileNotFoundError(args.output_dir / name)
    config = json.loads((args.output_dir / "config.json").read_text())
    if "chronos_config" not in config:
        raise ValueError("The downloaded config is not a Chronos checkpoint")
    print(f"Backbone ready at {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
