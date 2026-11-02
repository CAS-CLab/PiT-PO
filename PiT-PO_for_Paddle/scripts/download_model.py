#!/usr/bin/env python3
"""Download a Hugging Face checkpoint for PaddleFormers weight conversion."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        local_dir=str(Path(args.output).resolve()),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE", "README.md"],
        max_workers=4,
    )
    print(path)


if __name__ == "__main__":
    main()
