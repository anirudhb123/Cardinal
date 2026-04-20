#!/usr/bin/env python3
"""Upload rl_grpo_output/checkpoint-* to Hugging Face Hub (model weights + tokenizer, no optimizer)."""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, get_token


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--folder",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "rl_grpo_output" / "checkpoint-12",
        help="Local checkpoint directory",
    )
    p.add_argument(
        "--repo-id",
        default=os.environ.get("HF_REPO_ID", "abharadwaj123/sqlstorm-grpo-checkpoint"),
        help="HF model repo id (namespace/name)",
    )
    p.add_argument("--private", action="store_true", help="Create private repo")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit(
            "Set HF_TOKEN (https://huggingface.co/settings/tokens), e.g.\n"
            "  export HF_TOKEN=hf_...\n"
            "  export HF_REPO_ID=yourname/sqlstorm-grpo-checkpoint  # optional\n"
            f"  python {__file__}"
        )

    if not args.folder.is_dir():
        raise SystemExit(f"Missing folder: {args.folder}")

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=str(args.folder),
        path_in_repo=".",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="GRPO checkpoint: model.safetensors + tokenizer (no optimizer)",
    )
    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
