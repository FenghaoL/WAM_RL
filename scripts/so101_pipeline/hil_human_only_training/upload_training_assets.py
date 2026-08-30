#!/usr/bin/env python3
"""Upload the compact ZIP-based AHA-WAM training asset bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

from scripts.so101_pipeline.hil_human_only_training.training_assets import (
    ARCHIVE_ASSETS,
    REMOTE_DIRECT_FILES,
    REPO_ROOT,
    REPO_TYPE,
    assert_audited_local_bundle,
    configured_repo_id,
)


HERE = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remote_sizes(api: HfApi, repo_id: str, token: str | bool) -> dict[str, int]:
    return {
        entry.path: int(entry.size)
        for entry in api.list_repo_tree(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            recursive=True,
            expand=True,
            token=token,
        )
        if getattr(entry, "size", None) is not None
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=configured_repo_id())
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository instead of the public default.",
    )
    args = parser.parse_args()

    assert_audited_local_bundle()
    for archive in ARCHIVE_ASSETS:
        path = REPO_ROOT / archive.filename
        if not path.is_file() or path.stat().st_size != archive.size:
            raise RuntimeError(
                f"Missing or wrong-size archive: {path} ({archive.size} bytes expected)"
            )
        actual_hash = _sha256(path)
        if actual_hash != archive.sha256:
            raise RuntimeError(
                f"Archive SHA-256 mismatch for {archive.filename}: {actual_hash}"
            )

    token = os.environ.get("HF_TOKEN") or True
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=REPO_TYPE,
        private=args.private,
        exist_ok=True,
    )
    info = api.repo_info(repo_id=args.repo_id, repo_type=REPO_TYPE)
    if not args.private and info.private:
        print(
            "[upload] warning: existing repository is private; switch it to "
            "public in Hugging Face Settings after verification.",
            flush=True,
        )

    api.upload_file(
        repo_id=args.repo_id,
        repo_type=REPO_TYPE,
        path_or_fileobj=HERE / "HF_ASSETS_REPO_CARD.md",
        path_in_repo="README.md",
        commit_message="Document compact training assets",
    )
    remote = _remote_sizes(api, args.repo_id, token)
    expected = dict(REMOTE_DIRECT_FILES)
    expected.update({archive.filename: archive.size for archive in ARCHIVE_ASSETS})

    for relative, size in expected.items():
        if remote.get(relative) == size:
            print(f"[upload] skip verified remote file: {relative}", flush=True)
            continue
        path = REPO_ROOT / relative
        print(f"[upload] file: {relative} bytes={size}", flush=True)
        api.upload_file(
            repo_id=args.repo_id,
            repo_type=REPO_TYPE,
            path_or_fileobj=path,
            path_in_repo=relative,
            commit_message=f"Upload {relative}",
        )

    remote = _remote_sizes(api, args.repo_id, token)
    missing = sorted(path for path in expected if path not in remote)
    wrong_size = sorted(
        path for path, size in expected.items() if remote.get(path) not in (None, size)
    )
    if missing or wrong_size:
        raise RuntimeError(
            f"Remote verification failed: missing={missing}, wrong_size={wrong_size}"
        )

    inventory = json.dumps(
        {
            "repo_id": args.repo_id,
            "representation": "two ZIP archives plus direct runtime assets",
            "private": bool(info.private),
            "files": dict(sorted(expected.items())),
            "total_files": len(expected),
            "total_bytes": sum(expected.values()),
            "archives": {
                archive.filename: {
                    "sha256": archive.sha256,
                    "selected_file_count": archive.selected_file_count,
                    "selected_total_bytes": archive.selected_total_bytes,
                    "allowed_prefixes": list(archive.allowed_prefixes),
                }
                for archive in ARCHIVE_ASSETS
            },
        },
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    api.upload_file(
        repo_id=args.repo_id,
        repo_type=REPO_TYPE,
        path_or_fileobj=inventory,
        path_in_repo="assets_inventory.json",
        commit_message="Record verified compact asset inventory",
    )
    print(
        f"[upload] verified: https://huggingface.co/datasets/{args.repo_id} "
        f"files={len(expected)} bytes={sum(expected.values())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
