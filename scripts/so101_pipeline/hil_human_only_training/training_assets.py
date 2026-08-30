"""Shared asset-bundle definition for upload and training-time download."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPO_ID = "Jill111/AHA-WAM-SO101-HIL-training-assets"
REPO_TYPE = "dataset"

ASSET_FILES = (
    "checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt",
    "runs/so101_ahawam_plug_all/assets/dataset_stats.json",
)
ASSET_DIRECTORIES = (
    "data/so101_ahawam_plug/02_grasp_move_and_align",
    "data/so101_ahawam_plug/04_full_remove_and_insert",
    "data/ahawam_hil_raw",
    "data/so101_ahawam_square_v1_text_embeds_cache",
)

# Fingerprint of the bundle audited on 2026-08-30. This is a cheap offline
# fast path; the online sync writes a detailed per-file state after download.
AUDITED_FILE_COUNT = 696
AUDITED_TOTAL_BYTES = 16_331_861_500


@dataclass(frozen=True)
class ArchiveAsset:
    filename: str
    size: int
    sha256: str
    allowed_prefixes: tuple[str, ...]
    selected_file_count: int
    selected_total_bytes: int


# Compact Hub representation. The SO101 archive also contains task 01, which
# is intentionally ignored during extraction; only the audited 02+04 prefixes
# participate in this training bundle.
ARCHIVE_ASSETS = (
    ArchiveAsset(
        filename="so101_ahawam_plug.zip",
        size=965_095_995,
        sha256="83f48092b0db647435af7d9e0a1295d12a2d8c04e183a34c18318dfe485cbed0",
        allowed_prefixes=(
            "data/so101_ahawam_plug/02_grasp_move_and_align/",
            "data/so101_ahawam_plug/04_full_remove_and_insert/",
        ),
        selected_file_count=539,
        selected_total_bytes=826_712_473,
    ),
    ArchiveAsset(
        filename="ahawam_hil_raw.zip",
        size=797_366_154,
        sha256="c024b22b41a9346fe2c729e91e85f3155f83e847f3729d7f1029b73a2096bfcd",
        allowed_prefixes=("data/ahawam_hil_raw/",),
        selected_file_count=152,
        selected_total_bytes=802_697_695,
    ),
)

# Files stored directly on the Hub. Dataset payloads are represented by the
# two archives above and expanded locally by prepare_training_assets.py.
REMOTE_DIRECT_FILES = {
    "checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt": 14_699_235_086,
    "runs/so101_ahawam_plug_all/assets/dataset_stats.json": 60_639,
    (
        "data/so101_ahawam_square_v1_text_embeds_cache/"
        "0fa14c2feb74757091a46e23875866d7d97ba9c3213913ba14c6789a64c5933a."
        "t5_len128.wan22ti2v5b.pt"
    ): 1_051_869,
    (
        "data/so101_ahawam_square_v1_text_embeds_cache/"
        "60a776a4fbcef64e61ea7dce8f040dbeddcc83b40a5309d1dde93ded5eee3d12."
        "t5_len128.wan22ti2v5b.pt"
    ): 1_051_869,
    (
        "data/so101_ahawam_square_v1_text_embeds_cache/"
        "d4556692b31e502646b5af27d7d9d2e8b1f52d69b4772c1b014b325713d6eb76."
        "t5_len128.wan22ti2v5b.pt"
    ): 1_051_869,
}


def configured_repo_id() -> str:
    return os.environ.get("AHAWAM_TRAINING_ASSETS_REPO", DEFAULT_REPO_ID).strip()


def is_asset_path(path: str) -> bool:
    normalized = str(path).strip("/")
    if normalized in ASSET_FILES:
        return True
    return any(
        normalized == prefix or normalized.startswith(prefix + "/")
        for prefix in ASSET_DIRECTORIES
    )


def iter_local_asset_files(root: Path = REPO_ROOT) -> Iterable[tuple[str, Path]]:
    for relative in ASSET_FILES:
        path = root / relative
        if path.is_file():
            yield relative, path
    for relative in ASSET_DIRECTORIES:
        directory = root / relative
        if not directory.is_dir():
            continue
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            yield path.relative_to(root).as_posix(), path


def local_bundle_fingerprint(root: Path = REPO_ROOT) -> tuple[int, int]:
    count = 0
    total_bytes = 0
    for _, path in iter_local_asset_files(root):
        count += 1
        total_bytes += path.stat().st_size
    return count, total_bytes


def assert_audited_local_bundle(root: Path = REPO_ROOT) -> list[tuple[str, Path]]:
    files = list(iter_local_asset_files(root))
    count = len(files)
    total_bytes = sum(path.stat().st_size for _, path in files)
    if count != AUDITED_FILE_COUNT or total_bytes != AUDITED_TOTAL_BYTES:
        raise RuntimeError(
            "Local training asset bundle differs from the audited upload set: "
            f"files={count}/{AUDITED_FILE_COUNT}, "
            f"bytes={total_bytes}/{AUDITED_TOTAL_BYTES}. "
            "Do not upload an accidental partial or expanded dataset; audit the "
            "difference and update training_assets.py intentionally."
        )
    return files
