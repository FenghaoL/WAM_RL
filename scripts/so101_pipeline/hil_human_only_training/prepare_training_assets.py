#!/usr/bin/env python3
"""Restore the Hugging Face training bundle to repository paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from huggingface_hub import HfApi, hf_hub_download

from scripts.so101_pipeline.hil_human_only_training.training_assets import (
    ARCHIVE_ASSETS,
    AUDITED_FILE_COUNT,
    AUDITED_TOTAL_BYTES,
    REMOTE_DIRECT_FILES,
    REPO_ROOT,
    REPO_TYPE,
    ArchiveAsset,
    configured_repo_id,
    iter_local_asset_files,
    local_bundle_fingerprint,
)


STATE_PATH = REPO_ROOT / ".cache/huggingface/ahawam_training_assets_state.json"
ARCHIVE_CACHE_DIR = REPO_ROOT / ".cache/huggingface/ahawam_training_archives"


def _state_is_complete(repo_id: str, revision: str) -> bool:
    if STATE_PATH.is_file():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if state.get("repo_id") == repo_id and state.get("revision") == revision:
                files = state.get("files", {})
                if files and all(
                    (REPO_ROOT / relative).is_file()
                    and (REPO_ROOT / relative).stat().st_size == int(size)
                    for relative, size in files.items()
                ):
                    print(
                        f"[assets] verified local manifest: repo={repo_id} "
                        f"files={len(files)}"
                    )
                    return True
        except (OSError, ValueError, TypeError):
            pass

    count, total_bytes = local_bundle_fingerprint()
    if count == AUDITED_FILE_COUNT and total_bytes == AUDITED_TOTAL_BYTES:
        print(
            "[assets] audited bundle already exists at default paths: "
            f"files={count} bytes={total_bytes}"
        )
        return True
    return False


def _remote_files(
    api: HfApi,
    repo_id: str,
    revision: str,
    token: str | bool,
    required_archives: tuple[ArchiveAsset, ...] = ARCHIVE_ASSETS,
) -> dict[str, int]:
    entries = api.list_repo_tree(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        revision=revision,
        recursive=True,
        expand=True,
        token=token,
    )
    remote = {
        entry.path: int(entry.size)
        for entry in entries
        if getattr(entry, "size", None) is not None
    }
    expected = dict(REMOTE_DIRECT_FILES)
    expected.update({archive.filename: archive.size for archive in required_archives})
    missing = sorted(path for path in expected if path not in remote)
    wrong_size = sorted(
        path for path, size in expected.items() if remote.get(path) not in (None, size)
    )
    if missing or wrong_size:
        raise RuntimeError(
            "Remote archive bundle is incomplete: "
            f"missing={missing}, wrong_size={wrong_size}"
        )
    return expected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_archive(archive: ArchiveAsset) -> Path | None:
    """Return a manually placed archive at the repository root, if it looks intact.

    Operators who cannot reach the Hub upload path may copy the two ZIPs into
    the clone root out of band; the SHA-256 check in _archive_path still gates
    the contents before anything is extracted.
    """
    path = REPO_ROOT / archive.filename
    if path.is_file() and path.stat().st_size == archive.size:
        return path
    return None


def _archive_path(
    archive: ArchiveAsset,
    repo_id: str,
    revision: str,
    token: str | bool,
) -> Path:
    local_source = _local_archive(archive)
    if local_source is not None:
        path = local_source
    else:
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=archive.filename,
                repo_type=REPO_TYPE,
                revision=revision,
                local_dir=ARCHIVE_CACHE_DIR,
                token=token,
            )
        )
    actual_hash = _sha256(path)
    if actual_hash != archive.sha256:
        hint = (
            "The archive was copied in manually; transfer it again."
            if local_source is not None
            else "Delete the cached download and rerun."
        )
        raise RuntimeError(
            f"Archive SHA-256 mismatch for {archive.filename}: "
            f"{actual_hash} != {archive.sha256}. {hint}"
        )
    return path


def _selected_members(zip_file: ZipFile, archive: ArchiveAsset):
    selected = []
    seen: set[str] = set()
    for info in zip_file.infolist():
        if info.is_dir() or not info.filename.startswith(archive.allowed_prefixes):
            continue
        relative = PurePosixPath(info.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe archive member: {info.filename}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise RuntimeError(f"Archive symlink is not allowed: {info.filename}")
        normalized = relative.as_posix()
        if normalized in seen:
            raise RuntimeError(f"Duplicate archive member: {normalized}")
        seen.add(normalized)
        selected.append((info, relative))
    count = len(selected)
    total_bytes = sum(info.file_size for info, _ in selected)
    if (
        count != archive.selected_file_count
        or total_bytes != archive.selected_total_bytes
    ):
        raise RuntimeError(
            f"Unexpected contents in {archive.filename}: "
            f"files={count}/{archive.selected_file_count}, "
            f"bytes={total_bytes}/{archive.selected_total_bytes}"
        )
    return selected


def _extract_archive(
    path: Path, archive: ArchiveAsset, destination_root: Path = REPO_ROOT
) -> None:
    try:
        with ZipFile(path) as zip_file:
            selected = _selected_members(zip_file, archive)
            print(
                f"[assets] extracting {archive.filename}: "
                f"files={len(selected)} bytes={archive.selected_total_bytes}",
                flush=True,
            )
            for index, (info, relative) in enumerate(selected, start=1):
                target = destination_root.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(
                    f".{target.name}.ahawam-extract-{os.getpid()}"
                )
                try:
                    with zip_file.open(info, "r") as source, temporary.open("wb") as out:
                        shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
                    if temporary.stat().st_size != info.file_size:
                        raise RuntimeError(
                            f"Extracted size mismatch for {relative.as_posix()}"
                        )
                    os.replace(temporary, target)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                if index % 100 == 0 or index == len(selected):
                    print(
                        f"[assets] extracted {archive.filename}: "
                        f"{index}/{len(selected)} files",
                        flush=True,
                    )
    except BadZipFile as exc:
        raise RuntimeError(f"Invalid ZIP archive: {path}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=configured_repo_id())
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--force-online-check", action="store_true")
    args = parser.parse_args()

    if not args.force_online_check and _state_is_complete(args.repo_id, args.revision):
        return
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")

    # The canonical asset repository is public. Do not require a saved token
    # on training workers. HF_TOKEN remains supported for a private mirror or
    # a temporary visibility transition.
    token: str | bool = os.environ.get("HF_TOKEN") or False

    # Archives already sitting in the clone root are not required on the Hub.
    local_archives = tuple(a for a in ARCHIVE_ASSETS if _local_archive(a) is not None)
    for archive in local_archives:
        print(f"[assets] using local archive {REPO_ROOT / archive.filename}")
    hub_archives = tuple(a for a in ARCHIVE_ASSETS if a not in local_archives)

    try:
        remote = _remote_files(
            HfApi(token=token), args.repo_id, args.revision, token, hub_archives
        )
    except Exception as exc:
        raise RuntimeError(
            "Cannot read the AHA-WAM training asset repository. Verify that "
            f"the public dataset https://huggingface.co/datasets/{args.repo_id} "
            "exists and is complete. Archives that the Hub does not serve "
            f"({', '.join(a.filename for a in ARCHIVE_ASSETS)}) may instead be "
            f"copied into {REPO_ROOT} by hand. If you intentionally use a "
            "private mirror, export a read-only HF_TOKEN securely."
        ) from exc

    pending = [
        relative
        for relative, size in REMOTE_DIRECT_FILES.items()
        if not (REPO_ROOT / relative).is_file()
        or (REPO_ROOT / relative).stat().st_size != size
    ]
    print(
        f"[assets] repo={args.repo_id} required_remote_files={len(remote)} "
        f"direct_download_files={len(pending)}"
    )

    def download(relative: str) -> str:
        hf_hub_download(
            repo_id=args.repo_id,
            filename=relative,
            repo_type=REPO_TYPE,
            revision=args.revision,
            local_dir=REPO_ROOT,
            token=token,
            force_download=(REPO_ROOT / relative).exists(),
        )
        actual = (REPO_ROOT / relative).stat().st_size
        if actual != REMOTE_DIRECT_FILES[relative]:
            raise RuntimeError(
                f"Downloaded size mismatch for {relative}: "
                f"{actual} != {REMOTE_DIRECT_FILES[relative]}"
            )
        return relative

    if pending:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(download, relative): relative for relative in pending}
            completed = 0
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 25 == 0 or completed == len(pending):
                    print(f"[assets] downloaded {completed}/{len(pending)} files")

    count, total_bytes = local_bundle_fingerprint()
    if count != AUDITED_FILE_COUNT or total_bytes != AUDITED_TOTAL_BYTES:
        for archive in ARCHIVE_ASSETS:
            path = _archive_path(archive, args.repo_id, args.revision, token)
            _extract_archive(path, archive)

    count, total_bytes = local_bundle_fingerprint()
    if count != AUDITED_FILE_COUNT or total_bytes != AUDITED_TOTAL_BYTES:
        raise RuntimeError(
            "Expanded local training bundle failed final validation: "
            f"files={count}/{AUDITED_FILE_COUNT}, "
            f"bytes={total_bytes}/{AUDITED_TOTAL_BYTES}"
        )

    local_manifest = {
        relative: path.stat().st_size for relative, path in iter_local_asset_files()
    }

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "revision": args.revision,
                "files": dict(sorted(local_manifest.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[assets] ready at {REPO_ROOT}: files={count} bytes={total_bytes}"
    )


if __name__ == "__main__":
    main()
