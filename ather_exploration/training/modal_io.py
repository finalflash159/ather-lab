"""Explicit bank upload and atomic local mirrors. No training or image build here."""

import argparse
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from ather_exploration.worlds.scenarios import read_record, write_record

VOLUME_NAME = "ather-exploration"


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise ValueError("ID must be 1..96 letters/digits/underscore/hyphen; no paths")
    return value


def volume_handle():
    import modal

    return modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def upload_banks(config, volume):
    from ather_exploration.training.curriculum import WorldBank
    from ather_exploration.training.skill_runner import preflight
    from ather_exploration.worlds.scenarios import implementation_id

    identities = preflight(config) if config.skills.enabled else WorldBank(config.banks).identities
    paths = {} if config.skills.enabled and config.skills.stop_after != "P5" else config.banks
    dataset = f"bank-{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for group, path in paths.items():
            source = Path(path)
            if any(p.is_symlink() for p in source.rglob("*")):
                raise ValueError("Bank upload does not accept symlinks")
            shutil.copytree(source, root / group)
        if config.skills.enabled:
            from ather_exploration.worlds.skill_tasks import build_skill_suite

            build_skill_suite(config, root / "skills")
        write_record(
            root / "dataset.json",
            {
                "dataset": dataset,
                "bank_ids": identities,
                "source_revision": implementation_id(),
            },
        )
        with volume.batch_upload() as batch:
            batch.put_directory(root, f"/banks/{dataset}")
    return dataset


def fetch(volume, remote, target):
    """Download to a sibling temporary file, then replace (never expose partial bytes)."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".download-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            for chunk in volume.read_file(remote):
                stream.write(chunk)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def download_run(volume, run_id, output):
    from ather_exploration.training.checkpoints import FILES, inspect_checkpoint

    run_id = identifier(run_id)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "mirror.json"
    identity = {"volume": VOLUME_NAME, "run_id": run_id}
    if marker.exists():
        if read_record(marker) != identity:
            raise ValueError("Output belongs to another remote run")
    elif any(output.iterdir()):
        raise ValueError("Choose an empty output directory for the remote mirror")
    else:
        write_record(marker, identity)
    remote_root = f"/runs/{run_id}"
    # Volume API reads committed files, without a mounted reader/reload race.
    available = {entry.path.lstrip("/") for entry in volume.iterdir(remote_root)}
    with tempfile.TemporaryDirectory(prefix=".sync-", dir=output) as staging:
        staging = Path(staging)
        # Snapshot all published checkpoints, not only the latest pointer.
        prefix = f"runs/{run_id}/checkpoints/"
        relatives = sorted(
            {
                path.removeprefix(f"runs/{run_id}/").removesuffix("/READY")
                for path in available
                if path.startswith(prefix)
                and re.fullmatch(r"step_[0-9]+/READY", path[len(prefix) :])
            },
            key=lambda path: int(path.rsplit("_", 1)[1]),
        )
        if not relatives:
            raise ValueError("No published READY checkpoints for this run")
        for relative in relatives:
            destination = output / relative
            if not destination.exists():
                candidate = staging / "checkpoint"
                for name in sorted(FILES | {"checksums.json", "READY"}):
                    fetch(volume, f"{remote_root}/{relative}/{name}", candidate / name)
                inspect_checkpoint(candidate, inference=True)
                destination.parent.mkdir(exist_ok=True)
                candidate.rename(destination)
            else:
                inspect_checkpoint(destination, inference=True)
        latest = f"runs/{run_id}/latest.json"
        if latest in available:
            fetch(volume, f"/{latest}", staging / "latest.json")
            relative = read_record(staging / "latest.json")["checkpoint"]
            if relative not in relatives:
                raise ValueError(
                    "Remote checkpoint pointer not in downloaded snapshot; retry download"
                )
            write_record(output / "latest.json", {"checkpoint": relative}, replace=True)
        for name in (
            "manifest.json",
            "run_status.json",
            "remote.json",
            "tracking.json",
            "transfer.json",
            "best.json",
            "progress.jsonl",
            "train_episodes.jsonl",
            "resume_events.jsonl",
            "skill_evaluations.jsonl",
            "phase_transitions.jsonl",
        ):
            if f"runs/{run_id}/{name}" not in available:
                continue
            target = staging / name
            fetch(volume, f"{remote_root}/{name}", target)
            if name.endswith(".jsonl"):
                data = target.read_bytes()
                # An actively appended JSONL may end with an incomplete row.
                end = data.rfind(b"\n") + 1
                target.write_bytes(data[:end])
                for line in data[:end].splitlines():
                    json.loads(line)
            else:
                read_record(target)
            os.replace(target, output / name)
    return {
        "run_id": run_id,
        "output": str(output),
        "downloaded_at": time.time(),
        "checkpoints": len(relatives),
    }


def publish_run(source, destination):
    """Worker-local POSIX output -> Volume; caller owns a unique destination directory.

    Volume lacks hard links. Stage immutable checkpoint directories and publish the
    pointer last; no writer opens an in-progress checkpoint after publication.
    """
    source, destination = Path(source), Path(destination)
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for checkpoint in sorted((source / "checkpoints").glob("step_*")):
        target = destination / "checkpoints" / checkpoint.name
        if not (checkpoint / "READY").is_file() or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".pending-{uuid.uuid4().hex}"
        try:
            shutil.copytree(checkpoint, temporary)
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    for file in source.rglob("*"):
        relative = file.relative_to(source)
        if (
            not file.is_file()
            or file.is_symlink()
            or relative.parts[0] == "checkpoints"
            or file.name == "latest.json"
            or any(p.startswith(".pending-") for p in relative.parts)
        ):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".pending-{uuid.uuid4().hex}")
        try:
            shutil.copyfile(file, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    if (source / "latest.json").exists():
        pointer = read_record(source / "latest.json")
        relative = pointer["checkpoint"]
        if not re.fullmatch(r"checkpoints/step_[0-9]+", relative):
            raise ValueError("Invalid checkpoint pointer")
        if not (destination / relative / "READY").exists():
            raise ValueError("Cannot publish pointer before checkpoint")
        write_record(destination / "latest.json", pointer, replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    upload = sub.add_parser("upload")
    upload.add_argument("--config", required=True, type=Path)
    sync = sub.add_parser("download", help="Download all READY checkpoints and run logs")
    sync.add_argument("--run-id", required=True)
    sync.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "upload":
        from ather_exploration.training.config import read_training_config

        print(
            json.dumps(
                {"dataset": upload_banks(read_training_config(args.config), volume_handle())}
            )
        )
    else:
        identifier(args.run_id)
        print(json.dumps(download_run(volume_handle(), args.run_id, args.output)))


if __name__ == "__main__":
    main()
