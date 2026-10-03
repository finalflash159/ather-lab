"""Immutable update-boundary artifacts; inference and resume have distinct contracts."""

import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from ather_exploration.agents.learning import LearnedAgent, algorithm, schema_signature
from ather_exploration.worlds.scenarios import implementation_id, read_record

FILES = {"model.zip", "metadata.json", "runner_state.json", "rng_state.pt"}

# Audited active-phase-v1 P1 implementation; no arbitrary cross-source resume.
P1_TRANSFER_SOURCE = "3aa079a570354a8e2336bd7dfadb64bb45b202ec38fc2360f9aee73304fdbaf2"


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if (path / "latest.json").exists():
        relative = read_record(path / "latest.json")["checkpoint"]
        candidate = (path / relative).resolve()
        if not candidate.is_relative_to(path):
            raise ValueError("Checkpoint pointer escapes run directory")
        path = candidate
    return path


# UI accounting-only compatibility; environment/policy/training semantics unchanged.
VIEWER_COMPATIBLE_SOURCES = {"4859629ea0a94f3a1e17e1a331283f799a325a6a6889d85543e4b72c934ae1f3"}


def inspect_checkpoint(
    path,
    *,
    transfer_p1_to_p2=False,
    transfer_p2_to_p3=False,
    lr_trial=False,
    inference=False,
    unfinished_trial=False,
    p3_resume=False,
    recovery=False,
):
    path = resolve_checkpoint(path)
    if not (path / "READY").is_file():
        raise ValueError("Checkpoint is not READY")
    hashes = json.loads((path / "checksums.json").read_text())
    if set(hashes) != FILES:
        raise ValueError("Checkpoint file manifest mismatch")
    for name, expected in hashes.items():
        file = path / name
        if file.is_symlink() or not file.is_file() or _hash(file) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("artifact_schema") != "g4-checkpoint-v1":
        raise ValueError("Unsupported checkpoint schema")
    trial_source = "bed19cee2bd2ad77385fc5c8f72250b35b6efe3344c78c992d288764f9503e05"
    if meta.get("source_revision") != implementation_id() and (
        not (
            (lr_trial or inference)
            and meta.get("source_revision")
            in (trial_source, "9cba25c4713de6ca8f212b7b59e051b8e430371ed4f79b1a8ac9ce294d90d259")
        )
        and not (
            (unfinished_trial or inference)
            and meta.get("source_revision")
            == "b1caee3c156a4723864d0a465629765ac26a000ea7babc345404b1ffaf7a75be"
        )
        and not (
            p3_resume
            and meta.get("source_revision")
            == "b1caee3c156a4723864d0a465629765ac26a000ea7babc345404b1ffaf7a75be"
        )
        and not (
            (recovery or inference)
            and meta.get("source_revision")
            == "28e2dff54a78e80dd1b2f53ae1de6f667b1ddc8ef69aec554b5e9c0461e73e8b"
        )
        and not (transfer_p1_to_p2 and meta.get("source_revision") == P1_TRANSFER_SOURCE)
        and not (
            (inference or transfer_p2_to_p3)
            and meta.get("source_revision")
            in (VIEWER_COMPATIBLE_SOURCES | ({P1_TRANSFER_SOURCE} if inference else set()))
        )
    ):
        raise ValueError("Checkpoint source revision differs from current code")
    if transfer_p1_to_p2:
        state = json.loads((path / "runner_state.json").read_text())
        if (
            meta.get("curriculum_protocol")
            not in ("active-phase-v1", "active-phase-v2", "active-phase-v3")
            or meta.get("viewer_task") != "P1b"
            or meta.get("config", {}).get("skills", {}).get("stop_after") != "P1"
            or state.get("state") != "PHASE_COMPLETED"
            or state.get("skill_controller", {}).get("index") != 2
            or state.get("skill_controller", {}).get("failed")
        ):
            raise ValueError("P1 transfer requires a completed P1 checkpoint ready for P2a")
    if transfer_p2_to_p3:
        state = json.loads((path / "runner_state.json").read_text())
        controller = state.get("skill_controller", {})
        history = controller.get("history", [])
        if (
            meta.get("curriculum_protocol") not in ("active-phase-v2", "active-phase-v3")
            or meta.get("viewer_task") != "P2c"
            or meta.get("config", {}).get("skills", {}).get("stop_after") != "P2"
            or state.get("state") != "PHASE_COMPLETED"
            or controller.get("index") != 5
            or controller.get("failed")
            or len(history) < 2
            or any(h.get("task") != "P2c" or not h.get("passed") for h in history[-2:])
            or not history[-1].get("eligible")
            or history[-1].get("steps") != meta.get("env_steps")
            or history[-1].get("streak", 0) < 2
        ):
            raise ValueError("P2 transfer requires completed P2c with two passing gates")
    algorithm(meta["method"])
    return path, meta


def save_checkpoint(model, directory, config, runner_state, bank_ids):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=directory.parent))
    try:
        model.save(temporary / "model.zip")
        meta = {
            "artifact_schema": "g4-checkpoint-v1",
            "method": config.method,
            "learning_objective": "ppo_public_route_aux" if config.recovery else config.method,
            "env_steps": model.num_timesteps,
            "source_revision": implementation_id(),
            "schema": schema_signature(model.observation_space),
            "config": config.model_dump(mode="json"),
            "bank_ids": bank_ids,
            "environment_configs": runner_state.get("environment_configs", {}),
            "parameters": sum(p.numel() for p in model.policy.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.policy.parameters() if p.requires_grad
            ),
            "initialization": "SB3 ortho_init; recurrent LSTM uses PyTorch defaults",
            "versions": {
                n: importlib.metadata.version(n)
                for n in ("torch", "stable-baselines3", "sb3-contrib", "gymnasium", "numpy")
            },
            "skill_controller": runner_state.get("skill_controller"),
            "viewer_task": runner_state.get("viewer_task"),
            "transfer": runner_state.get("transfer"),
            "curriculum_protocol": "active-phase-v3" if config.skills.enabled else None,
            "boundary": "completed_update" if model._n_updates else "initialization_only",
            "optimizer_updates": model._n_updates,
            "resume": "optimizer/counters/RNG/sampler; reset episodes/LSTM",
        }
        (temporary / "metadata.json").write_text(json.dumps(meta, indent=2, allow_nan=False))
        (temporary / "runner_state.json").write_text(
            json.dumps(runner_state, indent=2, allow_nan=False)
        )
        torch.save(
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            temporary / "rng_state.pt",
        )
        hashes = {n: _hash(temporary / n) for n in sorted(FILES)}
        (temporary / "checksums.json").write_text(json.dumps(hashes, indent=2))
        for p in temporary.iterdir():
            with p.open("rb") as stream:
                os.fsync(stream.fileno())
        (temporary / "READY").write_text("g4-checkpoint-v1\n")
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return meta


def load_agent(path, observation_space, *, device="cpu"):
    path, meta = inspect_checkpoint(path, inference=True)
    if schema_signature(observation_space) != meta["schema"]:
        raise ValueError("Checkpoint observation/action schema incompatible with environment")
    from sb3_contrib import MaskablePPO

    cls = (
        MaskablePPO
        if meta["config"].get("skills", {}).get("wall_mask")
        else algorithm(meta["method"])
    )
    model = cls.load(path / "model.zip", device=device)
    if schema_signature(model.observation_space) != meta["schema"]:
        raise ValueError("Serialized policy schema differs from metadata")
    return LearnedAgent(model, {**meta, "checkpoint": str(path)})


def restore_rng(path):
    values = torch.load(Path(path) / "rng_state.pt", map_location="cpu", weights_only=False)
    random.setstate(values["python"])
    np.random.set_state(values["numpy"])
    torch.set_rng_state(values["torch"])
    if values["cuda"]:
        if not torch.cuda.is_available() or len(values["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG/device count mismatch for resume")
        torch.cuda.set_rng_state_all(values["cuda"])
