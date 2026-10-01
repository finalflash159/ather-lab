"""Modal entrypoint: check by default; learning requires explicit --command train."""

import json
import os
import uuid
from pathlib import Path

import modal

from ather_exploration.training.modal_io import VOLUME_NAME, identifier

ROOT = Path(__file__).resolve().parents[2]
MOUNT = Path("/data")
GPU = os.environ.get("ATHER_MODAL_GPU", "L4")
CPU = int(os.environ.get("ATHER_MODAL_CPU", "8"))
MEMORY = int(os.environ.get("ATHER_MODAL_MEMORY", "16384"))
if CPU < 1 or MEMORY < 1024:
    raise ValueError("Invalid Modal CPU/RAM allocation")

# Whitelist source: never upload .venv, credentials, artifacts or the whole home/project.
image = modal.Image.debian_slim(python_version="3.11").pip_install("uv==0.9.5")
for filename in ("pyproject.toml", "uv.lock", "README.md", "LICENSE"):
    image = image.add_local_file(ROOT / filename, f"/app/{filename}", copy=True)
sync_command = (
    "UV_DEFAULT_INDEX=https://pypi.org/simple UV_INDEX_URL=https://pypi.org/simple "
    "uv sync --locked --extra cuda --no-dev --python /usr/local/bin/python"
)
# Dependencies cached independently from rapidly changing project source.
image = image.workdir("/app").run_commands(sync_command + " --no-install-project")
for directory in ("ather_exploration", "minigrid"):
    image = image.add_local_dir(
        ROOT / directory,
        f"/app/{directory}",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
image = image.run_commands(sync_command).env(
    {
        "PATH": "/app/.venv/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": "/app/.venv/lib/python3.11/site-packages:/app",
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy",
    }
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("ather-exploration", image=image)


def remote_config(payload, dataset):
    from ather_exploration.training.config import TrainingConfig
    from ather_exploration.training.curriculum import WorldBank
    from ather_exploration.worlds.scenarios import implementation_id, read_record

    root = MOUNT / "banks" / identifier(dataset)
    manifest = read_record(root / "dataset.json")
    if manifest["source_revision"] != implementation_id():
        raise ValueError("Bank source differs; rebuild/upload with this revision")
    config = TrainingConfig.model_validate(
        {
            **payload,
            "device": "cuda",
            "banks": {group: str(root / group) for group in ("small", "medium", "large")},
        }
    )
    from ather_exploration.training.skill_runner import preflight

    identities = preflight(config) if config.skills.enabled else WorldBank(config.banks).identities
    if identities != manifest["bank_ids"]:
        raise ValueError("Remote bank identities differ from upload")
    return config


@app.function(
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY,
    volumes={MOUNT: volume},
    timeout=86400,
    retries=0,
    include_source=False,
    max_containers=1,
    secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
)
def execute(
    payload: dict,
    dataset: str,
    command: str = "check",
    run_id: str = "",
    resume: str = "",
    continue_curriculum: bool = False,
):
    import torch

    from ather_exploration.agents.learning import build_model
    from ather_exploration.training.environments import TrainingEnv
    from ather_exploration.training.runner import run_training
    from ather_exploration.worlds.scenarios import implementation_id

    if command not in ("check", "train"):
        raise ValueError("command must be check or train")
    volume.reload()  # Before opening files/writers; never reload the running trainer.
    config = remote_config(payload, dataset)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    resources = {
        "gpu": torch.cuda.get_device_name(0),
        "cpu_requested": CPU,
        "memory_mib_requested": MEMORY,
        "source_revision": implementation_id(),
        "dataset": dataset,
    }
    if command == "check":
        torch.set_num_threads(config.torch_threads)
        if config.skills.enabled:
            from ather_exploration.training.skill_environments import SkillTrainingEnv

            env = SkillTrainingEnv(config, 0)
        else:
            env = TrainingEnv(config.banks, config.seed, 0, 0)
        try:
            model = build_model(config, env)
            observation, _ = env.reset()
            action, _ = model.predict(observation, deterministic=True)
            env.step(int(action))
            torch.cuda.synchronize()
            return {
                **resources,
                "learning_executed": False,
                "cuda_forward_env_step": "pass",
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            }
        finally:
            env.close()
    output = MOUNT / "runs" / identifier(run_id)
    parent = None
    if resume:
        # Explicit checkpoint boundary only; immutable parent, new attempt/run ID.
        import re

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}/checkpoints/step_[0-9]+", resume):
            raise ValueError("resume must be RUN_ID/checkpoints/step_N")
        parent = MOUNT / "runs" / resume
    output.mkdir(parents=True, exist_ok=False)
    import tempfile

    from ather_exploration.training.modal_io import publish_run

    with tempfile.TemporaryDirectory(prefix="ather-modal-") as scratch:
        local_output = Path(scratch) / run_id

        def publish():
            publish_run(local_output, output)
            volume.commit()

        try:
            return run_training(
                config,
                local_output,
                resume=parent,
                continue_curriculum=continue_curriculum,
                on_boundary=publish,
                run_metadata=resources,
            )
        finally:
            publish()


@app.function(
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY,
    volumes={MOUNT: volume},
    timeout=600,
    retries=0,
    include_source=False,
)
def infrastructure_check():
    import torch

    from ather_exploration.agents.learning import build_model
    from ather_exploration.environment.env import make_fixture_env
    from ather_exploration.training.config import TrainingConfig
    from ather_exploration.worlds.scenarios import implementation_id, write_record

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(1)
    result = {
        "learning_executed": False,
        "gpu": torch.cuda.get_device_name(0),
        "source_revision": implementation_id(),
        "models": {},
    }
    env = make_fixture_env("poi_revisit", radius=4)
    try:
        for method in ("ppo", "recurrent_ppo"):
            config = TrainingConfig(
                method=method,
                device="cuda",
                banks={group: "/unused-in-fixture-probe" for group in ("small", "medium", "large")},
            )
            model = build_model(config, env)
            obs, _ = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            env.step(int(action))
            result["models"][method] = "forward/action/env.step pass; no learning"
        torch.cuda.synchronize()
        # Exercise Volume publishing with initialized weights, without learning.
        import tempfile

        from ather_exploration.training.checkpoints import save_checkpoint
        from ather_exploration.training.modal_io import publish_run

        probe_run = f"probe-{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory() as folder:
            local_run = Path(folder) / probe_run
            save_checkpoint(model, local_run / "checkpoints/step_0", config, {}, {})
            write_record(local_run / "latest.json", {"checkpoint": "checkpoints/step_0"})
            publish_run(local_run, MOUNT / "runs" / probe_run)
        result["initialized_checkpoint_run"] = probe_run
        probe = f"probes/{uuid.uuid4().hex}.json"
        result["volume_probe"] = probe
        write_record(MOUNT / probe, result, replace=True)
        volume.commit()
        return result
    finally:
        env.close()


@app.function(
    secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
    timeout=120,
    retries=0,
    include_source=False,
)
def tracking_check(payload):
    import tempfile

    from ather_exploration.training.config import TrainingConfig
    from ather_exploration.training.tracking import start_tracking

    config = TrainingConfig.model_validate(payload)
    if config.tracking.mode != "online":
        raise ValueError("tracking-check requires tracking.mode=online")
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder) / "tracking-check-no-learning"
        root.mkdir()
        run = start_tracking(config, root, {})
        try:
            run.log({"integration/no_learning": 1, "training/env_steps": 0}, step=0)
            return {"learning_executed": False, "wandb_url": run.url}
        finally:
            run.finish()


@app.local_entrypoint()
def main(
    config: str = "",
    dataset: str = "",
    command: str = "check",
    run_id: str = "",
    resume: str = "",
    continue_curriculum: bool = False,
):
    from ather_exploration.training.config import read_training_config

    if command == "tracking-check":
        print(tracking_check.remote(read_training_config(config).model_dump(mode="json")))
        return
    if command == "infrastructure":
        print(json.dumps(infrastructure_check.remote()))
        return
    if command not in ("check", "train"):
        raise ValueError("command must be check or train")
    identifier(dataset)
    run_id = identifier(run_id or f"run-{uuid.uuid4().hex}")
    payload = read_training_config(config).model_dump(mode="json")
    print(
        json.dumps(
            {"command": command, "run_id": run_id, "dataset": dataset, "method": payload["method"]}
        ),
        flush=True,
    )
    print(execute.remote(payload, dataset, command, run_id, resume, continue_curriculum))
