"""Explicit train entrypoint. Nothing executes until run_training is called."""

import json
import math
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ather_exploration.agents.learning import algorithm, build_model, schema_signature
from ather_exploration.training.checkpoints import inspect_checkpoint, restore_rng, save_checkpoint
from ather_exploration.training.config import normalize_method
from ather_exploration.training.curriculum import Curriculum, WorldBank
from ather_exploration.training.environments import TrainingEnv
from ather_exploration.worlds.scenarios import implementation_id, write_record


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


class TelemetryCallback(BaseCallback):
    def __init__(self, config, root, curriculum, previous_elapsed=0, on_boundary=None):
        super().__init__()
        self.config, self.root, self.curriculum = config, root, curriculum
        self.previous_elapsed = previous_elapsed
        self.on_boundary = on_boundary
        self.started = time.monotonic()
        self.episodes = []
        self.last_saved = -1
        self.last_telemetry = -1
        self.tracker = None
        self.environment_configs = {}

    def drain(self):
        for worker in self.training_env.env_method("drain"):
            for episode, trace in worker:
                append_jsonl(self.root / "train_episodes.jsonl", episode)
                self.episodes.append(episode)
                if trace:
                    write_record(
                        self.root / "replays" / f"{episode['episode_id']}.json",
                        {"schema": "g4-replay-v1", "source_revision": implementation_id(), **trace},
                    )
        self.episodes = self.episodes[-512:]

    def _on_step(self):
        if not np.isfinite(self.locals["rewards"]).all():
            raise ValueError("Nonfinite environment reward")
        completed = []
        for worker in self.training_env.env_method("drain"):
            for episode, trace in worker:
                append_jsonl(self.root / "train_episodes.jsonl", episode)
                self.episodes.append(episode)
                completed.append(episode)
                if trace:
                    write_record(
                        self.root / "replays" / f"{episode['episode_id']}.json",
                        {"schema": "g4-replay-v1", "source_revision": implementation_id(), **trace},
                    )
        self.episodes = self.episodes[-512:]
        old_stage = self.curriculum.stage
        self.curriculum.advance(self.num_timesteps, self.config.total_timesteps, completed)
        if self.curriculum.stage != old_stage:
            self.training_env.env_method("set_stage", self.curriculum.stage)
        return True

    def boundary(self):
        """Called on next rollout start or training end, AFTER the previous PPO update."""
        steps = self.model.num_timesteps
        if not steps or steps == self.last_telemetry:
            return
        for p in self.model.policy.parameters():
            if not torch.isfinite(p).all():
                raise ValueError("Nonfinite model parameter after update")
        elapsed = self.previous_elapsed + time.monotonic() - self.started
        measurements = {
            "training/env_steps": steps,
            "training/elapsed_seconds": elapsed,
            "training/steps_per_second": (steps - getattr(self, "starting_steps", 0))
            / max(time.monotonic() - self.started, 1e-9),
            "curriculum/stage": self.curriculum.stage,
        }
        for key, value in self.model.logger.name_to_value.items():
            if isinstance(value, (int, float, np.number)):
                if not math.isfinite(float(value)):
                    raise ValueError(f"Nonfinite optimizer measurement: {key}")
                measurements[key] = float(value)
        for group in ("small", "medium", "large"):
            subset = [
                e for e in self.episodes if e["group"] == group and e["status"] == "completed"
            ]
            if subset:
                measurements[f"{group}/return"] = float(np.mean([e["return"] for e in subset]))
                measurements[f"{group}/episode_length"] = float(np.mean([e["T"] for e in subset]))
                for key in subset[0]["metrics"]:
                    values = [
                        e["metrics"][key] for e in subset if e["metrics"].get(key) is not None
                    ]
                    if values:
                        measurements[f"{group}/{key}"] = float(np.mean(values))
                for key in subset[0]["reward_terms"]:
                    measurements[f"{group}/reward_{key}"] = float(
                        np.mean([e["reward_terms"].get(key, 0) for e in subset])
                    )
        if self.episodes:
            measurements["curriculum/fallback_fraction"] = float(
                np.mean([e["metadata"]["fallback"] for e in self.episodes])
            )
        rate = measurements["training/steps_per_second"]
        remaining = max(0, self.config.total_timesteps - steps)
        eta = f"{remaining / rate:.0f}s" if rate > 0 else "unknown"
        losses = " ".join(
            f"{label}={measurements[key]:.5g}"
            for label, key in (
                ("loss", "train/loss"),
                ("actor_loss", "train/policy_gradient_loss"),
                ("critic_loss", "train/value_loss"),
                ("kl", "train/approx_kl"),
            )
            if key in measurements
        )
        print(
            f"[train:{self.root.name}] {steps:,}/{self.config.total_timesteps:,} steps "
            f"({100 * steps / self.config.total_timesteps:.2f}%) | "
            f"rollout_update={steps // (self.config.n_envs * self.config.n_steps)} | "
            f"{rate:.1f} steps/s | elapsed={elapsed:.0f}s | ETA~{eta} | {losses}",
            flush=True,
        )
        for key, value in measurements.items():
            self.model.logger.record(key, value)
        self.model.logger.dump(steps)
        append_jsonl(
            self.root / "progress.jsonl",
            {
                "schema": "training-progress-v1",
                "run_id": self.root.name,
                "env_steps": steps,
                "elapsed_seconds": elapsed,
                "metrics": measurements,
            },
        )
        if self.tracker is not None:
            self.tracker.log(measurements, step=steps)
        self.last_telemetry = steps
        update = steps // (self.config.n_envs * self.config.n_steps)
        if update % self.config.checkpoint_updates == 0 or steps >= self.config.total_timesteps:
            states = self.training_env.env_method("checkpoint_state")
            relative = f"checkpoints/step_{steps}"
            save_checkpoint(
                self.model,
                self.root / relative,
                self.config,
                {
                    "environment_configs": self.environment_configs,
                    "workers": states,
                    "curriculum": asdict(self.curriculum),
                    "elapsed_seconds": elapsed,
                    "env_steps": steps,
                },
                self.bank_ids,
            )
            write_record(self.root / "latest.json", {"checkpoint": relative}, replace=True)
            self.last_saved = steps
        if self.on_boundary is not None:
            self.on_boundary()
        if self.last_saved == steps:
            location = "saved + published" if self.on_boundary is not None else "saved locally"
            print(f"[checkpoint:{self.root.name}] step_{steps} {location}", flush=True)

    def _on_rollout_start(self):
        self.boundary()

    def _on_training_end(self):
        self.boundary()


def run_training(
    config,
    output,
    *,
    resume=None,
    on_boundary=None,
    run_metadata=None,
    continue_curriculum=False,
    transfer_p1_to_p2=False,
):
    if config.skills.enabled:
        from ather_exploration.training.skill_runner import run_skill_training

        return run_skill_training(
            config,
            output,
            resume=resume,
            on_boundary=on_boundary,
            run_metadata=run_metadata,
            continue_curriculum=continue_curriculum,
            transfer_p1_to_p2=transfer_p1_to_p2,
        )
    if continue_curriculum or transfer_p1_to_p2:
        raise ValueError("Continuation is only for skills")
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(
            "Choose a NEW run directory; resume reads an immutable parent checkpoint"
        )
    print(
        f"[train:{root.name}] initializing | method={config.method} | "
        f"budget={config.total_timesteps:,} | device={config.device}",
        flush=True,
    )
    torch.set_num_threads(config.torch_threads)
    bank = WorldBank(config.banks)
    factories = [
        partial(TrainingEnv, config.banks, config.seed, i, config.trace_every)
        for i in range(config.n_envs)
    ]
    env = None
    model = None
    callback = None
    restored_steps = 0
    tracker = None
    succeeded = False
    root.mkdir(parents=True)
    write_record(
        root / "manifest.json",
        {
            "scope": "development_G4_not_formal",
            "config": config.model_dump(mode="json"),
            "bank_ids": bank.identities,
            "source_revision": implementation_id(),
            "resume": str(resume) if resume else None,
        },
    )
    if run_metadata is not None:
        write_record(root / "remote.json", run_metadata)
    write_record(root / "run_status.json", {"state": "INITIALIZING"}, replace=True)
    try:
        from ather_exploration.training.tracking import start_tracking

        tracker = start_tracking(config, root, bank.identities, resume)
        env = (
            DummyVecEnv(factories)
            if config.vec_backend == "dummy"
            else SubprocVecEnv(factories, start_method="spawn")
        )
        curriculum = Curriculum(config.curriculum_enabled)
        previous_elapsed = 0.0
        if resume:
            parent, metadata = inspect_checkpoint(resume)
            current = config.model_dump(mode="json")
            old = {**metadata["config"], "method": normalize_method(metadata["config"]["method"])}
            # Relocated banks may have different paths, but identities must match.
            for key in current:
                if key not in ("banks", "device", "tracking") and current[key] != old[key]:
                    raise ValueError(f"Resume config mismatch: {key}")
            if metadata["bank_ids"] != bank.identities or metadata["schema"] != schema_signature(
                env.observation_space
            ):
                raise ValueError("Resume bank/schema mismatch")
            state = json.loads((parent / "runner_state.json").read_text())
            if state["env_steps"] >= config.total_timesteps:
                raise ValueError("Checkpoint already exhausted the original budget")
            model = algorithm(config.method).load(
                parent / "model.zip", env=env, device=config.device, force_reset=True
            )
            curriculum = Curriculum(**state["curriculum"])
            for i, worker in enumerate(state["workers"]):
                env.env_method("restore", worker, indices=i)
                if worker["unfinished"]:
                    append_jsonl(
                        root / "train_episodes.jsonl",
                        {**worker["unfinished"], "cancellation_reason": "resume_reset"},
                    )
                    append_jsonl(
                        root / "resume_events.jsonl",
                        {"event": "cancel_unfinished", "worker": i, **worker["unfinished"]},
                    )
            restored_steps = state["env_steps"]
            if model.num_timesteps != restored_steps or len(state["workers"]) != config.n_envs:
                raise ValueError("Resume counters/worker count mismatch")
            previous_elapsed = state["elapsed_seconds"]
            restore_rng(parent)
        else:
            model = build_model(config, env)
        curriculum.advance(model.num_timesteps, config.total_timesteps)
        env.env_method("set_stage", curriculum.stage)
        model.set_logger(configure(str(root / "tensorboard"), ["tensorboard", "csv"]))
        callback = TelemetryCallback(config, root, curriculum, previous_elapsed, on_boundary)
        callback.bank_ids = bank.identities
        callback.tracker = tracker
        callback.environment_configs = {
            group: bank.worlds[group]["train"][0]["records"][0].config.model_dump(mode="json")
            for group in bank.worlds
        }
        # Avoid re-saving a loaded boundary under a new directory before any new update.
        callback.last_telemetry = model.num_timesteps
        callback.starting_steps = model.num_timesteps
        write_record(root / "run_status.json", {"state": "RUNNING"}, replace=True)
        print(
            f"[train:{root.name}] learning started at step {model.num_timesteps:,} | "
            f"rollout={config.n_envs * config.n_steps:,} transitions | "
            f"checkpoint every {config.checkpoint_updates} rollout updates | "
            "progress prints after each completed update",
            flush=True,
        )
        # THE ONLY LEARNING ENTRYPOINT. Not called by imports, preflight or unit tests.
        model.learn(
            total_timesteps=config.total_timesteps - model.num_timesteps,
            reset_num_timesteps=False,
            callback=callback,
            log_interval=None,
        )
        env.env_method("cancel_episode", "budget_end")
        callback.drain()
        result = {
            "state": "COMPLETED",
            "actual_env_steps": model.num_timesteps,
            "latest_checkpoint_steps": callback.last_saved,
            "curriculum": asdict(curriculum),
        }
        write_record(root / "run_status.json", result, replace=True)
        succeeded = True
        return result
    except BaseException as error:
        cleanup_error = None
        if env is not None:
            try:
                env.env_method("cancel_episode", "interrupted_or_failed")
                for worker in env.env_method("drain"):
                    for episode, _ in worker:
                        append_jsonl(root / "train_episodes.jsonl", episode)
            except Exception as cleanup:  # noqa: BLE001 -- preserve original worker failure
                cleanup_error = str(cleanup)
        actual = model.num_timesteps if model is not None else 0
        saved = max(restored_steps, callback.last_saved if callback else 0)
        write_record(
            root / "run_status.json",
            {
                "state": "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                "error": f"{type(error).__name__}: {error}",
                "actual_env_steps": actual,
                "last_recoverable_steps": saved,
                "unsaved_env_steps": max(0, actual - saved),
                "cleanup_error": cleanup_error,
                "recovery": "Resume from last READY update boundary; unfinished rollout is not saved",
            },
            replace=True,
        )
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if tracker is not None:
                tracker.finish(exit_code=0 if succeeded else 1)
