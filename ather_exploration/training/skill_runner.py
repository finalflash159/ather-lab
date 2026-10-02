"""Explicit headless P1-P5 learning; gates and checkpoints at completed updates."""

import json
import random
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ather_exploration.agents.learning import algorithm, build_model
from ather_exploration.evaluation.skills import evaluate_skill, evaluate_target
from ather_exploration.training.checkpoints import inspect_checkpoint, restore_rng, save_checkpoint
from ather_exploration.training.runner import append_jsonl
from ather_exploration.training.skill_curriculum import SkillController
from ather_exploration.training.skill_environments import SkillTrainingEnv, skill_identity
from ather_exploration.worlds.scenarios import implementation_id, write_record


class SkillStop(Exception):
    """Intentional gate stop at a fully saved optimizer boundary."""


def preflight(config):
    from ather_exploration.worlds.skill_tasks import TASKS, configured_skill_env

    ids = skill_identity(config)
    if config.skills.stop_after == "P5":
        from ather_exploration.training.curriculum import WorldBank

        ids.update(WorldBank(config.banks).identities)
    for task in TASKS:
        env = configured_skill_env(task, 0, config.skills)
        try:
            obs, _ = env.reset()
            assert env.observation_space.contains(obs)
            env.step(4)
        finally:
            env.close()
    return ids


class SkillCallback(BaseCallback):
    def __init__(self, config, root, controller, identities, publish=None):
        super().__init__()
        self.config, self.root, self.controller, self.identities = (
            config,
            root,
            controller,
            identities,
        )
        self.publish = publish
        self.last = -1
        self.saved = -1
        self.state = "RUNNING"
        self.started = time.monotonic()
        self.viewer_task = controller.task
        self.episodes = []
        self.transfer = None

    def _on_step(self):
        if not np.isfinite(self.locals["rewards"]).all():
            raise ValueError("Nonfinite rewards")
        for worker in self.training_env.env_method("drain"):
            for row in worker:
                append_jsonl(self.root / "train_episodes.jsonl", row)
                self.episodes.append(row)
        self.episodes = self.episodes[-256:]
        return True

    def save(self):
        steps = self.model.num_timesteps
        path = f"checkpoints/step_{steps}"
        if self.saved == steps:
            return
        configs = {}
        if self.controller.index >= 7 and self.state != "PHASE_COMPLETED":
            from ather_exploration.training.curriculum import WorldBank

            bank = WorldBank(self.config.banks)
            configs = {
                g: bank.worlds[g]["train"][0]["records"][0].config.model_dump(mode="json")
                for g in bank.worlds
            }
        save_checkpoint(
            self.model,
            self.root / path,
            self.config,
            {
                "skill_controller": asdict(self.controller),
                "viewer_task": self.viewer_task,
                "workers": self.training_env.env_method("checkpoint_state"),
                "env_steps": steps,
                "environment_configs": configs,
                "state": self.state,
                "transfer": self.transfer,
            },
            self.identities,
        )
        write_record(self.root / "latest.json", {"checkpoint": path}, replace=True)
        self.saved = steps
        if self.publish:
            self.publish()

    def evaluate(self, task):
        # Model evaluation may change training mode; preserve all global RNG streams.
        py, npstate, ts = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        mode = self.model.policy.training
        try:
            result = (
                evaluate_target(self.model, self.config)
                if task == "target"
                else evaluate_skill(self.model, task, self.config)
            )
        finally:
            self.model.policy.set_training_mode(mode)
            random.setstate(py)
            np.random.set_state(npstate)
            torch.set_rng_state(ts)
            if cuda:
                torch.cuda.set_rng_state_all(cuda)
        for check in result.get("checks", []):
            print(
                f"[gate:{task}] {check['name']}={check['value']:.4f} "
                f"{check['operator']} {check['threshold']:.4f}: "
                f"{'PASS' if check['passed'] else 'FAIL'}",
                flush=True,
            )
        self.controller.eval_steps += result["eval_steps"]
        append_jsonl(
            self.root / "skill_evaluations.jsonl",
            {"training_steps": self.model.num_timesteps, **result},
        )
        return result

    def boundary(self):
        steps = self.model.num_timesteps
        if steps == self.last or steps == 0:
            return
        self.last = steps
        previous = self.controller.task
        for p in self.model.policy.parameters():
            if not torch.isfinite(p).all():
                raise ValueError("Nonfinite policy")
        metrics = {
            "training/env_steps": steps,
            "skills/stage": self.controller.index,
            "skills/eval_steps": self.controller.eval_steps,
        }
        for k, v in self.model.logger.name_to_value.items():
            if isinstance(v, (int, float, np.number)):
                if k == "train/explained_variance" and np.isnan(v):
                    # SB3 reports undefined variance for constant-return rollouts.
                    continue
                if not np.isfinite(v):
                    raise ValueError(f"Nonfinite optimizer metric: {k}")
                metrics[k] = float(v)
        for task in {r.get("task") for r in self.episodes} - {None}:
            subset = [r for r in self.episodes if r.get("task") == task and not r.get("cancelled")]
            if subset:
                for key in ("return", "length", "intrinsic_return"):
                    metrics[f"skill_train/{task}/{key}"] = float(np.mean([r[key] for r in subset]))
                for component in (
                    "area",
                    "discovery",
                    "activation",
                    "death",
                    "intrinsic",
                    "step_cost",
                ):
                    values = [
                        r["reward_components"][component]
                        for r in subset
                        if component in r.get("reward_components", {})
                    ]
                    if values:
                        metrics[f"skill_train/{task}/reward_{component}"] = float(np.mean(values))
        changed = False
        if steps % self.config.skills.eval_interval == 0:
            if self.controller.index < 7:
                result = self.evaluate(self.controller.task)
                passed = result["passed"]
                # Advancement depends only on the current objective's validation.
                # Earlier geometry is training data, not a separate retention exam.
                for mode, values in result["summary"].items():
                    for k, v in values.items():
                        if v is not None:
                            metrics[f"skills/{self.controller.task}/{mode}/{k}"] = v
                previous = self.controller.task
                changed = self.controller.observe(passed, steps)
                if self.controller.failed:
                    self.state = "PHASE_GATE_FAILED"
                if (
                    changed
                    and previous[:2] == self.config.skills.stop_after
                    and self.controller.task[:2] != previous[:2]
                ):
                    self.state = "PHASE_COMPLETED"
            else:
                result = self.evaluate("target")
                current = result["summary"]["deterministic"]
                prior = self.controller.best.get("metrics")
                better = (
                    prior is None
                    or current["activation_survived"] > prior["activation_survived"] + 0.01
                )
                if (
                    prior is not None
                    and abs(current["activation_survived"] - prior["activation_survived"]) <= 0.01
                ):
                    better = (current["coverage_auc"], current["survival"]) > (
                        prior["coverage_auc"],
                        prior["survival"],
                    )
                if better:
                    self.controller.best = {
                        "run_id": self.root.name,
                        "checkpoint": f"checkpoints/step_{steps}",
                        "metrics": current,
                    }
                    write_record(self.root / "best.json", self.controller.best, replace=True)
                for mode, values in result["summary"].items():
                    for k, v in values.items():
                        metrics[f"target/{mode}/{k}"] = v
                if (
                    self.controller.task == "P5a"
                    and steps - self.controller.phase_start
                    >= (self.config.total_timesteps - self.controller.phase_start) // 2
                ):
                    self.controller.index = 8
                    changed = True
        self.viewer_task = (
            previous if changed and self.state == "PHASE_COMPLETED" else self.controller.task
        )
        if changed:
            self.training_env.env_method("cancel_episode", "phase_transition")
            self.training_env.env_method("set_controller", asdict(self.controller))
            if self.state == "RUNNING":
                self.model._last_obs = self.training_env.reset()
                self.model._last_episode_starts = np.ones(self.config.n_envs, dtype=bool)
            for records in self.training_env.env_method("drain"):
                for record in records:
                    append_jsonl(self.root / "train_episodes.jsonl", record)
            append_jsonl(
                self.root / "phase_transitions.jsonl",
                {"steps": steps, "controller": asdict(self.controller)},
            )
        metrics["skills/eval_steps"] = self.controller.eval_steps
        for key, value in metrics.items():
            self.model.logger.record(key, value)
        self.model.logger.dump(steps)
        if self.tracker:
            self.tracker.log(metrics, step=steps)
        append_jsonl(
            self.root / "progress.jsonl",
            {
                "schema": "training-progress-v1",
                "run_id": self.root.name,
                "env_steps": steps,
                "elapsed_seconds": time.monotonic() - self.started,
                "metrics": metrics,
            },
        )
        print(
            f"[skills:{self.root.name}] {steps:,}/{self.config.total_timesteps:,} task={self.controller.task} state={self.state} "
            + " ".join(
                f"{key}={metrics[key]:.5g}"
                for key in ("train/loss", "train/policy_gradient_loss", "train/value_loss")
                if key in metrics
            ),
            flush=True,
        )
        if (
            steps % (self.config.checkpoint_updates * self.config.n_envs * self.config.n_steps) == 0
            or steps % self.config.skills.eval_interval == 0
            or changed
            or self.state != "RUNNING"
            or steps >= self.config.total_timesteps
        ):
            self.save()
        if self.state != "RUNNING":
            raise SkillStop(self.state)

    def _on_rollout_start(self):
        self.boundary()

    def _on_training_end(self):
        self.boundary()


def run_skill_training(
    config,
    output,
    *,
    resume=None,
    on_boundary=None,
    run_metadata=None,
    continue_curriculum=False,
    transfer_p1_to_p2=False,
):
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError("Use a new run directory")
    if continue_curriculum and resume is None:
        raise ValueError("Continuation requires --resume")
    if transfer_p1_to_p2 and (not continue_curriculum or resume is None):
        raise ValueError("P1 transfer requires --resume and --continue-curriculum")
    identities = preflight(config)
    root.mkdir(parents=True)
    write_record(
        root / "manifest.json",
        {
            "config": config.model_dump(mode="json"),
            "source_revision": implementation_id(),
            "bank_ids": identities,
            "scope": "skill_curriculum_pilot",
            "curriculum_protocol": "active-phase-v1",
            "resume": str(resume) if resume else None,
        },
    )
    if run_metadata:
        write_record(root / "remote.json", run_metadata)
    write_record(root / "run_status.json", {"state": "INITIALIZING"}, replace=True)
    torch.set_num_threads(config.torch_threads)
    env = None
    tracker = None
    callback = None
    model = None
    from ather_exploration.training.tracking import start_tracking

    try:
        factories = [partial(SkillTrainingEnv, config, i) for i in range(config.n_envs)]
        env = (
            DummyVecEnv(factories)
            if config.vec_backend == "dummy"
            else SubprocVecEnv(factories, start_method="spawn")
        )
        controller = SkillController()
        transfer = None
        if transfer_p1_to_p2:
            from ather_exploration.training.skill_transfer import prepare_p1_transfer

            model, state, transfer = prepare_p1_transfer(resume, config, env)
            controller = SkillController(**state["skill_controller"])
            for i, worker in enumerate(state["workers"]):
                env.env_method("restore", worker, indices=i)
            restore_rng(Path(transfer["parent_checkpoint"]))
            write_record(root / "transfer.json", transfer)
        elif resume:
            parent, meta = inspect_checkpoint(resume)
            old = meta["config"]
            new = config.model_dump(mode="json")
            if continue_curriculum:
                if new["skills"]["stop_after"] <= old["skills"]["stop_after"]:
                    raise ValueError("Continuation must extend stop_after")
                old = {
                    **old,
                    "skills": {**old["skills"], "stop_after": new["skills"]["stop_after"]},
                }
            for k in new:
                if k not in ("banks", "device", "tracking") and new[k] != old.get(k):
                    raise ValueError(f"Resume config mismatch: {k}")
            if any(identities.get(k) != v for k, v in meta["bank_ids"].items()):
                raise ValueError("Resume bank mismatch")
            state = json.loads((parent / "runner_state.json").read_text())
            transfer = state.get("transfer")
            if state.get("state") == "PHASE_COMPLETED" and not continue_curriculum:
                raise ValueError(
                    "Completed phase requires --continue-curriculum and extended stop_after"
                )
            controller = SkillController(**state["skill_controller"])
            if controller.failed:
                raise ValueError("Failed gate requires a new experimental attempt")
            if continue_curriculum and state.get("state") != "PHASE_COMPLETED":
                raise ValueError("Continuation requires a completed phase checkpoint")
            cls = MaskablePPO if config.skills.wall_mask else algorithm(config.method)
            model = cls.load(parent / "model.zip", env=env, device=config.device)
            if model.num_timesteps >= config.total_timesteps:
                raise ValueError("Budget exhausted")
            for i, s in enumerate(state["workers"]):
                env.env_method("restore", s, indices=i)
            restore_rng(parent)
        else:
            model = build_model(config, env)
        env.env_method("set_controller", asdict(controller))
        tracker = start_tracking(config, root, identities, resume)
        model.set_logger(configure(str(root / "tensorboard"), ["tensorboard", "csv"]))
        callback = SkillCallback(config, root, controller, identities, on_boundary)
        callback.tracker = tracker
        callback.transfer = transfer
        callback.last = model.num_timesteps
        write_record(root / "run_status.json", {"state": "RUNNING"}, replace=True)
        try:
            model.learn(
                total_timesteps=config.total_timesteps - model.num_timesteps,
                reset_num_timesteps=False,
                callback=callback,
                log_interval=None,
            )
        except SkillStop:
            pass
        env.env_method("cancel_episode", "run_end")
        for worker in env.env_method("drain"):
            for row in worker:
                append_jsonl(root / "train_episodes.jsonl", row)
        result = {
            "state": callback.state
            if callback.state != "RUNNING"
            else "COMPLETED"
            if controller.index >= 7
            else "BUDGET_EXHAUSTED",
            "actual_env_steps": model.num_timesteps,
            "latest_checkpoint_steps": callback.saved,
            "skill_controller": asdict(controller),
        }
        write_record(root / "run_status.json", result, replace=True)
        return result
    except BaseException as e:
        write_record(
            root / "run_status.json",
            {
                "state": "FAILED",
                "error": str(e),
                "last_recoverable_steps": callback.saved if callback else 0,
            },
            replace=True,
        )
        raise
    finally:
        if env:
            env.close()
        if tracker:
            tracker.finish()
