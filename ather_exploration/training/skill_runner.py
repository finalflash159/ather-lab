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


def task_score(result):
    """Gate satisfaction first; quality tie-breaks belong to the current objective."""
    d = result["summary"]["deterministic"]
    if result["task"].startswith("P3") or result["task"] in ("P4b", "P4c"):
        quality = [d["joint_success"], d["success"], d["coverage_auc"]]
    elif result["task"] == "P2c":
        quality = [d["success"], d["coverage_auc"], d["coverage"]]
    else:
        quality = [
            d["success"],
            d["approach_efficiency"] if result["task"] == "P2b" else d["efficiency"],
        ]
    return [int(result["passed"]), *quality, -d.get("wall_block", 0.0)]


class SkillStop(Exception):
    """Intentional gate stop at a fully saved optimizer boundary."""


def preflight(config):
    from ather_exploration.worlds.skill_tasks import TASKS, configured_skill_env

    ids = skill_identity(config)
    if config.skills.stop_after == "P5":
        from ather_exploration.training.curriculum import WorldBank

        ids.update(WorldBank(config.banks).identities)
    for task in (*TASKS, *(("P4c",) if config.skills.p4.enabled else ())):
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
        self.route_seen = set()
        self.route_candidates = 0
        self.balanced_samples = None
        from ather_exploration.training.threat_training import SourceTelemetry

        self.source_telemetry = SourceTelemetry()

    def _on_step(self):
        if not np.isfinite(self.locals["rewards"]).all():
            raise ValueError("Nonfinite rewards")
        if self.config.p4_transfer:
            self.source_telemetry.observe(self.model, self.locals, self.n_calls)
            if self.config.skills.p4.recovery:
                if self.config.skills.p4.timing:
                    self.model.timing_teaching.active_task = self.controller.task
                self.model.threat_retention.collect(
                    self.model._last_obs, self.locals["infos"], self.n_calls
                )
                if self.config.skills.p4.timing:
                    self.model.timing_teaching.collect(self.model._last_obs, self.locals["infos"])
        if self.config.p4_transfer and self.config.skills.p4.teaching_batches:
            from ather_exploration.training.public_route import public_route

            for i, info in enumerate(self.locals["infos"]):
                if info.get("teaching_safe_source"):
                    self.model.route_memory.collection_steps += 1
                    obs = {key: value[i] for key, value in self.model._last_obs.items()}
                    route = public_route(obs)
                    if route:
                        self.model.route_memory.offer(
                            obs, route["actions"], ("P3c", info["teaching_seed"]), "learner"
                        )
        if self.config.recovery:
            import hashlib

            from ather_exploration.training.public_route import public_route

            aggregated = self.config.recovery.sampling == "aggregated_teaching"
            balanced = self.config.recovery.sampling == "disagreement_balanced"
            if balanced:
                # Frozen rollout policy distribution on PRE-action observations.
                with torch.no_grad():
                    tensor, _ = self.model.policy.obs_to_tensor(self.model._last_obs)
                    preferred = (
                        self.model.policy.get_distribution(tensor)
                        .distribution.probs.argmax(dim=1)
                        .cpu()
                        .numpy()
                    )
            for i, info in enumerate(self.locals["infos"]):
                if aggregated:
                    obs = {key: value[i] for key, value in self.model._last_obs.items()}
                    route = public_route(obs)
                    if route is not None:
                        self.model.route_memory.offer(
                            obs,
                            route["actions"],
                            (info["route_source_task"], info["route_seed"]),
                            "learner",
                        )
                    continue
                if not info.get("route_eligible") and not (
                    balanced and info.get("route_revisited")
                ):
                    continue
                obs = {key: value[i] for key, value in self.model._last_obs.items()}
                # Deduplicate geometry + goal + agent position, ignoring age/time/visit counts.
                digest = hashlib.sha256(obs["memory"][[0, 1, 2, 3, 4, 7]].tobytes()).digest()
                if not balanced:
                    if digest in self.route_seen:
                        continue
                    self.route_seen.add(digest)
                route = public_route(obs)
                if route is None:
                    continue
                if balanced:
                    self.balanced_samples.offer(
                        obs,
                        route["actions"],
                        disagreement=bool(
                            info.get("route_revisited") and not route["actions"][preferred[i]]
                        ),
                        ordinary=bool(info.get("route_eligible")),
                        source=(info["route_source_task"], info["route_seed"]),
                    )
                    continue
                self.route_candidates += 1
                sample = ({key: value.copy() for key, value in obs.items()}, route["actions"])
                limit = self.config.recovery.label_limit
                if len(self.model.route_samples) < limit:
                    self.model.route_samples.append(sample)
                else:
                    index = int(np.random.randint(self.route_candidates))
                    if index < limit:
                        self.model.route_samples[index] = sample
        for worker in self.training_env.env_method("drain"):
            for row in worker:
                append_jsonl(self.root / "train_episodes.jsonl", row)
                self.episodes.append(row)
                trial = self.config.unfinished_trial or self.config.p3_resume
                if (
                    trial
                    and row.get("restart_used")
                    and not row.get("cancelled")
                    and self.controller.observe_restart(
                        row["restart_level"],
                        row["restart_progress"],
                        trial.mastery_episodes,
                        trial.mastery_rate,
                    )
                ):
                    self.training_env.env_method("set_restart_level", self.controller.restart_level)
                    print(
                        f"[restart] unlocked prefix band {self.controller.restart_level}; "
                        "criterion: room or POI progress after replay, not the task gate",
                        flush=True,
                    )
        self.episodes = self.episodes[-256:]
        return True

    def save(self):
        steps = self.model.num_timesteps
        path = f"checkpoints/step_{steps}"
        if self.saved == steps:
            return
        configs = {}
        if self.controller.task.startswith("P5") and self.state != "PHASE_COMPLETED":
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
        if self.controller.best_by_task:
            write_record(
                self.root / "best.json",
                {"by_task": self.controller.best_by_task, "target": self.controller.best},
                replace=True,
            )
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
        if self.config.p4_transfer:
            metrics.update(self.source_telemetry.drain())
            if self.config.skills.p4.recovery:
                from ather_exploration.training.threat_training import update_threat_exploration

                metrics["threat_lesson/exploration_epsilon"] = update_threat_exploration(
                    self.model, self.config
                )
                if self.config.skills.p4.timing:
                    metrics["timing/temperature"] = self.model.policy.threat_temperature
        source_key = "source_task" if self.config.p4_transfer else "task"
        for task in {r.get(source_key) for r in self.episodes} - {None}:
            subset = [
                r for r in self.episodes if r.get(source_key) == task and not r.get("cancelled")
            ]
            if subset:
                for key in ("return", "length", "intrinsic_return"):
                    metrics[f"skill_train/{task}/{key}"] = float(np.mean([r[key] for r in subset]))
                for component in (
                    "area",
                    "discovery",
                    "activation",
                    "death",
                    "intrinsic",
                    "room_exploration",
                    "completion_bonus",
                    "step_cost",
                    "wall_penalty",
                ):
                    values = [
                        r["reward_components"][component]
                        for r in subset
                        if component in r.get("reward_components", {})
                    ]
                    if values:
                        metrics[f"skill_train/{task}/reward_{component}"] = float(np.mean(values))
        if self.config.unfinished_trial or self.config.p3_resume:
            completed = [r for r in self.episodes if not r.get("cancelled")]
            if completed:
                metrics["restart/used_fraction"] = float(
                    np.mean([r.get("restart_used", False) for r in completed])
                )
                metrics["restart/fallback_fraction"] = float(
                    np.mean(
                        [
                            r.get("restart_requested", False) and not r.get("restart_used", False)
                            for r in completed
                        ]
                    )
                )
                metrics["restart/progress_fraction"] = float(
                    np.mean(
                        [
                            r.get("restart_progress", False)
                            for r in completed
                            if r.get("restart_used")
                        ]
                    )
                    if any(r.get("restart_used") for r in completed)
                    else 0.0
                )
                metrics["restart/prefix_candidates"] = float(
                    np.mean([r.get("prefix_candidates", 0) for r in completed])
                )
                qualities = [
                    quality for row in completed for quality in row.get("prefix_qualities", [])
                ]
                if qualities:
                    metrics["restart/prefix_quality"] = float(np.mean(qualities))
            metrics["restart/level"] = self.controller.restart_level
            metrics["restart/reconstruction_steps"] = float(
                max((r.get("reconstruction_steps_total", 0) for r in completed), default=0)
            )
            if self.config.p3_resume:
                summaries = [
                    summary
                    for summary in self.training_env.env_method("archive_summary")
                    if summary is not None
                ]
                metrics["restart/archive_pool_items"] = float(
                    np.mean(
                        [
                            sum(sum(bands) for bands in summary["counts"].values())
                            for summary in summaries
                        ]
                    )
                )
                for task in ("P3b", "P3c"):
                    for band, count in enumerate(
                        zip(
                            *[summary["counts"][task] for summary in summaries],
                            strict=True,
                        )
                    ):
                        metrics[f"restart/archive_{task}_band_{band}"] = float(np.mean(count))
        if self.config.recovery:
            metrics["train/route_label_fraction"] = len(self.model.route_samples) / (
                self.config.n_envs * self.config.n_steps
            )
            metrics["training/probe_steps"] = self.transfer["probe_steps"]
            metrics["restart/reconstruction_steps"] = sum(
                s["replay_steps"] for s in self.training_env.env_method("archive_summary")
            )
            metrics["training/parent_steps"] = self.config.recovery.parent_steps
            metrics["training/branch_steps"] = steps - self.config.recovery.parent_steps
            completed = [r for r in self.episodes if not r.get("cancelled")]
            restarted = [r for r in completed if r.get("restart_used")]
            metrics["restart/resolved_fraction"] = (
                float(np.mean([r["recovery_resolved"] for r in restarted])) if restarted else 0.0
            )
            for source in ("P3a", "P3b", "P3c"):
                for kind in (False, True):
                    rows = [
                        r
                        for r in completed
                        if r.get("source_task") == source and r.get("restart_used") == kind
                    ]
                    if rows:
                        metrics[
                            f"skill_train/{source}/{'restarted' if kind else 'normal'}_success"
                        ] = float(np.mean([bool(r["success"]) for r in rows]))
        changed = False
        if steps % self.config.skills.eval_interval == 0:
            if not self.controller.task.startswith("P5"):
                result = self.evaluate(self.controller.task)
                passed = result["passed"]
                lesson_changed = False
                if self.config.skills.p4.recovery and self.controller.task == "P4a":
                    from ather_exploration.training.threat_lessons import (
                        observe_lesson,
                        probe_lesson,
                    )

                    mode = self.model.policy.training
                    try:
                        probe = probe_lesson(self.model, self.config, self.controller.threat_level)
                    finally:
                        self.model.policy.set_training_mode(mode)
                    self.controller.eval_steps += probe["steps"]
                    append_jsonl(
                        self.root / "threat_lessons.jsonl", {"training_steps": steps, **probe}
                    )
                    balanced = self.config.skills.p4.balanced_families
                    if not balanced:
                        lesson_changed = observe_lesson(self.controller, probe, steps)
                        ready = (
                            self.controller.threat_level == 3
                            and steps - self.controller.threat_level_start >= 16384
                        )
                        passed = passed and ready
                        metrics["threat_lesson/level"] = self.controller.threat_level
                    metrics["threat_lesson/train_probe_success"] = probe["success"]
                    print(
                        f"[encounter] probe={probe['success']:.3f} "
                        + (
                            "diagnostic_only=True"
                            if balanced
                            else f"level={self.controller.threat_level} probe_ready={ready}"
                        ),
                        flush=True,
                    )
                if self.config.skills.p4.recovery:
                    from ather_exploration.training.threat_lessons import retention_report

                    report = retention_report(self.model, self.config)
                    self.controller.eval_steps += report["steps"]
                    metrics["retention/validation_success"] = report["success"]
                    append_jsonl(
                        self.root / "retention_evaluations.jsonl",
                        {"training_steps": steps, **report},
                    )
                    print(
                        f"[retention:P3c] success={report['success']:.4f} (diagnostic, not promotion gate)",
                        flush=True,
                    )
                # Advancement depends only on the current objective's validation.
                # Earlier geometry is training data, not a separate retention exam.
                for mode, values in result["summary"].items():
                    for k, v in values.items():
                        if v is not None:
                            metrics[f"skills/{self.controller.task}/{mode}/{k}"] = v
                for mode, families in result.get("timing_adherence", {}).items():
                    for family, values in families.items():
                        for key, value in values.items():
                            if value is not None:
                                metrics[
                                    f"skills/{self.controller.task}/{mode}/timing/{family}/{key}"
                                ] = value
                        print(
                            f"[timing-follow:{self.controller.task}] mode={mode} family={family} "
                            f"WAIT={values['wait_follow']}/{values['wait_labels']} "
                            f"GO={values['go_follow']}/{values['go_labels']} "
                            f"exact={values['exact_follow']}/{values['labels']}",
                            flush=True,
                        )
                previous = self.controller.task
                score = task_score(result)
                prior = self.controller.best_by_task.get(previous)
                if prior is None or tuple(score) > tuple(prior["score"]):
                    self.controller.best_by_task[previous] = {
                        "run_id": self.root.name,
                        "checkpoint": f"checkpoints/step_{steps}",
                        "score": score,
                        "summary": result["summary"],
                        "passed": passed,
                    }
                gate_budget = self.controller.budget
                if self.config.lr_trial or self.config.unfinished_trial:
                    self.controller.passed = self.controller.passed + 1 if passed else 0
                    self.controller.history.append(
                        {
                            "task": previous,
                            "steps": steps,
                            "passed": bool(passed),
                            "elapsed": steps - self.controller.phase_start,
                            "minimum": self.controller.minimum,
                            "eligible": steps - self.controller.phase_start
                            >= self.controller.minimum,
                            "streak": self.controller.passed,
                            "promotion_disabled": True,
                        }
                    )
                else:
                    changed = self.controller.observe(passed, steps) or lesson_changed
                gate = self.controller.history[-1]
                print(
                    f"[gate:{previous}] raw_pass={result['passed']} promotion_pass={passed} eligible={gate['eligible']} "
                    f"streak={gate['streak']}/2 elapsed={gate['elapsed']} "
                    f"minimum={gate['minimum']} budget="
                    f"{gate_budget}",
                    flush=True,
                )
                if gate.get("forced_advance"):
                    metrics["curriculum/forced_advance"] = 1
                    metrics["curriculum/forced_advance_from"] = previous
                    print(
                        f"[forced-advance] {previous}->P4c after budget exhaustion; "
                        "gate remains failed and this run is experimental",
                        flush=True,
                    )
                if self.controller.failed:
                    self.state = "REVIEW_REQUIRED" if self.config.recovery else "PHASE_GATE_FAILED"
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
                    self.controller.index += 1
                    changed = True
        if (
            self.config.lr_trial
            and steps >= self.config.lr_trial.parent_steps + self.config.lr_trial.additional_steps
        ):
            self.state = "EXPERIMENT_COMPLETED"
        if (
            self.config.unfinished_trial
            and steps
            >= self.config.unfinished_trial.parent_steps
            + self.config.unfinished_trial.additional_steps
        ):
            self.state = "EXPERIMENT_COMPLETED"
        if (
            self.config.skills.p4.recovery
            and steps >= self.config.total_timesteps
            and self.state == "RUNNING"
        ):
            self.state = "BUDGET_EXHAUSTED"
        self.viewer_task = (
            previous if changed and self.state == "PHASE_COMPLETED" else self.controller.task
        )
        if changed:
            self.training_env.env_method("cancel_episode", "phase_transition")
            self.training_env.env_method("set_controller", asdict(self.controller))
            self.training_env.env_method("set_restart_level", self.controller.restart_level)
            if self.state == "RUNNING":
                self.seed_recovery()
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
        trial = self.config.unfinished_trial or self.config.lr_trial
        end_steps = (
            trial.parent_steps + trial.additional_steps if trial else self.config.total_timesteps
        )
        if self.config.recovery:
            end_steps = (
                self.controller.phase_start + self.controller.budget
                if self.controller.task in ("P3b", "P3c")
                else steps
            )
        timing_report = []
        if self.config.p4_transfer and self.config.skills.p4.timing:
            for kind in ("wait", "go"):
                before = metrics.get(f"timing/{kind}_target_mass_before")
                after = metrics.get(f"timing/{kind}_target_mass_after")
                margin = metrics.get(f"timing/{kind}_logit_margin_delta")
                if before is not None and after is not None and margin is not None:
                    timing_report.append(
                        f"{kind.upper()}_mass={before:.3f}->{after:.3f} dlogit={margin:+.3f}"
                    )
            if "timing/accepted" in metrics or "timing/rejected" in metrics:
                timing_report.append(
                    f"timing_update={int(metrics.get('timing/accepted', 0))} accepted/"
                    f"{int(metrics.get('timing/rejected', 0))} rejected"
                )
        print(
            f"[skills:{self.root.name}] {steps:,}/{end_steps:,} task={self.controller.task} state={self.state} "
            + " ".join(
                f"{key}={metrics[key]:.5g}"
                for key in ("train/loss", "train/policy_gradient_loss", "train/value_loss")
                if key in metrics
            ),
            flush=True,
        )
        if timing_report:
            print(f"[timing-update:{self.root.name}] " + " | ".join(timing_report), flush=True)
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

    def seed_recovery(self):
        if not self.config.recovery or self.controller.task not in ("P3b", "P3c"):
            return
        task = self.controller.task
        probed = self.training_env.env_method("recovery_probed")
        if all(task in tasks for tasks in probed):
            return
        from ather_exploration.agents.route_ppo import RoutePPO
        from ather_exploration.training.recovery import training_probes

        # Frozen original parent, including when entering P3c. No gradient or PPO samples.
        py, ns, ts = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        try:
            parent = RoutePPO.load(
                Path(self.transfer["parent_checkpoint"]) / "model.zip", device=self.config.device
            )
            items, count = training_probes(parent, self.config, task)
            for worker in range(self.config.n_envs):
                self.training_env.env_method(
                    "seed_recovery_archive",
                    task,
                    items[worker :: self.config.n_envs],
                    indices=worker,
                )
            self.transfer["probe_steps"] += count
            append_jsonl(
                self.root / "recovery_probes.jsonl",
                {"task": task, "steps": count, "items": len(items), "split": "train"},
            )
            print(
                f"[recovery] {task}: {count} frozen probe steps, {len(items)} prefixes; excluded from PPO",
                flush=True,
            )
        finally:
            random.setstate(py)
            np.random.set_state(ns)
            torch.set_rng_state(ts)
            if cuda:
                torch.cuda.set_rng_state_all(cuda)

    def _on_training_start(self):
        self.seed_recovery()
        if self.config.recovery and self.config.recovery.sampling == "aggregated_teaching":
            from ather_exploration.training.route_teaching import initialize_teaching

            initialize_teaching(self.model, self.config)

    def _on_rollout_start(self):
        self.boundary()
        if self.config.skills.p4.recovery:
            self.model.threat_retention.guard.buckets.clear()
            self.model.threat_retention.guard.counts.clear()
        if self.config.recovery:
            self.model.route_samples = []
            self.route_seen.clear()
            self.route_candidates = 0
            if self.config.recovery.sampling == "disagreement_balanced":
                from ather_exploration.training.route_sampling import BalancedRouteSamples

                self.balanced_samples = BalancedRouteSamples(self.config.recovery.label_limit)

    def _on_rollout_end(self):
        if self.balanced_samples is not None:
            self.model.route_samples, wrong = self.balanced_samples.select()
            self.model.logger.record("train/route_disagreement_labels", wrong)

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
    transfer_p2_to_p3=False,
):
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError("Use a new run directory")
    if continue_curriculum and resume is None:
        raise ValueError("Continuation requires --resume")
    if (transfer_p1_to_p2 or transfer_p2_to_p3) and (not continue_curriculum or resume is None):
        raise ValueError("Transfer requires --resume and --continue-curriculum")
    if transfer_p1_to_p2 and transfer_p2_to_p3:
        raise ValueError("Choose one transfer protocol")
    if config.p3_restart and (
        not resume or continue_curriculum or transfer_p1_to_p2 or transfer_p2_to_p3
    ):
        raise ValueError("P3 restart requires only --resume")
    if config.lr_trial and (
        resume is None or continue_curriculum or transfer_p1_to_p2 or transfer_p2_to_p3
    ):
        raise ValueError("LR trial requires only --resume; no curriculum transfer flags")
    if (config.p3_resume or config.recovery) and (
        resume is None or continue_curriculum or transfer_p1_to_p2 or transfer_p2_to_p3
    ):
        raise ValueError("P3 completion resume requires only --resume")
    if config.unfinished_trial and (
        resume is None or continue_curriculum or transfer_p1_to_p2 or transfer_p2_to_p3
    ):
        raise ValueError("Unfinished trial requires only --resume")
    if config.p4_transfer and (
        not resume or continue_curriculum or transfer_p1_to_p2 or transfer_p2_to_p3
    ):
        raise ValueError("P4 requires only --resume; no other transfer flags")
    identities = preflight(config)
    root.mkdir(parents=True)
    write_record(
        root / "manifest.json",
        {
            "config": config.model_dump(mode="json"),
            "source_revision": implementation_id(),
            "bank_ids": identities,
            "scope": "skill_curriculum_pilot",
            "learning_objective": (
                "ppo_public_timing_with_safe_retention"
                if config.p4_transfer and config.skills.p4.timing
                else "ppo_temporal_threat_with_safe_retention"
                if config.p4_transfer
                else "ppo_separate_public_teaching"
                if config.recovery and config.recovery.sampling == "aggregated_teaching"
                else "ppo_public_route_aux"
                if config.recovery
                else config.method
            ),
            "curriculum_protocol": "active-phase-v3",
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
        controller = SkillController(
            p2_task_budget=config.skills.p2_task_budget,
            p3_minimum=config.skills.p3_minimum,
            p3_task_budget=config.skills.p3_task_budget,
        )
        transfer = None
        if config.p4_transfer:
            from ather_exploration.training.p4_transfer import prepare_p4

            model, state, transfer = prepare_p4(resume, config, env)
            controller = SkillController(**state["skill_controller"])
            if state.get("workers"):
                for i, worker in enumerate(state["workers"]):
                    env.env_method("restore", worker, indices=i)
            restore_rng(Path(transfer["resume_rng_checkpoint"]))
            write_record(root / "transfer.json", transfer)
        elif config.recovery:
            from ather_exploration.training.recovery import prepare_recovery

            model, state, transfer = prepare_recovery(resume, config, env)
            controller = SkillController(**state["skill_controller"])
            for i, worker in enumerate(state["workers"]):
                env.env_method("restore", worker, indices=i)
            restore_rng(Path(transfer["resume_rng_checkpoint"]))
            write_record(root / "transfer.json", transfer)
        elif config.p3_resume:
            from ather_exploration.training.p3_completion import prepare_p3_resume

            model, state, transfer = prepare_p3_resume(resume, config, env)
            controller = SkillController(**state["skill_controller"])
            for i, worker in enumerate(state["workers"]):
                env.env_method("restore", worker, indices=i)
            restore_rng(Path(transfer["resume_rng_checkpoint"]))
            write_record(root / "transfer.json", transfer)
        elif config.unfinished_trial:
            from ather_exploration.training.unfinished_trial import prepare_unfinished

            model, state, transfer = prepare_unfinished(resume, config, env)
            controller = SkillController(**state["skill_controller"])
            for i, worker in enumerate(state["workers"]):
                env.env_method("restore", worker, indices=i)
            parent, _ = inspect_checkpoint(resume, unfinished_trial=True)
            restore_rng(parent)
            write_record(root / "transfer.json", transfer)
        elif (
            config.p3_restart
            and inspect_checkpoint(resume, lr_trial=True)[1]["schema"]["version"] == 1
        ):
            from ather_exploration.training.p3_restart import prepare_restart

            model, state, transfer = prepare_restart(resume, config, env)
            controller = SkillController(**state["skill_controller"])
            restore_rng(Path(transfer["parent_checkpoint"]))
            write_record(root / "transfer.json", transfer)
        elif config.lr_trial:
            from ather_exploration.training.lr_trial import prepare_trial

            model, state, transfer = prepare_trial(resume, config, env)
            controller = SkillController(**state["skill_controller"])
            for i, worker in enumerate(state["workers"]):
                env.env_method("restore", worker, indices=i)
            restore_rng(Path(transfer["parent_checkpoint"]))
            write_record(root / "transfer.json", transfer)
        elif transfer_p1_to_p2 or transfer_p2_to_p3:
            from ather_exploration.training.skill_transfer import (
                prepare_p1_transfer,
                prepare_p2_transfer,
            )

            prepare = prepare_p2_transfer if transfer_p2_to_p3 else prepare_p1_transfer
            model, state, transfer = prepare(resume, config, env)
            controller = SkillController(**state["skill_controller"])
            for i, worker in enumerate(state["workers"]):
                env.env_method("restore", worker, indices=i)
            restore_rng(Path(transfer["parent_checkpoint"]))
            write_record(root / "transfer.json", transfer)
        elif resume:
            parent, meta = inspect_checkpoint(resume)
            from ather_exploration.training.config import TrainingConfig

            old = TrainingConfig.model_validate(meta["config"]).model_dump(mode="json")
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
        env.env_method("set_restart_level", controller.restart_level)
        tracker = start_tracking(config, root, identities, resume)
        model.set_logger(configure(str(root / "tensorboard"), ["tensorboard", "csv"]))
        callback = SkillCallback(config, root, controller, identities, on_boundary)
        callback.tracker = tracker
        callback.transfer = transfer
        callback.last = model.num_timesteps
        write_record(root / "run_status.json", {"state": "RUNNING"}, replace=True)
        try:
            model.learn(
                total_timesteps=(
                    config.unfinished_trial.parent_steps
                    + config.unfinished_trial.additional_steps
                    - model.num_timesteps
                    if config.unfinished_trial
                    else config.lr_trial.parent_steps
                    + config.lr_trial.additional_steps
                    - model.num_timesteps
                    if config.lr_trial
                    else config.total_timesteps - model.num_timesteps
                ),
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
            if controller.task.startswith("P5")
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
