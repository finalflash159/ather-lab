"""Seed-bank skill sampling with isolated per-worker RNG and episode metrics."""

import copy

import gymnasium as gym
import numpy as np

from ather_exploration.config import ObservationConfig, RewardConfig
from ather_exploration.schema import observation_space
from ather_exploration.seeds import stage_rng
from ather_exploration.training.skill_curriculum import SkillController
from ather_exploration.worlds.scenarios import digest
from ather_exploration.worlds.skill_tasks import configured_skill_env, skill_pool, wall_mask


class SkillTrainingEnv(gym.Env):
    def __init__(self, config, worker=0):
        self.config, self.worker = config, worker
        self.observation_space = observation_space(ObservationConfig())
        if config.skills.frontier:
            from ather_exploration.environment.frontier import frontier_space

            self.observation_space = frontier_space(self.observation_space)
        if config.skills.p4.enabled:
            from ather_exploration.environment.threat_history import threat_space

            self.observation_space = threat_space(self.observation_space)
        self.action_space = gym.spaces.Discrete(5)
        self.rng = stage_rng(config.seed, "skill-worker", worker)
        self.controller = SkillController()
        self.env = None
        self.pending = []
        self.serial = 0
        self.active = False
        self.target = None
        self.archive = None
        if config.recovery:
            from ather_exploration.training.recovery_archive import RecoveryArchive

            self.archive = RecoveryArchive(
                config.recovery, stage_rng(config.seed, "recovery-worker", worker)
            )
        elif config.p3_resume:
            from ather_exploration.training.p3_completion import P3PrefixArchive

            self.archive = P3PrefixArchive(
                config.p3_resume, stage_rng(config.seed, "p3-prefix-worker", worker)
            )
        elif config.unfinished_trial:
            from ather_exploration.training.unfinished import PrefixArchive

            self.archive = PrefixArchive(
                config.unfinished_trial, stage_rng(config.seed, "unfinished-worker", worker)
            )

    def seed_recovery_archive(self, task, items):
        if not self.config.recovery or task in self.archive.probed:
            return
        for item in items:
            self.archive.offer(item)
        self.archive.probed.append(task)

    def recovery_probed(self):
        return list(self.archive.probed) if self.config.recovery else []

    def set_controller(self, state):
        self.controller = SkillController(**state)

    def set_restart_level(self, level):
        if self.archive:
            self.archive.level = level

    def archive_summary(self):
        if not self.archive:
            return None
        pools = getattr(self.archive, "pools", None)
        return {
            "level": self.archive.level,
            "replay_steps": self.archive.replay_steps,
            "counts": (
                {task: [len(pool) for pool in bands] for task, bands in pools.items()}
                if pools is not None
                else None
            ),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.active:
            self.cancel_episode("reset")
        if self.env is not None:
            self.env.close()
        mix = self.controller.mixture()
        self.phase = self.controller.task
        self.task = str(self.rng.choice([x[0] for x in mix], p=[x[1] for x in mix]))
        self.serial += 1
        self.total = 0.0
        self.intrinsic_total = 0.0
        self.actions = [0] * 5
        self.reward_totals = {}
        self.ticks = 0
        self.prefix_actions = []
        self.prefix_candidates = []
        self.prefix_qualities = []
        self.captured = set()
        self.restart_requested = self.restart_used = False
        self.prefix_length = 0
        self.restart_progress = False
        self.restart_level = self.archive.level if self.archive else 0
        if self.task == "target":
            from ather_exploration.training.environments import TrainingEnv

            if self.target is None:
                self.target = TrainingEnv(
                    self.config.banks,
                    self.config.seed,
                    self.worker,
                    0,
                    RewardConfig(
                        activation=self.config.skills.activation, death=self.config.skills.death
                    ),
                    self.config.skills.first_visit,
                )
            self.env = self.target
            obs, info = self.env.reset()
        else:
            task_seed = skill_pool(
                self.task, self.config.skills.train_count, p4=self.config.skills.p4.enabled
            )[int(self.rng.integers(self.config.skills.train_count))][0]
            item = None
            if self.archive and self.phase in ("P3b", "P3c"):
                if self.config.p3_resume or self.config.recovery:
                    item, self.restart_requested = self.archive.choose(self.phase)
                elif self.phase == self.task == "P3b":
                    item, self.restart_requested = self.archive.choose()
            if item:
                self.task = item.get("source_task", item.get("task", "P3b"))
            self.task_seed = item["seed"] if item else task_seed
            self.env = configured_skill_env(
                self.task,
                self.task_seed,
                self.config.skills,
                phase=(
                    "P3c" if self.config.skills.p4.enabled and self.task == "P3c" else self.phase
                ),
            )
            if item:
                obs, info = self.archive.replay(self.env, item)
                self.restart_used = True
                if self.config.recovery:
                    self.restart_level = 0
                elif self.config.p3_resume:
                    self.restart_level = item["band"]
                else:
                    from ather_exploration.training.unfinished import BANDS

                    self.restart_level = next(
                        i for i, (lo, hi) in enumerate(BANDS) if lo <= item["distance"] <= hi
                    )
                self.prefix_length = len(item["actions"])
            else:
                obs, info = self.env.reset()
            self.initial_floor = float(obs["memory"][2].sum())
            if self.config.p3_resume and self.phase in ("P3b", "P3c"):
                from ather_exploration.worlds.p3_tasks import (
                    room_coverage_fractions,
                    room_exploration_potential,
                )

                scenario = self.env.unwrapped.scenario
                self.restart_start_potential = room_exploration_potential(
                    room_coverage_fractions(scenario, obs["memory"])
                )
                self.restart_start_activated = len(
                    self.env.unwrapped.evaluator_snapshot().activated_pois
                )
        self.recovery_item = item if self.task != "target" else None
        self.recovery_resolved = False
        self.recovery_stale = 0
        self.recovery_capture = self.recovery_tracker = None
        if self.config.recovery and self.phase in ("P3b", "P3c"):
            from ather_exploration.training.recovery_archive import (
                ResolveTracker,
                StagnationCapture,
            )

            if self.restart_used:
                self.recovery_tracker = ResolveTracker(obs)
            elif self.phase == self.task:
                self.recovery_capture = StagnationCapture(
                    self.config.recovery, self.task, self.task_seed, obs
                )
        self.last_obs = obs
        self.active = True
        return obs, info

    def step(self, action):
        eligible = bool(
            self.config.recovery
            and self.phase in ("P3b", "P3c")
            and (self.restart_used or self.recovery_stale >= self.config.recovery.label_after)
        )
        before = self.last_obs
        obs, r, term, trunc, info = self.env.step(action)
        if self.config.p4_transfer:
            info["teaching_safe_source"] = (
                self.task == "P3c" and not self.env.unwrapped.scenario.routes
            )
            info["teaching_seed"] = self.task_seed
        if self.config.recovery:
            m = before["memory"]
            # Public log-scaled visit count > one visit; no hidden map facts.
            info["route_revisited"] = bool(
                self.phase in ("P3b", "P3c")
                and (m[6][m[7] > 0] > np.log1p(1) / np.log1p(1025) + 1e-6).any()
            )
            info["route_seed"] = self.task_seed
            info["route_source_task"] = self.task
            info["route_eligible"] = eligible  # Label is for BEFORE action, never autoreset obs.
            progress = (
                obs["memory"][2].sum() > before["memory"][2].sum()
                or obs["memory"][4].sum() > before["memory"][4].sum()
            )
            self.recovery_stale = 0 if progress else self.recovery_stale + 1
            if self.recovery_capture is not None:
                candidate = self.recovery_capture.step(action, obs, self.config.skills.p3_horizon)
                if candidate is not None:
                    self.archive.offer(candidate)
                    self.prefix_candidates.append(candidate)
            if self.recovery_tracker is not None:
                scenario = self.env.unwrapped.scenario
                complete = (
                    len(self.env.unwrapped.evaluator_snapshot().activated_pois)
                    == len(scenario.pois)
                ) and float(obs["memory"][2].sum()) == sum(
                    row.count(".") for row in scenario.terrain
                )
                self.recovery_resolved = self.recovery_tracker.step(obs, complete=complete)
                if term or trunc:
                    self.archive.outcome(self.recovery_item, self.recovery_resolved)
        self.last_obs = obs
        if self.config.p3_resume and self.archive and self.phase in ("P3b", "P3c"):
            from ather_exploration.worlds.p3_tasks import (
                room_coverage_fractions,
                room_exploration_potential,
            )

            scenario = self.env.unwrapped.scenario
            snapshot = self.env.unwrapped.evaluator_snapshot()
            activated_count = len(snapshot.activated_pois)
            potential = room_exploration_potential(room_coverage_fractions(scenario, obs["memory"]))
            if self.restart_used:
                self.restart_progress |= (
                    potential > self.restart_start_potential + 1e-9
                    or activated_count > self.restart_start_activated
                )
            else:
                actions = [*self.prefix_actions, int(action)]
                candidate = self.archive.capture(
                    self.phase,
                    self.task_seed,
                    actions,
                    obs,
                    scenario,
                    activated_count,
                    self.captured,
                )
                if candidate is not None:
                    candidate["source_task"] = self.task
                    self.prefix_candidates.append(candidate)
                self.prefix_actions.append(int(action))
                if term or trunc:
                    self.prefix_qualities = self.archive.complete(
                        self.phase,
                        self.prefix_candidates,
                        obs,
                        scenario,
                        activated_count,
                    )
        elif self.archive and not self.config.recovery and self.phase == self.task == "P3b":
            self.restart_progress |= bool(info["transition"]["new_floor"])
            if not self.restart_used and not (term or trunc):
                self.prefix_actions.append(int(action))
                self.archive.offer(
                    self.task_seed,
                    self.prefix_actions,
                    obs,
                    self.initial_floor,
                    self.captured,
                    self.config.skills.p3_horizon,
                )
        self.total += r
        self.actions[int(action)] += 1
        for key, value in info.get("skill", {}).get("reward_components", {}).items():
            self.reward_totals[key] = self.reward_totals.get(key, 0.0) + value
        self.intrinsic_total += info.get("skill", {}).get(
            "intrinsic_reward", info.get("reward_components", {}).get("intrinsic", 0.0)
        )
        self.ticks += 1
        if term or trunc:
            self.pending.append(
                {
                    "task": self.phase,
                    "source_task": self.task,
                    "return": self.total,
                    "intrinsic_return": self.intrinsic_total,
                    "action_counts": self.actions.copy(),
                    "reward_components": self.reward_totals.copy(),
                    "length": self.ticks,
                    "success": info.get("skill", {}).get("success"),
                    "cancelled": False,
                    "restart_requested": self.restart_requested,
                    "restart_used": self.restart_used,
                    "restart_level": self.restart_level,
                    "restart_progress": self.restart_progress,
                    "recovery_resolved": self.recovery_resolved,
                    "prefix_length": self.prefix_length,
                    "reconstruction_steps_total": self.archive.replay_steps if self.archive else 0,
                    "prefix_candidates": len(getattr(self, "prefix_candidates", [])),
                    "prefix_qualities": list(getattr(self, "prefix_qualities", [])),
                }
            )
            self.active = False
        if self.task == "target":
            for episode, _ in self.target.drain():
                self.pending.append({**episode, "phase": self.phase, "source_task": "target"})
        return obs, r, term, trunc, info

    def action_masks(self):
        return wall_mask(self.last_obs)

    def drain(self):
        records, self.pending = self.pending, []
        return records

    def cancel_episode(self, reason):
        if self.active:
            self.pending.append(
                {
                    "task": self.phase,
                    "source_task": self.task,
                    "return": self.total,
                    "intrinsic_return": self.intrinsic_total,
                    "action_counts": self.actions.copy(),
                    "reward_components": self.reward_totals.copy(),
                    "length": self.ticks,
                    "cancelled": True,
                    "reason": reason,
                }
            )
            if self.task == "target":
                self.target.cancel_episode(reason)
                for episode, _ in self.target.drain():
                    self.pending.append({**episode, "phase": self.phase, "source_task": "target"})
            self.active = False

    def checkpoint_state(self):
        state = {
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "serial": self.serial,
            "target": self.target.checkpoint_state() if self.target else None,
        }
        if self.config.recovery:
            state["recovery_archive"] = self.archive.state()
        elif self.config.p3_resume:
            state["prefix_archive"] = self.archive.state() if self.archive else None
        else:
            state["unfinished"] = self.archive.state() if self.archive else None
        return state

    def restore(self, state):
        self.rng.bit_generator.state = state["rng"]
        self.serial = state["serial"]
        archive_state = (
            state.get("recovery_archive")
            if self.config.recovery
            else state.get("prefix_archive", state.get("unfinished"))
        )
        if self.archive and archive_state:
            self.archive.restore(archive_state)
        if state["target"] is not None:
            from ather_exploration.training.environments import TrainingEnv

            self.target = TrainingEnv(
                self.config.banks,
                self.config.seed,
                self.worker,
                0,
                RewardConfig(
                    activation=self.config.skills.activation, death=self.config.skills.death
                ),
                self.config.skills.first_visit,
            )
            self.target.restore(state["target"])

    def close(self):
        if self.env is not None:
            self.env.close()


def skill_identity(config):
    # Count + task version + source fingerprint identify deterministic content-split pools.
    return {
        "skills": digest(
            {
                "version": 5 if config.skills.p4.enabled else 4,
                "train_count": config.skills.train_count,
                "validation_count": config.skills.validation_count,
                "split": "legacy_content_mod5+p3_terrain_dihedral_mod10",
            }
        )
    }
