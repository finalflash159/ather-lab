"""Seed-bank skill sampling with isolated per-worker RNG and episode metrics."""

import copy

import gymnasium as gym

from ather_exploration.config import ObservationConfig, RewardConfig
from ather_exploration.schema import observation_space
from ather_exploration.seeds import stage_rng
from ather_exploration.training.skill_curriculum import SkillController
from ather_exploration.worlds.scenarios import digest
from ather_exploration.worlds.skill_tasks import make_skill_env, skill_pool, wall_mask


class SkillTrainingEnv(gym.Env):
    def __init__(self, config, worker=0):
        self.config, self.worker = config, worker
        self.observation_space = observation_space(ObservationConfig())
        self.action_space = gym.spaces.Discrete(5)
        self.rng = stage_rng(config.seed, "skill-worker", worker)
        self.controller = SkillController()
        self.env = None
        self.pending = []
        self.serial = 0
        self.active = False
        self.target = None

    def set_controller(self, state):
        self.controller = SkillController(**state)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.active:
            self.cancel_episode("reset")
        if self.env is not None:
            self.env.close()
        mix = self.controller.mixture()
        self.task = str(self.rng.choice([x[0] for x in mix], p=[x[1] for x in mix]))
        self.serial += 1
        self.total = 0.0
        self.intrinsic_total = 0.0
        self.actions = [0] * 5
        self.ticks = 0
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
            task_seed = skill_pool(self.task, self.config.skills.train_count)[
                int(self.rng.integers(self.config.skills.train_count))
            ][0]
            reward = RewardConfig(
                activation=self.config.skills.activation, death=self.config.skills.death
            )
            self.env = make_skill_env(self.task, task_seed, reward, self.config.skills.first_visit)
            obs, info = self.env.reset()
        self.last_obs = obs
        self.active = True
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.last_obs = obs
        self.total += r
        self.actions[int(action)] += 1
        self.intrinsic_total += info.get("skill", {}).get(
            "intrinsic_reward", info.get("reward_components", {}).get("intrinsic", 0.0)
        )
        self.ticks += 1
        if term or trunc:
            self.pending.append(
                {
                    "task": self.task,
                    "return": self.total,
                    "intrinsic_return": self.intrinsic_total,
                    "action_counts": self.actions.copy(),
                    "length": self.ticks,
                    "success": info.get("skill", {}).get("success"),
                    "cancelled": False,
                }
            )
            self.active = False
        if self.task == "target":
            for episode, _ in self.target.drain():
                self.pending.append(episode)
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
                    "task": self.task,
                    "return": self.total,
                    "intrinsic_return": self.intrinsic_total,
                    "action_counts": self.actions.copy(),
                    "length": self.ticks,
                    "cancelled": True,
                    "reason": reason,
                }
            )
            if self.task == "target":
                self.target.cancel_episode(reason)
                for episode, _ in self.target.drain():
                    self.pending.append(episode)
            self.active = False

    def checkpoint_state(self):
        return {
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "serial": self.serial,
            "target": self.target.checkpoint_state() if self.target else None,
        }

    def restore(self, state):
        self.rng.bit_generator.state = state["rng"]
        self.serial = state["serial"]
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
                "version": 2,
                "train_count": config.skills.train_count,
                "validation_count": config.skills.validation_count,
                "split": "content_hash_mod5",
            }
        )
    }
