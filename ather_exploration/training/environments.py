"""Per-worker bank sampler and privileged metrics outside policy observations."""

import copy

import gymnasium as gym

from ather_exploration.config import ObservationConfig
from ather_exploration.environment.env import make_env
from ather_exploration.evaluation.metrics import EpisodeMetrics
from ather_exploration.schema import observation_space
from ather_exploration.seeds import stage_rng
from ather_exploration.training.curriculum import WorldBank


class TrainingEnv(gym.Env):
    def __init__(self, banks, seed, worker, trace_every=0):
        self.bank = WorldBank(banks)
        self.rng = stage_rng(seed, "training-worker", worker)
        self.worker, self.serial, self.stage = worker, 0, 2
        self.trace_every = trace_every
        self.observation_space = observation_space(
            ObservationConfig.model_validate(self.bank.schema)
        )
        self.action_space = gym.spaces.Discrete(5)
        self.env = None
        self.metrics = None
        self.pending = []
        self.active = False

    def set_stage(self, stage):
        self.stage = stage

    def reset(self, *, seed=None, options=None):
        # A caller seed must not silently restart the persisted worker stream.
        super().reset(seed=seed)
        if self.active:
            self.cancel_episode("reset")
        if self.env is not None:
            self.env.close()
        record, meta = self.bank.sample(self.rng, self.stage)
        self.env = make_env(generated=record)
        obs, info = self.env.reset()
        self.serial += 1
        self.meta = meta
        self.record = record
        self.metrics = EpisodeMetrics(
            self.env.unwrapped.evaluator_snapshot(),
            obs,
            record.config.reward,
            episode_id=f"worker{self.worker}-episode{self.serial}",
            group=meta["group"],
            metadata=meta,
        )
        self.active = True
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        core = self.env.unwrapped
        self.metrics.update(
            core.evaluator_snapshot(), obs, reward, info["transition"], core.collision_stage
        )
        if terminated or truncated:
            self._finish(cancelled=truncated)
        return obs, reward, terminated, truncated, info

    def _finish(self, cancelled=False, reason=None):
        record = self.metrics.finish(cancelled=cancelled)
        if reason:
            record["cancellation_reason"] = reason
        trace = None
        if self.trace_every and self.serial % self.trace_every == 0:
            trace = {
                "record": self.record.payload(),
                "steps": copy.deepcopy(self.metrics.steps),
                "result": record,
            }
        self.pending.append((record, trace))
        self.active = False

    def cancel_episode(self, reason="run_end"):
        if self.active:
            self._finish(cancelled=True, reason=reason)

    def drain(self):
        values, self.pending = self.pending, []
        return values

    def checkpoint_state(self):
        return {
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "serial": self.serial,
            "stage": self.stage,
            "unfinished": (
                copy.deepcopy(self.metrics).finish(cancelled=True) if self.active else None
            ),
        }

    def restore(self, state):
        if self.active:
            raise ValueError("Restore must happen before first reset")
        self.rng.bit_generator.state = state["rng"]
        self.serial, self.stage = state["serial"], state["stage"]

    def close(self):
        if self.env is not None:
            self.env.close()
