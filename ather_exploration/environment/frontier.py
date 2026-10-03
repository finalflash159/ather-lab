"""Frontier candidates from accumulated public terrain only; no oracle access."""

import gymnasium as gym
import numpy as np


def frontier_mask(memory):
    seen = memory[0] > 0
    unknown = ~seen
    adjacent = np.zeros_like(seen)
    adjacent[1:] |= unknown[:-1]
    adjacent[:-1] |= unknown[1:]
    adjacent[:, 1:] |= unknown[:, :-1]
    adjacent[:, :-1] |= unknown[:, 1:]
    return (seen & (memory[2] > 0) & ~(memory[1] > 0) & adjacent).astype(np.float32)


def frontier_space(space):
    spaces = dict(space.spaces)
    _, h, w = spaces["memory"].shape
    spaces["memory"] = gym.spaces.Box(0, 1, (12, h, w), np.float32)
    return gym.spaces.Dict(spaces)


class FrontierObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = frontier_space(env.observation_space)

    def observation(self, obs):
        return {
            **obs,
            "memory": np.concatenate((obs["memory"], frontier_mask(obs["memory"])[None])),
        }

    @property
    def phase(self):
        return self.env.phase

    @property
    def poi_seen(self):
        return self.env.poi_seen

    def action_masks(self):
        return self.env.action_masks()
