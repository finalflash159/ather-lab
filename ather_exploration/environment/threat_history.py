"""Public monster/visibility history in stable spawn-relative coordinates."""

from collections import deque

import gymnasium as gym
import numpy as np


def threat_space(space, frames=1):
    if frames not in (1, 2):
        raise ValueError("Threat history requires one or two past frames")
    spaces = dict(space.spaces)
    channels, height, width = spaces["memory"].shape
    if channels != 12:
        raise ValueError("Threat history requires frontier memory")
    spaces["memory"] = gym.spaces.Box(0, 1, (12 + 2 * frames, height, width), np.float32)
    return gym.spaces.Dict(spaces)


class ThreatHistory(gym.Wrapper):
    def __init__(self, env, frames=1):
        super().__init__(env)
        self.frames = frames
        self.observation_space = threat_space(env.observation_space, frames)
        self.history = deque(maxlen=frames)

    @staticmethod
    def frame(obs):
        memory = obs["memory"]
        return np.stack((memory[9] * memory[8], memory[8])).copy()

    def _observation(self, obs):
        result = {**obs, "memory": np.concatenate((obs["memory"], *self.history))}
        self.history.appendleft(self.frame(obs))
        return result

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.history.clear()
        self.history.extend(np.zeros_like(self.frame(obs)) for _ in range(self.frames))
        return self._observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._observation(obs), reward, terminated, truncated, info
