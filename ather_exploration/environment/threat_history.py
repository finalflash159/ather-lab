"""One previous public monster/visibility frame, in spawn-relative coordinates."""

import gymnasium as gym
import numpy as np


def threat_space(space):
    spaces = dict(space.spaces)
    channels, height, width = spaces["memory"].shape
    if channels != 12:
        raise ValueError("Threat history requires frontier memory")
    spaces["memory"] = gym.spaces.Box(0, 1, (14, height, width), np.float32)
    return gym.spaces.Dict(spaces)


class ThreatHistory(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = threat_space(env.observation_space)
        self.previous = None

    @staticmethod
    def frame(obs):
        memory = obs["memory"]
        return np.stack((memory[9] * memory[8], memory[8])).copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        history = np.zeros_like(self.frame(obs))
        self.previous = self.frame(obs)
        return {**obs, "memory": np.concatenate((obs["memory"], history))}, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        history, self.previous = self.previous, self.frame(obs)
        return (
            {**obs, "memory": np.concatenate((obs["memory"], history))},
            reward,
            terminated,
            truncated,
            info,
        )
