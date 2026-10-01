"""Explicit relative memory and Gym wrapper, updated exclusively from public history."""

import gymnasium as gym
import numpy as np

from ather_exploration.config import ObservationConfig
from ather_exploration.environment.public_state import public_cells
from ather_exploration.environment.visibility import disk_offsets
from ather_exploration.schema import observation_space
from ather_exploration.types import Motion, PublicObservation, PublicTransition


class PublicMemory:
    def __init__(self, config: ObservationConfig):
        self.config = config
        self._ready = False

    def _cells(self, local, position):
        if local.shape != self.config.local_shape:
            raise ValueError("local shape disagrees with observation config")
        cells = public_cells(local, position)
        limit = self.config.map_capacity - 1
        if any(abs(x) > limit or abs(y) > limit for x, y in [position, *cells]):
            raise ValueError("Public memory capacity exceeded; no clipping is allowed")
        return cells

    def reset(self, local: np.ndarray, horizon: int) -> None:
        if type(horizon) is not int or not 1 <= horizon <= self.config.horizon_capacity:
            raise ValueError("horizon exceeds observation capacity")
        cells = self._cells(local, (0, 0))
        side = self.config.memory_shape[1]
        self._base = np.zeros(self.config.memory_shape, dtype=np.float32)
        self._visits = np.zeros((side, side), dtype=np.int32)
        self._last_seen = np.full((side, side), -1, dtype=np.int32)
        self._position = (0, 0)
        self._tick = 0
        self._horizon = horizon
        self._event = None
        self._local = local.copy()
        center = self.config.map_capacity - 1
        self._visits[center, center] = 1
        self._ingest(cells)
        self._ready = True

    def _ingest(self, cells):
        center = self.config.map_capacity - 1
        self._base[7:9].fill(0)
        x, y = self._position
        self._base[7, y + center, x + center] = 1
        for (x, y), (wall, floor, pending, active, monster) in cells.items():
            row, col = y + center, x + center
            self._base[0:5, row, col] = (1, wall, floor, pending, active)
            self._base[8, row, col] = 1
            self._base[9, row, col] = monster
            self._last_seen[row, col] = self._tick

    def update(self, local: np.ndarray, event: PublicTransition) -> None:
        if not self._ready:
            raise RuntimeError("Memory must be reset first")
        if self._tick >= self._horizon or (self._event is not None and self._event.died):
            raise RuntimeError("Memory episode ended; reset first")
        position = tuple(a + b for a, b in zip(self._position, event.actual_delta, strict=True))
        cells = self._cells(local, position)  # Reject overflow before mutation.
        self._position = position
        self._tick += 1
        self._event = event
        self._local = local.copy()
        if event.motion is Motion.MOVED:
            center = self.config.map_capacity - 1
            self._visits[position[1] + center, position[0] + center] += 1
        self._ingest(cells)

    def observation(self) -> PublicObservation:
        if not self._ready:
            raise RuntimeError("Memory must be reset first")
        cap = self.config.horizon_capacity
        memory = self._base.copy()
        memory[5] = self._visits > 0
        memory[6] = np.log1p(self._visits) / np.log1p(cap + 1)
        seen = self._last_seen >= 0
        memory[10, seen] = np.log1p(self._tick - self._last_seen[seen]) / np.log1p(cap)
        state = np.zeros(17, dtype=np.float32)
        state[10] = (self._horizon - self._tick) / cap
        state[11] = self._horizon / cap
        state[16] = self._tick == 0
        if self._event is not None:
            e = self._event
            state[int(e.action)] = 1
            state[5 + (Motion.MOVED, Motion.WALL_BLOCKED, Motion.WAITED).index(e.motion)] = 1
            state[8:10] = e.actual_delta
            state[12] = e.new_floor / len(disk_offsets(self.config.radius))
            state[13] = e.new_poi / self.config.poi_capacity
            state[14:16] = (e.activated, e.died)
        return {"local": self._local.copy(), "memory": memory, "state": state}


class PublicMemoryWrapper(gym.Wrapper):
    """Create once per env, before vectorization; never read hidden state to build memory."""

    def __init__(self, env, config: ObservationConfig, horizon: int):
        super().__init__(env)
        self.memory = PublicMemory(config)
        self.horizon = horizon  # Public task budget, not inferred from hidden scenario.
        self.observation_space = observation_space(config)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.memory.reset(obs["local"], self.horizon)
        return self.memory.observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.memory.update(obs["local"], PublicTransition(**info["transition"]))
        return self.memory.observation(), reward, terminated, truncated, info

    def gen_obs(self):
        return self.memory.observation()
