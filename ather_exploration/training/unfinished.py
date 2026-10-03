"""Training-only restart curriculum using replayable public-history prefixes.

No policy targets, hidden terrain, POI coordinates, or expert actions are used.
A bounded per-worker archive is populated only by ordinary P3b episodes.
"""

import copy
import hashlib
from collections import deque

import numpy as np

from ather_exploration.environment.frontier import frontier_mask

BANDS = ((2, 4), (5, 8), (9, 16))


def observation_digest(obs):
    h = hashlib.sha256()
    for key in sorted(obs):
        a = np.ascontiguousarray(obs[key])
        h.update(key.encode())
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def frontier_distance(obs):
    """Shortest four-connected known-floor distance, in memory coordinates."""
    memory = obs["memory"]
    positions = np.argwhere(memory[7] > 0)
    if len(positions) != 1:
        raise ValueError("Expected one public agent position")
    start = tuple(int(x) for x in positions[0])
    frontiers = frontier_mask(memory) > 0
    floor = (memory[2] > 0) & ~(memory[1] > 0)
    queue, seen = deque([(start, 0)]), {start}
    while queue:
        (y, x), distance = queue.popleft()
        if frontiers[y, x]:
            return distance
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= ny < floor.shape[0]
                and 0 <= nx < floor.shape[1]
                and floor[ny, nx]
                and (ny, nx) not in seen
            ):
                seen.add((ny, nx))
                queue.append(((ny, nx), distance + 1))
    return None


class PrefixArchive:
    def __init__(self, config, rng):
        self.config, self.rng = config, rng
        self.pools = [[] for _ in BANDS]
        self.level = 0
        self.replay_steps = 0

    def offer(self, seed, actions, obs, initial_floor, captured, horizon=256):
        if (
            len(captured) == len(BANDS)
            or len(actions) < 16
            or len(actions) > horizon - self.config.minimum_remaining
            or float(obs["memory"][2].sum()) <= initial_floor
        ):
            return
        distance = frontier_distance(obs)
        band = next(
            (
                i
                for i, (lo, hi) in enumerate(BANDS)
                if distance is not None and lo <= distance <= hi
            ),
            None,
        )
        if band is None or band in captured:
            return
        captured.add(band)
        # One example per map/band, so long episodes cannot monopolize the pool.
        pool = self.pools[band]
        if any(item["seed"] == seed for item in pool):
            return
        item = {
            "seed": int(seed),
            "actions": list(actions),
            "observation": observation_digest(obs),
            "distance": distance,
        }
        if len(pool) < self.config.pool_per_band:
            pool.append(item)
        else:
            pool[int(self.rng.integers(len(pool)))] = item

    def choose(self):
        requested = bool(self.rng.random() < self.config.restart_probability)
        if not requested:
            return None, False
        # Retain easier examples when harder bands unlock; never silently use harder ones.
        available = [i for i in range(self.level + 1) if self.pools[i]]
        if not available:
            return None, True
        band = int(self.rng.choice(available))
        return copy.deepcopy(self.pools[band][int(self.rng.integers(len(self.pools[band])))]), True

    def replay(self, env, item):
        obs, info = env.reset()
        for action in item["actions"]:
            obs, _, terminated, truncated, info = env.step(action)
            self.replay_steps += 1
            if terminated or truncated:
                raise ValueError("Restart prefix terminated before its boundary")
        if observation_digest(obs) != item["observation"]:
            raise ValueError("Restart prefix observation differs; refusing stale history")
        return obs, info

    def state(self):
        return {
            "pools": copy.deepcopy(self.pools),
            "level": self.level,
            "replay_steps": self.replay_steps,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def restore(self, state):
        self.pools = copy.deepcopy(state["pools"])
        self.level = state["level"]
        self.replay_steps = state["replay_steps"]
        self.rng.bit_generator.state = state["rng"]
