"""Stagnation prefixes with seed quotas, failed examples and explicit resolution."""

import copy
from collections import deque

import numpy as np

from ather_exploration.training.p3_completion import _observation_digest
from ather_exploration.training.public_route import public_route


class RecoveryArchive:
    def __init__(self, config, rng):
        self.config, self.rng = config, rng
        self.pools = {task: [[], []] for task in ("P3b", "P3c")}
        self.level = 0  # Compatibility only; no mastery/unlock curriculum.
        self.replay_steps = 0
        self.clock = 0
        self.probed = []

    def choose(self, task):
        requested = self.rng.random() < self.config.restart_probability
        if not requested:
            return None, False
        available = [pool for pool in self.pools.get(task, []) if pool]
        if not available:
            return None, True
        pool = available[int(self.rng.integers(len(available)))]
        self.clock += 1
        # Half uniform, half uncertainty + staleness. Repeated unresolved items
        # lose uncertainty priority but never disappear from uniform sampling.
        weights = np.array(
            [
                1 / (1 + p["attempts"])
                + 4
                * ((p["resolved"] + 1) / (p["attempts"] + 2))
                * (1 - (p["resolved"] + 1) / (p["attempts"] + 2))
                + min(1, (self.clock - p["last_used"]) / 64)
                for p in pool
            ]
        )
        weights = 0.5 / len(pool) + 0.5 * weights / weights.sum()
        item = pool[int(self.rng.choice(len(pool), p=weights))]
        item["last_used"] = self.clock
        return copy.deepcopy(item), True

    def offer(self, item):
        task = item["task"]
        if task not in self.pools or item["source_task"] != task:
            raise ValueError("Recovery archive only accepts its active source task")
        kind = 0 if item["kind"] == "poi" else 1
        pool = self.pools[task][kind]
        if any(p["seed"] == item["seed"] for p in pool):
            return  # One prefix per seed and target kind.
        item = {**copy.deepcopy(item), "attempts": 0, "resolved": 0, "last_used": self.clock}
        if len(pool) >= self.config.pool_per_kind:
            # Evict a random item, rather than discarding all failed examples.
            pool.pop(int(self.rng.integers(len(pool))))
        pool.append(item)

    def outcome(self, item, resolved):
        for pool in self.pools[item["task"]]:
            for original in pool:
                if (
                    original["observation"] == item["observation"]
                    and original["seed"] == item["seed"]
                ):
                    original["attempts"] += 1
                    original["resolved"] += int(resolved)
                    return

    def replay(self, env, item):
        observation, info = env.reset()
        for action in item["actions"]:
            observation, _, terminated, truncated, info = env.step(action)
            self.replay_steps += 1
            if terminated or truncated:
                raise ValueError("Recovery prefix ended before boundary")
        if _observation_digest(observation) != item["observation"]:
            raise ValueError("Recovery prefix observation hash mismatch")
        return observation, info

    def state(self):
        return {
            "protocol": "stagnation-v1",
            "pools": copy.deepcopy(self.pools),
            "clock": self.clock,
            "probed": list(self.probed),
            "replay_steps": self.replay_steps,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def restore(self, state):
        if state.get("protocol") != "stagnation-v1":
            raise ValueError("Incompatible recovery archive")
        self.pools = copy.deepcopy(state["pools"])
        self.clock, self.probed = state["clock"], list(state["probed"])
        self.replay_steps = state["replay_steps"]
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])


class StagnationCapture:
    def __init__(self, config, task, seed, observation):
        self.config, self.task, self.seed = config, task, seed
        self.actions = []
        self.stale = 0
        self.history = deque(maxlen=config.lookback + 1)
        self.history.append((0, _observation_digest(observation), public_route(observation)))
        self.captured = set()
        self.last_floor = float(observation["memory"][2].sum())
        self.last_activated = float(observation["memory"][4].sum())

    def step(self, action, observation, horizon):
        self.actions.append(int(action))
        floor = float(observation["memory"][2].sum())
        activated = float(observation["memory"][4].sum())
        self.stale = (
            self.stale + 1 if floor == self.last_floor and activated == self.last_activated else 0
        )
        self.last_floor, self.last_activated = floor, activated
        self.history.append(
            (len(self.actions), _observation_digest(observation), public_route(observation))
        )
        if (
            self.stale < self.config.stagnation_steps
            or horizon - len(self.actions) < self.config.minimum_remaining
        ):
            return None
        length, digest, route = self.history[0]
        if not route or route["kind"] in self.captured:
            return None
        self.captured.add(route["kind"])
        return {
            "task": self.task,
            "source_task": self.task,
            "seed": int(self.seed),
            "actions": self.actions[:length],
            "observation": digest,
            "kind": route["kind"],
        }


class ResolveTracker:
    def __init__(self, observation):
        route = public_route(observation)
        self.kind = route["kind"] if route else None
        self.goals = set(route["goals"]) if route else set()
        self.initial_floor = float(observation["memory"][2].sum())
        self.visited_frontier = False
        self.steps = 0
        self.resolved = False

    def step(self, observation, *, complete=False):
        self.steps += 1
        if self.steps > 64 or self.resolved:
            return self.resolved
        m = observation["memory"]
        if self.kind == "poi":
            self.resolved = any(m[4, r, c] > 0 for r, c in self.goals)
        elif self.kind == "frontier":
            self.visited_frontier |= any(m[7, r, c] > 0 for r, c in self.goals)
            self.resolved = self.visited_frontier and (
                float(m[2].sum()) - self.initial_floor >= 4 or complete
            )
        return bool(self.resolved)
