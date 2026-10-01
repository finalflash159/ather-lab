"""Fair baseline adapters: observation/history only, never an env or scenario."""

import heapq
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import Field

from ather_exploration.config import FrozenConfig
from ather_exploration.environment.visibility import blocker_offsets, disk_offsets
from ather_exploration.types import ACTION_DELTAS, Action


class BaselineConfig(FrozenConfig):
    version: Literal["1"] = "1"
    poi_bonus: float = Field(default=25.0, ge=0, allow_inf_nan=False)
    stale_cost: float = Field(default=4.0, ge=0, allow_inf_nan=False)
    cooldown: int = Field(default=8, ge=1)
    blocked_limit: int = Field(default=8, ge=1)
    tie_tolerance: float = Field(default=1e-9, gt=0, allow_inf_nan=False)


@dataclass
class PlannerState:
    target: tuple[int, int] | None = None
    tick: int = 0
    blocked_steps: int = 0
    cooldown_until: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    information_key: bytes = b""
    information_cache: dict = field(default_factory=dict)


class RandomAgent:
    name = "random"

    def act(self, observation, state, *, deterministic, action_rng):
        # Uniform remains stochastic in the primary B0 protocol; deterministic is
        # an interface option for learned agents, not a hidden baseline mask.
        state.episode_start = False
        return Action(int(action_rng.integers(5))), state


def _neighbors(pos):
    return tuple((pos[0] + dx, pos[1] + dy) for dx, dy in ACTION_DELTAS[:4])


class PublicMap:
    def __init__(self, obs, config):
        self.tolerance = config.tie_tolerance
        self.memory = obs["memory"]
        m = self.memory
        positions = np.argwhere(m[7] > 0.5)
        if len(positions) != 1:
            raise ValueError("Public memory needs exactly one agent position")
        row, col = positions[0]
        self.position = (int(col), int(row))
        self.floor = {(int(c), int(r)) for r, c in np.argwhere(m[2] > 0.5)}
        self.wall = {(int(c), int(r)) for r, c in np.argwhere(m[1] > 0.5)}
        self.seen = self.floor | self.wall
        self.visible = {(int(c), int(r)) for r, c in np.argwhere(m[8] > 0.5)}
        self.monsters = {(int(c), int(r)) for r, c in np.argwhere((m[9] > 0.5) & (m[8] > 0.5))}
        stale = {(int(c), int(r)) for r, c in np.argwhere((m[9] > 0.5) & (m[8] < 0.5))}
        self.stale = {
            p: config.stale_cost
            / (1 + round(math.expm1(float(m[10, p[1], p[0]]) * math.log1p(1024))))
            for p in stale
        }
        self.pois = {(int(c), int(r)) for r, c in np.argwhere(m[3] > 0.5)}
        self.radius = obs["local"].shape[1] // 2

    def paths(self, start, *, reverse=False):
        costs = {start: 0.0}
        parents = {}
        queue = [(0.0, 0, start)]
        serial = 0
        while queue:
            distance, _, pos = heapq.heappop(queue)
            if distance > costs[pos]:
                continue
            for action, target in enumerate(_neighbors(pos)):
                if target not in self.floor or target in self.monsters:
                    continue
                cost = distance + 1 + self.stale.get(pos if reverse else target, 0.0)
                if cost < costs.get(target, math.inf) - self.tolerance:
                    costs[target] = cost
                    parents[target] = (pos, action)
                    serial += 1
                    heapq.heappush(queue, (cost, serial, target))
        return costs, parents

    def information(self, pos):
        result = 0
        for dx, dy in disk_offsets(self.radius):
            target = (pos[0] + dx, pos[1] + dy)
            if target in self.seen:
                continue
            if not (
                0 <= target[0] < self.memory.shape[2] and 0 <= target[1] < self.memory.shape[1]
            ):
                continue
            if not any(
                (pos[0] + bx, pos[1] + by) in self.wall for bx, by in blocker_offsets(dx, dy)
            ):
                result += 1
        return result

    def destination(self, action):
        dx, dy = ACTION_DELTAS[action]
        p = (self.position[0] + dx, self.position[1] + dy)
        return self.position if p in self.wall else p

    def safety(self, action):
        target = self.destination(action)
        immediate = target in self.monsters
        arrivals = sum(p in self.monsters for p in _neighbors(target))
        unknown = sum(p not in self.wall and p not in self.visible for p in _neighbors(target))
        safe = (
            target in self.floor
            and target in self.visible
            and not immediate
            and not arrivals
            and not unknown
        )
        return safe, (int(immediate), arrivals, unknown, self.stale.get(target, 0.0))


class FrontierAgent:
    name = "frontier"

    def __init__(self, config=None):
        self.config = config or BaselineConfig()

    def act(self, observation, state, *, deterministic, action_rng):
        if (
            state.episode_start
            or observation["state"][16] > 0.5
            or not isinstance(state.planner, PlannerState)
        ):
            state.planner = PlannerState()
        state.episode_start = False
        planner = state.planner
        public = PublicMap(observation, self.config)
        origin = public.position
        planner.tick += 1
        planner.cooldown_until = {
            p: t for p, t in planner.cooldown_until.items() if t > planner.tick
        }
        no_progress = not np.any(observation["state"][12:15])
        planner.blocked_steps = (
            planner.blocked_steps + 1 if observation["state"][6] > 0.5 and no_progress else 0
        )
        costs, parents = public.paths(origin)
        frontiers = {p for p in public.floor if any(n not in public.seen for n in _neighbors(p))}
        goals = (frontiers | public.pois) - {origin}
        goals = {p for p in goals if p in costs}
        if planner.target is not None and (
            planner.target not in goals or planner.blocked_steps >= self.config.blocked_limit
        ):
            planner.cooldown_until[planner.target] = planner.tick + self.config.cooldown
            planner.target = None
            planner.blocked_steps = 0
        if planner.target is None:
            key = observation["memory"][:2].tobytes()
            if key != planner.information_key:
                planner.information_key = key
                planner.information_cache.clear()
            best = None
            for goal in sorted(goals, key=lambda p: (p[1], p[0])):
                if goal in planner.cooldown_until:
                    continue
                if goal not in planner.information_cache:
                    planner.information_cache[goal] = public.information(goal)
                score = (
                    planner.information_cache[goal] + self.config.poi_bonus * (goal in public.pois)
                ) / max(1, costs[goal])
                candidate = (score, costs[goal], goal in public.pois, goal)
                if (
                    best is None
                    or score > best[0] + self.config.tie_tolerance
                    or (
                        abs(score - best[0]) <= self.config.tie_tolerance
                        and (
                            costs[goal] < best[1] - self.config.tie_tolerance
                            or (
                                abs(costs[goal] - best[1]) <= self.config.tie_tolerance
                                and (-(goal in public.pois), goal[1], goal[0])
                                < (-best[2], best[3][1], best[3][0])
                            )
                        )
                    )
                ):
                    best = candidate
            if best is not None:
                planner.target = best[3]
        planned = None
        if planner.target is not None:
            position = planner.target
            while parents[position][0] != origin:
                position = parents[position][0]
            planned = parents[position][1]
        safety = {a: public.safety(a) for a in range(5)}
        safe = [a for a in range(5) if safety[a][0]]
        # Reverse edge charges the original forward destination's stale cost.
        remaining = (
            public.paths(planner.target, reverse=True)[0]
            if planner.target is not None and planned not in safe
            else {}
        )

        def distance(action):
            return remaining.get(public.destination(action), math.inf)

        if planned in safe:
            chosen = planned
            reason = "planned_safe"
        else:
            reducing = [
                a
                for a in safe
                if planner.target is not None
                and distance(a) < costs[planner.target] - self.config.tie_tolerance
            ]
            if reducing:
                chosen = min(reducing, key=lambda a: (distance(a), a))
                reason = "safe_progress"
            elif 4 in safe:
                chosen = 4
                reason = "safe_wait"
            elif safe:
                chosen = min(safe, key=lambda a: (distance(a), a))
                reason = "safe_alternative"
            else:
                chosen = min(range(5), key=lambda a: (*safety[a][1], distance(a), a))
                reason = "uncertain_least_risk"
        planner.diagnostics = {
            "target": list(planner.target) if planner.target else None,
            "planned_action": planned,
            "reason": reason,
            "uncertain_action": not safety[chosen][0],
            "certified_safe_actions": safe,
            "known_floor_count": len(public.floor),
        }
        return Action(chosen), state


def make_baseline(name, config=None):
    if name == "random":
        return RandomAgent()
    if name == "frontier":
        return FrontierAgent(config)
    raise ValueError("Unknown baseline; choose random or frontier")
