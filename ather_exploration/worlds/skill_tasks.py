"""Versioned skill tasks using the existing MiniGrid dynamics and public sensor."""

from functools import lru_cache

import gymnasium as gym
import numpy as np

from ather_exploration.config import ObservationConfig, RewardConfig
from ather_exploration.environment.env import ExplorationEnv
from ather_exploration.environment.memory import PublicMemoryWrapper
from ather_exploration.types import Scenario, ValidatorStatus
from ather_exploration.worlds.topology import distances
from ather_exploration.worlds.validation import validate_scenario

TASKS = ("P1a", "P1b", "P2a", "P2b", "P3", "P4a", "P4b")
HORIZONS = dict(zip(TASKS, (32, 32, 96, 96, 256, 128, 256), strict=True))


def wall_mask(observation):
    local = observation["local"]
    r = local.shape[1] // 2
    return np.array(
        [
            not local[1, r - 1, r],
            not local[1, r + 1, r],
            not local[1, r, r + 1],
            not local[1, r, r - 1],
            True,
        ],
        dtype=bool,
    )


@lru_cache(maxsize=2048)
def skill_scenario(task, seed):
    if task not in TASKS or type(seed) is not int or seed < 0:
        raise ValueError("Invalid skill task/seed")
    rng = np.random.default_rng(seed)
    if task == "P4a":
        n = int(rng.choice((9, 11, 13)))
        grid = np.full((n, n), "#", dtype="<U1")
        y = n // 2
        x = int(rng.integers(3, n - 3))
        grid[y, 1:-1] = "."
        grid[y - 1 : y + 2, x] = "."
        spawn = (x - 1, y)
        poi = (x + int(rng.integers(1, 3)), y)
        route = ((x, y - 1), (x, y), (x, y + 1))
        turns = seed % 4

        def rotate(p):
            for _ in range(turns):
                p = (p[1], n - 1 - p[0])
            return p

        scenario = Scenario(
            tuple("".join(row) for row in np.rot90(grid, turns)),
            rotate(spawn),
            (rotate(poi),),
            (tuple(rotate(p) for p in route),),
            ((0, 2, 4)[seed % 3],),
            128,
            seed,
            skill_task=task,
        )
        if (
            validate_scenario(scenario, max_expansions=300000).status
            is not ValidatorStatus.VALIDATED
        ):
            raise ValueError("Invalid patrol crossing skill")
        return scenario
    for _ in range(4000):
        validation = seed >= 100000
        n = (
            int(rng.choice((7, 9)))
            if task.startswith("P1")
            else (11 if validation else int(rng.choice((9, 13))))
            if task.startswith("P2") or task == "P4a"
            else (17 if validation else int(rng.choice((13, 21))))
        )
        grid = np.full((n, n), ".", dtype="<U1")
        grid[[0, -1], :] = "#"
        grid[:, [0, -1]] = "#"
        if task.startswith("P2"):
            x = n // 2
            lo = int(rng.integers(2, n - 4))
            grid[lo : lo + 3, x] = "#"
        elif task in ("P3", "P4b"):
            grid[1:-1, n // 2] = "#"
            grid[int(rng.integers(2, n - 2)), n // 2] = "."
            grid[n // 2, 1 : n // 2] = "#"
            grid[n // 2, int(rng.integers(1, n // 2))] = "."
        grid = np.rot90(grid, int(rng.integers(4)))
        floors = [(int(x), int(y)) for y, x in np.argwhere(grid == ".")]
        spawn = floors[int(rng.integers(len(floors)))]
        dist = distances(tuple("".join(row) for row in grid), [spawn])
        candidates = [p for p in floors if p != spawn and p in dist]
        if not candidates:
            continue
        poi = candidates[int(rng.integers(len(candidates)))]
        d = dist[poi]
        if task.startswith("P1"):
            dx, dy = poi[0] - spawn[0], poi[1] - spawn[1]
            direction = (2 if dx > 0 else 3) if abs(dx) >= abs(dy) else (1 if dy > 0 else 0)
            if direction != seed % 4:
                continue
        if task == "P1a" and not (1 <= d <= 2 and (poi[0] == spawn[0] or poi[1] == spawn[1])):
            continue
        if task == "P1b" and not 2 <= d <= 4:
            continue
        if task.startswith("P2") and not 3 <= d <= 12:
            continue
        if task == "P2a":
            axis_path = [
                (x, spawn[1]) for x in range(min(spawn[0], poi[0]), max(spawn[0], poi[0]) + 1)
            ]
            axis_path += [
                (poi[0], y) for y in range(min(spawn[1], poi[1]), max(spawn[1], poi[1]) + 1)
            ]
            if not any(grid[y, x] == "#" for x, y in axis_path):
                continue
        pois = (poi,)
        if task in ("P3", "P4b") and seed % 2:
            other = [p for p in candidates if p != poi]
            pois += (other[int(rng.integers(len(other)))],)
        routes, phases = (), ()
        if task.startswith("P4"):
            choices = []
            for x, y in floors:
                route = tuple((x + i, y) for i in range(3))
                if all(p in floors for p in route) and spawn not in route:
                    choices.append(route)
            if not choices:
                continue
            route = choices[int(rng.integers(len(choices)))]
            routes, phases = (route,), (int(rng.integers(8)),)
        scenario = Scenario(
            tuple("".join(row) for row in grid),
            spawn,
            pois,
            routes,
            phases,
            HORIZONS[task],
            seed,
            skill_task=task,
        )
        from ather_exploration.environment.dynamics import make_grid
        from ather_exploration.environment.visibility import sense
        from ather_exploration.types import EpisodeState

        local = sense(make_grid(scenario), scenario, EpisodeState.from_scenario(scenario), 4)
        visible = bool(local[3].any())
        if visible != (task in ("P1a", "P1b", "P2a", "P4a")):
            continue
        # P4a must expose the threat, not accidentally create an empty-room lesson.
        if task == "P4a" and not local[5].any():
            continue
        if (
            routes
            and validate_scenario(scenario, max_expansions=300000).status
            is not ValidatorStatus.VALIDATED
        ):
            continue
        return scenario
    raise ValueError(f"Could not generate valid {task} seed {seed}; no silent fallback")


class SkillEnv(gym.Wrapper):
    def __init__(self, task, seed, reward=None, first_visit=False):
        self.task, self.task_seed = task, seed
        scenario = skill_scenario(task, seed)
        cfg = ObservationConfig()
        super().__init__(
            PublicMemoryWrapper(
                ExplorationEnv(
                    scenario, observation_config=cfg, reward_config=reward or RewardConfig()
                ),
                cfg,
                scenario.horizon,
            )
        )
        self.first_visit = first_visit
        self.finished = False

    def reset(self, *, seed=None, options=None):
        # Fixed bank record: reset does not silently select a different task.
        obs, info = self.env.reset(seed=seed, options=options)
        self.position = (0, 0)
        self.visited = {self.position}
        self.remaining_bonus = 0.10
        self.finished = False
        self.last_obs = obs
        return obs, info

    def action_masks(self):
        return wall_mask(self.last_obs)

    def step(self, action):
        if self.finished:
            raise RuntimeError("Skill episode ended; reset first")
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs
        e = info["transition"]
        self.position = tuple(a + b for a, b in zip(self.position, e["actual_delta"], strict=True))
        bonus = 0.0
        if any(e["actual_delta"]) and self.position not in self.visited:
            if self.first_visit:
                bonus = min(0.002, self.remaining_bonus)
                self.remaining_bonus -= bonus
            self.visited.add(self.position)
        core = self.unwrapped
        snap = core.evaluator_snapshot()
        all_pois = len(snap.activated_pois) == len(core.scenario.pois)
        success = all_pois and not e["died"]
        early = success and self.task != "P4b"
        self.finished = bool(terminated or truncated or early)
        info = {
            **info,
            "skill": {
                "task": self.task,
                "success": success,
                "intrinsic_reward": bonus,
                "task_reward": reward,
                "early_success": early,
            },
        }
        return obs, reward + bonus, bool(terminated or early), truncated, info


def make_skill_env(task, seed, reward=None, first_visit=False):
    return SkillEnv(task, seed, reward, first_visit)


@lru_cache(maxsize=64)
def skill_pool(task, count, validation=False):
    """Unique configurations with content-hash splitting, not merely different RNG seeds."""
    from dataclasses import asdict

    from ather_exploration.worlds.scenarios import digest

    records = []
    seen = set()
    start = 100000 if validation else 0
    for seed in range(start, start + max(10000, count * 200)):
        scenario = skill_scenario(task, seed)
        payload = asdict(scenario)
        payload.pop("seed")
        payload.pop("skill_task")
        identity = digest(payload)
        if (int(identity[:8], 16) % 5 == 0) != validation or identity in seen:
            continue
        seen.add(identity)
        records.append({"seed": seed, "identity": identity})
        if len(records) == count:
            return tuple((r["seed"], r["identity"]) for r in records)
    raise ValueError(f"Insufficient distinct {task} configurations; reduce pool count explicitly")


def build_skill_suite(config, output):
    from pathlib import Path

    from ather_exploration.worlds.scenarios import implementation_id, write_record

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "state": "BUILDING",
        "source_revision": implementation_id(),
        "version": 2,
        "tasks": {},
    }
    write_record(root / "manifest.json", manifest)
    try:
        for task in TASKS:
            train = skill_pool(task, config.skills.train_count)
            val = skill_pool(task, config.skills.validation_count, True)
            if {x[1] for x in train} & {x[1] for x in val}:
                raise ValueError("Skill split overlap")
            manifest["tasks"][task] = {"train": train, "validation": val}
            print(f"[skills-bank] {task}: train={len(train)} validation={len(val)}", flush=True)
        manifest["state"] = "READY"
        write_record(root / "manifest.json", manifest, replace=True)
    except BaseException:
        manifest["state"] = "FAILED"
        write_record(root / "manifest.json", manifest, replace=True)
        raise
    return manifest
