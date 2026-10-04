"""Versioned skill tasks using the existing MiniGrid dynamics and public sensor."""

from dataclasses import replace
from functools import lru_cache

import gymnasium as gym
import numpy as np

from ather_exploration.config import ObservationConfig, RewardConfig
from ather_exploration.environment.env import ExplorationEnv
from ather_exploration.environment.memory import PublicMemoryWrapper
from ather_exploration.types import Scenario, ValidatorStatus
from ather_exploration.worlds.topology import distances
from ather_exploration.worlds.validation import validate_scenario

TASKS = ("P1a", "P1b", "P2a", "P2b", "P2c", "P3", "P4a", "P4b")
P3_TASKS = ("P3a", "P3b", "P3c")
HORIZONS = dict(zip(TASKS, (32, 32, 96, 96, 192, 256, 128, 256), strict=True))
HORIZONS.update(dict.fromkeys(P3_TASKS, 256))
TASKS += P3_TASKS
# Easy geometry in P5a uses a small-target budget. Actual target banks retain their H.
PHASE_HORIZONS = {**HORIZONS, "P5a": 256, "P5b": 256, "P4c": 256}
EARLY_SUCCESS_PHASES = frozenset(("P1a", "P1b", "P2a", "P2b", "P4a"))


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
    if task in P3_TASKS:
        from ather_exploration.worlds.p3_tasks import p3_scenario

        return p3_scenario(task, seed)
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
    def __init__(
        self,
        task,
        seed,
        reward=None,
        first_visit=False,
        step_cost=0.0,
        *,
        phase=None,
        wall_penalty=0.0,
        horizon=None,
        visit_bonus=0.002,
        visit_cap=0.10,
        room_exploration=0.0,
        success_bonus=0.0,
        scenario_override=None,
        all_pois_required=False,
    ):
        self.task, self.task_seed = task, seed
        self.phase = phase or task
        if self.phase not in PHASE_HORIZONS:
            raise ValueError(f"Unknown curriculum phase: {self.phase}")
        scenario = replace(
            scenario_override or skill_scenario(task, seed),
            horizon=horizon or PHASE_HORIZONS[self.phase],
        )
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
        self.all_pois_required = all_pois_required
        self.first_visit = first_visit
        self.visit_bonus, self.visit_cap = visit_bonus, visit_cap
        self.room_exploration = room_exploration
        self.success_bonus = success_bonus
        self.step_cost = step_cost
        self.wall_penalty = wall_penalty
        self.finished = False

    def reset(self, *, seed=None, options=None):
        # Fixed bank record: reset does not silently select a different task.
        obs, info = self.env.reset(seed=seed, options=options)
        self.position = (0, 0)
        self.visited = {self.position}
        self.remaining_bonus = self.visit_cap
        self.finished = False
        self.last_obs = obs
        self.poi_seen = bool(obs["memory"][3:5].any())
        self.room_potential = self._room_potential(obs)
        return obs, info

    def _room_potential(self, observation):
        if (
            self.phase not in ("P3b", "P3c", "P4b", "P4c")
            or self.room_exploration <= 0
            or not self.unwrapped.scenario.room_labels
        ):
            return 0.0
        from ather_exploration.worlds.p3_tasks import (
            room_coverage_fractions,
            room_exploration_potential,
        )

        return room_exploration_potential(
            room_coverage_fractions(self.unwrapped.scenario, observation["memory"])
        )

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
        room_bonus = 0.0
        if any(e["actual_delta"]) and self.position not in self.visited:
            if self.first_visit:
                bonus = min(self.visit_bonus, self.remaining_bonus)
                self.remaining_bonus -= bonus
            self.visited.add(self.position)
        core = self.unwrapped
        # Reward the discovery transition, then latch search completion for the episode.
        if self.phase == "P2b" and self.poi_seen:
            reward -= core.reward_config.area * e["new_floor"]
            area_reward = 0.0
        else:
            area_reward = core.reward_config.area * e["new_floor"]
        self.poi_seen = self.poi_seen or bool(obs["memory"][3:5].any())
        next_room_potential = self._room_potential(obs)
        if self.room_exploration > 0:
            room_bonus = self.room_exploration * max(0.0, next_room_potential - self.room_potential)
        self.room_potential = next_room_potential
        wall_cost = self.wall_penalty * bool(obs["state"][6])
        snap = core.evaluator_snapshot()
        all_pois = len(snap.activated_pois) == len(core.scenario.pois)
        alive = not e["died"]
        completed_pois = (
            bool(snap.activated_pois)
            if self.phase == "P4b" and not self.all_pois_required
            else all_pois
        )
        early_phase = self.phase in EARLY_SUCCESS_PHASES
        success = (
            completed_pois and alive and (early_phase or snap.step_count == core.scenario.horizon)
        )
        early = success and early_phase
        completion_bonus = (
            self.success_bonus if success and self.phase in ("P4b", "P4c") else 0.0
        )
        self.finished = bool(terminated or truncated or early)
        info = {
            **info,
            "skill": {
                "task": self.phase,
                "phase": self.phase,
                "source_task": self.task,
                "success": success,
                "intrinsic_reward": bonus,
                "task_reward": reward,
                "reward_components": {
                    "area": area_reward,
                    "discovery": core.reward_config.discovery * e["new_poi"],
                    "activation": core.reward_config.activation * e["activated"],
                    "death": -core.reward_config.death * e["died"],
                    "intrinsic": bonus,
                    "room_exploration": room_bonus,
                    "completion_bonus": completion_bonus,
                    "step_cost": -self.step_cost,
                    "wall_penalty": -wall_cost,
                },
                "early_success": early,
            },
        }
        return (
            obs,
            reward + bonus + room_bonus + completion_bonus - self.step_cost - wall_cost,
            bool(terminated or early),
            truncated,
            info,
        )


def make_skill_env(task, seed, reward=None, first_visit=False, step_cost=0.0, *, phase=None):
    return SkillEnv(task, seed, reward, first_visit, step_cost, phase=phase)


def _configured_skill_env(task, seed, skills, *, phase=None, threat_lesson=None):
    """Sample geometry by task; apply the active phase objective in every caller.

    Phase is environment configuration only, never a policy observation.
    """
    phase = phase or task
    if skills.p4.enabled and task.startswith("P4"):
        from ather_exploration.worlds.p4_tasks import p4_scenario

        p3 = skills.p3_reward
        scenario = p4_scenario(task, seed)
        if threat_lesson is not None:
            from ather_exploration.training.threat_lessons import lesson_scenario

            scenario = lesson_scenario(scenario, threat_lesson, timing=skills.p4.timing)
        return SkillEnv(
            task,
            seed,
            RewardConfig(
                area=0.0 if phase == "P4a" else p3.area,
                discovery=0.0 if phase == "P4a" else p3.discovery,
                activation=p3.activation,
                death=skills.death,
            ),
            phase=phase,
            wall_penalty=p3.wall_penalty,
            room_exploration=skills.p4.room_exploration if phase != "P4a" else 0.0,
            success_bonus=skills.p4.success_bonus,
            first_visit=phase != "P4a" and skills.p3_visit_bonus > 0,
            visit_bonus=skills.p3_visit_bonus,
            visit_cap=skills.p3_visit_cap,
            scenario_override=scenario,
            step_cost=skills.p4.step_cost,
            all_pois_required=True,
        )
    reward = RewardConfig(activation=skills.activation, death=skills.death)
    if phase in ("P1a", "P1b"):
        p1 = skills.p1_reward
        reward = RewardConfig(
            area=p1.area, discovery=p1.discovery, activation=p1.activation, death=skills.death
        )
        return make_skill_env(task, seed, reward, False, p1.step_cost, phase=phase)
    if phase in ("P2a", "P2b", "P2c"):
        p2 = skills.p2_reward
        reward = RewardConfig(
            area=0.0 if phase == "P2a" else p2.area,
            discovery=0.0 if phase == "P2a" else p2.discovery,
            activation=p2.activation,
            death=skills.death,
        )
        return SkillEnv(
            task,
            seed,
            reward,
            False,
            p2.step_cost,
            phase=phase,
            wall_penalty=p2.wall_penalty,
            horizon=skills.p2c_horizon if phase == "P2c" else None,
        )
    if phase in P3_TASKS:
        p3 = skills.p3_reward
        return SkillEnv(
            task,
            seed,
            RewardConfig(
                area=p3.area, discovery=p3.discovery, activation=p3.activation, death=skills.death
            ),
            skills.p3_visit_bonus > 0,
            0.0,
            visit_bonus=skills.p3_visit_bonus,
            visit_cap=skills.p3_visit_cap,
            phase=phase,
            wall_penalty=p3.wall_penalty,
            horizon=skills.p3_horizon,
            room_exploration=0.0 if skills.p4.enabled else p3.room_exploration,
        )
    return make_skill_env(task, seed, reward, skills.first_visit, phase=phase)


@lru_cache(maxsize=64)
def skill_pool(task, count, validation=False, p4=False):
    """Unique configurations with content-hash splitting, not merely different RNG seeds."""
    from dataclasses import asdict

    if p4 and task.startswith("P4"):
        from ather_exploration.worlds.p4_tasks import p4_pool

        return p4_pool(task, count, validation)
    from ather_exploration.worlds.scenarios import digest

    if task in P3_TASKS:
        from ather_exploration.worlds.p3_tasks import p3_pool

        return p3_pool(task, count, "validation" if validation else "train")
    records = []
    seen = set()
    start = 100000 if validation else 0
    for seed in range(start, start + max(10000, count * 200)):
        scenario = skill_scenario(task, seed)
        payload = asdict(scenario)
        payload.pop("seed")
        payload.pop("skill_task")
        # P2b/c share geometry: changing the episode horizon must not move a map
        # across the content split when transferring from search to exploration.
        if task == "P2c":
            payload["horizon"] = HORIZONS["P2b"]
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
        "version": 6 if config.skills.p4.recovery else 5 if config.skills.p4.enabled else 4,
        "tasks": {},
    }
    write_record(root / "manifest.json", manifest)
    try:
        for task in (*TASKS, *(("P4c",) if config.skills.p4.enabled else ())):
            train = skill_pool(task, config.skills.train_count, p4=config.skills.p4.enabled)
            val = skill_pool(
                task, config.skills.validation_count, True, p4=config.skills.p4.enabled
            )
            if {x[1] for x in train} & {x[1] for x in val}:
                raise ValueError("Skill split overlap")
            manifest["tasks"][task] = {"train": train, "validation": val}
            if config.skills.p4.enabled and task.startswith("P4"):
                from dataclasses import asdict

                from ather_exploration.worlds.p4_tasks import p4_scenario

                for split, rows in (("train", train), ("validation", val)):
                    for seed, identity in rows:
                        write_record(
                            root / task / split / f"{seed}.json",
                            {
                                "identity": identity,
                                "scenario": asdict(p4_scenario(task, seed)),
                                "source_revision": manifest["source_revision"],
                            },
                        )
            print(f"[skills-bank] {task}: train={len(train)} validation={len(val)}", flush=True)
        if config.skills.p4.recovery:
            from ather_exploration.training.threat_lessons import export_lessons

            manifest["encounter_lessons"] = export_lessons(config, root)
        manifest["state"] = "READY"
        write_record(root / "manifest.json", manifest, replace=True)
    except BaseException:
        manifest["state"] = "FAILED"
        write_record(root / "manifest.json", manifest, replace=True)
        raise
    return manifest


def configured_skill_env(task, seed, skills, *, phase=None, threat_lesson=None):
    env = _configured_skill_env(task, seed, skills, phase=phase, threat_lesson=threat_lesson)
    if skills.frontier:
        from ather_exploration.environment.frontier import FrontierObservation

        env = FrontierObservation(env)
    if skills.p4.enabled:
        from ather_exploration.environment.threat_history import ThreatHistory

        env = ThreatHistory(env, frames=skills.p4.history_frames)
    return env
