"""Train-only encounter curriculum; full validation scenarios are never modified."""

from dataclasses import replace
from functools import lru_cache

import numpy as np
import torch

from ather_exploration.worlds.p4_tasks import group, p4_pool, p4_scenario, timing_geometry_identity

NAMES = ("near_crossing", "crossing_and_bypass", "near_yield", "full")
P4A_FAMILIES = ("crossing", "bypass", "yield_alcoves")


@lru_cache(maxsize=16)
def family_train_seeds(count, family):
    """Keep the existing whole-geometry holdout out of every family worker."""
    if family not in P4A_FAMILIES:
        raise ValueError("Unknown P4a encounter family")
    seeds = tuple(
        seed
        for seed in lesson_seeds(count, 3, probe=False)
        if group(p4_scenario("P4a", seed))["encounter_family"] == family
    )
    if not seeds:
        raise ValueError("Empty P4a family train split")
    return seeds


@lru_cache(maxsize=32)
def lesson_seeds(count, level, probe=False):
    if level not in range(4):
        raise ValueError("Unknown encounter lesson")
    families = (
        ("crossing",)
        if level == 0
        else ("crossing", "bypass")
        if level == 1
        else ("crossing", "bypass", "yield_alcoves")
    )
    # Reserve whole geometries (including all timing offsets) within TRAIN.
    held = set()
    for family in families:
        identities = list(
            dict.fromkeys(
                timing_geometry_identity(p4_scenario("P4a", seed))
                for seed, _ in p4_pool("P4a", count)
                if group(p4_scenario("P4a", seed))["encounter_family"] == family
            )
        )
        held.update(identities[: min(4, max(1, len(identities) // 4))])
    seeds = [
        seed
        for seed, _ in p4_pool("P4a", count)
        if group(p4_scenario("P4a", seed))["encounter_family"] in families
        and (timing_geometry_identity(p4_scenario("P4a", seed)) in held) == probe
    ]
    if not seeds:
        raise ValueError("Empty encounter lesson split")
    return tuple(seeds)


def lesson_scenario(scenario, level, *, timing=False):
    if scenario.skill_task != "P4a" or level not in range(4):
        raise ValueError("Encounter lessons apply only to P4a")
    family = group(scenario)["encounter_family"]
    route = scenario.routes[0]
    if level == 0 and family == "crossing":
        dx = int(np.sign(scenario.pois[0][0] - scenario.spawn[0]))
        dy = int(np.sign(scenario.pois[0][1] - scenario.spawn[1]))
        crossing = next(
            p for p in route if ((p[1] == scenario.spawn[1]) if dx else (p[0] == scenario.spawn[0]))
        )
        x, y = crossing
        if timing:
            # Geometry-derived, independent of patrol phase. Approach takes 2-3 steps.
            distance = min(
                abs(scenario.spawn[0] - x) + abs(scenario.spawn[1] - y), 2 + (len(route) % 2)
            )
            return replace(scenario, spawn=(x - distance * dx, y - distance * dy))
        return replace(scenario, spawn=(x - dx, y - dy), pois=((x + dx, y + dy),))
    if level == 2 and family == "yield_alcoves":
        x, y = route[-2]
        pockets = [
            (x + dx, y + dy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if (x + dx, y + dy) not in route and scenario.terrain[y + dy][x + dx] == "."
        ]
        if len(pockets) != 1:
            raise ValueError("Yield lesson requires a unique refuge")
        return replace(scenario, spawn=pockets[0])
    return scenario


def probe_lesson(model, config, level):
    """Greedy autonomous train holdout, with batched inference and no optimizer."""
    from ather_exploration.worlds.skill_tasks import configured_skill_env

    seeds = lesson_seeds(config.skills.train_count, level, probe=True)
    envs, observations, results = [], [], []
    try:
        for seed in seeds:
            env = configured_skill_env("P4a", seed, config.skills, threat_lesson=level)
            envs.append(env)
            observations.append(env.reset()[0])
            results.append(None)
        steps = 0
        with torch.no_grad():
            for _ in range(128):
                active = [i for i, result in enumerate(results) if result is None]
                if not active:
                    break
                batch = {k: np.stack([observations[i][k] for i in active]) for k in observations[0]}
                actions, _ = model.predict(batch, deterministic=True)
                for i, action in zip(active, actions, strict=True):
                    obs, _, term, trunc, info = envs[i].step(int(action))
                    observations[i] = obs
                    steps += 1
                    if term or trunc:
                        results[i] = bool(info["skill"]["success"])
        by_family = {}
        for family in {group(p4_scenario("P4a", s))["encounter_family"] for s in seeds}:
            values = [
                r
                for s, r in zip(seeds, results, strict=True)
                if group(p4_scenario("P4a", s))["encounter_family"] == family
            ]
            by_family[family] = float(np.mean(values))
        return {
            "split": "train_holdout",
            "level": level,
            "seeds": list(seeds),
            "success": float(np.mean(results)),
            "families": by_family,
            "steps": steps,
            "passed": all(v >= 0.75 for v in by_family.values()),
        }
    finally:
        for env in envs:
            env.close()


def observe_lesson(controller, result, steps):
    """Two consecutive train probes plus 16K experience; never changes the final gate."""
    if result["level"] != controller.threat_level or controller.threat_level == 3:
        return False
    controller.threat_streak = controller.threat_streak + 1 if result["passed"] else 0
    if controller.threat_streak >= 2 and steps - controller.threat_level_start >= 16384:
        controller.threat_level += 1
        controller.threat_level_start = steps
        controller.threat_streak = 0
        return True
    return False


def retention_report(model, config):
    """P3c deterministic validation report only; never samples for teaching."""
    from ather_exploration.worlds.skill_tasks import configured_skill_env, skill_pool

    envs, observations, results = [], [], []
    mode = model.policy.training
    try:
        for seed, _ in skill_pool("P3c", config.skills.validation_count, True):
            env = configured_skill_env("P3c", seed, config.skills, phase="P3c")
            envs.append(env)
            observations.append(env.reset()[0])
            results.append(None)
        steps = 0
        with torch.no_grad():
            for _ in range(config.skills.p3_horizon):
                active = [i for i, r in enumerate(results) if r is None]
                if not active:
                    break
                batch = {k: np.stack([observations[i][k] for i in active]) for k in observations[0]}
                actions, _ = model.predict(batch, deterministic=True)
                for i, action in zip(active, actions, strict=True):
                    observations[i], _, term, trunc, info = envs[i].step(int(action))
                    steps += 1
                    if term or trunc:
                        results[i] = bool(info["skill"]["success"])
        return {
            "split": "validation",
            "task": "P3c",
            "mode": "deterministic",
            "episodes": len(results),
            "success": float(np.mean(results)),
            "steps": steps,
        }
    finally:
        model.policy.set_training_mode(mode)
        for env in envs:
            env.close()


def export_lessons(config, root):
    """Persist derived scenarios and train-holdout split in the uploaded suite."""
    from dataclasses import asdict

    from ather_exploration.worlds.scenarios import digest, write_record

    manifest = {}
    for level, name in enumerate(NAMES):
        manifest[name] = {}
        for split, probe in (("train", False), ("train_holdout", True)):
            records = []
            for seed in lesson_seeds(config.skills.train_count, level, probe):
                scenario = asdict(
                    lesson_scenario(p4_scenario("P4a", seed), level, timing=config.skills.p4.timing)
                )
                identity = digest(scenario)
                write_record(
                    root / "P4a" / "lessons" / name / split / f"{seed}.json",
                    {"identity": identity, "scenario": scenario},
                )
                records.append((seed, identity))
            manifest[name][split] = records
    return manifest
