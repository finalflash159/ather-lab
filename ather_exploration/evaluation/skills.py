"""Frozen skill evaluation. No optimizer updates and no training RNG consumption."""

import numpy as np

from ather_exploration.agents.learning import LearnedAgent
from ather_exploration.types import AgentState
from ather_exploration.worlds.skill_tasks import configured_skill_env, skill_pool
from ather_exploration.worlds.topology import distances


class SearchDiagnostics:
    """Timing/coverage around first POI sighting, using public memory only.

    The discovery action belongs to the search interval (before/through discovery).
    Missing events remain None; conditional averages always include a sample count.
    """

    def __init__(self, observation, floors):
        self.floors = floors
        self.initial = float(observation["memory"][2].sum()) / floors
        self.coverage = self.initial
        self.first_seen = 0 if self.sees_poi(observation) else None
        self.coverage_at_seen = self.initial if self.first_seen == 0 else None
        self.first_activation = None
        self.block_streak = self.longest_block_streak = 0

    @staticmethod
    def sees_poi(observation):
        return bool(observation["memory"][3:5].any())

    def update(self, step, observation, transition):
        self.coverage = float(observation["memory"][2].sum()) / self.floors
        if self.first_seen is None and self.sees_poi(observation):
            self.first_seen = step
            self.coverage_at_seen = self.coverage
        if self.first_activation is None and transition["activated"]:
            self.first_activation = step
        self.block_streak = self.block_streak + 1 if observation["state"][6] else 0
        self.longest_block_streak = max(self.longest_block_streak, self.block_streak)

    def row(self):
        seen = self.first_seen is not None
        activated = self.first_activation is not None
        return {
            "poi_seen": float(seen),
            "first_poi_seen_step": self.first_seen,
            "first_poi_activation_step": self.first_activation,
            "steps_seen_to_activation": (
                self.first_activation - self.first_seen if seen and activated else None
            ),
            "activated_if_seen": float(activated) if seen else None,
            "coverage_gain_before_seen": (self.coverage_at_seen if seen else self.coverage)
            - self.initial,
            "coverage_gain_after_seen": (self.coverage - self.coverage_at_seen if seen else None),
            "longest_wall_block_streak": self.longest_block_streak,
        }


def evaluate_skill(model, task, config, *, agent=None, split="validation"):
    from ather_exploration.evaluation.p3 import ExplorationDiagnostics, add_p3_gates
    from ather_exploration.worlds.p3_tasks import map_group, p3_pool
    from ather_exploration.worlds.p4_tasks import group as threat_group

    agent = agent if agent is not None else LearnedAgent(model, {})
    is_p4 = config.skills.p4.enabled and task.startswith("P4")
    is_p3 = task in ("P3a", "P3b", "P3c")
    if split not in ("validation", "ood") or (split == "ood" and not is_p3):
        raise ValueError("Choose validation or P3 OOD; heldout test is not a tuning split")
    pool = (
        p3_pool(task, config.skills.validation_count, split)
        if is_p3
        else (
            skill_pool(task, config.skills.validation_count, True, p4=True)
            if config.skills.p4.enabled
            else skill_pool(task, config.skills.validation_count, True)
        )
    )
    rows = []
    for seed, _ in pool:
        for deterministic in (True, False):
            for action_seed in range(1 if deterministic else 3):
                env = configured_skill_env(task, seed, config.skills)
                try:
                    obs, _ = env.reset()
                    scenario = env.unwrapped.scenario
                    d = min(distances(scenario.terrain, [scenario.spawn])[p] for p in scenario.pois)
                    dx, dy = (scenario.pois[0][i] - scenario.spawn[i] for i in range(2))
                    direction = (
                        ("east" if dx > 0 else "west")
                        if abs(dx) >= abs(dy)
                        else ("south" if dy > 0 else "north")
                    )
                    state = AgentState()
                    rng = np.random.default_rng(action_seed)
                    blocked = moves = waits = revisits = 0
                    position = (0, 0)
                    visited = {position}
                    components = {}
                    floors = sum(row.count(".") for row in scenario.terrain)
                    coverage_history = []
                    approach_distance = d if SearchDiagnostics.sees_poi(obs) else None
                    actual_position = scenario.spawn
                    no_progress = longest_no_progress = 0
                    search = SearchDiagnostics(obs, floors)
                    exploration = (
                        ExplorationDiagnostics(scenario, obs)
                        if is_p3 or is_p4 and task != "P4a"
                        else None
                    )
                    initial_coverage = (
                        exploration.coverage
                        if exploration
                        else float(obs["memory"][2].sum()) / floors
                    )
                    for t in range(scenario.horizon):
                        action, state = agent.act(
                            obs, state, deterministic=deterministic, action_rng=rng
                        )
                        obs, _, term, trunc, info = env.step(action)
                        search.update(t + 1, obs, info["transition"])
                        actual_position = tuple(
                            a + b
                            for a, b in zip(
                                actual_position, info["transition"]["actual_delta"], strict=True
                            )
                        )
                        if approach_distance is None and search.first_seen is not None:
                            approach_distance = min(
                                distances(scenario.terrain, [actual_position])[p]
                                for p in scenario.pois
                            )
                        progress = (
                            info["transition"]["new_floor"] or info["transition"]["activated"]
                        )
                        if exploration:
                            exploration.update(
                                t + 1,
                                obs,
                                env.unwrapped.evaluator_snapshot().activated_pois,
                                progress=progress,
                            )
                        no_progress = 0 if progress else no_progress + 1
                        longest_no_progress = max(longest_no_progress, no_progress)
                        coverage_history.append(
                            exploration.coverage
                            if exploration
                            else float(obs["memory"][2].sum()) / floors
                        )
                        blocked += int(obs["state"][6])
                        moves += int(action != 4)
                        waits += int(action == 4)
                        delta = info["transition"]["actual_delta"]
                        position = tuple(a + b for a, b in zip(position, delta, strict=True))
                        revisits += int(any(delta) and position in visited)
                        visited.add(position)
                        for key, value in info["skill"]["reward_components"].items():
                            components[key] = components.get(key, 0.0) + value
                        if term or trunc:
                            break
                    alive = not info["transition"]["died"]
                    success = info["skill"]["success"]
                    snapshot = env.unwrapped.evaluator_snapshot()
                    coverage = coverage_history[-1]
                    tile_coverage = float(obs["memory"][2].sum()) / floors
                    room_coverage_min = (
                        exploration.room_coverage_min if exploration else tile_coverage
                    )
                    coverage_threshold = (
                        config.skills.p4.room_coverage
                        if is_p4
                        else (
                            config.skills.p3_gates.room_coverage
                            if task in ("P3b", "P3c", "P4b", "P4c")
                            else config.skills.p3_gates.coverage
                        )
                    )
                    coverage_auc = (
                        sum(coverage_history)
                        + (scenario.horizon - len(coverage_history)) * coverage
                    ) / scenario.horizon
                    rows.append(
                        {
                            **search.row(),
                            **(exploration.row() if exploration else {}),
                            **(
                                map_group(scenario)
                                if is_p3
                                else threat_group(scenario)
                                if is_p4
                                else {}
                            ),
                            "joint_success": float(success and coverage >= coverage_threshold),
                            "approach_efficiency": (
                                approach_distance
                                / max(
                                    search.first_activation - search.first_seen,
                                    approach_distance,
                                    1,
                                )
                                if search.first_activation is not None
                                and approach_distance is not None
                                else 0.0
                            ),
                            "longest_no_progress_streak": longest_no_progress,
                            "seed": seed,
                            "deterministic": deterministic,
                            "action_seed": action_seed,
                            "direction": direction,
                            "poi_count": len(scenario.pois),
                            "timing_bucket": seed % 3,
                            "success": float(success),
                            "coverage": coverage,
                            "tile_coverage": tile_coverage,
                            "room_coverage_min": room_coverage_min,
                            "coverage_gain": coverage - initial_coverage,
                            "coverage_auc": coverage_auc,
                            "activation": len(snapshot.activated_pois) / len(scenario.pois),
                            "survival": float(alive and t + 1 == scenario.horizon),
                            "alive_at_end": float(alive),
                            "timeout": float(alive and t + 1 == scenario.horizon and not success),
                            "death": float(not alive),
                            "wait_fraction": waits / (t + 1),
                            "revisit_fraction": revisits / (t + 1),
                            **{f"reward_{k}": v for k, v in components.items()},
                            "efficiency": float(success * d / max(t + 1, d)),
                            "steps": t + 1,
                            "wall_block": blocked / max(moves, 1),
                        }
                    )
                finally:
                    env.close()
    summary = {}
    for mode in (True, False):
        selected = [r for r in rows if r["deterministic"] == mode]
        summary["deterministic" if mode else "stochastic"] = {
            key: float(np.mean([r.get(key, 0.0) for r in selected]))
            for key in (
                "success",
                "joint_success",
                "coverage",
                "tile_coverage",
                "room_coverage_min",
                "coverage_gain",
                "coverage_auc",
                "activation",
                "efficiency",
                "wall_block",
                "alive_at_end",
                "timeout",
                "death",
                "steps",
                "wait_fraction",
                "revisit_fraction",
                "reward_area",
                "reward_discovery",
                "reward_activation",
                "reward_death",
                "reward_intrinsic",
                "reward_room_exploration",
                "reward_step_cost",
                "reward_wall_penalty",
                "approach_efficiency",
                "longest_no_progress_streak",
            )
        }
        mode_summary = summary["deterministic" if mode else "stochastic"]
        for key in (
            "poi_seen",
            "first_poi_seen_step",
            "first_poi_activation_step",
            "steps_seen_to_activation",
            "activated_if_seen",
            "coverage_gain_before_seen",
            "coverage_gain_after_seen",
            "longest_wall_block_streak",
        ):
            values = [r.get(key, 0.0) for r in selected if r[key] is not None]
            mode_summary[key] = float(np.mean(values)) if values else None
            mode_summary[f"{key}_count"] = len(values)
        if task in ("P3", "P4b", "P4c"):
            summary["deterministic" if mode else "stochastic"]["survival"] = float(
                np.mean([r["survival"] for r in selected])
            )
    checks = []

    def check(name, value, threshold, *, maximum=False):
        ok = value <= threshold if maximum else value >= threshold
        checks.append(
            {
                "name": name,
                "value": float(value),
                "threshold": threshold,
                "operator": "<=" if maximum else ">=",
                "passed": bool(ok),
            }
        )
        return bool(ok)

    # Populated below with each actual gate requirement.
    thresholds = {
        "P1a": (0.95, 0.90),
        "P1b": (0.95, 0.90),
        "P2a": (0.90, 0.85),
        "P2b": (0.90, 0.85),
        "P2c": (0.90, 0.85),
        "P3": (0.80, 0.75),
        "P4a": (0.85, 0.80),
        "P4b": (0.65, 0.65),
    }
    if is_p4:
        thresholds[task] = (0.85, 0.80) if task == "P4a" else (config.skills.p4.success,) * 2
    if is_p3:
        thresholds[task] = (
            config.skills.p3_gates.success,
            config.skills.p3_gates.stochastic_success,
        )
    passed = True
    for i, m in enumerate(("deterministic", "stochastic")):
        passed &= check(f"{m}/success", summary[m]["success"], thresholds[task][i])
    if task.startswith("P1"):
        for direction in ("north", "south", "east", "west"):
            bucket = [
                r["success"] for r in rows if r["deterministic"] and r["direction"] == direction
            ]
            passed &= check(
                f"direction/{direction}", float(np.mean(bucket)) if bucket else 0.0, 0.90
            )
        passed &= check("deterministic/efficiency", summary["deterministic"]["efficiency"], 0.75)
    if task.startswith("P2"):
        d = summary["deterministic"]
        if task == "P2a":
            passed &= check("deterministic/efficiency", d["efficiency"], 0.60)
        elif task == "P2b":
            passed &= check(
                "deterministic/poi_seen", d["poi_seen"], config.skills.p2_gates.discovery_success
            )
            passed &= check(
                "deterministic/approach_efficiency",
                d["approach_efficiency"],
                config.skills.p2_gates.approach_efficiency,
            )
        else:
            passed &= check(
                "deterministic/coverage", d["coverage"], config.skills.p2_gates.coverage
            )
            passed &= check(
                "deterministic/coverage_auc", d["coverage_auc"], config.skills.p2_gates.coverage_auc
            )
        if not config.skills.wall_mask:
            passed &= check(
                "deterministic/wall_block",
                summary["deterministic"]["wall_block"],
                0.10,
                maximum=True,
            )
    if task == "P4b" and not is_p4:
        for m in summary:
            passed &= check(f"{m}/survival", summary[m]["survival"], 0.80)
    if task in ("P3", "P4a"):
        field = "poi_count" if task == "P3" else "timing_bucket"
        buckets = (1, 2) if task == "P3" else (0, 1, 2)
        for bucket in buckets:
            values = [r["success"] for r in rows if r["deterministic"] and r[field] == bucket]
            passed &= check(
                f"{field}/{bucket}",
                float(np.mean(values)) if values else 0.0,
                0.70 if task == "P3" else 0.75,
            )
    if is_p4 and task != "P4a":
        g = config.skills.p4
        for mode in summary:
            passed &= check(f"{mode}/survival", summary[mode]["survival"], g.survival)
        d = summary["deterministic"]
        passed &= check("deterministic/coverage", d["coverage"], g.room_coverage)
        passed &= check("deterministic/coverage_auc", d["coverage_auc"], g.coverage_auc)
        passed &= check("deterministic/joint_success", d["joint_success"], g.joint)
        passed &= check("deterministic/wall_block", d["wall_block"], 0.10, maximum=True)
    groups = {}
    if is_p4:
        for field in ("size", "monster_count", "poi_count"):
            for value in sorted({r[field] for r in rows}):
                selected = [r for r in rows if r["deterministic"] and r[field] == value]
                groups[f"{field}/{value}"] = {
                    "count": len(selected),
                    "success": float(np.mean([r["success"] for r in selected])),
                    "death": float(np.mean([r["death"] for r in selected])),
                }
    if is_p3:
        p3_passed, groups = add_p3_gates(task, rows, summary, check, config)
        passed &= p3_passed
    return {
        "split": split,
        "groups": groups,
        "task": task,
        "validation_scope": "current_phase_only",
        "gate_status": "pilot_thresholds"
        if task in ("P2b", "P2c", "P3a", "P3b", "P3c") or is_p4
        else "configured",
        "approach_reference": "evaluator-only geodesic distance at first sighting; failures zero",
        "efficiency_reference": "full-map shortest path from spawn; POI location is oracle",
        "discovery_metrics": (
            "v2: P3 activated_if_seen pooled over all seen POIs; other timing metrics first POI"
            if is_p3
            else "first POI; discovery action included in before_seen coverage"
        ),
        "coverage_definition": (
            "minimum per-room coverage and its time average for P3b/P3c/P4b/P4c; aggregate tile coverage for P3a"
            if task in ("P3a", "P3b", "P3c") or is_p4 and task != "P4a"
            else "aggregate tile coverage"
        ),
        "passed": bool(passed),
        "checks": checks,
        "failed_checks": [c["name"] for c in checks if not c["passed"]],
        "summary": summary,
        "episodes": rows,
        "eval_steps": sum(r["steps"] for r in rows),
    }


def evaluate_target(model, config):
    from ather_exploration.environment.env import make_env
    from ather_exploration.evaluation.learned import ModeAdapter
    from ather_exploration.evaluation.runner import evaluate_episode
    from ather_exploration.training.curriculum import WorldBank

    bank = WorldBank(config.banks)
    agent = LearnedAgent(model, {})
    rows = []
    for group in ("small", "medium", "large"):
        for world in bank.worlds[group]["validation_quick"]:
            env = make_env(generated=world["records"][0])
            try:
                for deterministic in (True, False):
                    for seed in range(1 if deterministic else 3):
                        result, _ = evaluate_episode(
                            env,
                            ModeAdapter(agent, deterministic),
                            episode_id=f"{group}-{len(rows)}",
                            group=group,
                            action_seed=seed,
                        )
                        if result["status"] != "completed":
                            raise ValueError("Target evaluation failed")
                        rows.append({"deterministic": deterministic, **result})
            finally:
                env.close()
    summaries = {}
    for mode in (True, False):
        selected = [r for r in rows if r["deterministic"] == mode]
        summaries["deterministic" if mode else "stochastic"] = {
            k: float(np.mean([r["metrics"][k] for r in selected]))
            for k in ("coverage", "coverage_auc", "activation", "survival")
        }
        summaries["deterministic" if mode else "stochastic"]["activation_survived"] = float(
            np.mean([r["metrics"]["activation"] * r["metrics"]["survival"] for r in selected])
        )
    return {
        "task": "target",
        "summary": summaries,
        "episodes": rows,
        "eval_steps": sum(r["T"] for r in rows),
    }
