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


def evaluate_skill(model, task, config):
    agent = LearnedAgent(model, {})
    rows = []
    for seed, _ in skill_pool(task, config.skills.validation_count, True):
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
                    initial_coverage = float(obs["memory"][2].sum()) / floors
                    coverage_history = []
                    search = SearchDiagnostics(obs, floors)
                    for t in range(scenario.horizon):
                        action, state = agent.act(
                            obs, state, deterministic=deterministic, action_rng=rng
                        )
                        obs, _, term, trunc, info = env.step(action)
                        search.update(t + 1, obs, info["transition"])
                        coverage_history.append(float(obs["memory"][2].sum()) / floors)
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
                    coverage_auc = (
                        sum(coverage_history)
                        + (scenario.horizon - len(coverage_history)) * coverage
                    ) / scenario.horizon
                    rows.append(
                        {
                            **search.row(),
                            "seed": seed,
                            "deterministic": deterministic,
                            "action_seed": action_seed,
                            "direction": direction,
                            "poi_count": len(scenario.pois),
                            "timing_bucket": seed % 3,
                            "success": float(success),
                            "coverage": coverage,
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
            key: float(np.mean([r[key] for r in selected]))
            for key in (
                "success",
                "coverage",
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
                "reward_step_cost",
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
            values = [r[key] for r in selected if r[key] is not None]
            mode_summary[key] = float(np.mean(values)) if values else None
            mode_summary[f"{key}_count"] = len(values)
        if task in ("P3", "P4b"):
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
        "P3": (0.80, 0.75),
        "P4a": (0.85, 0.80),
        "P4b": (0.65, 0.65),
    }
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
        passed &= check("deterministic/efficiency", summary["deterministic"]["efficiency"], 0.60)
        if not config.skills.wall_mask:
            passed &= check(
                "deterministic/wall_block",
                summary["deterministic"]["wall_block"],
                0.10,
                maximum=True,
            )
    if task == "P4b":
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
    return {
        "task": task,
        "validation_scope": "current_phase_only",
        "efficiency_reference": "full-map shortest path from spawn; POI location is oracle",
        "discovery_metrics": "first POI; discovery action included in before_seen coverage",
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
