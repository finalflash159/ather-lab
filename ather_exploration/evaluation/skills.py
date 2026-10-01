"""Frozen skill evaluation. No optimizer updates and no training RNG consumption."""

import numpy as np

from ather_exploration.agents.learning import LearnedAgent
from ather_exploration.config import RewardConfig
from ather_exploration.types import AgentState
from ather_exploration.worlds.skill_tasks import make_skill_env, skill_pool
from ather_exploration.worlds.topology import distances


def evaluate_skill(model, task, config):
    agent = LearnedAgent(model, {})
    rows = []
    for seed, _ in skill_pool(task, config.skills.validation_count, True):
        for deterministic in (True, False):
            for action_seed in range(1 if deterministic else 3):
                env = make_skill_env(
                    task,
                    seed,
                    RewardConfig(activation=config.skills.activation, death=config.skills.death),
                    config.skills.first_visit,
                )
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
                    blocked = moves = 0
                    for t in range(scenario.horizon):
                        action, state = agent.act(
                            obs, state, deterministic=deterministic, action_rng=rng
                        )
                        obs, _, term, trunc, info = env.step(action)
                        blocked += int(obs["state"][6])
                        moves += int(action != 4)
                        if term or trunc:
                            break
                    alive = not info["transition"]["died"]
                    success = info["skill"]["success"]
                    if task == "P4b":
                        success = (
                            bool(env.unwrapped.evaluator_snapshot().activated_pois)
                            and alive
                            and t + 1 == scenario.horizon
                        )
                    rows.append(
                        {
                            "seed": seed,
                            "deterministic": deterministic,
                            "action_seed": action_seed,
                            "direction": direction,
                            "poi_count": len(scenario.pois),
                            "timing_bucket": seed % 3,
                            "success": float(success),
                            "survival": float(alive and t + 1 == scenario.horizon),
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
            for key in ("success", "efficiency", "wall_block", "survival")
        }
    thresholds = {
        "P1a": (0.95, 0.90),
        "P1b": (0.95, 0.90),
        "P2a": (0.90, 0.85),
        "P2b": (0.90, 0.85),
        "P3": (0.80, 0.75),
        "P4a": (0.85, 0.80),
        "P4b": (0.65, 0.65),
    }
    passed = all(
        summary[m]["success"] >= thresholds[task][i]
        for i, m in enumerate(("deterministic", "stochastic"))
    )
    if task.startswith("P1"):
        for direction in ("north", "south", "east", "west"):
            bucket = [
                r["success"] for r in rows if r["deterministic"] and r["direction"] == direction
            ]
            passed &= bool(bucket) and np.mean(bucket) >= 0.90
        passed &= summary["deterministic"]["efficiency"] >= 0.75
    if task.startswith("P2"):
        passed &= summary["deterministic"]["efficiency"] >= 0.60
        if not config.skills.wall_mask:
            passed &= summary["deterministic"]["wall_block"] <= 0.10
    if task == "P4b":
        passed &= all(summary[m]["survival"] >= 0.80 for m in summary)
    if task in ("P3", "P4a"):
        field = "poi_count" if task == "P3" else "timing_bucket"
        buckets = (1, 2) if task == "P3" else (0, 1, 2)
        for bucket in buckets:
            values = [r["success"] for r in rows if r["deterministic"] and r[field] == bucket]
            passed &= bool(values) and np.mean(values) >= (0.70 if task == "P3" else 0.75)
    return {
        "task": task,
        "passed": bool(passed),
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
