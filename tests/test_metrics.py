import pytest

from ather_exploration.environment.env import make_fixture_env
from ather_exploration.evaluation.metrics import (
    EpisodeMetrics,
    aggregate_episodes,
    select_checkpoint,
)


def trace(name, actions):
    env = make_fixture_env(name)
    obs, _ = env.reset(seed=0)
    metrics = EpisodeMetrics(
        env.unwrapped.evaluator_snapshot(),
        obs,
        env.unwrapped.reward_config,
        episode_id=name,
        group="small",
    )
    for action in actions:
        obs, reward, done, truncated, info = env.step(action)
        metrics.update(
            env.unwrapped.evaluator_snapshot(),
            obs,
            reward,
            info["transition"],
            env.unwrapped.collision_stage,
        )
        if done or truncated:
            break
    result = metrics.finish()
    env.close()
    return result, metrics.steps


def test_collision2_keeps_activation_and_death_and_horizon_auc():
    result, steps = trace("collision2_poi", [2])
    assert result["T"] == 1 and result["H"] == 8
    assert result["metrics"]["survival"] == 0
    assert result["metrics"]["activation"] == 1
    assert result["metrics"]["activation_auc"] == 1
    assert result["metrics"]["all_pois_and_survived_H"] == 0
    assert result["milestones"]["all_pois"] == 1
    assert result["reward_terms"]["activation"] == 0.5
    assert result["reward_terms"]["death"] == -2
    assert len(steps) == 2 and steps[-1]["collision_stage"] == 2


def test_wait_denominators_and_initial_milestones():
    result, _ = trace("wall_wait", [4] * 8)
    assert result["metrics"]["wall_block_rate"] is None
    assert result["metrics"]["revisit_ratio"] is None
    assert result["metrics"]["wait_fraction"] == 1
    assert result["metrics"]["cycle_rate"] == 0


def test_partial_episode_is_cancelled_not_survival_failure():
    result, _ = trace("poi_revisit", [4])
    assert result["status"] == "cancelled" and result["metrics"]["survival"] is None
    assert aggregate_episodes([result])["completed"] == 0


def test_duplicate_episode_ids_rejected():
    result, _ = trace("collision2_poi", [2])
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_episodes([result, result])


def test_selection_gate_ties_and_diagnostic_fallback():
    def candidate(name, q, survival, step):
        return {
            "checkpoint_id": name,
            "step": step,
            "groups": {
                g: {"survival": survival, "q": q, "coverage_auc": 0.5}
                for g in ("small", "medium", "large")
            },
        }

    result = select_checkpoint([candidate("bad", 0.99, 0.8, 1), candidate("good", 0.6, 0.9, 2)])
    assert result["checkpoint_id"] == "good" and not result["gate_failed"]
    assert (
        select_checkpoint([candidate("z", 0.6, 0.9, 1), candidate("a", 0.6, 0.9, 1)])[
            "checkpoint_id"
        ]
        == "a"
    )
    result = select_checkpoint([candidate("a", 0.9, 0.7, 1), candidate("b", 0.5, 0.8, 2)])
    assert result["checkpoint_id"] == "b" and result["gate_failed"]


def test_macro_averages_groups_not_episode_counts():
    from copy import deepcopy

    a, _ = trace("collision2_poi", [2])
    b = deepcopy(a)
    b["episode_id"] = "b"
    b["metrics"]["coverage"] = 0
    b["group"] = "large"
    a["metrics"]["coverage"] = 1
    c = deepcopy(a)
    c["episode_id"] = "c"
    result = aggregate_episodes([a, b, c])
    assert result["macro"]["coverage"] == 0.5


def test_hand_calculated_early_death_progress_auc():
    from ather_exploration.evaluation.metrics import recompute_progress_metrics

    steps = [
        {
            "episode_id": "x",
            "t": t,
            "counts": {"seen": n, "activated": a},
            "reward": r,
            "terminated": t == 2,
        }
        for t, n, a, r in [(0, 2, 0, 0), (1, 4, 0, 0.02), (2, 5, 1, -1.49)]
    ]
    rebuilt = recompute_progress_metrics(steps, horizon=4, floor_count=10, poi_count=1)
    assert rebuilt["coverage_auc"] == pytest.approx((0.4 + 0.5 + 0.5 + 0.5) / 4)
    assert rebuilt["activation_auc"] == 0.75
    assert rebuilt["coverage_gain"] == pytest.approx(0.3)


def test_cycles_require_movement_and_stagnation_is_not_always_cycle():
    from dataclasses import replace

    from ather_exploration.config import ObservationConfig
    from ather_exploration.environment.env import ExplorationEnv
    from ather_exploration.environment.memory import PublicMemoryWrapper
    from ather_exploration.fixtures import fixture_scenario

    s = replace(fixture_scenario("poi_revisit"), horizon=40)
    cfg = ObservationConfig()
    env = PublicMemoryWrapper(ExplorationEnv(s), cfg, s.horizon)
    obs, _ = env.reset()
    m = EpisodeMetrics(
        env.unwrapped.evaluator_snapshot(),
        obs,
        env.unwrapped.reward_config,
        episode_id="cycle",
        group="small",
    )
    for action in [2, 3] * 20:
        obs, r, _, _, info = env.step(action)
        m.update(env.unwrapped.evaluator_snapshot(), obs, r, info["transition"])
    result = m.finish()
    assert result["metrics"]["cycle_rate"] > 0
    assert result["metrics"]["stagnation_rate_16"] > 0
    assert result["metrics"]["backtrack_count"] == 39
    env.close()


def test_checkpoint_bridge_rejects_missing_group_or_failed_trials():
    from ather_exploration.evaluation.metrics import checkpoint_candidate

    result, _ = trace("collision2_poi", [2])
    with pytest.raises(ValueError, match="all ID groups"):
        checkpoint_candidate("x", 100, [result])
