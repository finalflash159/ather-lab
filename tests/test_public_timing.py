"""Public timing safety, useful gradients, isolation and persisted teaching."""

import copy

import numpy as np
import pytest
import torch

from ather_exploration.agents.learning import build_model
from ather_exploration.training.config import read_training_config
from ather_exploration.training.public_timing import (
    _shelter_action,
    public_p4_timing,
    public_p4a_timing,
    public_timing,
)
from ather_exploration.training.threat_lessons import lesson_seeds
from ather_exploration.worlds.skill_tasks import configured_skill_env, skill_pool


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return read_training_config("ather_exploration/resources/training/skills_p4.yaml")


def test_temperature_gradient_no_threat_identity_and_likelihood(config):
    env = configured_skill_env("P4a", 19, config.skills)
    model = build_model(config, env)
    try:
        policy = model.policy
        policy.threat_temperature = 16
        with torch.no_grad():
            policy.action_net.weight.zero_()
            policy.action_net.bias.copy_(torch.tensor([0.0, -2.0, -3.0, -4.0, -30.0]))
        obs, _ = env.reset()
        obs["memory"][9, 40, 40] = obs["memory"][8, 40, 40] = 1
        tensor, _ = policy.obs_to_tensor(obs)
        dist = policy.get_distribution(tensor)
        (-dist.log_prob(torch.tensor([4]))).backward()
        assert abs(policy.action_net.bias.grad[4]) > 0.01
        assert dist.distribution.probs[0, 4] > 0.02
        action, value, lp = policy(tensor)
        checked, checked_lp, _ = policy.evaluate_actions(tensor, action)
        torch.testing.assert_close(lp, checked_lp)
        torch.testing.assert_close(value, checked)
        obs["memory"][9] = 0
        obs["memory"][12:] = 0
        tensor, _ = policy.obs_to_tensor(obs)
        expected = policy.action_net.bias.softmax(0)
        torch.testing.assert_close(policy.get_distribution(tensor).distribution.probs[0], expected)
        # A recent public sighting still enables the learnable distribution.
        obs["memory"][12, 40, 40] = 1
        tensor, _ = policy.obs_to_tensor(obs)
        assert policy.get_distribution(tensor).distribution.probs[0, 4] > 0.02
    finally:
        env.close()


@pytest.mark.parametrize("task", ["P4a", "P4b", "P4c"])
def test_public_labels_never_collide_on_replayed_train_trajectories(config, task):
    wait = go = 0
    seeds = (
        lesson_seeds(256, 0)[::8][:10]
        if task == "P4a"
        else [s for s, _ in skill_pool(task, 256, p4=True)][::25][:10]
    )
    for seed in seeds:
        env = configured_skill_env(
            task, seed, config.skills, threat_lesson=0 if task == "P4a" else None
        )
        try:
            obs, _ = env.reset()
            for _ in range(64):
                label = public_timing(obs)
                if label:
                    wait += label["kind"] == "wait"
                    go += label["kind"] == "go"
                    for action in np.flatnonzero(label["actions"]):
                        clone = copy.deepcopy(env)
                        try:
                            _, _, _, _, info = clone.step(int(action))
                            assert not info["transition"]["died"], (task, seed, action)
                        finally:
                            clone.close()
                    action = int(np.flatnonzero(label["actions"])[0])
                else:
                    from ather_exploration.training.public_route import public_route

                    route = public_route(obs)
                    action = int(np.flatnonzero(route["actions"])[0]) if route else 4
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
        finally:
            env.close()
    assert go > 0
    if task == "P4a":
        assert wait > 0


def test_teacher_rejects_validation_and_checkpoint_persists(config, tmp_path):
    from ather_exploration.agents.route_ppo import RoutePPO
    from ather_exploration.training.timing_teaching import TimingTeaching

    env = configured_skill_env("P4a", 19, config.skills)
    try:
        model = build_model(config, env)
        model.timing_teaching = TimingTeaching(config)
        obs, _ = env.reset()
        validation_seed = skill_pool("P4a", 64, True, p4=True)[0][0]
        with pytest.raises(ValueError, match="train-only"):
            model.timing_teaching.offer(obs, "P4a", validation_seed)
        model.policy.threat_temperature = 16
        model.policy_kwargs["threat_temperature"] = 16
        model.save(tmp_path / "timing")
        loaded = RoutePPO.load(tmp_path / "timing", device="cpu")
        assert loaded.policy.threat_mode == "temperature"
        assert loaded.policy.threat_temperature == 16
        assert loaded.timing_teaching.seeds == model.timing_teaching.seeds
        np.testing.assert_array_equal(
            model.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0]
        )
    finally:
        env.close()


def test_timing_lesson_approach_not_single_step(config):
    from ather_exploration.training.threat_lessons import lesson_scenario
    from ather_exploration.worlds.p4_tasks import p4_scenario

    distances = set()
    for seed in lesson_seeds(256, 0):
        base = p4_scenario("P4a", seed)
        variant = lesson_scenario(base, 0, timing=True)
        distance = min(
            abs(variant.spawn[0] - x) + abs(variant.spawn[1] - y) for x, y in variant.routes[0]
        )
        distances.add(distance)
        assert distance >= 2
        assert variant.pois == base.pois
        assert variant.routes == base.routes and variant.phases == base.phases
    assert distances == {2, 3}


def test_timing_update_guard_rolls_back_weights_and_optimizer(config, monkeypatch):
    from types import SimpleNamespace

    from test_recovery import prepared
    from test_route_teaching import assert_nested_equal

    from ather_exploration.agents.route_ppo import RoutePPO
    from ather_exploration.training.route_teaching import RouteMemory
    from ather_exploration.training.timing_teaching import teach_timing

    model = prepared(RoutePPO)
    rows = [{"x": np.ones(3, np.float32), "labels": np.array([0, 0, 0, 0, 1], bool)}] * 64
    memory = RouteMemory()
    monkeypatch.setattr(memory, "sample", lambda n: rows[:n])
    model.threat_retention = SimpleNamespace(memory=memory)
    model.timing_teaching = SimpleNamespace(
        sample=lambda n: rows[:n],
        batches=2,
        coefficient=0.1,
        max_kl=1.0,
        memories={},
        p4a_family_counts=dict,
    )
    trained = teach_timing(model)
    assert trained["accepted"] == 2
    assert trained["wait_label_count"] == 64
    assert trained["go_label_count"] == 0
    assert trained["wait_target_mass_delta"] > 0
    assert trained["wait_logit_margin_delta"] > 0
    weights = copy.deepcopy(model.policy.state_dict())
    optimizer = copy.deepcopy(model.policy.optimizer.state_dict())
    model.timing_teaching.max_kl = 0
    result = teach_timing(model)
    assert result["rejected"] == 1 and result["accepted"] == 0
    assert result["wait_target_mass_delta"] == pytest.approx(0.0, abs=1e-7)
    assert result["wait_logit_margin_delta"] == pytest.approx(0.0, abs=1e-7)
    assert_nested_equal(weights, model.policy.state_dict())
    assert_nested_equal(optimizer, model.policy.optimizer.state_dict())


def test_timing_guard_keeps_safe_minibatch_before_later_rejection(monkeypatch):
    from types import SimpleNamespace

    from test_recovery import prepared
    from test_route_teaching import assert_nested_equal

    from ather_exploration.agents.route_ppo import RoutePPO
    from ather_exploration.training import timing_teaching
    from ather_exploration.training.route_teaching import RouteMemory

    model = prepared(RoutePPO)
    labels = np.array([False, False, False, False, True])
    rows = [{"x": np.ones(3, np.float32), "labels": labels} for _ in range(64)]
    memory = RouteMemory()
    monkeypatch.setattr(memory, "sample", lambda count: rows[:count])
    model.threat_retention = SimpleNamespace(memory=memory)

    sample_calls = 0
    accepted_snapshot = {}

    def sample(count):
        nonlocal sample_calls
        sample_calls += 1
        if sample_calls == 2:
            accepted_snapshot["weights"] = copy.deepcopy(model.policy.state_dict())
            accepted_snapshot["optimizer"] = copy.deepcopy(model.policy.optimizer.state_dict())
        return rows[:count]

    model.timing_teaching = SimpleNamespace(
        sample=sample,
        batches=2,
        coefficient=0.1,
        max_kl=0.5,
        memories={},
        p4a_family_counts=dict,
    )
    kl_values = iter((0.1, 0.6))
    monkeypatch.setattr(
        timing_teaching,
        "divergence",
        lambda reference, current: torch.full((len(reference),), next(kl_values)),
    )

    result = timing_teaching.teach_timing(model)

    assert result["accepted"] == 1 and result["rejected"] == 1
    assert result["wait_target_mass_delta"] > 0
    assert result["wait_logit_margin_delta"] > 0
    assert_nested_equal(accepted_snapshot["weights"], model.policy.state_dict())
    assert_nested_equal(accepted_snapshot["optimizer"], model.policy.optimizer.state_dict())


def test_unknown_approach_is_not_certified_safe():
    memory = np.zeros((16, 9, 9), np.float32)
    memory[0:1] = 1
    memory[2] = 1
    memory[8] = 1
    memory[7, 4, 4] = 1
    memory[3, 4, 6] = 1
    memory[9, 2, 4] = 1
    obs = {"memory": memory}
    label = public_timing(obs)
    assert label is not None and label["actions"][2]
    # East destination could be reached by an unseen monster in this cell.
    memory[8, 3, 5] = 0
    memory[0, 3, 5] = 0
    label = public_timing(obs)
    assert label is None or not label["actions"][2]


def test_p4bc_timing_does_not_label_consecutive_waits():
    memory = np.zeros((16, 9, 9), np.float32)
    memory[0] = memory[2] = memory[8] = 1
    memory[7, 4, 4] = 1
    memory[3, 4, 8] = 1
    memory[9, 4, 6] = 1
    obs = {"memory": memory, "state": np.zeros(17, np.float32)}

    first = public_p4_timing(obs)
    assert first is not None and first["kind"] == "wait"

    obs["state"][4] = 1
    legacy = public_timing(obs)
    after_wait = public_p4_timing(obs)
    assert legacy is not None and legacy["kind"] == "wait"
    assert after_wait is None or not after_wait["actions"][4]


def test_p4bc_timing_sampler_prioritizes_active_task_and_previous_threat_task():
    from ather_exploration.training.route_teaching import RouteMemory
    from ather_exploration.training.timing_teaching import TimingTeaching

    teacher = object.__new__(TimingTeaching)
    teacher.memories = {
        "wait": RouteMemory(per_map=4, seed=11),
        "go": RouteMemory(per_map=4, seed=12),
    }
    teacher.active_task = "P4b"
    teacher.balanced_families = False
    for task in ("P4a", "P4b", "P4c"):
        for kind, memory in teacher.memories.items():
            obs = {
                "local": np.zeros((6, 9, 9), np.uint8),
                "memory": np.zeros((16, 81, 81), np.float32),
                "state": np.zeros(17, np.float32),
            }
            obs["memory"][0, 0, 0] = ("P4a", "P4b", "P4c").index(task) + 1
            labels = np.zeros(5, bool)
            labels[4 if kind == "wait" else 0] = True
            memory.offer(obs, labels, (task, 1), "learner")

    teacher.sample(64)
    counts = teacher.last_sample_counts
    assert set(counts) <= {"P4a", "P4b"}
    assert counts["P4b"] / sum(counts.values()) >= 0.65

    teacher.active_task = "P4c"
    teacher.sample(64)
    counts = teacher.last_sample_counts
    assert set(counts) <= {"P4b", "P4c"}
    assert counts["P4c"] / sum(counts.values()) >= 0.65


def test_shelter_helper_handles_monster_on_observation_edge():
    memory = np.zeros((16, 9, 9), np.float32)
    memory[0] = memory[2] = memory[8] = 1
    memory[7, 4, 4] = 1
    memory[3, 4, 6] = 1
    memory[9, 0, 0] = 1
    assert _shelter_action({"memory": memory}) in range(5)


def test_shelter_helper_leaves_wait_for_safe_exit_after_revisits():
    """A public revisit signal breaks the WAIT-vs-detour tie without hidden state."""
    memory = np.zeros((16, 9, 9), np.float32)
    memory[0] = memory[2] = memory[8] = 1
    memory[7, 4, 4] = 1
    memory[3, 4, 8] = 1
    # The monster is two cells east. Its adjacent lane cell blocks the shortest
    # move, but north/south are known safe floor; the POI remains farther east.
    memory[9, 4, 6] = 1
    memory[6, 4, 4] = np.log1p(5) / np.log1p(1025)
    action = _shelter_action({"memory": memory})
    assert action in range(4), "after repeated visits, take a safe exit instead of waiting"


def test_shelter_helper_routes_around_repeated_corridor_cells():
    """Route-level revisit cost should beat a repeatedly traversed short path."""
    memory = np.zeros((16, 9, 9), np.float32)
    floor = [(4, col) for col in range(2, 7)]
    floor += [(5, col) for col in range(2, 7)]
    for row, col in floor:
        memory[0, row, col] = memory[2, row, col] = 1
    memory[8] = 1
    memory[7, 4, 2] = 1
    memory[3, 4, 6] = 1
    memory[9, 0, 0] = 1
    for col in (3, 4, 5):
        memory[6, 4, col] = np.log1p(8) / np.log1p(1025)

    action = _shelter_action({"memory": memory})

    assert action == 1, "take the longer fresh southern route instead of the looped corridor"


def test_shelter_helper_selects_less_revisited_frontier_target():
    """Frontier choice should account for the full visited route, not map extremity."""
    memory = np.zeros((16, 9, 9), np.float32)
    floor = [(4, col) for col in range(4, 8)] + [(row, 4) for row in range(4, 7)]
    for row, col in floor:
        memory[0, row, col] = memory[2, row, col] = 1
    memory[8] = 1
    memory[7, 4, 4] = 1
    memory[9, 0, 0] = 1
    # The far eastern frontier has been traversed repeatedly. The nearer
    # southern frontier is still available and should become the target.
    memory[0, 4, 8] = 0
    memory[2, 4, 8] = 0
    memory[0, 7, 4] = 0
    memory[2, 7, 4] = 0
    for col in (5, 6, 7):
        memory[6, 4, col] = np.log1p(8) / np.log1p(1025)

    action = _shelter_action({"memory": memory})

    assert action == 1, "select the less-revisited southern frontier route"


def test_timing_label_diagnostics_measure_wait_and_safe_go_logit_margins():
    from ather_exploration.training.timing_teaching import _timing_label_diagnostics

    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, -1.0], [0.0, 0.0, -1.0, -1.0, -1.0]])
    labels = torch.tensor([[False, False, False, False, True], [True, True, False, False, False]])

    metrics = _timing_label_diagnostics(logits, labels)

    assert metrics["wait_label_count"] == 1
    assert metrics["go_label_count"] == 1
    assert metrics["wait_target_mass"] == pytest.approx(torch.softmax(logits[0], 0)[4].item())
    assert metrics["go_target_mass"] == pytest.approx(torch.softmax(logits[1], 0)[:2].sum().item())
    assert metrics["wait_logit_margin"] < 0
    assert metrics["go_logit_margin"] > 0


def test_public_shelter_helper_solves_each_yield_timing_bucket(config):
    """Closed-loop train holdout, including occlusion after stepping into a pocket."""
    from collections import Counter

    from ather_exploration.training.public_route import public_route
    from ather_exploration.worlds.p4_tasks import group, p4_scenario

    outcomes = Counter()
    fatal_labels = 0
    longest_wait = 0
    for seed in lesson_seeds(config.skills.train_count, 3, probe=True):
        if group(p4_scenario("P4a", seed))["encounter_family"] != "yield_alcoves":
            continue
        env = configured_skill_env("P4a", seed, config.skills)
        try:
            obs, _ = env.reset()
            consecutive_wait = 0
            for _ in range(128):
                label = public_p4a_timing(obs)
                route = public_route(obs) if label is None else None
                action = (
                    int(np.flatnonzero(label["actions"])[0])
                    if label
                    else int(np.flatnonzero(route["actions"])[0])
                    if route
                    else 4
                )
                consecutive_wait = consecutive_wait + 1 if action == 4 else 0
                longest_wait = max(longest_wait, consecutive_wait)
                obs, _, terminated, truncated, info = env.step(action)
                fatal_labels += bool(label and info["transition"]["died"])
                if terminated or truncated:
                    outcomes[(seed % 3, bool(info["skill"]["success"]))] += 1
                    break
        finally:
            env.close()
    assert sum(outcomes[(bucket, True)] for bucket in range(3)) >= 9
    assert all(outcomes[(bucket, True)] > 0 for bucket in range(3))
    assert fatal_labels == 0
    assert longest_wait <= 8  # No episode is labeled to WAIT indefinitely.


def test_p4a_validation_reports_wait_and_go_adherence_by_family(config, monkeypatch):
    from ather_exploration.evaluation.skills import evaluate_skill
    from ather_exploration.types import Action
    from ather_exploration.worlds import skill_tasks
    from ather_exploration.worlds.p4_tasks import group, p4_scenario

    original_pool = skill_tasks.skill_pool
    selected = {}
    for seed, record in original_pool("P4a", config.skills.validation_count, True, p4=True):
        family = group(p4_scenario("P4a", seed))["encounter_family"]
        selected.setdefault(family, (seed, record))
    short_pool = tuple(selected.values())
    monkeypatch.setattr(skill_tasks, "skill_pool", lambda *args, **kwargs: short_pool)

    class AlwaysWait:
        def act(self, observation, state, **kwargs):
            return Action.WAIT, state

    result = evaluate_skill(None, "P4a", config, agent=AlwaysWait())
    assert set(result["timing_adherence"]) == {"deterministic", "stochastic"}
    for mode in result["timing_adherence"].values():
        assert set(mode) == {"crossing", "bypass", "yield_alcoves"}
        assert sum(row["labels"] for row in mode.values()) > 0
        assert all(row["wait_follow_rate"] == 1.0 for row in mode.values() if row["wait_labels"])
        assert all(row["go_follow_rate"] == 0.0 for row in mode.values() if row["go_labels"])


def test_balanced_timing_bootstrap_keeps_successful_current_task_only(config):
    from collections import Counter
    from types import SimpleNamespace

    from ather_exploration.training.timing_teaching import TimingTeaching, initialize_timing

    teacher = TimingTeaching(config)
    initialize_timing(SimpleNamespace(timing_teaching=teacher), config)
    assert all(value > 0 for value in teacher.bootstrap_success.values())
    sources = Counter(
        teacher.family_by_seed[key[2]]
        for memory in teacher.memories.values()
        for key, bucket in memory.buckets.items()
        if bucket
    )
    assert set(sources) == {"crossing", "bypass", "yield_alcoves"}
    assert all(value > 0 for value in teacher.p4a_family_counts().values())
    assert all(key[1] == "P4a" for memory in teacher.memories.values() for key in memory.buckets)
    assert len(teacher.sample(60)) == 60


def test_audited_old_recovery_keeps_mixture():
    from pathlib import Path

    from ather_exploration.agents.route_ppo import RoutePPO
    from ather_exploration.training.checkpoints import inspect_checkpoint

    path = Path("artifacts/diagnostics/p4_recovery_audit/checkpoints/step_1654784")
    if not path.exists():
        pytest.skip("Audited historical checkpoint unavailable")
    checked, _ = inspect_checkpoint(path, inference=True)
    model = RoutePPO.load(checked / "model.zip", device="cpu")
    assert model.policy.threat_mode == "mixture"
    assert model.policy.threat_exploration == pytest.approx(0.0875)
