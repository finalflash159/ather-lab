"""P4 data allocation and transfer tests; no optimizer updates."""

from collections import Counter
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ather_exploration.training.config import TrainingConfig, read_training_config
from ather_exploration.training.skill_curriculum import SkillController
from ather_exploration.training.skill_environments import SkillTrainingEnv
from ather_exploration.training.threat_training import (
    SourceTelemetry,
    calibrate_actor,
    distribution_summary,
    p4a_worker_family,
    worker_source,
)


@pytest.mark.parametrize(
    "task,expected",
    [
        ("P4a", {"P4a": 12, "P3c": 4}),
        ("P4b", {"P4b": 12, "P4a": 2, "P3c": 2}),
        ("P4c", {"P4c": 12, "P4b": 2, "P3c": 2}),
    ],
)
def test_quota(task, expected):
    assert Counter(worker_source(task, i, 16) for i in range(16)) == expected
    assert Counter(worker_source(task, i, 32) for i in range(32)) == {
        k: v * 2 for k, v in expected.items()
    }


def test_balanced_p4a_workers_expose_all_families_from_first_rollout():
    config = read_training_config("ather_exploration/resources/training/skills_p4.yaml")
    assert Counter(p4a_worker_family(i, 16) for i in range(12)) == {
        "crossing": 4,
        "bypass": 4,
        "yield_alcoves": 4,
    }
    for worker in range(12):
        env = SkillTrainingEnv(config, worker)
        env.set_controller(asdict(SkillController(index=8, p4_enabled=True)))
        try:
            env.reset()
            assert env.encounter_family == p4a_worker_family(worker, 16)
            _, _, _, _, info = env.step(4)
            assert info["encounter_family"] == env.encounter_family
        finally:
            env.close()


def test_telemetry_reports_real_p4a_family_transitions():
    telemetry = SourceTelemetry()
    telemetry.observe(
        None,
        {
            "infos": [
                {"source_task": "P4a", "encounter_family": "crossing"},
                {"source_task": "P4a", "encounter_family": "bypass"},
                {"source_task": "P4a", "encounter_family": "yield_alcoves"},
                {"source_task": "P3c"},
            ],
            "actions": np.array([0, 1, 4, 2]),
        },
        calls=1,
    )
    metrics = telemetry.drain()
    assert metrics["skill_train/P4a/transition_fraction"] == 0.75
    for family in ("crossing", "bypass", "yield_alcoves"):
        assert metrics[f"skill_train/P4a/{family}/transitions"] == 1
        assert metrics[f"skill_train/P4a/{family}/transition_fraction"] == 0.25
    assert metrics["skill_train/P4a/yield_alcoves/action_wait_fraction"] == 1.0
    assert telemetry.drain() == {}


def test_reject_invalid_quota():
    for task, worker, n in (("P4a", 0, 1), ("P3c", 0, 16), ("P4a", 16, 16)):
        with pytest.raises(ValueError):
            worker_source(task, worker, n)
    config = read_training_config("ather_exploration/resources/training/skills_p4.yaml")
    payload = config.model_dump(mode="json")
    payload["n_envs"] = 4
    with pytest.raises(ValueError, match="quota"):
        TrainingConfig.model_validate(payload)


def test_workers_do_not_change_source_on_short_episode_reset():
    config = read_training_config("ather_exploration/resources/training/skills_p4.yaml")
    for worker, source in ((0, "P4a"), (12, "P3c")):
        env = SkillTrainingEnv(config, worker)
        env.set_controller(asdict(SkillController(index=8, p4_enabled=True)))
        try:
            for _ in range(6):
                env.reset()
                assert env.task == source
                _, _, _, _, info = env.step(4)
                assert info["source_task"] == source
        finally:
            env.close()


def test_telemetry_counts_unfinished_transitions_and_resets():
    telemetry = SourceTelemetry()
    locals_ = {
        "infos": [{"source_task": "P4a"}] * 12 + [{"source_task": "P3c"}] * 4,
        "actions": np.array([4] * 12 + [0] * 4),
    }
    for _ in range(256):
        telemetry.observe(None, locals_, 1)  # Skip optional entropy probe.
    result = telemetry.drain()
    assert result["skill_train/P4a/transitions"] == 3072
    assert result["skill_train/P3c/transitions"] == 1024
    assert result["skill_train/P4a/transition_fraction"] == 0.75
    assert result["skill_train/P4a/action_wait_fraction"] == 1
    assert telemetry.drain() == {}


def test_calibration_uses_train_only_preserves_order_and_value(monkeypatch):
    import ather_exploration.worlds.skill_tasks as tasks

    calls = []

    class Env:
        def reset(self):
            return {"x": np.array([1.0], dtype=np.float32)}, {}

        def step(self, action):
            assert action == 4
            return self.reset()[0], 0, False, False, {}

        def close(self):
            pass

    def pool(task, count, **kwargs):
        calls.append(kwargs)
        return ((0, "a"), (1, "b"))

    monkeypatch.setattr(tasks, "skill_pool", pool)
    monkeypatch.setattr(tasks, "configured_skill_env", lambda *a: Env())

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_net = torch.nn.Linear(1, 5)
            self.critic = torch.nn.Linear(1, 1)
            with torch.no_grad():
                self.action_net.weight.copy_(
                    torch.tensor([[100.0], [50.0], [0.0], [-50.0], [-100.0]])
                )
                self.action_net.bias.zero_()

        def set_training_mode(self, mode):
            self.train(mode)

        def obs_to_tensor(self, obs):
            return torch.tensor(obs["x"]).reshape(1, 1), False

        def get_distribution(self, x):
            return SimpleNamespace(
                distribution=torch.distributions.Categorical(logits=self.action_net(x))
            )

    policy = Policy()
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    audit = calibrate_actor(
        SimpleNamespace(policy=policy), SimpleNamespace(skills=SimpleNamespace(train_count=2))
    )
    assert calls == [{"p4": True}]
    assert 0 < audit["scale"] < 1
    assert audit["observations"] == 6
    assert audit["after"]["wait_p10"] >= 0.01
    assert audit["after"]["entropy_mean"] >= 0.8
    assert policy.training
    for k, v in policy.state_dict().items():
        torch.testing.assert_close(
            v, before[k] * audit["scale"] if k.startswith("action_net") else before[k]
        )
    assert policy.get_distribution(torch.ones(1, 1)).distribution.probs.argmax() == 0
    assert distribution_summary(torch.zeros(2, 5))["wait_mean"] == pytest.approx(0.2)


def test_disabled_teaching_never_calls_auxiliary(monkeypatch):
    from stable_baselines3 import PPO

    from ather_exploration.agents.route_ppo import RoutePPO
    from ather_exploration.training import route_teaching

    monkeypatch.setattr(PPO, "train", lambda self: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Auxiliary training is disabled")

    monkeypatch.setattr(route_teaching, "teach", forbidden)
    model = RoutePPO.__new__(RoutePPO)
    model.route_memory = route_teaching.RouteMemory(seed=0)
    model.teaching_batches = 0
    model._logger = SimpleNamespace(record=lambda *args: None)
    model.train()
    assert model.route_memory.aux_updates == 0
