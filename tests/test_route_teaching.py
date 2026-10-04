"""Teaching reservoirs, separate optimization and checkpoint continuation."""

import copy

import numpy as np
import pytest
import torch
from test_recovery import observation, prepared

from ather_exploration.agents.route_ppo import RoutePPO
from ather_exploration.training.config import read_training_config
from ather_exploration.training.route_teaching import RouteMemory, teach


def test_reservoir_considers_late_states_and_keeps_origins():
    memory = RouteMemory(per_map=4, seed=1)
    labels = np.array([1, 0, 0, 0, 0], bool)
    for i in range(100):
        obs = observation()
        obs["memory"][6, 4, 4] = i / 100
        memory.offer(obs, labels, ("P3c", 17), "learner")
    memory.offer(observation(), labels, ("P3c", 17), "teacher")
    assert len(memory) == 5
    rows = memory.sample(40)
    assert any(r["memory"][6, 4, 4] > 0.5 for r in rows)
    assert sum(r["memory"][6, 4, 4] == 0 for r in rows) >= 20
    for r in rows:
        assert r["memory"].dtype == np.float32
        assert np.array_equal(r["labels"], labels)


def test_dataset_and_rng_survive_model_checkpoint(tmp_path):
    model = prepared(RoutePPO)
    model.route_memory = RouteMemory(seed=7)
    model.route_memory.offer(observation(), np.array([1, 0, 0, 0, 0], bool), ("P3c", 2), "teacher")
    model.route_memory.initialized = True
    model.route_memory.aux_updates = 12
    model.save(tmp_path / "model")
    loaded = RoutePPO.load(tmp_path / "model")
    assert loaded.route_memory.initialized
    assert loaded.route_memory.aux_updates == 12
    assert loaded.route_memory.buckets == model.route_memory.buckets
    assert loaded.route_memory.rng.integers(10000) == model.route_memory.rng.integers(10000)


def teaching_model(monkeypatch):
    model = prepared(RoutePPO)
    model.route_memory = RouteMemory()
    rows = [{"x": np.ones(3, np.float32), "labels": np.array([0, 1, 0, 0, 0], bool)}] * 128
    monkeypatch.setattr(model.route_memory, "sample", lambda n: rows[:n])
    return model


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_nested_equal(a[k], b[k])
    elif isinstance(a, list):
        for x, y in zip(a, b, strict=True):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_aux_rolls_back_weights_and_adam_after_excessive_update(monkeypatch):
    model = teaching_model(monkeypatch)
    teach(model, batches=1, max_kl=1)  # Populate Adam moments before rollback.
    weights = copy.deepcopy(model.policy.state_dict())
    optimizer = copy.deepcopy(model.policy.optimizer.state_dict())
    count = model.route_memory.aux_updates
    result = teach(model, batches=1, max_kl=0)
    assert result["rejected"] == 1 and result["accepted"] == 0
    assert_nested_equal(weights, model.policy.state_dict())
    assert_nested_equal(optimizer, model.policy.optimizer.state_dict())
    assert model.route_memory.aux_updates == count


def test_separate_aux_improves_label_without_touching_rollout(monkeypatch):
    model = teaching_model(monkeypatch)
    old = {
        k: getattr(model.rollout_buffer, k).copy()
        for k in ("actions", "rewards", "log_probs", "advantages")
    }
    x, _ = model.policy.obs_to_tensor({"x": np.ones(3, np.float32)})
    before = float(model.policy.get_distribution(x).distribution.probs[0, 1].detach())
    result = teach(model, batches=8, max_kl=0.03)
    after = float(model.policy.get_distribution(x).distribution.probs[0, 1].detach())
    assert result["accepted"] > 0 and after > before
    assert model.num_timesteps == 0
    for k, v in old.items():
        assert np.array_equal(v, getattr(model.rollout_buffer, k))


def test_teaching_budget_aligns_evaluation_and_rollout():
    cfg = read_training_config("ather_exploration/resources/training/skills_p3c_teaching.yaml")
    assert cfg.recovery.additional_steps == 507904
    assert cfg.recovery.additional_steps % cfg.skills.eval_interval == 0
    assert cfg.recovery.additional_steps % (cfg.n_envs * cfg.n_steps) == 0
    assert cfg.recovery.parent_steps + cfg.recovery.additional_steps == 2080768
    data = cfg.model_dump()
    data["recovery"]["parent_steps"] = 1835008
    with pytest.raises(ValueError):
        type(cfg).model_validate(data)


def test_invalid_post_update_distribution_restores_state(monkeypatch):
    model = teaching_model(monkeypatch)
    weights = copy.deepcopy(model.policy.state_dict())
    optimizer = copy.deepcopy(model.policy.optimizer.state_dict())

    def corrupt(*args, **kwargs):
        with torch.no_grad():
            model.policy.action_net.weight.fill_(float("nan"))

    monkeypatch.setattr(model.policy.optimizer, "step", corrupt)
    with pytest.raises(ValueError):
        teach(model, batches=1)
    assert_nested_equal(weights, model.policy.state_dict())
    assert_nested_equal(optimizer, model.policy.optimizer.state_dict())


def test_ppo_then_auxiliary_and_no_legacy_loss(monkeypatch):
    model = teaching_model(monkeypatch)
    model.route_coefficient = 0.02  # Must not enable the legacy combined objective.
    model.route_samples = [({"x": np.ones(3, np.float32)}, np.ones(5, bool))]
    before = model._n_updates
    model.train()
    assert model._n_updates == before + model.n_epochs
    assert model.route_memory.aux_updates > 0
    assert "teaching/accepted" in model.logger.name_to_value
    assert "train/route_loss" not in model.logger.name_to_value


def test_warmup_excludes_holdout_and_does_not_repeat(monkeypatch):
    from ather_exploration.training import recovery, route_teaching
    from ather_exploration.worlds import skill_tasks

    model = prepared(RoutePPO)
    model.route_memory = RouteMemory()
    config = read_training_config("ather_exploration/resources/training/skills_p3c_teaching.yaml")
    monkeypatch.setattr(recovery, "probe_seeds", lambda config, task: list(range(32)))
    checked = []
    monkeypatch.setattr(
        route_teaching, "check_teaching_policy", lambda m, c, seeds: checked.append(seeds) or {}
    )
    monkeypatch.setattr(route_teaching, "teach", lambda *a, **kw: {})
    created = []

    class Env:
        def reset(self):
            return observation(), {}

        def step(self, action):
            return observation(), 0, True, False, {}

        def close(self):
            pass

    def make(task, seed, skills, phase):
        created.append(seed)
        return Env()

    monkeypatch.setattr(skill_tasks, "configured_skill_env", make)
    route_teaching.initialize_teaching(model, config)
    assert created == list(range(16))
    assert checked == [list(range(16, 32))] * 2
    assert model.route_memory.collection_steps == 16
    assert model.num_timesteps == 0
    route_teaching.initialize_teaching(model, config)
    assert len(created) == 16
