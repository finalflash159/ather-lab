"""Temporal input, train-only lessons, policy likelihoods and retention safety."""

import copy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ather_exploration.agents.learning import build_model, schema_signature
from ather_exploration.training.config import TrainingConfig, read_training_config
from ather_exploration.training.skill_curriculum import SkillController
from ather_exploration.training.threat_lessons import lesson_scenario, lesson_seeds, observe_lesson
from ather_exploration.worlds.p4_tasks import p4_pool, p4_scenario, timing_geometry_identity
from ather_exploration.worlds.skill_tasks import configured_skill_env


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return read_training_config("ather_exploration/resources/training/skills_p4.yaml")


def test_actual_alias_resolved_and_reset_clean(config):
    a, b = [configured_skill_env("P4a", seed, config.skills) for seed in (75, 77)]
    try:
        a.reset()
        b.reset()
        for action in (4, 4, 2):
            oa, *_ = a.step(action)
            ob, *_ = b.step(action)
        # v2 saw precisely the same input. The added public frame resolves it.
        for key in ("local", "state"):
            np.testing.assert_array_equal(oa[key], ob[key])
        np.testing.assert_array_equal(oa["memory"][:14], ob["memory"][:14])
        assert not np.array_equal(oa["memory"][14:], ob["memory"][14:])
        assert not a.reset()[0]["memory"][12:].any()
        assert schema_signature(a.observation_space)["version"] == 4
    finally:
        a.close()
        b.close()


def test_history_masks_stale_monsters_and_old_schema(config):
    from ather_exploration.environment.threat_history import ThreatHistory

    memory = np.zeros((12, 81, 81), np.float32)
    memory[9, 40, 40] = 1  # Remembered, not currently visible.
    assert not ThreatHistory.frame({"memory": memory}).any()
    skills = config.skills.model_copy(
        update={
            "p4": config.skills.p4.model_copy(
                update={
                    "history_frames": 1,
                    "recovery": False,
                    "step_cost": 0.0,
                }
            )
        }
    )
    env = configured_skill_env("P4a", 0, skills)
    try:
        obs, _ = env.reset()
        old = ThreatHistory.frame(obs)
        after, *_ = env.step(4)
        assert after["memory"].shape[0] == 14
        np.testing.assert_array_equal(after["memory"][12:], old)
        assert schema_signature(env.observation_space)["version"] == 3
    finally:
        env.close()


@pytest.mark.parametrize("level", range(4))
def test_train_probe_geometry_disjoint_and_lessons_reproducible(config, level):
    train = lesson_seeds(256, level)
    probe = lesson_seeds(256, level, True)
    identities = lambda seeds: {timing_geometry_identity(p4_scenario("P4a", s)) for s in seeds}
    assert not identities(train) & identities(probe)
    assert set(train) | set(probe) <= {s for s, _ in p4_pool("P4a", 256)}
    assert not identities(train) & identities([s for s, _ in p4_pool("P4a", 64, True)])
    for seed in (*train[:6], *probe[:6]):
        base = p4_scenario("P4a", seed)
        lesson = lesson_scenario(base, level)
        assert lesson == lesson_scenario(base, level)
        assert lesson.routes == base.routes and lesson.phases == base.phases
        assert lesson.spawn not in lesson.routes[0]
        assert lesson.terrain[lesson.spawn[1]][lesson.spawn[0]] == "."


def test_curriculum_promotion_resume_and_no_gate_reduction():
    controller = SkillController(index=8, p4_enabled=True)
    result = {"level": 0, "passed": True}
    assert not observe_lesson(controller, result, 1654784)
    restored = SkillController(**asdict(controller))
    assert observe_lesson(restored, result, 1671168)
    assert restored.task == "P4a" and restored.threat_level == 1
    assert restored.passed == 0 and restored.threat_level_start == 1671168
    assert not observe_lesson(restored, result, 1687552)  # stale result
    assert restored.budget == 1048576


@pytest.mark.parametrize("task", ("P4a", "P4b", "P4c"))
def test_step_cost_exactly_once_including_wait(config, task):
    env = configured_skill_env(task, 0, config.skills)
    try:
        env.reset()
        _, reward, _, _, info = env.step(4)
        terms = info["skill"]["reward_components"]
        assert terms["step_cost"] == -0.002
        assert reward == pytest.approx(sum(terms.values()))
    finally:
        env.close()


def test_threat_exploration_is_on_policy_public_and_preserves_greedy(config):
    config = config.model_copy(
        update={
            "skills": config.skills.model_copy(
                update={"p4": config.skills.p4.model_copy(update={"timing": False})}
            )
        }
    )
    env = configured_skill_env("P4a", 0, config.skills)
    model = build_model(config.model_copy(update={"batch_size": 256}), env)
    try:
        obs, _ = env.reset()
        obs["memory"][9] = 0
        tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            before = model.policy.get_distribution(tensor).distribution.probs.clone()
            obs["memory"][9, 40, 40] = 1
            obs["memory"][8, 40, 40] = 1
            tensor, _ = model.policy.obs_to_tensor(obs)
            latent = model.policy.mlp_extractor.forward_actor(model.policy.extract_features(tensor))
            raw = model.policy._get_action_dist_from_latent(latent).distribution.probs.clone()
            mixed = model.policy.get_distribution(tensor).distribution.probs
            torch.testing.assert_close(mixed, 0.9 * raw + 0.02)
            assert mixed.argmax() == raw.argmax()
            action, values, logprob = model.policy(tensor)
            checked_values, checked_logprob, _ = model.policy.evaluate_actions(tensor, action)
            torch.testing.assert_close(logprob, checked_logprob)
            torch.testing.assert_close(values, checked_values)
            obs["memory"][8, 40, 40] = 0
            tensor, _ = model.policy.obs_to_tensor(obs)
            latent = model.policy.mlp_extractor.forward_actor(model.policy.extract_features(tensor))
            expected = model.policy._get_action_dist_from_latent(latent).distribution.probs.clone()
            torch.testing.assert_close(
                model.policy.get_distribution(tensor).distribution.probs, expected
            )
            assert torch.isfinite(before).all()
    finally:
        env.close()


def test_guarded_retention_updates_and_rolls_back_adam(monkeypatch):
    from test_recovery import prepared
    from test_route_teaching import assert_nested_equal

    from ather_exploration.agents.route_ppo import RoutePPO
    from ather_exploration.training.route_teaching import RouteMemory
    from ather_exploration.training.threat_retention import retain

    model = prepared(RoutePPO)
    model.threat_retention = SimpleNamespace(
        memory=RouteMemory(), guard=RouteMemory(), batches=2, max_kl=1
    )
    rows = [{"x": np.ones(3, np.float32), "labels": np.array([0, 1, 0, 0, 0], np.float32)}] * 64
    for memory in (model.threat_retention.memory, model.threat_retention.guard):
        monkeypatch.setattr(memory, "sample", lambda n: rows[:n])
    model.threat_retention.memory.buckets = {"test": [None]}
    result = retain(model)
    assert result["accepted"] == 2
    weights = copy.deepcopy(model.policy.state_dict())
    optimizer = copy.deepcopy(model.policy.optimizer.state_dict())
    updates = model.threat_retention.memory.aux_updates
    model.threat_retention.max_kl = 0
    result = retain(model)
    assert result["rejected"] == 1 and result["accepted"] == 0
    assert_nested_equal(weights, model.policy.state_dict())
    assert_nested_equal(optimizer, model.policy.optimizer.state_dict())
    assert model.threat_retention.memory.aux_updates == updates


def test_invalid_recovery_config_rejected(config):
    payload = config.model_dump(mode="json")
    payload["skills"]["p4"]["soften_actor"] = True
    with pytest.raises(ValueError, match="recovery requires"):
        TrainingConfig.model_validate(payload)


def test_real_rollout_retention_save_load_and_teacher_unchanged(config, tmp_path):
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from ather_exploration.agents.route_ppo import RoutePPO
    from ather_exploration.training.skill_environments import SkillTrainingEnv
    from ather_exploration.training.threat_retention import ThreatRetention
    from ather_exploration.worlds.skill_tasks import skill_pool

    config = config.model_copy(update={"n_envs": 8, "n_steps": 16, "batch_size": 32, "n_epochs": 1})

    def factory(worker):
        env = SkillTrainingEnv(config, worker)
        env.set_controller(asdict(SkillController(index=8, p4_enabled=True)))
        return env

    env = DummyVecEnv([lambda i=i: factory(i) for i in range(8)])
    model = build_model(config, env)
    seeds = [s for s, _ in skill_pool("P3c", config.skills.train_count)]
    model.threat_retention = ThreatRetention(model.policy, 0, seeds)
    frozen = copy.deepcopy(model.threat_retention.teacher.state_dict())
    model.threat_retention.batches = 1

    class Collector(BaseCallback):
        def _on_step(self):
            self.model.threat_retention.collect(
                self.model._last_obs, self.locals["infos"], self.n_calls
            )
            return True

    try:
        model.learn(128, callback=Collector())  # Tiny disposable integration test, no run/W&B.
        assert model.num_timesteps == 128
        assert len(model.threat_retention.memory) > 0
        assert "retention/accepted" in model.logger.name_to_value
        for key, value in frozen.items():
            torch.testing.assert_close(
                value, model.threat_retention.teacher.state_dict()[key], rtol=0, atol=0
            )
        model.policy.threat_exploration = 0.05
        model.policy_kwargs["threat_exploration"] = 0.05
        model.save(tmp_path / "model")
        restored = RoutePPO.load(tmp_path / "model", device="cpu")
        assert restored.policy.threat_exploration == 0.05
        assert restored.threat_retention.memory.buckets == model.threat_retention.memory.buckets
        assert restored.threat_retention.train_seeds == tuple(seeds)
        obs = env.reset()
        np.testing.assert_array_equal(
            model.predict(obs, deterministic=True)[0], restored.predict(obs, deterministic=True)[0]
        )
        assert next(restored.threat_retention.teacher.parameters()).device.type == "cpu"
    finally:
        env.close()


def test_exploration_schedule_persists_policy_constructor(config):
    config = config.model_copy(
        update={
            "skills": config.skills.model_copy(
                update={"p4": config.skills.p4.model_copy(update={"timing": False})}
            )
        }
    )
    from ather_exploration.training.threat_training import update_threat_exploration

    model = SimpleNamespace(num_timesteps=1638400, policy=SimpleNamespace(), policy_kwargs={})
    assert update_threat_exploration(model, config) == pytest.approx(0.1)
    model.num_timesteps += 65536
    assert update_threat_exploration(model, config) == pytest.approx(0.05)
    assert model.policy_kwargs["threat_exploration"] == model.policy.threat_exploration
    model.num_timesteps += 65536
    assert update_threat_exploration(model, config) == 0
    model.num_timesteps += 1000000
    assert update_threat_exploration(model, config) == 0


def test_retention_dedup_includes_state_and_rejects_validation(config):
    from ather_exploration.training.threat_retention import PolicyMemory, ThreatRetention

    memory = PolicyMemory(per_map=4)
    obs = {"memory": np.zeros((16, 3, 3), np.float32), "state": np.zeros(17, np.float32)}
    target = np.ones(5, np.float32) / 5
    memory.offer(obs, target, ("P3c", 1), "teacher")
    obs["state"][0] = 1
    memory.offer(obs, target, ("P3c", 1), "teacher")
    assert len(memory) == 2
    retention = ThreatRetention.__new__(ThreatRetention)
    retention.memory = memory
    retention.train_seeds = (1,)
    retention.targets = lambda obs: [target]
    batch = {key: value[None] for key, value in obs.items()}
    with pytest.raises(ValueError, match="train observations"):
        retention.collect(
            batch, [{"teaching_safe_source": True, "teaching_seed": 999, "source_task": "P3c"}], 16
        )
    assert len(memory) == 2


def test_uploaded_lessons_record_actual_variant(config, tmp_path):
    from ather_exploration.training.threat_lessons import export_lessons
    from ather_exploration.worlds.scenarios import read_record

    manifest = export_lessons(config, tmp_path)
    seed, identity = manifest["near_crossing"]["train"][0]
    record = read_record(tmp_path / "P4a" / "lessons" / "near_crossing" / "train" / f"{seed}.json")
    assert record["identity"] == identity
    assert (
        tuple(record["scenario"]["spawn"])
        == lesson_scenario(p4_scenario("P4a", seed), 0, timing=True).spawn
    )
    assert {s for s, _ in manifest["full"]["train"]}.isdisjoint(
        s for s, _ in manifest["full"]["train_holdout"]
    )


def test_branch_budget_is_one_million_shared_across_tasks(config):
    assert config.total_timesteps - 1638400 == 1048576
    assert (config.total_timesteps - 1638400) % config.skills.eval_interval == 0
    assert config.total_timesteps % (config.n_envs * config.n_steps) == 0
    TrainingConfig.model_validate(config.model_dump(mode="json"))
    payload = config.model_dump(mode="json")
    payload["total_timesteps"] = 1638400
    with pytest.raises(ValueError, match="budgets"):
        TrainingConfig.model_validate(payload)
