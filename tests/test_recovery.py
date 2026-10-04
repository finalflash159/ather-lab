"""Recovery routing, archive, optimizer isolation and bounded promotion."""

import copy

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from ather_exploration.agents.route_ppo import RoutePPO
from ather_exploration.training.config import RecoveryConfig, read_training_config
from ather_exploration.training.public_route import public_route, route_loss
from ather_exploration.training.recovery_archive import (
    RecoveryArchive,
    ResolveTracker,
    StagnationCapture,
)
from ather_exploration.training.skill_curriculum import SkillController


def observation():
    m = np.zeros((12, 9, 9), dtype=np.float32)
    m[0, 2:7, 2:7] = 1
    m[2, 2:7, 2:7] = 1
    m[7, 4, 4] = 1
    m[3, 2, 6] = 1
    return {
        "memory": m,
        "local": np.zeros((6, 9, 9), np.float32),
        "state": np.zeros(17, np.float32),
    }


def test_routes_ties_pending_priority_and_unknown():
    o = observation()
    r = public_route(o)
    assert r["kind"] == "poi" and r["distance"] == 4
    assert r["actions"].tolist() == [True, False, True, False, False]
    # Wall blocks north; east remains shortest. A hidden pending marker is ignored.
    o["memory"][1, 3, 4] = 1
    assert public_route(o)["actions"].tolist() == [False, False, True, False, False]
    o["memory"][3, 2, 6] = 0
    o["memory"][3, 0, 0] = 1
    assert public_route(o)["kind"] == "frontier"
    o["memory"][2] = 0
    assert public_route(o) is None


def test_route_labels_reduce_distance_with_frozen_memory():
    o = observation()
    route = public_route(o)
    for action, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, 1), (0, -1))):
        if route["actions"][action]:
            updated = copy.deepcopy(o)
            updated["memory"][7] = 0
            updated["memory"][7, 4 + dr, 4 + dc] = 1
            assert public_route(updated)["distance"] == route["distance"] - 1


def test_loss_uses_set_and_actor_gradient():
    logits = torch.zeros((1, 5), requires_grad=True)
    labels = torch.tensor([[True, False, True, False, False]])
    loss = route_loss(logits, labels)
    assert float(loss.detach()) == pytest.approx(-np.log(0.4))
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0
    assert logits.grad[0, 0] == logits.grad[0, 2]
    with pytest.raises(ValueError, match="Empty"):
        route_loss(logits, torch.zeros_like(labels))


def test_archive_captures_failed_stagnation_lookback_and_roundtrip():
    config = RecoveryConfig()
    o = observation()
    capture = StagnationCapture(config, "P3b", 17, o)
    item = None
    for _ in range(16):
        item = capture.step(4, o, 256)
    assert len(item["actions"]) == 8
    archive = RecoveryArchive(config, np.random.default_rng(1))
    archive.offer(item)  # No successful suffix requirement.
    archive.outcome(item, False)
    assert archive.pools["P3b"][0][0]["attempts"] == 1
    with pytest.raises(ValueError, match="active source"):
        archive.offer({**item, "source_task": "P3a"})
    restored = RecoveryArchive(config, np.random.default_rng(2))
    restored.restore(archive.state())
    assert archive.state() == restored.state()
    assert len(restored.pools["P3c"][0]) == 0


def test_resolve_requires_original_pending_poi():
    o = observation()
    tracker = ResolveTracker(o)
    o["memory"][4, 3, 3] = 1
    assert not tracker.step(o)
    o["memory"][4, 2, 6] = 1
    assert tracker.step(o)


def test_resolve_window_expires():
    o = observation()
    tracker = ResolveTracker(o)
    for _ in range(64):
        assert not tracker.step(o)
    o["memory"][4, 2, 6] = 1
    assert not tracker.step(o)


def test_budget_reset_streak_and_promotion():
    controller = SkillController(index=6, phase_start=1048576, recovery_p3b_budget=131072)
    assert not controller.observe(True, 1048576 + 16384)
    assert not controller.observe(True, 1048576 + 32768)
    assert controller.observe(True, 1048576 + 65536)
    assert controller.task == "P3c" and controller.budget == 524288 and controller.passed == 0
    controller = SkillController(index=6, phase_start=1048576, recovery_p3b_budget=131072)
    assert not controller.observe(False, 1179648)
    assert controller.failed


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Dict({"x": gym.spaces.Box(-1, 1, (3,), dtype=np.float32)})
    action_space = gym.spaces.Discrete(5)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {"x": np.ones(3, np.float32)}, {}

    def step(self, action):
        return {"x": np.ones(3, np.float32)}, 0.0, False, False, {}


def prepared(cls):
    torch.set_num_threads(1)
    model = cls(
        "MultiInputPolicy",
        TinyEnv(),
        n_steps=8,
        batch_size=4,
        n_epochs=2,
        seed=17,
        learning_rate=0.001,
        target_kl=0.03,
        ent_coef=0,
    )
    model.set_logger(configure(None, []))
    model._current_progress_remaining = 1
    b = model.rollout_buffer
    obs = {"x": np.ones((1, 3), np.float32)}
    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(obs)
        value, logprob, _ = model.policy.evaluate_actions(tensor, torch.tensor([0]))
    for _ in range(8):
        b.add(obs, np.array([0]), np.array([0.0]), np.array([False]), value, logprob)
    b.compute_returns_and_advantage(value, np.array([False]))
    return model


def test_zero_aux_matches_upstream_exactly():
    reference, candidate = prepared(PPO), prepared(RoutePPO)
    candidate.route_coefficient = 0
    candidate.route_samples = [({"x": np.ones(3, np.float32)}, np.array([1, 0, 0, 0, 0], bool))]
    np.random.seed(4)
    reference.train()
    np.random.seed(4)
    candidate.train()
    for key, tensor in reference.policy.state_dict().items():
        assert torch.equal(tensor, candidate.policy.state_dict()[key])


def test_aux_same_optimizer_does_not_mutate_rollout_and_roundtrips(tmp_path):
    model = prepared(RoutePPO)
    model.route_coefficient = 0.02
    model.route_samples = [({"x": np.ones(3, np.float32)}, np.array([0, 1, 0, 0, 0], bool))]
    old = {
        key: getattr(model.rollout_buffer, key).copy()
        for key in ("actions", "log_probs", "advantages", "rewards")
    }
    before = model.policy.action_net.weight.detach().clone()
    model.train()
    assert not torch.equal(before, model.policy.action_net.weight)
    for key, value in old.items():
        assert np.array_equal(value.reshape(-1), getattr(model.rollout_buffer, key).reshape(-1))
    assert model.logger.name_to_value["train/route_loss"] > 0
    model.save(tmp_path / "model")
    loaded = RoutePPO.load(tmp_path / "model")
    assert loaded.route_samples is None
    assert loaded.route_coefficient == 0.02
    assert (
        loaded.policy.optimizer.state_dict()["state"].keys()
        == model.policy.optimizer.state_dict()["state"].keys()
    )


def test_recovery_config_is_separate():
    config = read_training_config("ather_exploration/resources/training/skills_p3_recovery.yaml")
    assert config.recovery and config.p3_resume is None
    data = config.model_dump()
    data["skills"]["stop_after"] = "P4"
    with pytest.raises(ValueError):
        type(config).model_validate(data)


def test_real_environment_replay_preserves_observation_and_excludes_prefix(monkeypatch):
    from dataclasses import asdict

    from ather_exploration.training.skill_environments import SkillTrainingEnv

    config = read_training_config("ather_exploration/resources/training/skills_p3_recovery.yaml")
    env = SkillTrainingEnv(config)
    env.set_controller(asdict(SkillController(index=6)))
    try:
        obs, _ = env.reset()
        while env.task != "P3b":
            obs, _ = env.reset()
        snapshots = [copy.deepcopy(obs)]
        for i in range(16):
            obs, _, _, _, info = env.step(4)
            snapshots.append(copy.deepcopy(obs))
            assert info["route_eligible"] == (i >= 8)
        items = [item for pool in env.archive.pools["P3b"] for item in pool]
        assert items
        item = items[0]
        assert len(item["actions"]) == 8
        monkeypatch.setattr(env.archive, "choose", lambda task: (copy.deepcopy(item), True))
        replayed, _ = env.reset()
        assert env.ticks == 0 and env.total == 0 and env.prefix_length == 8
        for key in replayed:
            assert np.array_equal(replayed[key], snapshots[8][key])
        _, _, _, _, info = env.step(4)
        assert info["route_eligible"] and env.ticks == 1
        item["observation"] = "bad"
        with pytest.raises(ValueError, match="hash mismatch"):
            env.reset()
    finally:
        env.close()


def test_callback_labels_before_action_and_clears_each_rollout(tmp_path):
    from types import SimpleNamespace

    from ather_exploration.training.skill_runner import SkillCallback

    config = read_training_config("ather_exploration/resources/training/skills_p3_recovery.yaml")
    callback = SkillCallback(config, tmp_path, SkillController(index=6), {})
    o = observation()

    class Env:
        def env_method(self, name):
            assert name == "drain"
            return [[]]

    model = SimpleNamespace(
        _last_obs={k: v[None] for k, v in o.items()},
        route_samples=[],
        num_timesteps=0,
        get_env=lambda: Env(),
    )
    callback.model = model
    callback.locals = {"infos": [{"route_eligible": True}], "rewards": np.zeros(1)}
    callback._on_step()
    assert len(model.route_samples) == 1
    assert model.route_samples[0][1].tolist() == [True, False, True, False, False]
    callback._on_step()
    assert len(model.route_samples) == 1
    callback._on_rollout_start()
    assert model.route_samples == []


def test_target_kl_prevents_aux_optimizer_step():
    model = prepared(RoutePPO)
    model.route_coefficient = 0.02
    model.route_samples = [({"x": np.ones(3, np.float32)}, np.array([0, 1, 0, 0, 0], bool))]
    model.rollout_buffer.log_probs[:] = -100
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    model.train()
    for key, tensor in before.items():
        assert torch.equal(tensor, model.policy.state_dict()[key])


@pytest.mark.parametrize("task", ["P3b", "P3c"])
def test_probes_cover_all_training_strata(task):
    from ather_exploration.training.recovery import probe_seeds
    from ather_exploration.worlds.p3_tasks import map_group, p3_scenario
    from ather_exploration.worlds.skill_tasks import skill_pool

    config = read_training_config("ather_exploration/resources/training/skills_p3_recovery.yaml")
    seeds = probe_seeds(config, task)
    pool = [s for s, _ in skill_pool(task, config.skills.train_count)]
    assert len(seeds) == len(set(seeds)) == 64
    assert set(seeds) <= set(pool)

    def group(seed):
        return tuple(map_group(p3_scenario(task, seed)).values())

    assert {group(s) for s in seeds} == {group(s) for s in pool}


def test_balanced_samples_quota_shortage_and_dedup():
    from ather_exploration.training.route_sampling import BalancedRouteSamples

    sampler = BalancedRouteSamples(8)
    for i in range(20):
        obs = observation()
        obs["state"][0] = i  # Ignored for map-state dedup.
        sampler.offer(
            obs,
            np.array([1, 0, 0, 0, 0], bool),
            disagreement=True,
            ordinary=True,
            source=("P3c", 0),
        )
    assert len(sampler.select()[0]) == 1
    for i in range(1, 12):
        obs = observation()
        obs["memory"][0, 0, 0] = i
        sampler.offer(
            obs,
            np.array([1, 0, 0, 0, 0], bool),
            disagreement=i < 6,
            ordinary=True,
            source=("P3c", i),
        )
    rows, wrong = sampler.select()
    assert len(rows) == 8 and wrong == 4
    only = BalancedRouteSamples(32)
    for i in range(40):
        obs = observation()
        obs["memory"][0, 0, 0] = i
        only.offer(
            obs,
            np.array([1, 0, 0, 0, 0], bool),
            disagreement=True,
            ordinary=False,
            source=("P3c", 7),
        )
    rows, wrong = only.select()
    assert len(rows) == wrong == 16  # Same seed cannot fill the rollout.


def test_p3c_branch_budget_persists_without_extension():
    from dataclasses import asdict

    controller = SkillController(index=7, phase_start=1572864, recovery_p3c_budget=262144)
    controller.observe(False, 1572864 + 131072)
    restored = SkillController(**asdict(controller))
    assert restored.budget == 262144 and restored.phase_start == 1572864
    restored.observe(False, 1835008)
    assert restored.failed
    passed = SkillController(index=7, phase_start=1572864, recovery_p3c_budget=262144)
    passed.observe(True, 1572864 + 49152)
    assert passed.observe(True, 1572864 + 65536)
    assert passed.task == "P4a"


def test_p3c_branch_keeps_reward_and_fixed_label_budget():
    old = read_training_config("ather_exploration/resources/training/skills_p3_recovery.yaml")
    new = read_training_config("ather_exploration/resources/training/skills_p3c_recovery.yaml")
    assert new.skills == old.skills
    assert new.recovery.route_coefficient == old.recovery.route_coefficient == 0.02
    assert new.recovery.label_limit == old.recovery.label_limit == 256
    assert new.recovery.parent_steps + new.recovery.additional_steps == 1835008


def test_balanced_unaccepted_early_visit_can_be_labeled_later():
    from ather_exploration.training.route_sampling import BalancedRouteSamples

    sampler = BalancedRouteSamples(8)
    obs = observation()
    labels = np.array([1, 0, 0, 0, 0], bool)
    sampler.offer(obs, labels, disagreement=False, ordinary=False, source=("P3c", 1))
    assert not sampler.select()[0]
    sampler.offer(obs, labels, disagreement=False, ordinary=True, source=("P3c", 1))
    assert len(sampler.select()[0]) == 1


def test_balanced_callback_does_not_discard_later_eligible_state(tmp_path):
    from types import SimpleNamespace

    from ather_exploration.training.route_sampling import BalancedRouteSamples
    from ather_exploration.training.skill_runner import SkillCallback

    config = read_training_config("ather_exploration/resources/training/skills_p3c_recovery.yaml")
    cb = SkillCallback(config, tmp_path, SkillController(index=7), {})
    cb.balanced_samples = BalancedRouteSamples(256)
    obs = observation()

    class Env:
        def env_method(self, name):
            return [[]]

    policy = SimpleNamespace(
        obs_to_tensor=lambda x: (x, True),
        get_distribution=lambda x: SimpleNamespace(
            distribution=SimpleNamespace(probs=torch.tensor([[0.9, 0.01, 0.07, 0.01, 0.01]]))
        ),
    )
    cb.model = SimpleNamespace(
        _last_obs={k: v[None] for k, v in obs.items()}, policy=policy, get_env=lambda: Env()
    )
    info = {
        "route_revisited": True,
        "route_eligible": False,
        "route_seed": 1,
        "route_source_task": "P3c",
    }
    cb.locals = {"infos": [info], "rewards": np.zeros(1)}
    cb._on_step()
    assert not cb.balanced_samples.select()[0]
    info["route_eligible"] = True
    cb._on_step()
    assert len(cb.balanced_samples.select()[0]) == 1
