"""Restart equivalence, public-only sampling and fixed-task branch regressions."""

import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from ather_exploration.training.config import UnfinishedTrial, read_training_config
from ather_exploration.training.skill_curriculum import SkillController
from ather_exploration.training.skill_environments import SkillTrainingEnv
from ather_exploration.training.unfinished import (
    PrefixArchive,
    frontier_distance,
    observation_digest,
)
from ather_exploration.worlds.skill_tasks import configured_skill_env

CONFIG = "ather_exploration/resources/training/skills_p3_unfinished.yaml"


def memory_observation():
    memory = np.zeros((12, 9, 9), np.float32)
    # Known walls surround a narrow floor corridor, except beyond its endpoint.
    memory[0, 3:6, :7] = 1
    memory[1, 3:6, :7] = 1
    memory[1, 4, 1:7] = 0
    memory[2, 4, 1:7] = 1
    memory[7, 4, 2] = 1
    return {"memory": memory}


def test_frontier_known_path_and_wall_occlusion():
    obs = memory_observation()
    assert frontier_distance(obs) == 4
    obs["memory"][1, 4, 4] = 1
    obs["memory"][2, 4, 4] = 0
    assert frontier_distance(obs) is None


def test_archive_bounded_difficulty_and_checkpoint_rng():
    obs = memory_observation()
    cfg = UnfinishedTrial(restart_probability=0.5, pool_per_band=2)
    archive = PrefixArchive(cfg, np.random.default_rng(3))
    for seed in range(10):
        archive.offer(seed, [2] * 16, obs, 0, set())
    assert len(archive.pools[0]) == 2
    assert not archive.pools[1]
    other = PrefixArchive(cfg, np.random.default_rng(99))
    other.restore(archive.state())
    assert [archive.choose() for _ in range(20)] == [other.choose() for _ in range(20)]
    empty = PrefixArchive(cfg, np.random.default_rng(3))
    assert all(empty.choose()[0] is None for _ in range(20))


def test_replay_restores_observation_reward_history_clock():
    config = read_training_config(CONFIG)
    original = configured_skill_env("P3b", 42, config.skills)
    restored = configured_skill_env("P3b", 42, config.skills)
    archive = PrefixArchive(config.unfinished_trial, np.random.default_rng(0))
    try:
        obs, _ = original.reset()
        actions = [0, 1, 2, 3, 4] * 5
        for action in actions:
            obs, _, term, trunc, _ = original.step(action)
            assert not (term or trunc)
        item = {"actions": actions, "observation": observation_digest(obs)}
        reconstructed, _ = archive.replay(restored, item)
        assert observation_digest(reconstructed) == observation_digest(obs)
        assert restored.unwrapped.evaluator_snapshot().step_count == len(actions)
        # Every subsequent reward/event must agree, including remaining novelty budget.
        for action in [2, 2, 1, 3, 0, 4] * 4:
            a, ar, at, ax, ai = original.step(action)
            b, br, bt, bx, bi = restored.step(action)
            assert observation_digest(a) == observation_digest(b)
            assert (ar, at, ax, ai["skill"]) == (br, bt, bx, bi["skill"])
        assert archive.replay_steps == len(actions)
    finally:
        original.close()
        restored.close()


def test_replay_rejects_stale_or_terminal_prefix():
    config = read_training_config(CONFIG)
    env = configured_skill_env("P3b", 42, config.skills)
    archive = PrefixArchive(config.unfinished_trial, np.random.default_rng(0))
    try:
        with pytest.raises(ValueError, match="observation differs"):
            archive.replay(env, {"actions": [4], "observation": "wrong"})
        with pytest.raises(ValueError, match="terminated"):
            archive.replay(env, {"actions": [4] * 256, "observation": "wrong"})
    finally:
        env.close()


def test_restart_training_counts_suffix_only_and_expires_original_horizon():
    config = read_training_config(CONFIG)
    env = SkillTrainingEnv(config)
    env.set_controller(asdict(SkillController(index=6)))
    env.controller.mixture = lambda: [("P3b", 1.0)]
    try:
        # Construct a real training episode prefix, not a validation seed.
        obs, _ = env.reset()
        seed = env.task_seed
        actions = [4] * 20
        for action in actions:
            obs, *_ = env.step(action)
        item = {
            "seed": seed,
            "actions": actions,
            "observation": observation_digest(obs),
            "distance": 2,
        }
        env.archive.choose = lambda: (copy.deepcopy(item), True)
        env.reset()
        assert env.prefix_length == 20
        assert env.ticks == 0 and env.total == 0
        for _ in range(236):
            _, _, term, trunc, _ = env.step(4)
        assert term and not trunc  # Finite-horizon task uses terminal budget semantics.
        row = env.drain()[-1]
        assert row["length"] == 236 and row["prefix_length"] == 20 and row["restart_used"]
        assert row["return"] == 0
    finally:
        env.close()


def test_validation_env_not_affected_by_trial():
    c = read_training_config(CONFIG)
    env = configured_skill_env("P3b", 100021, c.skills)
    try:
        env.reset()
        assert env.unwrapped.evaluator_snapshot().step_count == 0
        assert not hasattr(env, "archive")
    finally:
        env.close()


def test_config_rejects_other_transfer_and_misalignment():
    c = read_training_config(CONFIG).model_dump()
    for key, value in [("p3_restart", True), ("learning_rate", 0.0002)]:
        with pytest.raises(ValueError):
            type(read_training_config(CONFIG)).model_validate({**c, key: value})
    c["unfinished_trial"]["additional_steps"] = 100
    with pytest.raises(ValueError):
        type(read_training_config(CONFIG)).model_validate(c)


def test_real_parent_preserves_parameters_and_optimizer():
    import torch
    from stable_baselines3 import PPO

    from ather_exploration.training.unfinished_trial import prepare_unfinished

    path = Path("artifacts/modal/skills-p3-frontier-01/checkpoints/step_1212416")
    if not path.exists():
        pytest.skip("Local real checkpoint unavailable")
    config = read_training_config(CONFIG)
    env = SkillTrainingEnv(config)
    try:
        model, state, audit = prepare_unfinished(path, config, env)
        source = PPO.load(path / "model.zip", device="cpu")
        assert model.num_timesteps == 1212416
        assert state["skill_controller"]["index"] == 6
        assert state["skill_controller"]["passed"] == 0
        assert audit["promotion_disabled"]
        for k, v in source.policy.state_dict().items():
            assert torch.equal(v, model.policy.state_dict()[k])
        a = source.policy.optimizer.state_dict()["state"]
        b = model.policy.optimizer.state_dict()["state"]
        for k, slot in a.items():
            for key, v in slot.items():
                assert torch.equal(v, b[k][key]) if torch.is_tensor(v) else v == b[k][key]
        bad = config.model_copy(update={"gamma": 0.99})
        with pytest.raises(ValueError, match="gamma"):
            prepare_unfinished(path, bad, env)
    finally:
        env.close()


def test_trial_boundary_does_not_promote_or_fail_at_original_cap(tmp_path):
    from stable_baselines3.common.logger import configure

    from ather_exploration.training.skill_runner import SkillCallback, SkillStop
    from ather_exploration.training.unfinished_trial import prepare_unfinished

    path = Path("artifacts/modal/skills-p3-frontier-01/checkpoints/step_1212416")
    if not path.exists():
        pytest.skip("Local real checkpoint unavailable")
    config = read_training_config(CONFIG)
    env = SkillTrainingEnv(config)
    try:
        model, state, _ = prepare_unfinished(path, config, env)
        model.set_logger(configure(str(tmp_path), []))
        controller = SkillController(**state["skill_controller"])
        callback = SkillCallback(config, tmp_path, controller, {})
        callback.model = model
        callback.tracker = None
        callback.save = lambda: None
        callback.evaluate = lambda task: {
            "task": task,
            "passed": True,
            "summary": {
                "deterministic": {
                    "success": 1.0,
                    "joint_success": 1.0,
                    "coverage_auc": 0.9,
                    "wall_block": 0.0,
                }
            },
        }
        for step in (1228800, 1245184, 1261568):
            model.num_timesteps = step
            callback.boundary()
            assert controller.task == "P3b" and not controller.failed
        model.num_timesteps = 1277952
        with pytest.raises(SkillStop, match="EXPERIMENT_COMPLETED"):
            callback.boundary()
        assert controller.task == "P3b"
        assert controller.phase_start == 737280
    finally:
        env.close()


def test_poi_metric_pools_all_seen_pois():
    from ather_exploration.evaluation.p3 import add_p3_gates

    config = read_training_config(CONFIG)
    rows = []
    for deterministic in (True, False):
        for events in (
            [{"seen_step": 1, "activated_step": 2}, {"seen_step": 4, "activated_step": None}],
            [{"seen_step": 0, "activated_step": 1}, {"seen_step": None, "activated_step": None}],
        ):
            rows.append(
                {
                    "deterministic": deterministic,
                    "poi_events": events,
                    "joint_success": 0,
                    "unfinished_no_progress_streak": 10,
                    "unfinished_no_progress_fraction": 0.5,
                    "all_pois_activated_step": None,
                    "coverage_after_first_activation": 0,
                    "size": 13,
                    "topology": "three_rooms",
                    "poi_count": 2,
                    "success": 0,
                    "coverage": 0.8,
                    "coverage_auc": 0.7,
                    "wall_block": 0,
                }
            )
    summary = {
        mode: {"coverage": 0.8, "coverage_auc": 0.7, "wall_block": 0}
        for mode in ("deterministic", "stochastic")
    }
    add_p3_gates("P3b", rows, summary, lambda *a, **k: True, config)
    assert summary["deterministic"]["activated_if_seen"] == 2 / 3
    assert summary["deterministic"]["activated_if_seen_count"] == 3


def test_restart_mastery_does_not_count_easy_examples_for_harder_level():
    c = SkillController(index=6)
    for _ in range(7):
        assert not c.observe_restart(0, True, 8, 0.75)
    assert c.observe_restart(0, False, 8, 0.75)
    assert c.restart_level == 1 and c.task == "P3b"
    for _ in range(20):
        assert not c.observe_restart(0, True, 8, 0.75)
    assert c.restart_results == []
    for _ in range(7):
        assert not c.observe_restart(1, False, 8, 0.75)
    assert not c.observe_restart(1, True, 8, 0.75)


def test_mid_trial_checkpoint_reload_preserves_archive(tmp_path):
    import torch

    from ather_exploration.training.checkpoints import save_checkpoint
    from ather_exploration.training.skill_environments import skill_identity
    from ather_exploration.training.unfinished_trial import prepare_unfinished

    source = Path("artifacts/modal/skills-p3-frontier-01/checkpoints/step_1212416")
    if not source.exists():
        pytest.skip("Local real checkpoint unavailable")
    config = read_training_config(CONFIG)
    env = SkillTrainingEnv(config)
    try:
        model, state, audit = prepare_unfinished(source, config, env)
        env.restore(state["workers"][0])
        env.archive.offer(42, [2] * 16, memory_observation(), 0, set())
        worker = env.checkpoint_state()
        model.num_timesteps = 1228800
        state.update(
            env_steps=1228800,
            state="RUNNING",
            transfer=audit,
            viewer_task="P3b",
            workers=[worker] * config.n_envs,
        )
        path = tmp_path / "checkpoint"
        save_checkpoint(model, path, config, state, skill_identity(config))
        restored, again, _ = prepare_unfinished(path, config, env)
        env.restore(again["workers"][0])
        assert env.archive.state() == worker["unfinished"]
        assert restored.num_timesteps == 1228800
        for k, v in model.policy.state_dict().items():
            assert torch.equal(v, restored.policy.state_dict()[k])
    finally:
        env.close()
