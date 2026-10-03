"""Public frontier, bonus, and legacy-to-frontier transfer without learning."""

from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from ather_exploration.environment.frontier import frontier_mask
from ather_exploration.training.config import SkillConfig, read_training_config
from ather_exploration.training.p3_restart import prepare_restart
from ather_exploration.training.skill_environments import SkillTrainingEnv
from ather_exploration.worlds.skill_tasks import configured_skill_env


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("Training is forbidden in these tests")

    monkeypatch.setattr(PPO, "learn", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    torch.set_num_threads(1)


def test_frontier_cardinal_occlusion_memory_and_edges():
    m = np.zeros((11, 5, 5), np.float32)
    m[0] = 1  # fully seen
    m[2, 2, 2] = 1
    m[0, 1, 3] = 0  # unknown only diagonally
    assert not frontier_mask(m).any()
    m[0, 2, 3] = 0
    assert frontier_mask(m)[2, 2] == 1
    m[0, 2, 3] = 1
    m[1, 2, 3] = 1  # known wall shields farther unknown
    m[0, 2, 4] = 0
    assert not frontier_mask(m).any()
    m[8] = 0  # no current visibility must not erase accumulated seen
    assert not frontier_mask(m).any()
    m[2, 0, 0] = 1
    m[0, -1, 0] = 0  # array edges do not wrap around
    assert frontier_mask(m)[0, 0] == 0
    m[1, 2, 2] = 1
    m[0, 2, 1] = 0
    assert frontier_mask(m)[2, 2] == 0  # walls never frontier


def test_frontier_does_not_change_existing_channels_or_rewards():
    settings = SkillConfig(enabled=True, stop_after="P3")
    old = configured_skill_env("P3a", 65, settings)
    new = configured_skill_env("P3a", 65, settings.model_copy(update={"frontier": True}))
    try:
        a, _ = old.reset()
        b, _ = new.reset()
        for action in [4, 0, 1, 2, 3] * 3:
            assert np.array_equal(a["memory"], b["memory"][:11])
            assert np.array_equal(a["local"], b["local"])
            assert np.array_equal(a["state"], b["state"])
            assert new.observation_space.contains(b)
            a, r1, t1, u1, _ = old.step(action)
            b, r2, t2, u2, _ = new.step(action)
            assert (r1, t1, u1) == (r2, t2, u2)
    finally:
        old.close()
        new.close()


def test_visit_bonus_wait_wall_revisit_cap_and_reset():
    from ather_exploration.types import ACTION_DELTAS

    settings = SkillConfig(enabled=True, stop_after="P3", p3_visit_bonus=0.002, p3_visit_cap=0.002)
    env = configured_skill_env("P3a", 65, settings)
    try:
        env.reset()
        assert env.step(4)[4]["skill"]["reward_components"]["intrinsic"] == 0
        sc = env.unwrapped.scenario
        x, y = sc.spawn
        move = next(
            i for i, (dx, dy) in enumerate(ACTION_DELTAS[:4]) if sc.terrain[y + dy][x + dx] == "."
        )
        reverse = [1, 0, 3, 2][move]
        for _ in range(2):
            env.reset()
            assert env.step(move)[4]["skill"]["reward_components"]["intrinsic"] == 0.002
            assert env.step(reverse)[4]["skill"]["reward_components"]["intrinsic"] == 0
            assert env.step(move)[4]["skill"]["reward_components"]["intrinsic"] == 0
            assert env.remaining_bonus == 0
    finally:
        env.close()


PARENT = Path("artifacts/diagnostics/skills-p3-01/checkpoints/step_655360")


@pytest.mark.skipif(not PARENT.exists(), reason="Local checkpoint unavailable")
def test_real_transfer_logits_values_adam_and_schema(tmp_path):
    from ather_exploration.training.checkpoints import load_agent, save_checkpoint
    from ather_exploration.training.skill_environments import skill_identity

    cfg = read_training_config("ather_exploration/resources/training/skills_p3.yaml")
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda i=i: SkillTrainingEnv(cfg, i) for i in range(16)])
    before = (PARENT / "checksums.json").read_bytes()
    try:
        old = PPO.load(PARENT / "model.zip", device="cpu")
        new, state, _audit = prepare_restart(PARENT, cfg, env)
        env.env_method("set_controller", state["skill_controller"])
        assert state["skill_controller"]["phase_start"] == 655360
        assert state["skill_controller"]["passed"] == 0
        for seed in (42, 65, 100033):
            probe = configured_skill_env("P3a", seed, cfg.skills)
            obs, _ = probe.reset()
            legacy = {**obs, "memory": obs["memory"][:11]}
            with torch.no_grad():
                a, _ = old.policy.obs_to_tensor(legacy)
                b, _ = new.policy.obs_to_tensor(obs)
                torch.testing.assert_close(
                    old.policy.get_distribution(a).distribution.logits,
                    new.policy.get_distribution(b).distribution.logits,
                    atol=1e-5,
                    rtol=1e-5,
                )
                torch.testing.assert_close(
                    old.policy.predict_values(a), new.policy.predict_values(b), atol=1e-5, rtol=1e-5
                )
            probe.close()
        old_params = dict(old.policy.named_parameters())
        for name, param in new.policy.named_parameters():
            for k, v in old.policy.optimizer.state[old_params[name]].items():
                actual = new.policy.optimizer.state[param][k]
                if isinstance(v, torch.Tensor):
                    if v.shape != actual.shape:
                        assert torch.equal(v, actual[:, :11])
                        assert not actual[:, 11].any()
                    else:
                        assert torch.equal(v, actual)
        assert new.n_envs == 16 and new.batch_size == 1024
        path = tmp_path / "checkpoint"
        save_checkpoint(
            new, path, cfg, {**state, "viewer_task": "P3a", "state": "RUNNING"}, skill_identity(cfg)
        )
        load_agent(path, env.observation_space)
        assert (PARENT / "checksums.json").read_bytes() == before
    finally:
        env.close()


@pytest.mark.parametrize("run,step", [("skills-p1-01", 98304), ("skills-p2-02-resume-01", 376832)])
def test_real_legacy_ui_still_uses_eleven_channels(run, step):
    from ather_exploration.ui.session import EpisodeSession, SessionSpec

    parent = Path("artifacts/modal") / run / "checkpoints" / f"step_{step}"
    if not parent.exists():
        pytest.skip("Local checkpoint unavailable")
    session = EpisodeSession(SessionSpec(agent="checkpoint", checkpoint=str(parent), seed=42))
    try:
        assert session.obs["memory"].shape[0] == 11
        for _ in range(3):
            if not session.done:
                session.step()
    finally:
        session.close()
