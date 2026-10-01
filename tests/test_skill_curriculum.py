"""Skill contracts, no learning or optimizer updates."""

import numpy as np
import pytest

from ather_exploration.training.skill_curriculum import SkillController
from ather_exploration.worlds.skill_tasks import make_skill_env


@pytest.mark.parametrize("phase", ["P1a", "P1b", "P2a", "P2b", "P3", "P4a", "P4b"])
def test_skill_reset_and_space(phase):
    env = make_skill_env(phase, 17)
    a, _ = env.reset(seed=17)
    b, _ = env.reset(seed=17)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert env.observation_space.contains(a)
    assert env.action_masks()[4]
    assert bool(a["local"][3].any()) == (phase in ("P1a", "P1b", "P2a", "P4a"))
    env.close()


def test_gates_require_two_evaluations():
    c = SkillController()
    assert not c.observe(True, 32768)
    assert c.observe(True, 49152)
    assert c.task == "P1b"


def test_gate_failure_never_skips():
    c = SkillController()
    c.observe(False, 131072)
    assert c.failed
    assert c.task == "P1a"


def test_bonus_does_not_reward_wait_or_revisit():
    env = make_skill_env("P1a", 17, first_visit=True)
    env.reset()
    for _ in range(3):
        _, _, _, _, info = env.step(4)
        assert info["skill"]["intrinsic_reward"] == 0
    assert env.remaining_bonus == 0.10
    env.close()


def test_pool_content_split():
    from ather_exploration.worlds.skill_tasks import skill_pool

    train = skill_pool("P1a", 8)
    val = skill_pool("P1a", 8, True)
    assert len({x[1] for x in train}) == 8
    assert not {x[1] for x in train} & {x[1] for x in val}


def test_p1_success_requires_reset():
    from ather_exploration.types import ACTION_DELTAS
    from ather_exploration.worlds.topology import distances

    env = make_skill_env("P1a", 17)
    env.reset()
    scenario = env.unwrapped.scenario
    target = scenario.pois[0]
    d = distances(scenario.terrain, [target])
    pos = scenario.spawn
    while pos != target:
        action = next(
            i
            for i, (dx, dy) in enumerate(ACTION_DELTAS[:4])
            if d.get((pos[0] + dx, pos[1] + dy), 100) < d[pos]
        )
        dx, dy = ACTION_DELTAS[action]
        pos = (pos[0] + dx, pos[1] + dy)
        _, _, done, _, info = env.step(action)
    assert done and info["skill"]["success"]
    with pytest.raises(RuntimeError):
        env.step(4)
    env.close()


def test_p4_collision_is_not_success():
    # Find a crossing phase where stepping into the monster causes collision1.
    from ather_exploration.types import ACTION_DELTAS

    for seed in range(12):
        env = make_skill_env("P4a", seed)
        obs, _ = env.reset()
        r = obs["local"].shape[1] // 2
        for action, (dx, dy) in enumerate(ACTION_DELTAS[:4]):
            if obs["local"][5, r + dy, r + dx]:
                _, _, done, _, info = env.step(action)
                assert done and not info["skill"]["success"]
                env.close()
                return
        env.close()
    pytest.fail("No exposed crossing collision case")


def test_mask_and_checkpoint_roundtrip(tmp_path):
    from ather_exploration.agents.learning import build_model
    from ather_exploration.training.checkpoints import load_agent, save_checkpoint
    from ather_exploration.training.config import SkillConfig, TrainingConfig
    from ather_exploration.types import AgentState

    env = make_skill_env("P1a", 17)
    obs, _ = env.reset()
    cfg = TrainingConfig(
        banks={g: "unused" for g in ("small", "medium", "large")},
        skills=SkillConfig(enabled=True, wall_mask=True),
    )
    model = build_model(cfg, env)
    save_checkpoint(
        model, tmp_path / "cp", cfg, {"skill_controller": {"index": 0}, "viewer_task": "P1a"}, {}
    )
    agent = load_agent(tmp_path / "cp", env.observation_space)
    for seed in range(8):
        action, _ = agent.act(
            obs, AgentState(), deterministic=False, action_rng=np.random.default_rng(seed)
        )
        assert env.action_masks()[action]
    env.close()


def test_curriculum_lifecycle_without_learning(tmp_path, monkeypatch):
    import torch
    from stable_baselines3 import PPO

    from ather_exploration.training.checkpoints import inspect_checkpoint
    from ather_exploration.training.config import SkillConfig, TrainingConfig
    from ather_exploration.training.skill_runner import SkillCallback, run_skill_training

    def forbidden(*args, **kwargs):
        raise AssertionError("Optimizer update forbidden")

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)

    def fake_evaluate(self, task):
        return {
            "passed": True,
            "summary": {"deterministic": {"success": 1.0}, "stochastic": {"success": 1.0}},
        }

    monkeypatch.setattr(SkillCallback, "evaluate", fake_evaluate)

    def fake_learn(self, *, callback, **kwargs):
        callback.init_callback(self)
        self._last_obs = self.env.reset()
        for step in (32768, 49152, 81920, 98304):
            self.num_timesteps = step
            callback.boundary()
        return self

    monkeypatch.setattr(PPO, "learn", fake_learn)
    cfg = TrainingConfig(
        total_timesteps=4063232,
        banks={g: "unused" for g in ("small", "medium", "large")},
        skills=SkillConfig(enabled=True, stop_after="P1", train_count=8, validation_count=4),
    )
    result = run_skill_training(cfg, tmp_path / "run")
    assert result["state"] == "PHASE_COMPLETED"
    assert result["skill_controller"]["index"] == 2
    _, meta = inspect_checkpoint(tmp_path / "run/checkpoints/step_98304")
    assert meta["viewer_task"] == "P1b"
    assert meta["skill_controller"]["index"] == 2
    with pytest.raises(ValueError, match="Completed phase requires"):
        run_skill_training(
            cfg, tmp_path / "invalid_resume", resume=tmp_path / "run/checkpoints/step_98304"
        )

    def continue_learn(self, *, callback, **kwargs):
        callback.init_callback(self)
        self._last_obs = self.env.reset()
        for step in (131072, 147456, 180224, 196608):
            self.num_timesteps = step
            callback.boundary()
        return self

    monkeypatch.setattr(PPO, "learn", continue_learn)
    extended = cfg.model_copy(update={"skills": cfg.skills.model_copy(update={"stop_after": "P2"})})
    continued = run_skill_training(
        extended,
        tmp_path / "continued",
        resume=tmp_path / "run/checkpoints/step_98304",
        continue_curriculum=True,
    )
    assert continued["state"] == "PHASE_COMPLETED"
    assert continued["skill_controller"]["index"] == 4


def test_skill_viewer_loads_task_from_checkpoint(tmp_path):
    from ather_exploration.agents.learning import build_model
    from ather_exploration.training.checkpoints import save_checkpoint
    from ather_exploration.training.config import SkillConfig, TrainingConfig
    from ather_exploration.ui.session import EpisodeSession, SessionSpec

    env = make_skill_env("P1b", 17)
    config = TrainingConfig(
        banks={g: "unused" for g in ("small", "medium", "large")}, skills=SkillConfig(enabled=True)
    )
    model = build_model(config, env)
    save_checkpoint(
        model,
        tmp_path / "step_0",
        config,
        {"skill_controller": {"index": 2}, "viewer_task": "P1b"},
        {},
    )
    session = EpisodeSession(
        SessionSpec(agent="checkpoint", checkpoint=str(tmp_path / "step_0"), seed=17)
    )
    try:
        assert session.env.unwrapped.scenario.skill_task == "P1b"
        session.step()
        assert session.frame()["phase"] == "P1b"
    finally:
        session.env.close()
        env.close()


def test_eval_is_frozen(tmp_path, monkeypatch):
    import torch

    from ather_exploration.agents.learning import build_model
    from ather_exploration.evaluation.skills import evaluate_skill
    from ather_exploration.training.config import SkillConfig, TrainingConfig

    def forbidden(*a, **kw):
        raise AssertionError("Evaluation must never learn")

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    env = make_skill_env("P1a", 17)
    cfg = TrainingConfig(
        banks={g: "unused" for g in ("small", "medium", "large")},
        skills=SkillConfig(enabled=True, validation_count=4),
    )
    model = build_model(cfg, env)
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    result = evaluate_skill(model, "P1a", cfg)
    assert len(result["episodes"]) == 16
    assert all(torch.equal(v, before[k]) for k, v in model.policy.state_dict().items())
    env.close()
