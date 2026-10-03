"""Controlled LR branching checks; never run learning."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from stable_baselines3 import PPO

from ather_exploration.training.config import read_training_config
from ather_exploration.training.lr_trial import prepare_trial
from ather_exploration.training.skill_curriculum import SkillController
from ather_exploration.training.skill_environments import SkillTrainingEnv
from ather_exploration.training.skill_runner import SkillCallback, SkillStop

CONFIG = "ather_exploration/resources/training/skills_p3_lr_trial.yaml"
PARENT = Path("artifacts/diagnostics/skills-p3-01/checkpoints/step_655360")


@pytest.fixture(autouse=True)
def no_learning(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Learning is forbidden")

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    monkeypatch.setattr(PPO, "learn", forbidden)
    torch.set_num_threads(1)


def test_trial_boundary_keeps_task_and_stops_exactly(tmp_path):
    cfg = read_training_config(CONFIG)
    controller = SkillController(index=5, phase_start=376832)
    cb = SkillCallback(cfg, tmp_path, controller, {})
    cb.model = SimpleNamespace(
        num_timesteps=0, policy=torch.nn.Linear(1, 1), logger=Mock(name_to_value={})
    )
    cb.evaluate = Mock(
        return_value={
            "task": "P3a",
            "passed": True,
            "summary": {
                "deterministic": {"joint_success": 0.9, "success": 0.9, "coverage_auc": 0.8}
            },
        }
    )
    cb.save = Mock()
    cb.tracker = None
    for step in (671744, 688128, 704512):
        cb.model.num_timesteps = step
        cb.boundary()
        assert controller.task == "P3a"
        assert cb.state == "RUNNING"
    cb.model.num_timesteps = 720896
    with pytest.raises(SkillStop, match="EXPERIMENT_COMPLETED"):
        cb.boundary()
    assert cb.evaluate.call_count == 4
    assert cb.save.call_count == 4
    assert controller.task == "P3a" and controller.passed == 4
    assert all(h["promotion_disabled"] for h in controller.history)


@pytest.mark.skipif(not PARENT.exists(), reason="Real checkpoint is local-only")
def test_real_parent_preserves_weights_and_optimizer():
    cfg = read_training_config(CONFIG)
    env = SkillTrainingEnv(cfg)
    try:
        original = PPO.load(PARENT / "model.zip", device="cpu")
        model, state, audit = prepare_trial(PARENT, cfg, env)
        assert model.num_timesteps == original.num_timesteps == 655360
        assert model._n_updates == original._n_updates
        for key, value in original.policy.state_dict().items():
            assert torch.equal(value, model.policy.state_dict()[key])
        old = original.policy.optimizer.state_dict()["state"]
        new = model.policy.optimizer.state_dict()["state"]
        for key, slot in old.items():
            for name, value in slot.items():
                assert (
                    torch.equal(value, new[key][name])
                    if isinstance(value, torch.Tensor)
                    else value == new[key][name]
                )
        assert all(model.lr_schedule(x) == 1e-4 for x in (0, 0.5, 1))
        assert all(g["lr"] == 1e-4 for g in model.policy.optimizer.param_groups)
        assert state["skill_controller"]["passed"] == 0
        assert audit["end_steps"] == 720896
        changed = cfg.model_copy(update={"ent_coef": 0.02})
        with pytest.raises(ValueError, match="ent_coef"):
            prepare_trial(PARENT, changed, env)
    finally:
        env.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"final_learning_rate": 3e-5},
        {"lr_trial": {"parent_steps": 655360, "additional_steps": 1, "task": "P3a"}},
        {"lr_trial": {"parent_steps": 655360, "additional_steps": 4063232, "task": "P3a"}},
    ],
)
def test_reject_invalid_trial_schedule(changes):
    from ather_exploration.training.config import TrainingConfig

    data = read_training_config(CONFIG).model_dump()
    data.update(changes)
    with pytest.raises(ValueError):
        TrainingConfig.model_validate(data)
