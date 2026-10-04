"""Explicit continuation changes LR only, preserving optimizer and curriculum state."""

import json
from pathlib import Path

import pytest
import torch

from ather_exploration.agents.route_ppo import RoutePPO
from ather_exploration.training.checkpoints import inspect_checkpoint, save_checkpoint
from ather_exploration.training.config import TrainingConfig, read_training_config
from ather_exploration.training.p4_transfer import prepare_p4
from ather_exploration.training.skill_environments import SkillTrainingEnv, skill_identity

PARENT = Path("artifacts/diagnostics/p4_timing_audit/checkpoints/step_1703936")


def config():
    torch.set_num_threads(1)
    return read_training_config("ather_exploration/resources/training/skills_p4_lr.yaml")


def test_branch_config_requires_declared_constant_rate():
    payload = config().model_dump(mode="json")
    payload["learning_rate"] = 0.0003
    with pytest.raises(ValueError, match="constant learning rate"):
        TrainingConfig.model_validate(payload)


def test_audited_continuation_preserves_state_and_can_resume(tmp_path):
    from test_route_teaching import assert_nested_equal

    if not PARENT.exists():
        pytest.skip("Audited timing checkpoint unavailable")
    c = config()
    env = SkillTrainingEnv(c)
    try:
        parent, _ = inspect_checkpoint(PARENT, inference=True)
        before = json.loads((parent / "runner_state.json").read_text())
        old = RoutePPO.load(parent / "model.zip", device="cpu")
        model, state, audit = prepare_p4(parent, c, env)
        assert_nested_equal(old.policy.state_dict(), model.policy.state_dict())
        assert_nested_equal(
            old.policy.optimizer.state_dict()["state"], model.policy.optimizer.state_dict()["state"]
        )
        assert_nested_equal(before["workers"], state["workers"])
        assert model.num_timesteps == 1703936
        assert model.lr_schedule(0.3) == 0.0002
        assert all(group["lr"] == 0.0002 for group in model.policy.optimizer.param_groups)
        assert model.policy.threat_temperature == old.policy.threat_temperature == 8
        assert (
            model.timing_teaching.memories["wait"].buckets
            == old.timing_teaching.memories["wait"].buckets
        )
        assert model.threat_retention.memory.buckets == old.threat_retention.memory.buckets
        assert state["skill_controller"]["phase_start"] == before["skill_controller"]["phase_start"]
        assert (
            state["skill_controller"]["threat_level"] == before["skill_controller"]["threat_level"]
        )
        assert not state["skill_controller"]["best_by_task"]
        assert audit["lr_branch"]["to"] == 0.0002
        for k, v in old.threat_retention.teacher.state_dict().items():
            torch.testing.assert_close(
                v, model.threat_retention.teacher.state_dict()[k], rtol=0, atol=0
            )
        save_checkpoint(model, tmp_path / "branch", c, state, skill_identity(c))
        resumed, resumed_state, _ = prepare_p4(tmp_path / "branch", c, env)
        assert resumed.lr_schedule(0.8) == 0.0002
        assert_nested_equal(
            model.policy.optimizer.state_dict(), resumed.policy.optimizer.state_dict()
        )
        assert resumed_state["skill_controller"] == state["skill_controller"]
        # Map/reward edits cannot ride along with this LR-only branch.
        altered = c.model_copy(update={"skills": c.skills.model_copy(update={"death": 3.0})})
        with pytest.raises(ValueError, match="config mismatch"):
            prepare_p4(parent, altered, env)
        normal = c.model_copy(update={"p4_lr_branch": False})
        with pytest.raises(ValueError, match="config mismatch"):
            prepare_p4(parent, normal, env)
    finally:
        env.close()
