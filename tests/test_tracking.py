"""Tracking contract checks; fake SDK, no network, learning, or credentials."""

import sys
from types import SimpleNamespace

import pytest

from ather_exploration.training.config import TrackingConfig, TrainingConfig
from ather_exploration.training.tracking import start_tracking
from ather_exploration.worlds.scenarios import read_record


def cfg(mode):
    return TrainingConfig(
        banks={g: "/unused" for g in ("small", "medium", "large")},
        tracking=TrackingConfig(mode=mode),
    )


def test_disabled_and_missing_secret(tmp_path, monkeypatch):
    assert start_tracking(cfg("disabled"), tmp_path, {}) is None
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        start_tracking(cfg("online"), tmp_path, {})


def test_tracking_registers_axis_and_persists_url_without_key(tmp_path, monkeypatch):
    calls = []
    run = SimpleNamespace(
        id="test",
        url="https://wandb.ai/test/project/runs/test",
        define_metric=lambda *a, **kw: calls.append((a, kw)),
        finish=lambda **kw: None,
    )

    def init(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    tracker = start_tracking(cfg("offline"), tmp_path, {"small": "bank-id"}, "parent")
    assert tracker is run
    assert calls[0]["config"]["parent_checkpoint"] == "parent"
    assert calls[2][1]["step_metric"] == "training/env_steps"
    assert read_record(tmp_path / "tracking.json")["id"] == "test"


def test_callback_sends_real_measurements_with_env_step_axis(tmp_path, capsys):
    import torch

    from ather_exploration.training.curriculum import Curriculum
    from ather_exploration.training.runner import TelemetryCallback

    config = cfg("disabled").model_copy(update={"checkpoint_updates": 2})
    logged = []
    callback = TelemetryCallback(config, tmp_path, Curriculum(False))
    callback.model = SimpleNamespace(
        num_timesteps=256,
        policy=torch.nn.Linear(2, 1),
        logger=SimpleNamespace(
            name_to_value={"train/loss": 0.4}, record=lambda *a: None, dump=lambda *a: None
        ),
    )
    callback.tracker = SimpleNamespace(log=lambda values, step: logged.append((values, step)))
    callback.episodes = [
        {
            "group": "small",
            "status": "completed",
            "return": 1.5,
            "T": 40,
            "metrics": {"coverage": 0.4, "activation": 0.5, "survival": 1, "undefined": None},
            "reward_terms": {"activation": 1.0},
            "metadata": {"fallback": False},
        }
    ]
    callback.boundary()
    values, step = logged[0]
    assert step == values["training/env_steps"] == 256
    assert values["train/loss"] == 0.4
    assert values["small/return"] == 1.5
    assert values["small/coverage"] == 0.4
    assert "small/undefined" not in values
    callback.boundary()
    assert len(logged) == 1
    output = capsys.readouterr().out
    assert output.count("256/4,096 steps") == 1
    assert "6.25%" in output and "loss=0.4" in output
    assert "ETA~" in output and "steps/s" in output
    assert "[checkpoint:" not in output


def test_progress_reports_resume_eta_and_only_published_checkpoint(tmp_path, monkeypatch, capsys):
    import torch

    from ather_exploration.training import runner
    from ather_exploration.training.curriculum import Curriculum

    monkeypatch.setattr(runner.time, "monotonic", lambda: 20.0)
    config = cfg("disabled")
    callback = runner.TelemetryCallback(config, tmp_path, Curriculum(False), previous_elapsed=100)
    callback.started = 10.0
    callback.starting_steps = 256
    callback.model = SimpleNamespace(
        num_timesteps=512,
        policy=torch.nn.Linear(2, 1),
        logger=SimpleNamespace(
            name_to_value={"train/policy_gradient_loss": -0.2, "train/value_loss": 0.3},
            record=lambda *a: None,
            dump=lambda *a: None,
        ),
        get_env=lambda: SimpleNamespace(env_method=lambda *a: []),
    )
    callback.bank_ids = {}
    monkeypatch.setattr(runner, "save_checkpoint", lambda *a, **kw: None)

    def publish():
        assert "published" not in capsys.readouterr().out
        raise RuntimeError("publish failed")

    callback.on_boundary = publish
    with pytest.raises(RuntimeError, match="publish failed"):
        callback.boundary()
    assert "saved + published" not in capsys.readouterr().out
    callback.last_telemetry = -1
    callback.on_boundary = lambda: None
    callback.boundary()
    output = capsys.readouterr().out
    assert "25.6 steps/s" in output and "ETA~140s" in output
    assert "elapsed=110s" in output
    assert "actor_loss=-0.2" in output and "critic_loss=0.3" in output
    assert "step_512 saved + published" in output
