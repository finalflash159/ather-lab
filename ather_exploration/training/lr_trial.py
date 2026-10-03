"""Audited fixed-task LR experiment; no learning in preparation or validation."""

import json

import torch
from stable_baselines3 import PPO

from ather_exploration.agents.learning import LinearSchedule, build_model, schema_signature
from ather_exploration.training.checkpoints import inspect_checkpoint, restore_rng
from ather_exploration.training.skill_curriculum import STAGES
from ather_exploration.training.skill_environments import SkillTrainingEnv, skill_identity
from ather_exploration.worlds.scenarios import implementation_id


def prepare_trial(path, config, env):
    if config.lr_trial is None:
        raise ValueError("Explicit lr_trial config required")
    parent, meta = inspect_checkpoint(path, lr_trial=True)
    from ather_exploration.training.config import TrainingConfig

    old = TrainingConfig.model_validate(meta["config"]).model_dump(mode="json")
    new = config.model_dump(mode="json")
    if old["lr_trial"] is not None:
        raise ValueError("Trial must branch from the original run, not another trial")
    for key in new:
        if key not in (
            "learning_rate",
            "final_learning_rate",
            "lr_trial",
            "banks",
            "device",
            "tracking",
        ) and new[key] != old.get(key):
            raise ValueError(f"LR trial config mismatch: {key}")
    state = json.loads((parent / "runner_state.json").read_text())
    c = state["skill_controller"]
    trial = config.lr_trial
    if (
        meta["env_steps"] != trial.parent_steps
        or state["env_steps"] != trial.parent_steps
        or state["state"] != "RUNNING"
        or meta.get("curriculum_protocol") != "active-phase-v3"
        or c != meta["skill_controller"]
        or STAGES[c["index"]] != trial.task
        or meta["viewer_task"] != trial.task
        or c["failed"]
        or len(state["workers"]) != config.n_envs
        or meta["bank_ids"] != skill_identity(config)
    ):
        raise ValueError("LR trial parent task/state/counters/bank mismatch")
    if meta["schema"] != schema_signature(env.observation_space):
        raise ValueError("LR trial schema mismatch")
    model = PPO.load(parent / "model.zip", env=env, device=config.device)
    if (
        model.num_timesteps != trial.parent_steps
        or model._n_updates != meta["optimizer_updates"]
        or model.action_space.n != 5
    ):
        raise ValueError("LR trial model counters/actions mismatch")
    expected = build_model(config, env)
    actual = model.policy.state_dict()
    wanted = expected.policy.state_dict()
    if {k: (v.shape, v.dtype) for k, v in actual.items()} != {
        k: (v.shape, v.dtype) for k, v in wanted.items()
    } or any(not torch.isfinite(v).all() for v in actual.values()):
        raise ValueError("LR trial architecture/nonfinite weights")
    slots = model.policy.optimizer.state_dict()["state"]
    if not slots or any(
        isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
        for slot in slots.values()
        for v in slot.values()
    ):
        raise ValueError("LR trial missing/nonfinite optimizer state")
    del expected
    old_schedule = model.lr_schedule
    old_rate = float(old_schedule(1 - model.num_timesteps / old["total_timesteps"]))
    if config.learning_rate >= old_rate:
        raise ValueError("LR trial must lower parent effective learning rate")
    model.learning_rate = LinearSchedule(config.learning_rate, config.final_learning_rate)
    model.lr_schedule = model.learning_rate
    for group in model.policy.optimizer.param_groups:
        group["lr"] = config.learning_rate
    # A parent PASS does not count as an independent PASS of this experiment.
    c["passed"] = 0
    c["best_by_task"] = {}
    c["best"] = {}
    audit = {
        "protocol": "lr-trial-v1",
        "parent_checkpoint": str(parent),
        "parent_source_revision": meta["source_revision"],
        "destination_source_revision": implementation_id(),
        "parent_checksums": json.loads((parent / "checksums.json").read_text()),
        "parent_env_steps": model.num_timesteps,
        "parent_optimizer_updates": model._n_updates,
        "learning_rate": {"parent_effective": old_rate, "trial_constant": config.learning_rate},
        "end_steps": trial.parent_steps + trial.additional_steps,
        "task": trial.task,
        "promotion_disabled": True,
        "gate_streak_reset": True,
        "preserved": [
            "weights",
            "optimizer moments",
            "counters",
            "worker sampler",
            "reward",
            "maps",
            "gate thresholds",
            "entropy",
            "rollout",
        ],
        "episodes": "reset; RNG restored at execution; not bit-for-bit episode continuation",
    }
    return model, state, audit


def check_trial(path, config):
    torch.set_num_threads(config.torch_threads)
    env = SkillTrainingEnv(config)
    try:
        model, state, audit = prepare_trial(path, config, env)
        cuda = config.device.startswith("cuda")
        if cuda:
            restore_rng(audit["parent_checkpoint"])
        env.set_controller(state["skill_controller"])
        env.restore(state["workers"][0])
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        env.step(int(action))
        return {
            "status": "valid",
            "learning_executed": False,
            "rng_restoration_checked": cuda,
            "phase": env.phase,
            "transfer": audit,
        }
    finally:
        env.close()
