"""Audited P3b continuation; fixed validation, optimizer, reward and architecture."""

import copy
import json

import torch
from stable_baselines3 import PPO

from ather_exploration.agents.learning import schema_signature
from ather_exploration.training.checkpoints import inspect_checkpoint
from ather_exploration.training.config import TrainingConfig
from ather_exploration.training.skill_environments import SkillTrainingEnv, skill_identity
from ather_exploration.worlds.scenarios import implementation_id


def prepare_unfinished(path, config, env):
    trial = config.unfinished_trial
    if trial is None:
        raise ValueError("Explicit unfinished_trial required")
    parent, meta = inspect_checkpoint(path, unfinished_trial=True)
    old = TrainingConfig.model_validate(meta["config"]).model_dump(mode="json")
    new = config.model_dump(mode="json")
    for key in new:
        if (
            key not in ("unfinished_trial", "p3_restart", "banks", "device", "tracking")
            and new[key] != old[key]
        ):
            raise ValueError(f"Unfinished trial config mismatch: {key}")
    state = json.loads((parent / "runner_state.json").read_text())
    c = state["skill_controller"]
    same_trial = old["unfinished_trial"] is not None
    if same_trial and old["unfinished_trial"] != new["unfinished_trial"]:
        raise ValueError("Cannot change an existing unfinished trial")
    if (
        state["env_steps"] != meta["env_steps"]
        or c != meta["skill_controller"]
        or c["index"] != 6
        or meta["viewer_task"] != "P3b"
        or meta["bank_ids"] != skill_identity(config)
        or len(state["workers"]) != config.n_envs
        or meta["schema"] != schema_signature(env.observation_space)
        or meta.get("curriculum_protocol") != "active-phase-v3"
        or state["state"] != "RUNNING"
        or c["failed"]
    ):
        raise ValueError(
            "Unfinished trial requires a RUNNING P3b checkpoint with matching schema/banks"
        )
    end = trial.parent_steps + trial.additional_steps
    if (
        not same_trial and meta["env_steps"] != trial.parent_steps
    ) or not trial.parent_steps <= meta["env_steps"] < end:
        raise ValueError("Unfinished trial parent/budget mismatch")
    model = PPO.load(parent / "model.zip", env=env, device=config.device)
    if (
        model.num_timesteps != meta["env_steps"]
        or model._n_updates != meta["optimizer_updates"]
        or schema_signature(model.observation_space) != meta["schema"]
        or any(not torch.isfinite(v).all() for v in model.policy.state_dict().values())
        or not model.policy.optimizer.state
    ):
        raise ValueError("Unfinished trial model counters/weights/optimizer mismatch")
    if any(
        isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
        for slot in model.policy.optimizer.state.values()
        for v in slot.values()
    ):
        raise ValueError("Nonfinite optimizer state")
    if same_trial:
        audit = state["transfer"]
    else:
        c = copy.deepcopy(c)
        c.update(passed=0, best={}, best_by_task={}, restart_level=0, restart_results=[])
        state["skill_controller"] = c
        audit = {
            "protocol": "unfinished-p3b-v1",
            "parent_checkpoint": str(parent),
            "parent_source_revision": meta["source_revision"],
            "destination_source_revision": implementation_id(),
            "parent_checksums": json.loads((parent / "checksums.json").read_text()),
            "parent_env_steps": meta["env_steps"],
            "end_steps": end,
            "changes": {k: {"from": old[k], "to": new[k]} for k in new if old[k] != new[k]},
            "preserved": [
                "weights",
                "Adam moments",
                "counters",
                "worker sampling RNG",
                "reward",
                "map generator",
                "validation",
                "gate thresholds",
            ],
            "promotion_disabled": True,
            "phase_start_preserved": True,
            "episodes": "reset; training prefixes reconstructed outside PPO transitions",
        }
    return model, state, audit


def check_unfinished(path, config):
    torch.set_num_threads(config.torch_threads)
    env = SkillTrainingEnv(config)
    try:
        model, state, audit = prepare_unfinished(path, config, env)
        env.set_controller(state["skill_controller"])
        env.restore(state["workers"][0])
        env.set_restart_level(state["skill_controller"].get("restart_level", 0))
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        env.step(int(action))
        return {
            "status": "valid",
            "learning_executed": False,
            "transfer": audit,
            "phase": env.phase,
            "scope": "checkpoint load and inference; no optimizer update",
        }
    finally:
        env.close()
