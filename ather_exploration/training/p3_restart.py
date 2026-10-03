"""Explicit P3 branch with frontier expansion and preserved Adam moments."""

import copy
import json

import torch
from stable_baselines3 import PPO

from ather_exploration.agents.learning import build_model, schema_signature
from ather_exploration.training.checkpoints import inspect_checkpoint, restore_rng
from ather_exploration.training.config import TrainingConfig
from ather_exploration.training.skill_curriculum import SkillController
from ather_exploration.training.skill_environments import SkillTrainingEnv, skill_identity
from ather_exploration.worlds.scenarios import implementation_id


def expand_tensor(value, shape):
    if value.shape == shape:
        return value.clone()
    if value.ndim == 4 and value.shape[1] == 11 and shape == (value.shape[0], 12, *value.shape[2:]):
        expanded = value.new_zeros(shape)
        expanded[:, :11] = value
        return expanded
    raise ValueError(f"Unsupported tensor transfer: {value.shape} -> {shape}")


def prepare_restart(path, config, env):
    parent, meta = inspect_checkpoint(path, lr_trial=True)
    old_config = TrainingConfig.model_validate(meta["config"])
    state = json.loads((parent / "runner_state.json").read_text())
    if (
        not config.p3_restart
        or meta["env_steps"] != 655360
        or meta["viewer_task"] != "P3a"
        or meta["schema"]["version"] != 1
        or state["state"] != "RUNNING"
        or meta["bank_ids"] != skill_identity(config)
        or state["env_steps"] != meta["env_steps"]
        or state["skill_controller"] != meta["skill_controller"]
        or state["skill_controller"]["index"] != 5
        or state["skill_controller"]["failed"]
        or old_config.lr_trial
    ):
        raise ValueError("P3 frontier branch requires original P3a step_655360")
    allowed = {
        "p3_restart",
        "learning_rate",
        "final_learning_rate",
        "n_envs",
        "batch_size",
        "checkpoint_updates",
        "torch_threads",
        "device",
        "banks",
        "tracking",
        "skills",
    }
    old, new = old_config.model_dump(mode="json"), config.model_dump(mode="json")
    for k in new:
        if k not in allowed and new[k] != old[k]:
            raise ValueError(f"P3 branch config mismatch: {k}")
    for k in new["skills"]:
        if (
            k not in ("frontier", "p3_visit_bonus", "p3_visit_cap")
            and new["skills"][k] != old["skills"][k]
        ):
            raise ValueError(f"P3 branch skill mismatch: {k}")
    source = PPO.load(parent / "model.zip", device=config.device)
    if (
        source.num_timesteps != meta["env_steps"]
        or source._n_updates != meta["optimizer_updates"]
        or schema_signature(source.observation_space) != meta["schema"]
    ):
        raise ValueError("P3 source counters/schema mismatch")
    model = build_model(config, env)
    src, dst = source.policy.state_dict(), model.policy.state_dict()
    if src.keys() != dst.keys() or any(not torch.isfinite(v).all() for v in src.values()):
        raise ValueError("P3 source architecture/weights invalid")
    model.policy.load_state_dict({k: expand_tensor(src[k], v.shape) for k, v in dst.items()})
    old_params, new_params = (
        dict(source.policy.named_parameters()),
        dict(model.policy.named_parameters()),
    )
    if old_params.keys() != new_params.keys() or not source.policy.optimizer.state:
        raise ValueError("Missing optimizer state or parameter mismatch")
    for name, param in new_params.items():
        slots = source.policy.optimizer.state.get(old_params[name], {})
        copied = {}
        for key, value in slots.items():
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite optimizer state")
                copied[key] = (
                    value.clone() if value.ndim == 0 else expand_tensor(value, param.shape)
                )
            else:
                copied[key] = copy.deepcopy(value)
        model.policy.optimizer.state[param] = copied
    model.num_timesteps = source.num_timesteps
    model._n_updates = source._n_updates
    # Fresh rollout buffers and independent worker seeds are intentional with a new n_envs.
    controller = SkillController(
        index=5,
        phase_start=model.num_timesteps,
        family_start=model.num_timesteps,
        p2_task_budget=config.skills.p2_task_budget,
        p3_minimum=config.skills.p3_minimum,
        p3_task_budget=config.skills.p3_task_budget,
    )
    from dataclasses import asdict

    state = {"skill_controller": asdict(controller), "workers": []}
    audit = {
        "protocol": "p3-frontier-v1",
        "parent_checkpoint": str(parent),
        "parent_source_revision": meta["source_revision"],
        "destination_source_revision": implementation_id(),
        "parent_checksums": json.loads((parent / "checksums.json").read_text()),
        "changes": {k: {"from": old[k], "to": new[k]} for k in new if old[k] != new[k]},
        "preserved": ["11-channel weights", "Adam moments", "optimizer/env counters"],
        "reset": ["P3 controller and best", "episodes", "worker samplers"],
        "new_channel": "frontier convolution and Adam moments initialized to zero",
    }
    return model, state, audit


def check_restart(path, config):
    import time
    from functools import partial

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    torch.set_num_threads(config.torch_threads)
    factories = [partial(SkillTrainingEnv, config, worker=i) for i in range(config.n_envs)]
    env = (
        SubprocVecEnv(factories, start_method="spawn")
        if config.vec_backend == "subproc"
        else DummyVecEnv(factories)
    )
    try:
        model, state, audit = prepare_restart(path, config, env)
        cuda = config.device.startswith("cuda")
        if cuda:
            restore_rng(audit["parent_checkpoint"])
        env.env_method("set_controller", state["skill_controller"])
        obs = env.reset()
        started = time.monotonic()
        for _ in range(16):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(action)
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        return {
            "status": "valid",
            "learning_executed": False,
            "rng_restoration_checked": cuda,
            "transfer": audit,
            "vector_check": {
                "n_envs": env.num_envs,
                "backend": config.vec_backend,
                "transitions": 16 * env.num_envs,
                "inference_env_steps_per_second": 16 * env.num_envs / elapsed,
                "scope": "forward and env stepping only; excludes PPO backward/evaluation/checkpoint IO",
            },
        }
    finally:
        env.close()
