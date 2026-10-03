"""Explicit P3 teaching completion -> temporal public-input monster curriculum."""

import copy
import json
from dataclasses import asdict

import torch

from ather_exploration.agents.learning import build_model, schema_signature
from ather_exploration.agents.route_ppo import RoutePPO
from ather_exploration.training.checkpoints import inspect_checkpoint
from ather_exploration.training.config import TrainingConfig
from ather_exploration.training.route_teaching import RouteMemory
from ather_exploration.training.skill_curriculum import SkillController
from ather_exploration.training.skill_environments import skill_identity
from ather_exploration.worlds.scenarios import implementation_id

PROTOCOL = "public-threat-curriculum-v1"
SOURCE = "17e3e864172cc7cd52e2a70cc6b293258192b0f39b9d748aa643e4de790797c8"


def prepare_p4(path, config, env):
    if not path or not config.p4_transfer or not config.skills.p4.enabled:
        raise ValueError("P4 requires --resume")
    parent, meta = inspect_checkpoint(path, inference=True)
    state = json.loads((parent / "runner_state.json").read_text())
    if (
        state["env_steps"] != meta["env_steps"]
        or state["skill_controller"] != meta["skill_controller"]
    ):
        raise ValueError("P4 parent controller/counter mismatch")
    if meta["config"].get("p4_transfer"):
        old = TrainingConfig.model_validate(meta["config"]).model_dump(mode="json")
        new = config.model_dump(mode="json")
        if any(old[k] != new[k] for k in new if k not in ("banks", "device", "tracking")):
            raise ValueError("P4 resume config mismatch")
        if (
            meta["source_revision"] != implementation_id()
            or state["state"] != "RUNNING"
            or state.get("transfer", {}).get("protocol") != PROTOCOL
            or meta["bank_ids"] != skill_identity(config)
            or len(state["workers"]) != config.n_envs
            or not state["skill_controller"].get("p4_enabled")
        ):
            raise ValueError("P4 resume source/state/suite mismatch")
        model = RoutePPO.load(parent / "model.zip", env=env, device=config.device)
        if not hasattr(model, "route_memory"):
            raise ValueError("Missing retention memory")
        audit = copy.deepcopy(state["transfer"])
    else:
        c = state["skill_controller"]
        if (
            meta["source_revision"] != SOURCE
            or meta["env_steps"] != 1638400
            or state["state"] != "PHASE_COMPLETED"
            or c["index"] != 8
            or c["failed"]
            or meta["schema"]["version"] != 2
        ):
            raise ValueError("P4 requires audited completed P3c step_1638400")
        old = RoutePPO.load(parent / "model.zip", device=config.device)
        if old.num_timesteps != meta["env_steps"] or old._n_updates != meta["optimizer_updates"]:
            raise ValueError("P4 parent serialized counters mismatch")
        model = build_model(config, env)
        weights = old.policy.state_dict()
        new = model.policy.state_dict()
        if set(weights) != set(new):
            raise ValueError("P4 parent architecture keys mismatch")
        for key, value in weights.items():
            if value.shape == new[key].shape:
                new[key] = value
            elif (
                key.endswith("memory.0.weight") and new[key].shape[1] == 14 and value.shape[1] == 12
            ):
                new[key].zero_()
                new[key][:, :12] = value
            else:
                raise ValueError(f"Unexpected parameter migration: {key}")
        model.policy.load_state_dict(new)
        model.num_timesteps = old.num_timesteps
        model._n_updates = old._n_updates
        model.route_memory = RouteMemory(seed=config.seed)
        model.route_memory.initialized = True
        controller = SkillController(
            index=8,
            phase_start=1638400,
            family_start=1638400,
            p4_enabled=True,
            p4_task_budget=config.skills.p4.task_budget,
            p4_minimum=config.skills.p4.minimum,
        )
        state = {
            "state": "RUNNING",
            "env_steps": 1638400,
            "skill_controller": asdict(controller),
            "workers": [],
        }
        audit = {
            "protocol": PROTOCOL,
            "parent_checkpoint": str(parent),
            "parent_source_revision": meta["source_revision"],
            "parent_checksums": json.loads((parent / "checksums.json").read_text()),
            "migration": "12 to 14 memory channels, additional input weights zero",
            "optimizer": "fresh Adam for changed input shape",
            "retention": "new memory, labels only in monster-free P3c replay",
        }
    if schema_signature(env.observation_space) != schema_signature(model.observation_space):
        raise ValueError("P4 schema mismatch")
    if (
        model.num_timesteps != meta["env_steps"]
        or model._n_updates != meta["optimizer_updates"]
        or model.num_timesteps >= config.total_timesteps
        or any(not torch.isfinite(v).all() for v in model.policy.state_dict().values())
    ):
        raise ValueError("P4 policy/counters invalid or budget exhausted")
    audit["resume_rng_checkpoint"] = str(parent)
    state["transfer"] = audit
    return model, state, audit


def check_p4(path, config):
    from ather_exploration.training.skill_environments import SkillTrainingEnv

    torch.set_num_threads(config.torch_threads)
    env = SkillTrainingEnv(config)
    try:
        model, state, audit = prepare_p4(path, config, env)
        env.set_controller(state["skill_controller"])
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        env.step(int(action))
        return {
            "status": "valid",
            "learning_executed": False,
            "transfer": audit,
            "schema": schema_signature(env.observation_space),
        }
    finally:
        env.close()
