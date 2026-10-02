"""Explicit, audited completed-P1 transfer. Never updates weights or rewrites a parent."""

import json

import torch
from stable_baselines3 import PPO

from ather_exploration.agents.learning import build_model, schema_signature
from ather_exploration.training.checkpoints import _hash, inspect_checkpoint, restore_rng
from ather_exploration.training.skill_environments import skill_identity
from ather_exploration.worlds.scenarios import implementation_id


def prepare_p1_transfer(path, config, env):
    """Validate configuration, spaces, architecture and counters; load optimizer unchanged."""
    parent, meta = inspect_checkpoint(path, transfer_p1_to_p2=True)
    old, new = meta["config"], config.model_dump(mode="json")
    if not config.skills.enabled or config.skills.stop_after != "P2":
        raise ValueError("P1 transfer destination must be skills.stop_after=P2")
    if config.method != "ppo" or config.skills.wall_mask:
        raise ValueError("P1 transfer supports unmasked feedforward PPO only")
    # Only the objective for the not-yet-trained P2 may change.
    old_skills = {k: v for k, v in old["skills"].items() if k not in ("stop_after", "p2_reward")}
    new_skills = {k: v for k, v in new["skills"].items() if k not in ("stop_after", "p2_reward")}
    if old_skills != new_skills:
        raise ValueError("P1 transfer skill config mismatch outside stop_after/p2_reward")
    for key in set(old) | set(new):
        if key not in ("skills", "banks", "device", "tracking") and old.get(key) != new.get(key):
            raise ValueError(f"P1 transfer config mismatch: {key}")
    if meta["bank_ids"] != skill_identity(config):
        raise ValueError("P1 transfer skill bank identity mismatch")
    signature = schema_signature(env.observation_space)
    if meta["schema"] != signature or env.action_space.n != 5:
        raise ValueError("P1 transfer observation/action schema mismatch")
    state = json.loads((parent / "runner_state.json").read_text())
    controller = state["skill_controller"]
    if (
        len(state["workers"]) != config.n_envs
        or controller["phase_start"] != meta["env_steps"]
        or controller["family_start"] != meta["env_steps"]
        or controller["passed"] != 0
        or state["env_steps"] != meta["env_steps"]
        or meta["skill_controller"] != controller
    ):
        raise ValueError("P1 transfer controller/worker counters mismatch")
    model = PPO.load(parent / "model.zip", env=env, device=config.device)
    if (
        schema_signature(model.observation_space) != signature
        or model.action_space.n != 5
        or model.num_timesteps != meta["env_steps"]
        or model._n_updates != meta["optimizer_updates"]
        or model.num_timesteps >= config.total_timesteps
    ):
        raise ValueError("P1 transfer serialized policy/counters mismatch or budget exhausted")
    # Architecture fingerprint includes buffers as well as weights. It checks the current
    # encoder/actor/critic, not just observation shapes or total parameter count.
    expected = build_model(config, env)
    actual_state, expected_state = model.policy.state_dict(), expected.policy.state_dict()
    if {k: (v.shape, v.dtype) for k, v in actual_state.items()} != {
        k: (v.shape, v.dtype) for k, v in expected_state.items()
    }:
        raise ValueError("P1 transfer network architecture mismatch")
    if any(not torch.isfinite(v).all() for v in actual_state.values()):
        raise ValueError("P1 transfer nonfinite policy")
    del expected
    optimizer_state = model.policy.optimizer.state_dict()["state"]
    if not optimizer_state:
        raise ValueError("P1 transfer requires a trained optimizer state")
    if any(
        isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
        for slots in optimizer_state.values()
        for value in slots.values()
    ):
        raise ValueError("P1 transfer nonfinite optimizer state")
    provenance = {
        "protocol": "p1-to-p2-reward-v1",
        "parent_checkpoint": str(parent),
        "parent_source_revision": meta["source_revision"],
        "destination_source_revision": implementation_id(),
        "parent_checksums": json.loads((parent / "checksums.json").read_text()),
        "parent_manifest_sha256": _hash(parent / "checksums.json"),
        "parent_env_steps": model.num_timesteps,
        "parent_optimizer_updates": model._n_updates,
        "changes": {
            "stop_after": {"from": "P1", "to": "P2"},
            "p2_reward": {
                "from": old["skills"].get(
                    "p2_reward",
                    {
                        "area": 0.01,
                        "discovery": 0.05,
                        "activation": old["skills"]["activation"],
                        "step_cost": 0.0,
                    },
                ),
                "to": new["skills"]["p2_reward"],
            },
        },
        "preserved": [
            "weights",
            "optimizer",
            "env_steps",
            "learning_rate_schedule",
            "curriculum_controller",
            "worker_sampler",
        ],
        "episodes": "reset; no promise of bit-for-bit continuation",
        "rng": "restored on training start; device topology must match",
    }
    return model, state, provenance


def check_p1_transfer(path, config):
    """Local or remote no-learning preflight, including a real P2 forward/env step."""
    from ather_exploration.training.skill_environments import SkillTrainingEnv

    torch.set_num_threads(config.torch_threads)
    env = SkillTrainingEnv(config)
    try:
        model, state, provenance = prepare_p1_transfer(path, config, env)
        rng_checked = False
        if config.device.startswith("cuda"):
            restore_rng(provenance["parent_checkpoint"])
            rng_checked = True
        env.set_controller(state["skill_controller"])
        env.restore(state["workers"][0])
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, info = env.step(int(action))
        return {
            "status": "valid",
            "learning_executed": False,
            "rng_restoration_checked": rng_checked,
            "phase": env.phase,
            "source_task": env.task,
            "step_cost": env.env.step_cost,
            "sample_reward": reward,
            "sample_reward_components": info["skill"]["reward_components"],
            "transfer": provenance,
        }
    finally:
        env.close()
