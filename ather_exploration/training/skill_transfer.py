"""Audited completed-skill transfers. Never updates weights or rewrites a parent."""

import json

import torch
from stable_baselines3 import PPO

from ather_exploration.agents.learning import build_model, schema_signature
from ather_exploration.training.checkpoints import _hash, inspect_checkpoint, restore_rng
from ather_exploration.training.skill_environments import skill_identity
from ather_exploration.worlds.scenarios import implementation_id


def _prepare_transfer(path, config, env, *, source_phase):
    """Validate configuration, spaces, architecture and counters; load optimizer unchanged."""
    _p3 = source_phase == "P2"
    parent, meta = inspect_checkpoint(path, transfer_p1_to_p2=not _p3, transfer_p2_to_p3=_p3)
    old, new = meta["config"], config.model_dump(mode="json")
    old = {"lr_trial": None, "p3_restart": False, **old}
    old = {
        **old,
        "skills": {"frontier": False, "p3_visit_bonus": 0.0, "p3_visit_cap": 0.1, **old["skills"]},
    }
    if not config.skills.enabled or config.skills.stop_after != ("P3" if _p3 else "P2"):
        raise ValueError("Transfer destination must be P2 for a P1 parent, or P3 for a P2 parent")
    if config.method != "ppo" or config.skills.wall_mask:
        raise ValueError("Completed-skill transfer supports unmasked feedforward PPO only")
    # Only settings for not-yet-trained objectives may change.
    allowed = {"stop_after", "p3_reward", "p3_gates", "p3_horizon", "p3_minimum", "p3_task_budget"}
    if not _p3:
        allowed.update(("p2_reward", "p2_gates", "p2c_horizon", "p2_task_budget"))
    old_skills = {k: v for k, v in old["skills"].items() if k not in allowed}
    new_skills = {k: v for k, v in new["skills"].items() if k not in allowed}
    if old_skills != new_skills:
        raise ValueError(
            "Completed-skill transfer skill config mismatch outside stop_after/p2_reward"
        )
    for key in set(old) | set(new):
        if key not in ("skills", "banks", "device", "tracking") and old.get(key) != new.get(key):
            raise ValueError(f"Completed-skill transfer config mismatch: {key}")
    from ather_exploration.worlds.scenarios import digest

    legacy_identity = {
        "skills": digest(
            {
                "version": 3 if _p3 else 2,
                "train_count": config.skills.train_count,
                "validation_count": config.skills.validation_count,
                "split": "content_hash_mod5",
            }
        )
    }
    if meta["bank_ids"] not in (legacy_identity, skill_identity(config)):
        raise ValueError("Completed-skill transfer skill bank identity mismatch")
    signature = schema_signature(env.observation_space)
    if meta["schema"] != signature or env.action_space.n != 5:
        raise ValueError("Completed-skill transfer observation/action schema mismatch")
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
        raise ValueError("Completed-skill transfer controller/worker counters mismatch")
    model = PPO.load(parent / "model.zip", env=env, device=config.device)
    if (
        schema_signature(model.observation_space) != signature
        or model.action_space.n != 5
        or model.num_timesteps != meta["env_steps"]
        or model._n_updates != meta["optimizer_updates"]
        or model.num_timesteps >= config.total_timesteps
    ):
        raise ValueError(
            "Completed-skill transfer serialized policy/counters mismatch or budget exhausted"
        )
    # Architecture fingerprint includes buffers as well as weights. It checks the current
    # encoder/actor/critic, not just observation shapes or total parameter count.
    expected = build_model(config, env)
    actual_state, expected_state = model.policy.state_dict(), expected.policy.state_dict()
    if {k: (v.shape, v.dtype) for k, v in actual_state.items()} != {
        k: (v.shape, v.dtype) for k, v in expected_state.items()
    }:
        raise ValueError("Completed-skill transfer network architecture mismatch")
    if any(not torch.isfinite(v).all() for v in actual_state.values()):
        raise ValueError("Completed-skill transfer nonfinite policy")
    del expected
    optimizer_state = model.policy.optimizer.state_dict()["state"]
    if not optimizer_state:
        raise ValueError("Completed-skill transfer requires a trained optimizer state")
    if any(
        isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
        for slots in optimizer_state.values()
        for value in slots.values()
    ):
        raise ValueError("Completed-skill transfer nonfinite optimizer state")
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
    controller["p2_task_budget"] = config.skills.p2_task_budget
    controller["best_by_task"] = {}
    provenance["protocol"] = "p1-to-p2-reward-v2"
    provenance["changes"]["curriculum_protocol"] = {
        "from": meta["curriculum_protocol"],
        "to": "active-phase-v3",
    }
    provenance["changes"]["p2_settings"] = {
        k: new["skills"][k] for k in ("p2_gates", "p2c_horizon", "p2_task_budget")
    }
    provenance["changes"]["bank_ids"] = {"from": meta["bank_ids"], "to": skill_identity(config)}
    controller["p3_minimum"] = config.skills.p3_minimum
    controller["p3_task_budget"] = config.skills.p3_task_budget
    if _p3:
        provenance["protocol"] = "p2-to-p3-multiroom-v1"
        provenance["changes"] = {
            "stop_after": {"from": "P2", "to": "P3"},
            "curriculum_protocol": {"from": meta["curriculum_protocol"], "to": "active-phase-v3"},
            "settings": {k: new["skills"][k] for k in allowed if k != "stop_after"},
            "bank_ids": {"from": meta["bank_ids"], "to": skill_identity(config)},
            "task_mapping": "completed P2c -> P3a; preserve absolute learning counters",
        }
    return model, state, provenance


def _check_transfer(path, config, *, source_phase):
    """Local or remote no-learning preflight, including a real P2 forward/env step."""
    from ather_exploration.training.skill_environments import SkillTrainingEnv

    torch.set_num_threads(config.torch_threads)
    env = SkillTrainingEnv(config)
    try:
        model, state, provenance = _prepare_transfer(path, config, env, source_phase=source_phase)
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


def prepare_p2_transfer(path, config, env):
    """Audited completed P2c -> P3a; retain trained PPO and optimizer."""
    return _prepare_transfer(path, config, env, source_phase="P2")


def check_p2_transfer(path, config):
    return _check_transfer(path, config, source_phase="P2")


def prepare_p1_transfer(path, config, env):
    return _prepare_transfer(path, config, env, source_phase="P1")


def check_p1_transfer(path, config):
    return _check_transfer(path, config, source_phase="P1")
