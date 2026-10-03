"""Audited recovery transfer, frozen training probes and no-learning preflight."""

import copy
import hashlib
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from ather_exploration.agents.learning import schema_signature
from ather_exploration.agents.route_ppo import RoutePPO
from ather_exploration.training.checkpoints import inspect_checkpoint
from ather_exploration.training.config import TrainingConfig
from ather_exploration.training.public_route import public_route
from ather_exploration.training.recovery_archive import StagnationCapture
from ather_exploration.training.skill_curriculum import SkillController
from ather_exploration.training.skill_environments import skill_identity
from ather_exploration.worlds.scenarios import implementation_id
from ather_exploration.worlds.skill_tasks import configured_skill_env, skill_pool

PROTOCOL = "public-route-recovery-v1"
PARENT_SOURCE = "28e2dff54a78e80dd1b2f53ae1de6f667b1ddc8ef69aec554b5e9c0461e73e8b"


def prepare_recovery(path, config, env):
    parent, metadata = inspect_checkpoint(path, recovery=True)
    old = TrainingConfig.model_validate(metadata["config"]).model_dump(mode="json")
    new = config.model_dump(mode="json")
    state = json.loads((parent / "runner_state.json").read_text())
    balanced = config.recovery.sampling != "stale_uniform"
    aggregated = config.recovery.sampling == "aggregated_teaching"
    protocol = (
        "public-route-teaching-v3"
        if aggregated
        else "public-route-disagreement-v2"
        if balanced
        else PROTOCOL
    )
    branch = (
        balanced and old["recovery"] is not None and old["recovery"]["sampling"] == "stale_uniform"
    )
    initial = old["recovery"] is None or branch
    allowed = {"banks", "device", "tracking"} | ({"recovery", "p3_resume"} if initial else set())
    if any(old[key] != new[key] for key in new if key not in allowed):
        raise ValueError("Recovery config differs outside the audited transfer fields")
    controller = SkillController(**state["skill_controller"])
    if (
        metadata["schema"] != schema_signature(env.observation_space)
        or metadata["bank_ids"] != skill_identity(config)
        or metadata.get("curriculum_protocol") != "active-phase-v3"
        or state["state"] != "RUNNING"
        or controller.failed
        or controller.task not in ("P3b", "P3c")
        or state["env_steps"] != metadata["env_steps"]
        or len(state["workers"]) != config.n_envs
        or metadata["env_steps"] % (config.n_envs * config.n_steps)
    ):
        raise ValueError("Recovery parent state/schema/split/boundary mismatch")
    if branch:
        if (
            metadata["source_revision"]
            != "b8ab7598e58dcf56820a13aba394e16a88e055cdb28777e0ed79c52dd9421a88"
            or metadata["env_steps"] != 1572864
            or controller.task != "P3c"
            or state.get("transfer", {}).get("protocol") != PROTOCOL
        ):
            raise ValueError("Balanced recovery requires audited P3c best step_1572864")
        allowed_recovery = {"sampling", "parent_steps", "additional_steps"}
        if any(
            old["recovery"][k] != new["recovery"][k]
            for k in new["recovery"]
            if k not in allowed_recovery
        ):
            raise ValueError("Balanced branch changes more than sampling and budget")
    elif initial:
        if (
            metadata["source_revision"] != PARENT_SOURCE
            or metadata["env_steps"] != config.recovery.parent_steps
            or controller.task != "P3b"
            or not old["p3_resume"]
        ):
            raise ValueError("Recovery starts from audited completion checkpoint step_1048576")
    elif (
        metadata["source_revision"] != implementation_id()
        or state.get("transfer", {}).get("protocol") != protocol
    ):
        raise ValueError("Recovery resume protocol/source mismatch")
    model = RoutePPO.load(parent / "model.zip", env=env, device=config.device)
    if (
        model.num_timesteps != metadata["env_steps"]
        or model._n_updates != metadata["optimizer_updates"]
        or schema_signature(model.observation_space) != metadata["schema"]
        or not model.policy.optimizer.state
        or any(not torch.isfinite(p).all() for p in model.policy.parameters())
    ):
        raise ValueError("Recovery policy/optimizer/counters mismatch")
    if any(
        isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
        for slot in model.policy.optimizer.state.values()
        for v in slot.values()
    ):
        raise ValueError("Nonfinite parent Adam state")
    model.route_coefficient = 0.0 if aggregated else config.recovery.route_coefficient
    if aggregated:
        from ather_exploration.training.route_teaching import RouteMemory

        if initial:
            model.route_memory = RouteMemory(seed=config.seed)
        elif not hasattr(model, "route_memory") or not model.route_memory.initialized:
            raise ValueError("Teaching resume is missing its persistent dataset")
    model.route_samples = []
    if initial:
        old_controller = copy.deepcopy(state["skill_controller"])
        controller.phase_start = model.num_timesteps
        controller.passed = 0
        controller.restart_level = 0
        controller.restart_results = []
        controller.best_by_task = {}
        if branch:
            controller.recovery_p3c_budget = config.recovery.additional_steps
        else:
            controller.recovery_p3b_budget = config.recovery.additional_steps
        state["skill_controller"] = asdict(controller)
        # No carry-over of biased early-prefix pools. Preserve sampler RNG only.
        for worker in state["workers"]:
            worker.pop("prefix_archive", None)
            worker.pop("unfinished", None)
        audit = {
            "protocol": protocol,
            "learning_objective": "ppo_separate_public_teaching"
            if aggregated
            else "ppo_public_route_aux",
            "parent_env_steps": model.num_timesteps,
            "parent_source_revision": metadata["source_revision"],
            "parent_checksums": json.loads((parent / "checksums.json").read_text()),
            "parent_controller": old_controller,
            "parent_checkpoint": str(parent),
            "reward_map_gate_changed": False,
            "reset": (
                ["gate streak", "current best selection", "live episodes"]
                if branch
                else ["prefix archive", "gate streak", "current best selection", "live episodes"]
            ),
            "archive_preserved": branch,
            "sampling": config.recovery.sampling,
            "preserved": ["weights", "Adam", "global counters", "sampler RNG", "validation split"],
            "probe_steps": state.get("transfer", {}).get("probe_steps", 0) if branch else 0,
        }
    else:
        audit = copy.deepcopy(state["transfer"])
    audit["resume_rng_checkpoint"] = str(parent)
    state["transfer"] = audit
    return model, state, audit


def probe_seeds(config, task):
    """Round-robin strata within the actual train pool, never a grouped prefix."""
    from collections import defaultdict, deque

    from ather_exploration.worlds.p3_tasks import map_group, p3_scenario

    groups = defaultdict(deque)
    for seed, _ in skill_pool(task, config.skills.train_count):
        group = map_group(p3_scenario(task, seed))
        groups[tuple(group[key] for key in ("size", "topology", "poi_count"))].append(seed)
    selected = []
    while len(selected) < config.recovery.probe_count:
        for group in sorted(groups):
            if groups[group] and len(selected) < config.recovery.probe_count:
                selected.append(groups[group].popleft())
    return selected


def training_probes(model, config, task):
    """Deterministic frozen policy, TRAIN seeds only; never part of PPO buffer."""

    seeds = probe_seeds(config, task)
    items, steps = [], 0
    mode = model.policy.training
    try:
        for seed in seeds:
            env = configured_skill_env(task, seed, config.skills, phase=task)
            try:
                obs, _ = env.reset()
                capture = StagnationCapture(config.recovery, task, seed, obs)
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, term, trunc, _ = env.step(int(action))
                    steps += 1
                    item = capture.step(int(action), obs, config.skills.p3_horizon)
                    if item:
                        items.append(item)
                    if term or trunc:
                        break
            finally:
                env.close()
    finally:
        model.policy.set_training_mode(mode)
    return items, steps


def check_recovery(path, config):
    """Validate transfer plus frozen public-planner shadow on both supported tasks."""
    from ather_exploration.training.skill_environments import SkillTrainingEnv

    torch.set_num_threads(config.torch_threads)
    env = SkillTrainingEnv(config)
    try:
        model, state, audit = prepare_recovery(path, config, env)
        env.set_controller(state["skill_controller"])
        env.restore(state["workers"][0])
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, _ = env.step(int(action))
        if not np.isfinite(reward):
            raise ValueError("Nonfinite recovery transition")
        baseline = frozen_baseline(model, config)
        shadow = shadow_routes(config)
        if any(
            row["success"] != row["episodes"] for key, row in shadow.items() if key != "seconds"
        ):
            raise ValueError("Public teacher failed structural shadow preflight")
        return {
            "status": "valid",
            "learning_executed": False,
            "transfer": audit,
            "shadow": shadow,
            "frozen_parent_probe": baseline,
            "sample_reward": float(reward),
        }
    finally:
        env.close()


def shadow_routes(config):
    """No optimizer: exercise teacher on train and validation without retaining labels."""
    results = {}
    started = time.monotonic()
    for task in ("P3b", "P3c"):
        for split in ("train", "validation"):
            # These are structural preflight samples, not the model gate.
            rows = skill_pool(task, 8, validation=split == "validation")
            successes, steps = 0, 0
            for row in rows:
                env = configured_skill_env(task, row[0], config.skills, phase=task)
                try:
                    obs, _ = env.reset()
                    while True:
                        route = public_route(obs)
                        action = int(np.flatnonzero(route["actions"])[0]) if route else 4
                        obs, _, term, trunc, info = env.step(action)
                        steps += 1
                        if term or trunc:
                            successes += bool(info.get("skill", {}).get("success"))
                            break
                finally:
                    env.close()
            results[f"{task}/{split}"] = {
                "success": successes,
                "episodes": len(rows),
                "steps": steps,
            }
    results["seconds"] = time.monotonic() - started
    return results


def frozen_baseline(model, config):
    """Known CPU/CUDA sensitivity seed: report native policy, no teacher intervention."""
    env = configured_skill_env("P3b", 100051, config.skills, phase="P3b")
    actions = []
    try:
        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            actions.append(int(action))
            obs, _, term, trunc, info = env.step(int(action))
            if term or trunc:
                return {
                    "seed": 100051,
                    "success": bool(info["skill"]["success"]),
                    "device": str(model.device),
                    "steps": len(actions),
                    "actions_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
                    "learning_executed": False,
                }
    finally:
        env.close()
