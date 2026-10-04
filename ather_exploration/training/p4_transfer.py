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

PROTOCOL = "public-threat-timing-v4"
SOURCE = "17e3e864172cc7cd52e2a70cc6b293258192b0f39b9d748aa643e4de790797c8"
FORCED_P4A_SOURCE = "1954d1c44e80c66da92d538811d0de6a0cdb32799086d292df516aed48e7dd82"
FORCED_P4A_STEPS = 1851392


def prepare_p4(path, config, env):
    if not path or not config.p4_transfer or not config.skills.p4.enabled:
        raise ValueError("P4 requires --resume")
    parent, meta = inspect_checkpoint(
        path,
        inference=True,
        forced_p4_branch=config.p4_force_advance,
    )
    state = json.loads((parent / "runner_state.json").read_text())
    if (
        state["env_steps"] != meta["env_steps"]
        or state["skill_controller"] != meta["skill_controller"]
    ):
        raise ValueError("P4 parent controller/counter mismatch")
    if meta["config"].get("p4_transfer"):
        old = TrainingConfig.model_validate(meta["config"]).model_dump(mode="json")
        new = config.model_dump(mode="json")
        branching = config.p4_lr_branch and not old["p4_lr_branch"]
        forcing = config.p4_force_advance and not old.get("p4_force_advance", False)
        allowed = {"banks", "device", "tracking"}
        if branching:
            allowed |= {"learning_rate", "final_learning_rate", "p4_lr_branch"}
        if forcing:
            allowed.add("p4_force_advance")
        if any(old[k] != new[k] for k in new if k not in allowed):
            raise ValueError("P4 resume config mismatch")
        if (
            (meta["source_revision"] != implementation_id() and not (branching or forcing))
            or state["state"] != "RUNNING"
            or state.get("transfer", {}).get("protocol") != PROTOCOL
            or meta["bank_ids"] != skill_identity(config)
            or len(state["workers"]) != config.n_envs
            or not state["skill_controller"].get("p4_enabled")
        ):
            raise ValueError("P4 resume source/state/suite mismatch")
        if branching and (
            meta["source_revision"]
            != "ddc453e5b1bdf88cb70d72e5c1c73b5f744c60097e7d4046d0d6ccc078ea95ba"
            or meta["env_steps"] != 1703936
            or meta["viewer_task"] != "P4a"
            or old["learning_rate"] != 0.0001
            or old["final_learning_rate"] != 0.0001
            or state["skill_controller"]["failed"]
        ):
            raise ValueError("P4 LR branch requires audited timing step_1703936")
        if forcing:
            c = state["skill_controller"]
            if (
                branching
                or meta["source_revision"] != FORCED_P4A_SOURCE
                or meta["env_steps"] != FORCED_P4A_STEPS
                or meta["viewer_task"] != "P4a"
                or meta["schema"].get("version") != 4
                or c["index"] != 8
                or c["failed"]
                or c["best_by_task"].get("P4a", {}).get("checkpoint")
                != f"checkpoints/step_{FORCED_P4A_STEPS}"
                or old.get("p4_force_advance", False)
            ):
                raise ValueError("Forced P4 continuation requires the audited best P4a checkpoint")
        model = RoutePPO.load(parent / "model.zip", env=env, device=config.device)
        if not hasattr(model, "route_memory"):
            raise ValueError("Missing retention memory")
        audit = copy.deepcopy(state["transfer"])
        if branching:
            from ather_exploration.agents.learning import LinearSchedule

            if not model.policy.optimizer.state:
                raise ValueError("P4 LR branch requires saved optimizer moments")
            model.learning_rate = LinearSchedule(config.learning_rate, config.final_learning_rate)
            model.lr_schedule = model.learning_rate
            for group in model.policy.optimizer.param_groups:
                group["lr"] = config.learning_rate
            controller = state["skill_controller"]
            controller["passed"] = 0
            controller["threat_streak"] = 0
            controller["best_by_task"] = {}
            controller["best"] = {}
            audit["lr_branch"] = {
                "protocol": "p4-lr-branch-v1",
                "parent_checkpoint": str(parent),
                "parent_checksums": json.loads((parent / "checksums.json").read_text()),
                "parent_source_revision": meta["source_revision"],
                "parent_steps": meta["env_steps"],
                "from": old["learning_rate"],
                "to": config.learning_rate,
                "shared_optimizer": ["PPO", "timing", "retention"],
                "preserved": [
                    "weights",
                    "Adam moments",
                    "temperature",
                    "teaching reservoirs",
                    "teacher",
                    "sampler state",
                    "task elapsed",
                    "total budget",
                ],
                "reset": ["gate streak", "lesson streak", "best selector", "episodes"],
            }
        if forcing:
            c.update(
                {
                    "index": 9,
                    "phase_start": meta["env_steps"],
                    "family_start": meta["env_steps"],
                    "passed": 0,
                    "failed": False,
                    "p4_force_advance": True,
                }
            )
            state["state"] = "RUNNING"
            state["viewer_task"] = "P4b"
            # Do not carry active P4a episodes into the forced P4b branch.
            state["workers"] = []
            audit["forced_curriculum"] = {
                "protocol": "p4-forced-advance-v1",
                "parent_run": "skills-p4-balanced-full-20261005-001904-22555",
                "parent_checkpoint": str(parent),
                "parent_steps": meta["env_steps"],
                "source_task": "P4a",
                "start_task": "P4b",
                "p4a_gate": "failed; not represented as passed",
                "p4b_gate": "kept unchanged; advance to P4c at budget cap if it fails",
                "p4c_gate": "kept unchanged; failure at its cap remains a failed gate",
                "active_episodes": "reset at branch start",
                "interpretation": "experimental continuation; not a full-P4 pass",
            }
    else:
        if config.p4_force_advance:
            raise ValueError("Forced P4 continuation requires an in-progress P4a checkpoint")
        if config.p4_lr_branch:
            raise ValueError("P4 LR branch requires a timing checkpoint, not P3")
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
                key.endswith("memory.0.weight")
                and new[key].shape[1] == 12 + 2 * config.skills.p4.history_frames
                and value.shape[1] == 12
            ):
                new[key].zero_()
                new[key][:, :12] = value
            else:
                raise ValueError(f"Unexpected parameter migration: {key}")
        model.policy.load_state_dict(new)
        from ather_exploration.training.threat_training import calibrate_actor

        calibration = calibrate_actor(model, config) if config.skills.p4.soften_actor else None
        if config.skills.p4.recovery:
            from ather_exploration.training.threat_retention import (
                ThreatRetention,
                initialize_retention,
            )
            from ather_exploration.worlds.skill_tasks import skill_pool

            seeds = [seed for seed, _ in skill_pool("P3c", config.skills.train_count)]
            model.threat_retention = ThreatRetention(old.policy, config.seed, seeds)
            model.threat_retention.batches = config.skills.p4.retention_batches
            model.threat_retention.max_kl = config.skills.p4.retention_max_kl
            initialize_retention(model.threat_retention, config)
            if config.skills.p4.timing:
                from ather_exploration.training.threat_training import calibrate_threat_temperature
                from ather_exploration.training.timing_teaching import (
                    TimingTeaching,
                    initialize_timing,
                )

                calibration = calibrate_threat_temperature(model, config)
                model.timing_teaching = TimingTeaching(config)
                initialize_timing(model, config)
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
            "migration": f"12 to {12 + 2 * config.skills.p4.history_frames} memory channels, additional input weights zero",
            "optimizer": "fresh Adam for changed input shape",
            "retention": "frozen parent KL on P3c train only"
            if config.skills.p4.recovery
            else "P3c PPO replay",
            "threat_exploration": 0.0
            if config.skills.p4.timing
            else config.skills.p4.threat_exploration
            if config.skills.p4.recovery
            else 0,
            "actor_calibration": calibration,
            "worker_quota": config.skills.p4.worker_quota,
            "teaching_batches": config.skills.p4.teaching_batches,
        }
    if config.skills.p4.recovery and not hasattr(model, "threat_retention"):
        raise ValueError("Missing frozen parent retention state")
    if config.skills.p4.timing and not hasattr(model, "timing_teaching"):
        raise ValueError("Missing timing teaching state")
    model.teaching_batches = config.skills.p4.teaching_batches
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
