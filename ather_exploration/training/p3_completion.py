"""Productive unfinished-frontier replay and audited P3a-boundary resume."""

import copy
import hashlib
import json

import numpy as np
import torch
from stable_baselines3 import PPO

from ather_exploration.agents.learning import build_model, schema_signature
from ather_exploration.training.checkpoints import inspect_checkpoint
from ather_exploration.training.config import P3ResumeConfig, TrainingConfig
from ather_exploration.training.skill_curriculum import STAGES
from ather_exploration.training.skill_environments import skill_identity
from ather_exploration.worlds.p3_tasks import (
    room_coverage_fractions,
    room_exploration_potential,
)
from ather_exploration.worlds.scenarios import implementation_id

BANDS = ((0, 1), (2, 4), (5, 8))
P3_BOUNDARY_SOURCE = "b1caee3c156a4723864d0a465629765ac26a000ea7babc345404b1ffaf7a75be"


def prefix_quality(before_rooms, after_rooms, activated_before, activated_after, poi_count):
    """Score only progress that occurred after a candidate replay prefix."""
    before = np.asarray(before_rooms, dtype=np.float64)
    after = np.asarray(after_rooms, dtype=np.float64)
    if before.ndim != 1 or before.shape != after.shape or not len(before):
        raise ValueError("Prefix room coverage vectors must be nonempty and match")
    if poi_count < 1 or activated_before < 0 or activated_after < activated_before:
        raise ValueError("Invalid prefix POI progress counts")
    room_gain = max(
        0.0,
        room_exploration_potential(after) - room_exploration_potential(before),
    )
    newly_opened = sum((before <= 0) & (after > 0)) / len(before)
    new_activations = min(1.0, (activated_after - activated_before) / poi_count)
    return float(0.65 * room_gain + 0.20 * newly_opened + 0.15 * new_activations)


def _observation_digest(observation):
    digest = hashlib.sha256()
    for key in sorted(observation):
        array = np.ascontiguousarray(observation[key])
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _frontier_distance(observation):
    from ather_exploration.training.unfinished import frontier_distance

    return frontier_distance(observation)


class P3PrefixArchive:
    """Per-phase, quality-ranked prefixes with a bounded three-level curriculum."""

    def __init__(self, config: P3ResumeConfig, rng):
        self.config = config
        self.rng = rng
        self.pools = {"P3b": [[] for _ in BANDS], "P3c": [[] for _ in BANDS]}
        self.level = 0
        self.replay_steps = 0

    def capture(self, task, seed, actions, observation, scenario, activated_count, captured):
        """Capture one candidate per frontier band; score is assigned at episode end."""
        if task not in self.pools or len(actions) < 16:
            return None
        if len(actions) > scenario.horizon - self.config.minimum_remaining:
            return None
        distance = _frontier_distance(observation)
        band = next(
            (
                index
                for index, (minimum, maximum) in enumerate(BANDS)
                if distance is not None and minimum <= distance <= maximum
            ),
            None,
        )
        if band is None or band in captured:
            return None
        captured.add(band)
        return {
            "task": task,
            "seed": int(seed),
            "band": band,
            "distance": int(distance),
            "actions": list(actions),
            "observation": _observation_digest(observation),
            "room_coverage": room_coverage_fractions(scenario, observation["memory"]).tolist(),
            "activated_count": int(activated_count),
        }

    def complete(self, task, candidates, observation, scenario, activated_count):
        """Archive candidates only if their continuation produced room/POI progress."""
        if task not in self.pools:
            return []
        final_rooms = room_coverage_fractions(scenario, observation["memory"])
        activated_after = int(activated_count)
        qualities = []
        for candidate in candidates:
            quality = prefix_quality(
                candidate["room_coverage"],
                final_rooms,
                candidate["activated_count"],
                activated_after,
                len(scenario.pois),
            )
            if quality <= 1e-6:
                continue
            item = {**candidate, "quality": quality}
            self.offer(task, item)
            qualities.append(quality)
        return qualities

    def offer(self, task, item):
        if task not in self.pools or item.get("task") != task:
            raise ValueError("Prefix task does not match its phase archive")
        band = item.get("band")
        quality = item.get("quality")
        if (
            type(band) is not int
            or not 0 <= band < len(BANDS)
            or not isinstance(quality, (int, float))
            or not np.isfinite(quality)
            or quality <= 0
        ):
            raise ValueError("Archive prefix requires a valid band and positive finite quality")
        pool = self.pools[task][band]
        for index, previous in enumerate(pool):
            if (
                previous.get("source_task", task) == item.get("source_task", task)
                and previous["seed"] == item["seed"]
            ):
                if quality > previous["quality"]:
                    pool[index] = copy.deepcopy(item)
                return
        if len(pool) < self.config.pool_per_band:
            pool.append(copy.deepcopy(item))
            return

        # Eviction is weighted toward weaker items; occasionally retain a diverse weaker map.
        victim_weights = np.asarray([1.0 / (0.02 + entry["quality"]) for entry in pool])
        victim_weights /= victim_weights.sum()
        victim_index = int(self.rng.choice(len(pool), p=victim_weights))
        victim_quality = pool[victim_index]["quality"]
        replacement_probability = min(1.0, quality / max(victim_quality, 1e-6))
        if self.rng.random() < replacement_probability:
            pool[victim_index] = copy.deepcopy(item)

    def choose(self, task):
        requested = bool(self.rng.random() < self.config.restart_probability)
        if not requested:
            return None, False
        if task not in self.pools:
            return None, True
        available = [index for index in range(self.level + 1) if self.pools[task][index]]
        if not available:
            return None, True
        band = int(self.rng.choice(available))
        pool = self.pools[task][band]
        quality = np.sqrt([entry["quality"] for entry in pool])
        weighted = 0.5 / len(pool) + 0.5 * quality / quality.sum()
        item = copy.deepcopy(pool[int(self.rng.choice(len(pool), p=weighted))])
        return item, True

    def replay(self, env, item):
        observation, info = env.reset()
        for action in item["actions"]:
            observation, _, terminated, truncated, info = env.step(action)
            self.replay_steps += 1
            if terminated or truncated:
                raise ValueError("P3 replay prefix terminated before its boundary")
        if _observation_digest(observation) != item["observation"]:
            raise ValueError("P3 replay observation differs; refusing stale prefix")
        return observation, info

    def state(self):
        return {
            "pools": copy.deepcopy(self.pools),
            "level": self.level,
            "replay_steps": self.replay_steps,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def restore(self, state):
        self.pools = copy.deepcopy(state["pools"])
        self.level = state["level"]
        self.replay_steps = state["replay_steps"]
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])


def _assert_config_compatible(old, new):
    mutable_top_level = {"p3_restart", "p3_resume", "banks", "device", "tracking"}
    for key in new:
        if key not in mutable_top_level | {"skills"} and new[key] != old[key]:
            raise ValueError(f"P3 resume config mismatch: {key}")

    allowed_skill_changes = {
        "p3_reward": {"room_exploration"},
        "p3_gates": {
            "room_coverage",
            "room_coverage_auc",
        },
    }
    for key in new["skills"]:
        previous, current = old["skills"][key], new["skills"][key]
        allowed = allowed_skill_changes.get(key, set())
        if key in allowed_skill_changes:
            if any(
                previous[name] != current[name]
                for name in set(previous) | set(current)
                if name not in allowed
            ):
                raise ValueError(f"P3 resume config mismatch: skills.{key}")
        elif previous != current:
            raise ValueError(f"P3 resume config mismatch: skills.{key}")


def prepare_p3_resume(path, config, env):
    """Load the exact passed P3a boundary, retaining policy, Adam, and controller."""
    if config.p3_resume is None:
        raise ValueError("p3_resume configuration is required")
    parent, metadata = inspect_checkpoint(path, p3_resume=True)
    old = TrainingConfig.model_validate(metadata["config"]).model_dump(mode="json")
    new = config.model_dump(mode="json")
    _assert_config_compatible(old, new)
    state = json.loads((parent / "runner_state.json").read_text())
    controller = state["skill_controller"]
    history = controller.get("history", [])
    source_revision = metadata["source_revision"]
    if source_revision == P3_BOUNDARY_SOURCE:
        resume_kind = "p3a_boundary"
        valid_boundary = (
            metadata["env_steps"] == config.p3_resume.parent_steps
            and metadata["viewer_task"] == "P3b"
            and controller.get("index") == 6
            and controller.get("phase_start") == metadata["env_steps"]
            and len(history) >= 2
            and all(
                item.get("task") == "P3a" and item.get("passed") and item.get("eligible")
                for item in history[-2:]
            )
            and history[-1].get("steps") == metadata["env_steps"]
            and history[-1].get("streak", 0) >= 2
        )
    elif source_revision == implementation_id():
        resume_kind = "p3_completion_checkpoint"
        locked_old = {
            key: value for key, value in old.items() if key not in {"banks", "device", "tracking"}
        }
        locked_new = {
            key: value for key, value in new.items() if key not in {"banks", "device", "tracking"}
        }
        valid_boundary = (
            locked_old == locked_new
            and old.get("p3_resume") == new.get("p3_resume")
            and metadata["env_steps"] > config.p3_resume.parent_steps
            and metadata["env_steps"] < config.total_timesteps
            and controller.get("index") in (6, 7)
            and metadata["viewer_task"] == STAGES[controller["index"]]
            and controller.get("phase_start", metadata["env_steps"]) <= metadata["env_steps"]
            and state.get("transfer", {}).get("protocol") == "p3a-boundary-room-completion-v1"
        )
    else:
        valid_boundary = False
        resume_kind = "unsupported_source"
    if (
        not valid_boundary
        or state["env_steps"] != metadata["env_steps"]
        or metadata["bank_ids"] != skill_identity(config)
        or len(state["workers"]) != config.n_envs
        or metadata["schema"] != schema_signature(env.observation_space)
        or metadata.get("curriculum_protocol") != "active-phase-v3"
        or state["state"] != "RUNNING"
        or controller.get("failed")
        or metadata["env_steps"] % (config.n_envs * config.n_steps)
    ):
        raise ValueError(
            "P3 resume requires the passed P3a boundary or a compatible active P3b/P3c checkpoint"
        )

    model = PPO.load(parent / "model.zip", env=env, device=config.device)
    # This temporary model is used only for architecture comparison. A single-env
    # no-learning check should not emit a false batch/rollout-size warning for the
    # production 16-env training config.
    check_config = config.model_copy(update={"batch_size": config.n_steps})
    expected = build_model(check_config, env)
    actual_state = model.policy.state_dict()
    expected_state = expected.policy.state_dict()
    if (
        model.num_timesteps != metadata["env_steps"]
        or model._n_updates != metadata["optimizer_updates"]
        or schema_signature(model.observation_space) != metadata["schema"]
        or actual_state.keys() != expected_state.keys()
        or any(actual_state[key].shape != expected_state[key].shape for key in actual_state)
        or any(not torch.isfinite(value).all() for value in actual_state.values())
        or not model.policy.optimizer.state
    ):
        raise ValueError("P3 resume model counters/schema/weights/optimizer mismatch")
    optimizer_state = model.policy.optimizer.state_dict()["state"]
    if any(
        isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
        for slots in optimizer_state.values()
        for value in slots.values()
    ):
        raise ValueError("P3 resume has a nonfinite optimizer state")

    controller = copy.deepcopy(controller)
    if resume_kind == "p3a_boundary":
        controller["restart_level"] = 0
        controller["restart_results"] = []
        audit = {
            "protocol": "p3a-boundary-room-completion-v1",
            "parent_checkpoint": str(parent),
            "parent_source_revision": source_revision,
            "destination_source_revision": implementation_id(),
            "parent_env_steps": metadata["env_steps"],
            "parent_optimizer_updates": metadata["optimizer_updates"],
            "parent_checksums": json.loads((parent / "checksums.json").read_text()),
            "resume_kind": resume_kind,
            "start_phase": "P3b",
            "promotion": "normal two-consecutive-gate promotion P3b -> P3c -> P4a",
            "preserved": [
                "policy weights and 12-channel architecture",
                "Adam optimizer state",
                "environment and optimizer counters",
                "P3a pass history and P3b controller boundary",
                "worker sampling RNG state",
                "map and validation identities",
            ],
            "changed": [
                "room-progress reward in P3b/P3c only",
                "room-balanced P3b/P3c coverage gate semantics",
                "training-only productive-prefix replay in separate P3b/P3c archives",
            ],
            "reset": [
                "restart mastery level for room-focused curriculum",
                "live episodes; replay prefixes begin fresh PPO suffixes",
            ],
        }
    else:
        audit = copy.deepcopy(state["transfer"])
        audit["resumed_from_checkpoint"] = str(parent)
        audit["resume_kind"] = resume_kind
    audit["resume_rng_checkpoint"] = str(parent)
    audit["training"] = "not started by this preparation check"
    state["skill_controller"] = controller
    state["transfer"] = audit
    return model, state, audit


def check_p3_resume(path, config):
    """No-learning checkpoint check with one real P3b inference/environment step."""
    from ather_exploration.training.checkpoints import restore_rng
    from ather_exploration.training.skill_environments import SkillTrainingEnv

    torch.set_num_threads(config.torch_threads)
    env = SkillTrainingEnv(config, worker=0)
    try:
        model, state, audit = prepare_p3_resume(path, config, env)
        env.set_controller(state["skill_controller"])
        env.restore(state["workers"][0])
        env.set_restart_level(0)
        if config.device.startswith("cuda"):
            restore_rng(audit["resume_rng_checkpoint"])
        observation, _ = env.reset()
        action, _ = model.predict(observation, deterministic=True)
        _, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated or not np.isfinite(reward):
            raise ValueError("P3 resume sample transition failed")
        return {
            "status": "valid",
            "learning_executed": False,
            "phase": env.phase,
            "source_task": env.task,
            "sample_reward": float(reward),
            "sample_reward_components": info["skill"]["reward_components"],
            "transfer": audit,
            "scope": "checkpoint/schema/optimizer validation plus one P3b transition; no optimizer update",
        }
    finally:
        env.close()
