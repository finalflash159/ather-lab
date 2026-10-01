"""Verified action replay, independent of model inference or training."""

import tempfile
from pathlib import Path

from ather_exploration.config import ObservationConfig, RewardConfig
from ather_exploration.environment.env import ExplorationEnv, make_env
from ather_exploration.environment.memory import PublicMemoryWrapper
from ather_exploration.evaluation.metrics import EpisodeMetrics, observation_hash
from ather_exploration.worlds.generation import load_generated
from ather_exploration.worlds.scenarios import (
    implementation_id,
    read_record,
    scenario_from_dict,
    write_record,
)


def replay_env(payload):
    if payload.get("schema") == "g4-replay-v1":
        if payload.get("source_revision") != implementation_id():
            raise ValueError("Replay source revision mismatch")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenario.json"
            write_record(path, payload["record"])
            return make_env(generated=load_generated(path))
    if payload.get("schema") == "interactive-trace-v1":
        if payload.get("implementation_id") != implementation_id():
            raise ValueError("Debug trace source revision mismatch")
        scenario = scenario_from_dict(payload["scenario"])
        # A debug export lacks the generated certificate. Only fixtures may use this route.
        if not scenario.fixture_only:
            raise ValueError(
                "Procedural debug trace lacks validation record; use G4 verified replay"
            )
        config = ObservationConfig.model_validate(payload["observation_config"])
        core = ExplorationEnv(
            scenario,
            observation_config=config,
            reward_config=RewardConfig.model_validate(payload["reward_config"]),
        )
        return PublicMemoryWrapper(core, config, scenario.horizon)
    raise ValueError("Unsupported replay schema")


def verify_replay(path):
    payload = read_record(path)
    steps = payload["steps"]
    if not steps or steps[0]["t"] != 0:
        raise ValueError("Replay needs reset record")
    env = replay_env(payload)
    try:
        obs, _ = env.reset()
        meta = payload.get("result") or {}
        metrics = EpisodeMetrics(
            env.unwrapped.evaluator_snapshot(),
            obs,
            env.unwrapped.reward_config,
            episode_id=steps[0]["episode_id"],
            group=meta.get("group", "replay"),
            metadata=meta.get("metadata"),
        )
        for index, expected in enumerate(steps):
            if index:
                obs, reward, terminated, truncated, info = env.step(expected["action"])
                actual = metrics.update(
                    env.unwrapped.evaluator_snapshot(),
                    obs,
                    reward,
                    info["transition"],
                    env.unwrapped.collision_stage,
                )
                actual["terminated"], actual["truncated"] = terminated, truncated
            else:
                actual = metrics.steps[0]
            if actual != {k: expected[k] for k in actual}:
                raise ValueError(f"Replay mismatch at tick {index}")
        if payload.get("result"):
            result = metrics.finish(cancelled=payload["result"]["status"] != "completed")
            for key in ("metrics", "reward_terms", "final_counts", "end_reason", "return"):
                if result[key] != payload["result"][key]:
                    raise ValueError(f"Replay summary mismatch: {key}")
        return {
            "status": "verified",
            "steps": len(steps) - 1,
            "completed": steps[-1]["terminated"],
            "final_observation_hash": observation_hash(obs),
        }
    finally:
        env.close()
