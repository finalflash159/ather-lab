"""Development baseline evaluation with explicit failures and opt-in raw artifacts."""

import json
import os
from pathlib import Path
from time import perf_counter

from ather_exploration.agents.baselines import BaselineConfig, make_baseline
from ather_exploration.config import load_preset
from ather_exploration.environment.env import make_env
from ather_exploration.evaluation.metrics import EpisodeMetrics, aggregate_episodes
from ather_exploration.seeds import derive_seed, stage_rng
from ather_exploration.types import AgentState
from ather_exploration.worlds.generation import generate_scenario, load_generated
from ather_exploration.worlds.scenarios import digest, read_record, write_record
from ather_exploration.worlds.suites import SPLITS, _record_path


def evaluate_episode(env, agent, *, episode_id, group, seed=0, action_seed=0, metadata=None):
    """Caller owns env. Agent receives detached public tensors and its own state."""
    obs, _ = env.reset(seed=seed)
    metadata = {**(metadata or {}), "agent": agent.name, "action_seed": action_seed}
    metrics = EpisodeMetrics(
        env.unwrapped.evaluator_snapshot(),
        obs,
        env.unwrapped.reward_config,
        episode_id=episode_id,
        group=group,
        metadata=metadata,
    )
    state = AgentState()
    rng = stage_rng(action_seed, "baseline-action")
    decision_seconds = 0.0
    step_seconds = 0.0
    try:
        for _ in range(metrics.H):
            started = perf_counter()
            action, state = agent.act(
                {k: v.copy() for k, v in obs.items()}, state, deterministic=True, action_rng=rng
            )
            decision_seconds += perf_counter() - started
            started = perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            step_seconds += perf_counter() - started
            row = metrics.update(
                env.unwrapped.evaluator_snapshot(),
                obs,
                reward,
                info["transition"],
                env.unwrapped.collision_stage,
            )
            row["terminated"] = bool(terminated)
            row["truncated"] = bool(truncated)
            row["agent_diagnostics"] = (
                dict(state.planner.diagnostics) if state.planner is not None else {}
            )
            if terminated or truncated:
                result = metrics.finish(cancelled=truncated)
                if truncated:
                    result["cancellation_reason"] = "external_truncation"
                break
        else:
            result = metrics.finish()
    except Exception as error:  # noqa: BLE001 -- record failed trial explicitly, never rank it
        # A failed evaluation must remain in the trial, never silently disappear.
        result = metrics.finish(failure=f"{type(error).__name__}: {error}")
    result["timing"] = {
        "decision_seconds": decision_seconds,
        "env_step_seconds": step_seconds,
        "decisions": metrics.steps[-1]["t"],
    }
    return result, metrics.steps


def _jsonl(path, rows, *, append=False):
    # Writer operates only within the newly-created, single-writer trial folder.
    with path.open("a" if append else "x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def evaluate_baselines(
    *,
    agents=("random", "frontier"),
    presets=("small",),
    seeds=(0,),
    action_repeats=3,
    root_seed=0,
    output=None,
    bank=None,
    split="validation_quick",
    baseline_config=None,
):
    """Same fixed scenarios for B0/B1. No held-out performance filtering or tuning.

    Bank mode visits each requested split world once (its recorded primary start),
    rather than sampling with replacement. Defaults are development-sized only.
    """
    if (
        not agents
        or len(set(agents)) != len(agents)
        or any(a not in ("random", "frontier") for a in agents)
    ):
        raise ValueError("Choose unique baseline names random/frontier")
    if type(action_repeats) is not int or action_repeats < 1:
        raise ValueError("action_repeats must be positive")
    if (
        not presets
        or len(set(presets)) != len(presets)
        or not seeds
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Presets and seeds must be nonempty and unique")
    derive_seed(root_seed, "evaluation")
    config = baseline_config or BaselineConfig()
    for seed in seeds:
        derive_seed(seed, "scenario")
    for name in presets:
        load_preset(name)
    if split not in SPLITS:
        raise ValueError("Unknown bank split")
    destination = Path(output) if output is not None else None
    request = {
        "namespace": "development",
        "agents": list(agents),
        "presets": list(presets),
        "seeds": list(seeds),
        "action_repeats": action_repeats,
        "root_seed": root_seed,
        "baseline_config": config.model_dump(mode="json"),
        "bank": str(bank) if bank is not None else None,
        "split": split,
    }
    manifest = {"state": "BUILDING", "request": request, "scenarios": []}
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
        write_record(destination / "manifest.json", manifest)
        _jsonl(destination / "episodes.jsonl", [])
        _jsonl(destination / "steps.jsonl", [])
    records = []
    try:
        scenarios = []
        if bank is not None:
            source = Path(bank)
            data = read_record(source / "manifest.json")
            if data["state"] != "READY" or data["request"]["namespace"] != "development":
                raise ValueError("Need a READY development bank")
            worlds = [w for w in data["worlds"] if w["split"] == split]
            if len(worlds) != data["request"]["count_per_split"]:
                raise ValueError("Incomplete bank split")
            for world in worlds:
                scenarios.append(load_generated(_record_path(source, world["starts"][0])))
        else:
            for name in presets:
                for seed in seeds:
                    scenarios.append(generate_scenario(load_preset(name), seed))
        for index, record in enumerate(scenarios):
            scenario_id = record.validation.scenario_hash
            manifest["scenarios"].append(
                {
                    "index": index,
                    "scenario_hash": scenario_id,
                    "group": record.config.preset,
                    "seed": record.scenario.seed,
                }
            )
            if destination is not None:
                write_record(destination / "scenarios" / f"{index}.json", record.payload())
            for name in agents:
                for repeat in range(action_repeats if name == "random" else 1):
                    action_seed = derive_seed(
                        root_seed, "evaluation-action", scenario_id, name, repeat
                    )
                    episode_id = digest(
                        [scenario_id, name, repeat, action_seed, config.model_dump(mode="json")]
                    )
                    env = make_env(generated=record)
                    try:
                        result, steps = evaluate_episode(
                            env,
                            make_baseline(name, config),
                            episode_id=episode_id,
                            group=record.config.preset,
                            action_seed=action_seed,
                            metadata={
                                "mode": "stochastic" if name == "random" else "deterministic",
                                "repeat": repeat,
                                "baseline_config": config.model_dump(mode="json"),
                            },
                        )
                    finally:
                        env.close()
                    records.append(result)
                    if destination is not None:
                        _jsonl(destination / "episodes.jsonl", [result], append=True)
                        _jsonl(destination / "steps.jsonl", steps, append=True)
        if len({r["episode_id"] for r in records}) != len(records):
            raise ValueError("Duplicate episode ID in trial")
        summaries = {
            name: aggregate_episodes([r for r in records if r["metadata"]["agent"] == name])
            for name in agents
        }
        summary = {
            "status": "failed" if any(r["status"] != "completed" for r in records) else "pass",
            "scope": "development baseline check, not formal held-out evidence",
            "request": request,
            "scenarios": len(scenarios),
            "episodes": len(records),
            "agents": summaries,
        }
        if destination is not None:
            write_record(destination / "summary.json", summary)
            manifest["state"] = (
                "FAILED" if any(r["status"] != "completed" for r in records) else "READY"
            )
            manifest["files"] = {
                name: digest((destination / name).read_text())
                for name in ("episodes.jsonl", "steps.jsonl", "summary.json")
            }
            write_record(destination / "manifest.json", manifest, replace=True)
        return summary
    except BaseException as error:
        if destination is not None:
            manifest["state"] = "FAILED"
            manifest["error"] = f"{type(error).__name__}: {error}"
            write_record(destination / "manifest.json", manifest, replace=True)
        raise


def read_evaluation(output):
    """Read complete raw evidence; reject corrupt/incomplete output and duplicate IDs."""
    root = Path(output)
    manifest = read_record(root / "manifest.json")
    if manifest["state"] != "READY":
        raise ValueError("Evaluation is not READY; inspect recorded failure")
    for name in ("episodes.jsonl", "steps.jsonl", "summary.json"):
        if digest((root / name).read_text()) != manifest["files"][name]:
            raise ValueError("Evaluation checksum mismatch")
    episodes = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]
    steps = [json.loads(line) for line in (root / "steps.jsonl").read_text().splitlines()]
    aggregate_episodes(episodes)  # also rejects duplicate IDs
    return episodes, steps, read_record(root / "summary.json")
