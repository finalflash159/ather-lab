"""Checkpoint evaluation uses held-apart bank records and fresh policy state per episode."""

from pathlib import Path

from ather_exploration.environment.env import make_env
from ather_exploration.evaluation.metrics import aggregate_episodes
from ather_exploration.evaluation.runner import evaluate_episode
from ather_exploration.seeds import derive_seed
from ather_exploration.training.checkpoints import load_agent
from ather_exploration.training.runner import append_jsonl
from ather_exploration.worlds.generation import load_generated
from ather_exploration.worlds.scenarios import implementation_id, read_record, write_record
from ather_exploration.worlds.suites import _record_path


def evaluate_checkpoint(
    checkpoint, bank, output, *, split="validation_quick", action_seed=0, deterministic=True
):
    if split not in ("validation_quick", "validation_selection", "heldout_id"):
        raise ValueError("Use a held-apart evaluation split")
    bank, root = Path(bank), Path(output)
    manifest = read_record(bank / "manifest.json")
    if manifest["state"] != "READY":
        raise ValueError("Bank is not READY")
    worlds = [w for w in manifest["worlds"] if w["split"] == split]
    if len(worlds) != manifest["request"]["count_per_split"]:
        raise ValueError("Evaluation split incomplete")
    root.mkdir(parents=True, exist_ok=False)
    write_record(root / "status.json", {"state": "RUNNING"})
    results = []
    try:
        agent = None
        for i, world in enumerate(worlds):
            record = load_generated(_record_path(bank, world["starts"][0]))
            env = make_env(generated=record)
            try:
                if agent is None:
                    agent = load_agent(checkpoint, env.observation_space)

                result, steps = evaluate_episode(
                    env,
                    ModeAdapter(agent, deterministic),
                    episode_id=f"{split}-{i}",
                    group=record.config.preset,
                    action_seed=derive_seed(action_seed, "policy-eval", i),
                    metadata={
                        "checkpoint": agent.metadata["checkpoint"],
                        "deterministic": deterministic,
                    },
                )
                results.append(result)
                append_jsonl(root / "episodes.jsonl", result)
                for row in steps:
                    append_jsonl(root / "steps.jsonl", row)
                write_record(
                    root / "replays" / f"{i}.json",
                    {
                        "schema": "g4-replay-v1",
                        "source_revision": implementation_id(),
                        "record": record.payload(),
                        "steps": steps,
                        "result": result,
                    },
                )
            finally:
                env.close()
        summary = aggregate_episodes(results)
        write_record(root / "summary.json", summary)
        write_record(
            root / "status.json",
            {"state": "READY" if all(r["status"] == "completed" for r in results) else "FAILED"},
            replace=True,
        )
        return summary
    except BaseException as error:
        write_record(root / "status.json", {"state": "FAILED", "error": str(error)}, replace=True)
        raise


class ModeAdapter:
    def __init__(self, agent, deterministic):
        self.agent, self.deterministic = agent, deterministic
        self.name = agent.name

    def act(self, observation, state, *, action_rng, **kwargs):
        return self.agent.act(
            observation, state, deterministic=self.deterministic, action_rng=action_rng
        )
