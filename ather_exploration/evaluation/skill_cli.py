"""Explicit no-learning evaluation; immutable raw episode evidence."""

import torch

from ather_exploration.agents.baselines import make_baseline
from ather_exploration.evaluation.skills import evaluate_skill
from ather_exploration.training.checkpoints import load_agent
from ather_exploration.training.config import read_training_config
from ather_exploration.worlds.scenarios import implementation_id, write_record
from ather_exploration.worlds.skill_tasks import configured_skill_env


def evaluate_cli(args):
    config = read_training_config(args.config)
    if args.count is not None:
        config = config.model_copy(
            update={"skills": config.skills.model_copy(update={"validation_count": args.count})}
        )
    torch.set_num_threads(config.torch_threads)
    if args.output.exists():
        raise FileExistsError("Use a new evaluation output path")
    env = configured_skill_env(args.tasks[0], 0, config.skills)
    try:
        agent = (
            load_agent(args.checkpoint, env.observation_space, device=config.device)
            if args.checkpoint
            else make_baseline(args.agent)
        )
    finally:
        env.close()
    results = []
    for task in args.tasks:
        print(
            f"[evaluate:{task}] source={args.checkpoint or args.agent} split={args.split}",
            flush=True,
        )
        result = evaluate_skill(None, task, config, agent=agent, split=args.split)
        results.append(result)
        print(
            f"[evaluate:{task}] passed={result['passed']} {result['summary']['deterministic']}",
            flush=True,
        )
    record = {
        "source_revision": implementation_id(),
        "source": str(args.checkpoint or args.agent),
        "config": config.model_dump(mode="json"),
        "split": args.split,
        "learning_executed": False,
        "results": results,
    }
    write_record(args.output, record)
    return {
        "output": str(args.output),
        "learning_executed": False,
        "gates": {r["task"]: r["passed"] for r in results},
    }
