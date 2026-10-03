"""One W&B run per training attempt; local records remain recovery evidence."""

import os

from ather_exploration.worlds.scenarios import implementation_id, write_record


def start_tracking(config, root, bank_ids, parent_checkpoint=None):
    if config.tracking.mode == "disabled":
        return None
    if config.tracking.mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise ValueError("W&B online requires WANDB_API_KEY (Modal Secret wandb)")
    import wandb

    run = wandb.init(
        project=config.tracking.project,
        entity=config.tracking.entity,
        name=root.name,
        mode=config.tracking.mode,
        dir=str(root),
        config={
            "training": config.model_dump(mode="json"),
            "learning_objective": "ppo_public_route_aux" if config.recovery else config.method,
            "bank_ids": bank_ids,
            "source_revision": implementation_id(),
            "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
        },
        tags=[config.method, "development"] + (["public-route-aux"] if config.recovery else []),
    )
    try:
        run.define_metric("training/env_steps")
        run.define_metric("*", step_metric="training/env_steps")
        write_record(
            root / "tracking.json",
            {
                "provider": "wandb",
                "mode": config.tracking.mode,
                "id": run.id,
                "url": run.url,
                "project": config.tracking.project,
            },
        )
        print(f"W&B: {run.url or 'offline recording'}", flush=True)
        return run
    except Exception:
        run.finish(exit_code=1)
        raise
