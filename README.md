# Ather Exploration

Partially observed exploration with persistent POIs and patrolling threats, built on MiniGrid 3.1.0. The project includes procedural maps, public spatial memory, random/frontier baselines, PPO/RecurrentPPO, checkpoint evaluation, a local inference viewer, and headless Modal training with W&B metrics.

The skill curriculum adds PPO tasks P1–P5: approach visible POIs, navigate obstacles, explore hidden rooms, manage threats, then train on the target map distribution. Promotion requires validation gates for the active phase only; easier maps inherit that phase’s reward, horizon and termination rules. P3 continues after POI activation until its horizon. No task-ID input or old-task retention exams are used; hitting a skill budget without passing stops the run. A single-seed P1 pilot passed both validation gates at 98,304 environment steps. P2 learning quality remains to be evaluated.

## Setup and checks

Use Python 3.11 and the locked CPU dependencies locally:

```bash
uv sync --locked --extra cpu --extra modal
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ather_exploration train-check --config ather_exploration/resources/training/skills_p1.yaml
```

These checks do not train. Always use `.venv/bin/python` (including `-m modal`) to avoid a globally installed CLI using another Python version. CPU and CUDA extras are mutually exclusive.

## First skill run

```bash
.venv/bin/python -m ather_exploration build-skills --config ather_exploration/resources/training/skills_p1.yaml --output artifacts/skills-p1
.venv/bin/python -m ather_exploration train --config ather_exploration/resources/training/skills_p1.yaml --output artifacts/runs/skills-p1-01
```

The second command **trains**. Configure W&B credentials or set tracking mode to offline before running. The P1 config stops after both P1 gates pass; the large total budget preserves the learning-rate schedule for later continuation. It does not train for the entire budget in P1. No target world bank is required for P1–P4.

`skills.yaml` enables the full sequence and requires target banks at its configured paths. Use `build-suite --help` to create them. Continuing from a completed phase requires a new output directory, `--resume PATH_TO_STEP`, `--continue-curriculum`, and a config with a later `skills.stop_after`. Architecture and other training settings must stay compatible.

## P2 continuation

`skills_p2.yaml` stops after P2 and adds a 0.005 cost to every action while retaining first-observation coverage and POI rewards. It preserves PPO settings and the full learning-rate schedule. Diagnostics separate searching for a POI from reaching it after discovery; missing events are null with explicit sample counts.

For the audited legacy P1 source, use `--transfer-p1-to-p2` together with `--resume` and `--continue-curriculum` on training commands. A no-learning local check is:

```bash
.venv/bin/python -m ather_exploration train-check --config ather_exploration/resources/training/skills_p2.yaml --resume artifacts/modal/skills-p1-01/checkpoints/step_98304 --transfer-p1-to-p2
```

Modal supports the same transfer flag for `--command check` and `--command train`; remote paths use `RUN_ID/checkpoints/step_N`. Check mode also requires `--continue-curriculum`. Transfer validates the parent integrity, source allowlist, completed phase, observation/action schema, network architecture, optimizer and counters. It preserves weights/optimizer/schedule and resets episodes. `transfer.json` and checkpoint metadata record provenance. Ordinary resume/inference retain strict source checks. Upload a fresh dataset after Python source changes. Tests simulate training boundaries without optimizer updates.

## Modal and inference

Use `python -m ather_exploration.training.modal_io upload --help` to upload datasets and `python -m modal run -m ather_exploration.training.modal_app --help` for remote check/train. Substitute `.venv/bin/python` for `python` in your shell. Upload returns the dataset ID to use; do not reuse an unrelated example ID. Training is headless. Download a finished run with:

```bash
.venv/bin/python -m ather_exploration.training.modal_io download --run-id YOUR_RUN_ID --output artifacts/modal/YOUR_RUN_ID
.venv/bin/python -m ather_exploration ui --checkpoint-dir artifacts/modal/YOUR_RUN_ID
```

The viewer offers checkpoint selection, new map and reset. Skill checkpoints select their task automatically. Config and experiment setup stay in CLI/YAML; metrics are logged to W&B and local files.

## Code map

- `environment/`: Gymnasium environment, collisions, visibility, memory and rewards.
- `worlds/`: target map generation/validation and skill scenarios.
- `agents/`: baseline policies and learned models.
- `training/`: config, runner, curriculum, checkpoints, tracking and Modal integration.
- `evaluation/`: task metrics and frozen-policy validation.
- `ui/`: offline inference viewer.

These directories live under `ather_exploration/`. `resources/training/` contains executable example configurations. `tests/` tests project behavior. `notes/` is intentionally local and ignored, as are datasets, checkpoints, credentials and virtual environments.

Source fingerprints guard dataset/checkpoint compatibility. Changed environment code may require new banks; do not edit fingerprints to force compatibility. Baseline revision `fff13b4` preserves the implementation used before the skill curriculum.

## Attribution

Based on [Farama Foundation MiniGrid v3.1.0](https://github.com/Farama-Foundation/Minigrid/tree/v3.1.0), commit `90928729376741a41222a257911343b97103b548`. The upstream source remains in `minigrid/`; see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Other dependencies are installed into the virtual environment, not copied into the project.
