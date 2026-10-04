# Ather Exploration

Partially observed exploration with persistent POIs and patrolling threats, built on MiniGrid 3.1.0. The project includes procedural maps, public spatial memory, random/frontier baselines, PPO/RecurrentPPO, checkpoint evaluation, a local inference viewer, and headless Modal training with W&B metrics.

The skill curriculum adds PPO tasks P1–P5: approach visible POIs, navigate obstacles, explore hidden rooms, manage threats, then train on the target map distribution. Promotion requires validation gates for the active phase only; easier maps inherit that phase’s reward, horizon and termination rules. P3 continues after POI activation until its horizon. No task-ID input or old-task retention exams are used; hitting a skill budget without passing stops the run. A single-seed P1 pilot passed both validation gates at 98,304 environment steps. The revised exploration curriculum uses provisional pilot gates; its learning quality remains to be evaluated.

## Setup and checks

Use Python 3.11 and the locked CPU dependencies locally. The main pinned runtime versions are Gymnasium 1.3.0, PyTorch 2.11.0, Stable-Baselines3 2.9.0 and sb3-contrib 2.9.0; `uv.lock` pins the full environment. MiniGrid 3.1.0's source is included under `minigrid/` and is installed by the project environment; no second MiniGrid checkout is needed.

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

## Multi-room continuation

`skills_p3.yaml` transfers a completed exploration checkpoint into three lessons: two connected rooms, three rooms with two POIs, then four rooms with chain/cycle layouts. Terrain splits are invariant to rotation/reflection and independent of spawn/POI placement. The active reward keeps area/discovery/activation bonuses and a blocked-move penalty, with no per-step cost. Episodes continue to their horizon. Joint POI-and-coverage gates and unfinished-task diagnostics distinguish exploration failures from time spent after completion.

```bash
.venv/bin/python -m ather_exploration train-check --config ather_exploration/resources/training/skills_p3.yaml --resume artifacts/modal/skills-p2-02-resume-01/checkpoints/step_376832 --transfer-p2-to-p3
```

For training, supply `--transfer-p2-to-p3`, `--continue-curriculum` and the same parent checkpoint with a new output/run ID. Modal accepts the same flags. Upload a fresh dataset for the changed source. `evaluate-skills` supports P1–P4 validation and P3 OOD evaluation for a checkpoint or a baseline; it never trains. These are pilot gates, not demonstrated learning results.

## P2 continuation

`skills_p2.yaml` runs visible-goal navigation, hidden-goal search, then full-horizon exploration. Every action costs 0.005; blocked moves cost an additional 0.02. Coverage rewards stop after discovery in the search task, remain active in full-horizon exploration, and are disabled for visible-goal navigation. It preserves PPO settings and the full learning-rate schedule. Diagnostics separate searching for a POI from reaching it after discovery; missing events are null with explicit sample counts.

For the audited legacy P1 source, use `--transfer-p1-to-p2` together with `--resume` and `--continue-curriculum` on training commands. A no-learning local check is:

```bash
.venv/bin/python -m ather_exploration train-check --config ather_exploration/resources/training/skills_p2.yaml --resume artifacts/modal/skills-p1-01/checkpoints/step_98304 --transfer-p1-to-p2
```

Modal supports the same transfer flag for `--command check` and `--command train`; remote paths use `RUN_ID/checkpoints/step_N`. Check mode also requires `--continue-curriculum`. Transfer validates the parent integrity, source allowlist, completed phase, observation/action schema, network architecture, optimizer and counters. It preserves weights/optimizer/schedule and resets episodes. `transfer.json` and checkpoint metadata record provenance. Ordinary resume/inference retain strict source checks. Upload a fresh dataset after Python source changes. Tests simulate training boundaries without optimizer updates.

## Evaluation

Build a development bank with distinct train, validation-selection and held-out map splits. `count` is the number of maps **per split**. Choose a fresh output path for each immutable bank:

```bash
.venv/bin/python -m ather_exploration build-suite \
  --preset large --root-seed 4401 --count 16 --train-starts 1 \
  --output artifacts/banks/eval-large-4401
```

Compare the random and deterministic frontier baselines on the same held-out maps:

```bash
.venv/bin/python -m ather_exploration evaluate \
  --agents random frontier --bank artifacts/banks/eval-large-4401 \
  --split heldout_id --action-repeats 3 --root-seed 0 \
  --output artifacts/evaluations/baselines-large-4401
```

The random baseline samples actions without wall masking. The frontier baseline uses public memory, routes over known floor toward a pending POI or frontier, and uses deterministic tie-breaking. Neither receives hidden map cells, hidden POIs, or hidden monster routes. In bank mode, each requested agent uses the same recorded scenario/start; random gets the requested action repeats. These commands are development evaluation, not by themselves a multi-training-seed final result.

Evaluate a checkpoint trained for the **same main environment observation/action schema** on that same held-out bank:

```bash
.venv/bin/python -m ather_exploration evaluate-policy \
  --checkpoint artifacts/runs/RUN_ID/checkpoints/step_123456 \
  --bank artifacts/banks/eval-large-4401 --split heldout_id \
  --output artifacts/evaluations/policy-large-4401
```

`evaluate-policy` writes `status.json`, `summary.json`, per-episode and per-step JSONL, and replay files. `evaluate` writes a manifest, scenario records, episode/step JSONL and summary. Use `validation_selection` while making choices and reserve `heldout_id` for the final report. Do not compare on different banks or split records. A checkpoint with a skill-specific observation schema (for example P4's 16-channel policy) is not a main-environment checkpoint; use the skill evaluator below.

Evaluate skill checkpoints or public-information baselines on their fixed validation pools:

```bash
CONFIG=ather_exploration/resources/training/skills_p4_balanced.yaml
CHECKPOINT=artifacts/modal/RUN_ID/checkpoints/step_2000000
.venv/bin/python -m ather_exploration evaluate-skills \
  --config "$CONFIG" --tasks P4a P4b P4c \
  --checkpoint "$CHECKPOINT" --split validation \
  --output artifacts/evaluations/p4-validation-RUN_ID.json
```

For a diagnostic subset use `--count 4`; omit it to evaluate the configured validation count (64 maps, one deterministic and three stochastic episodes per map/task). To run the same task suite with a baseline, replace `--checkpoint "$CHECKPOINT"` with `--agent frontier` or `--agent random`. This writes one JSON result with the fixed pool's metrics and gate results; it does not train or select a checkpoint. `evaluate-skills` offers P3 OOD maps only; P4 has train/validation pools but no separate held-out P4 pool yet. P4's training-time gates use validation maps and must not be presented as held-out final evidence. The P4 gate thresholds are exploratory pilot settings.

## Inference, replay and map review

Download a finished Modal run and open its checkpoint viewer:

```bash
.venv/bin/python -m ather_exploration.training.modal_io download \
  --run-id RUN_ID --output artifacts/modal/RUN_ID
.venv/bin/python -m ather_exploration ui \
  --checkpoint-dir artifacts/modal/RUN_ID --seed 42 --fps 8
```

The viewer has a checkpoint dropdown, auto-plays the selected policy, `New map` advances to the next seed, and `Reset same map` replays the current seed. Skill checkpoints select their task from checkpoint metadata. For a single checkpoint use `--checkpoint PATH`; for a no-training smoke test use `--agent frontier --seed 42`. `--stochastic` enables sampled actions. Close the window to stop the viewer; this never affects a finished training run.

To inspect the exact generated skill maps before or after training, open the separate dataset preview:

```bash
.venv/bin/python -m ather_exploration preview-maps \
  --config ather_exploration/resources/training/skills_p4_balanced.yaml \
  --task P4a --split validation --index 0
```

The preview UI lets you choose P1a–P4c, train or validation, and a zero-based map index; it shows the full map and hidden patrol routes, so this is a privileged generator/debug view, not agent input or inference. `Reset` sets patrol time to zero; `Play patrol` and `Tick +1` animate the route without simulating agent actions or collisions. This preview command does not show P5 target-bank maps.

`evaluate-policy` output contains exact scenarios and replay traces. Verify and render one recorded episode with:

```bash
.venv/bin/python -m ather_exploration replay \
  --file artifacts/evaluations/policy-large-4401/replays/0.json
.venv/bin/python -m ather_exploration ui --agent replay \
  --replay artifacts/evaluations/policy-large-4401/replays/0.json --fps 8
```

Local training writes to the directory passed to `train`; its checkpoints are under `OUTPUT/checkpoints/step_N`. After Modal download, use `artifacts/modal/RUN_ID/checkpoints/step_N`. A `READY` checkpoint is loadable without retraining, subject to the recorded source/schema compatibility checks.

## Modal training

Training on Modal is headless and uses one terminal. Build the skill bank first as described above, then upload it, check the remote configuration, and train using the returned dataset ID. This example runs P1; replace the config and run ID for another task:

```bash
set -euo pipefail
CONFIG=ather_exploration/resources/training/skills_p1.yaml
RUN_ID=skills-p1-$(date +%Y%m%d-%H%M%S)
UPLOAD_OUTPUT=$(.venv/bin/python -m ather_exploration.training.modal_io upload --config "$CONFIG")
export ATHER_DATASET=$(printf '%s\n' "$UPLOAD_OUTPUT" | .venv/bin/python -c 'import json,sys; print(json.loads(sys.stdin.read().splitlines()[-1])["dataset"])')

.venv/bin/python -m modal run -m ather_exploration.training.modal_app \
  --command check --config "$CONFIG" --dataset "$ATHER_DATASET"
.venv/bin/python -m modal run -m ather_exploration.training.modal_app \
  --command train --config "$CONFIG" --dataset "$ATHER_DATASET" --run-id "$RUN_ID"

.venv/bin/python -m ather_exploration.training.modal_io download \
  --run-id "$RUN_ID" --output "artifacts/modal/$RUN_ID"
.venv/bin/python -m ather_exploration ui \
  --checkpoint-dir "artifacts/modal/$RUN_ID"
```

If upload or remote check fails, stop there and fix the reported issue before training. Re-upload after source or config changes; use a new run ID for each run. Metrics are logged to W&B and local run files. P4's current config is `ather_exploration/resources/training/skills_p4_balanced.yaml`; its training-start checkpoint and thresholds are experiment-specific, so verify them against the artifacts and config before launching.


## Current limitations against the assignment

The remaining items reflect the project's current compute and development-time limits and are still required for full compliance.

- The Python viewer is autoplay-only: it does not currently accept a typed/random seed in the window, edit environment parameters, pause, or step the policy manually. Set the initial seed/FPS via CLI; `New map` increments the seed. This does **not** yet meet every UI control requested in assignment §8.
- P4 skill evaluation currently uses a fixed validation pool; it has no independent held-out P4 split. Multi-training-seed mean/variation and held-out final results are still needed before claiming final generalization.
- The map preview covers P1–P4 but not P5 target-bank maps.
- The multi-room `room_exploration` reward uses room-coverage information unavailable in the agent's observation. That is a privileged-reward issue against assignment §4 and must be resolved before describing those runs as fully compliant.
- P4 thresholds are exploratory, and previous P3/P4 pilots showed incomplete exploration and weak yield/threat behavior. Gate passage alone does not establish final task performance.
- Model checkpoints and experiment artifacts are not stored in Git. The full `artifacts/` directory will be uploaded separately to the [Drive artifacts folder](https://drive.google.com/drive/u/2/folders/1Cw4dGizwmsfqtLVQBDoo5njk7tA4sG08); include the exact run/config/checkpoint/evaluation paths in the final handoff. The demo still needs to be recorded, and the prebuilt no-retraining package is not yet documented as complete.
- Current README examples do not establish formal assignment completion: final held-out comparisons across multiple independent training seeds, mean/variation, final latency report, and a complete demo/package handoff still need to be produced and linked.

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
