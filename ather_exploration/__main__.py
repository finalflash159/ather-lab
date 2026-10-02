"""Inspect contracts, execute fixtures, and build G2 validated scenarios."""

import argparse
import json
from pathlib import Path

from ather_exploration import __version__
from ather_exploration.config import PRESET_NAMES, config_hash, load_config, load_preset
from ather_exploration.fixtures import fixture_catalog
from ather_exploration.schema import LOCAL_CHANNELS, MEMORY_CHANNELS, STATE_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Ather Exploration — G0/G1/G2/G3")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    skills_parser = commands.add_parser(
        "build-skills", help="Validate and record skill seed pools; no training"
    )
    skills_parser.add_argument("--config", type=Path, required=True)
    skills_parser.add_argument("--output", type=Path, required=True)
    config_parser = commands.add_parser("config", help="Validate and print resolved env config")
    source = config_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=PRESET_NAMES)
    source.add_argument("--file", type=Path)
    commands.add_parser("schema", help="Print the version-1 policy channel order")
    fixture_parser = commands.add_parser("fixtures", help="List hand-specified G1+ examples")
    fixture_parser.add_argument("--name", help="Show one fixture with expected outcomes")
    rollout_parser = commands.add_parser("rollout", help="Execute a G1 fixture and print a trace")
    rollout_parser.add_argument("--fixture", required=True)
    rollout_parser.add_argument(
        "--actions", nargs="+", help="NORTH SOUTH EAST WEST WAIT or IDs0..4"
    )
    rollout_parser.add_argument("--seed", type=int, default=0)
    setup_parser = commands.add_parser(
        "setup-check", help="Run vanilla dependency integration smoke"
    )
    setup_parser.add_argument("--output", type=Path)
    generate_parser = commands.add_parser(
        "generate", help="Generate and validate a main-task dungeon"
    )
    generate_parser.add_argument("--preset", choices=PRESET_NAMES, default="small")
    generate_parser.add_argument("--seed", type=int, required=True)
    generate_parser.add_argument(
        "--stratum", choices=("room_quiet", "room_threat", "corridor_quiet", "corridor_threat")
    )
    generate_parser.add_argument("--output", type=Path)
    generate_parser.add_argument("--cache", type=Path)
    validate_parser = commands.add_parser(
        "validate", help="Replay stored witness and independently search again"
    )
    validate_parser.add_argument("--scenario", type=Path, required=True)
    validate_parser.add_argument("--max-expansions", type=int, default=4000000)
    suite_parser = commands.add_parser(
        "build-suite", help="Build a small development bank, four disjoint splits"
    )
    suite_parser.add_argument("--preset", choices=PRESET_NAMES, default="small")
    suite_parser.add_argument("--root-seed", type=int, required=True)
    suite_parser.add_argument("--count", type=int, default=2, help="Worlds per split")
    suite_parser.add_argument("--train-starts", type=int, default=1)
    suite_parser.add_argument("--attempts-per-world", type=int, default=32)
    suite_parser.add_argument("--output", type=Path, required=True)
    eval_parser = commands.add_parser(
        "evaluate", help="Run fair development baselines and raw metrics"
    )
    eval_parser.add_argument(
        "--agents", nargs="+", choices=("random", "frontier"), default=["random", "frontier"]
    )
    eval_parser.add_argument("--presets", nargs="+", choices=PRESET_NAMES, default=["small"])
    eval_parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    eval_parser.add_argument("--action-repeats", type=int, default=3)
    eval_parser.add_argument("--root-seed", type=int, default=0)
    eval_parser.add_argument("--output", type=Path)
    eval_parser.add_argument("--bank", type=Path)
    eval_parser.add_argument(
        "--split",
        choices=("train", "validation_quick", "validation_selection", "heldout_id"),
        default="validation_quick",
    )
    ui_parser = commands.add_parser(
        "ui", help="Open autoplay agent viewer; configuration through CLI"
    )
    ui_parser.add_argument("--preset", choices=PRESET_NAMES, default="small")
    ui_parser.add_argument("--seed", type=int, default=42)
    ui_parser.add_argument(
        "--agent",
        choices=("random", "frontier", "ppo", "recurrent", "replay", "checkpoint"),
        default="checkpoint",
    )
    ui_parser.add_argument("--fixture", help="Use a named diagnostic fixture instead of generation")
    ui_parser.add_argument("--config", type=Path, help="Validated env config; overrides preset")
    ui_parser.add_argument("--action-seed", type=int, default=0)
    ui_parser.add_argument("--fps", type=int, default=8, help="Viewer steps per second (1..60)")
    checkpoint_choice = ui_parser.add_mutually_exclusive_group()
    checkpoint_choice.add_argument("--checkpoint", type=Path, help="One READY checkpoint")
    checkpoint_choice.add_argument(
        "--checkpoint-dir", type=Path, help="Run or checkpoints directory; enables dropdown"
    )
    ui_parser.add_argument("--stochastic", action="store_true")
    ui_parser.add_argument("--replay", type=Path)
    train_parser = commands.add_parser("train", help="Explicitly start learning (never implicit)")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--continue-curriculum", action="store_true")
    train_parser.add_argument("--transfer-p1-to-p2", action="store_true")
    preflight = commands.add_parser(
        "train-check", help="Validate training config/banks; NO learning"
    )
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--resume", type=Path)
    preflight.add_argument("--transfer-p1-to-p2", action="store_true")
    replay_parser = commands.add_parser("replay", help="Verify recorded actions without a policy")
    replay_parser.add_argument("--file", type=Path, required=True)
    policy_eval = commands.add_parser(
        "evaluate-policy", help="Evaluate a READY learned checkpoint; no training"
    )
    policy_eval.add_argument("--checkpoint", type=Path, required=True)
    policy_eval.add_argument("--bank", type=Path, required=True)
    policy_eval.add_argument(
        "--split",
        choices=("validation_quick", "validation_selection", "heldout_id"),
        default="validation_quick",
    )
    policy_eval.add_argument("--output", type=Path, required=True)
    policy_eval.add_argument("--action-seed", type=int, default=0)
    policy_eval.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "build-skills":
            from ather_exploration.training.config import read_training_config
            from ather_exploration.worlds.skill_tasks import build_skill_suite

            manifest = build_skill_suite(read_training_config(args.config), args.output)
            result = {
                "state": manifest["state"],
                "output": str(args.output),
                "tasks": list(manifest["tasks"]),
            }
        elif args.command in ("train", "train-check"):
            from ather_exploration.training.config import read_training_config

            config = read_training_config(args.config)
            if args.command == "train-check":
                from ather_exploration.training.curriculum import WorldBank
                from ather_exploration.training.skill_runner import preflight

                identities = (
                    preflight(config)
                    if config.skills.enabled
                    else WorldBank(config.banks).identities
                )
                if args.resume and not args.transfer_p1_to_p2:
                    raise ValueError("train-check --resume requires --transfer-p1-to-p2")
                if args.transfer_p1_to_p2 and not args.resume:
                    raise ValueError("P1 transfer check requires --resume")
                transfer_check = {}
                if args.transfer_p1_to_p2:
                    from ather_exploration.training.skill_transfer import check_p1_transfer

                    transfer_check = check_p1_transfer(args.resume, config)
                result = {
                    **transfer_check,
                    "status": "valid",
                    "learning_executed": False,
                    "config": config.model_dump(mode="json"),
                    "bank_ids": identities,
                }
            else:
                from ather_exploration.training.runner import run_training

                result = run_training(
                    config,
                    args.output,
                    resume=args.resume,
                    continue_curriculum=args.continue_curriculum,
                    transfer_p1_to_p2=args.transfer_p1_to_p2,
                )
        elif args.command == "evaluate-policy":
            from ather_exploration.evaluation.learned import evaluate_checkpoint

            result = evaluate_checkpoint(
                args.checkpoint,
                args.bank,
                args.output,
                split=args.split,
                action_seed=args.action_seed,
                deterministic=not args.stochastic,
            )
        elif args.command == "replay":
            from ather_exploration.evaluation.replay import verify_replay

            result = verify_replay(args.file)
        elif args.command == "ui":
            from ather_exploration.ui.app import run_ui
            from ather_exploration.ui.session import SessionSpec

            run_ui(
                SessionSpec(
                    args.preset,
                    args.seed,
                    args.agent,
                    args.fixture,
                    str(args.config) if args.config else None,
                    args.action_seed,
                    str(args.checkpoint) if args.checkpoint else None,
                    not args.stochastic,
                    str(args.replay) if args.replay else None,
                ),
                fps=args.fps,
                checkpoint_dir=args.checkpoint_dir,
            )
            return
        elif args.command == "config":
            config = load_preset(args.preset) if args.preset else load_config(args.file)
            result = {
                "resolved": config.model_dump(mode="json"),
                "config_hash": config_hash(config),
            }
        elif args.command == "schema":
            result = {
                "version": "1",
                "local": LOCAL_CHANNELS,
                "memory": MEMORY_CHANNELS,
                "state": STATE_FIELDS,
            }
        elif args.command == "fixtures":
            cases = {case["name"]: case for case in fixture_catalog()["cases"]}
            if args.name and args.name not in cases:
                raise ValueError(f"Unknown fixture: {args.name}")
            result = (
                cases[args.name] if args.name else {name: c["purpose"] for name, c in cases.items()}
            )
        elif args.command == "rollout":
            result = fixture_rollout(args.fixture, args.actions, args.seed)
        elif args.command == "generate":
            from ather_exploration.environment.env import make_env
            from ather_exploration.worlds.generation import generate_scenario
            from ather_exploration.worlds.scenarios import scenario_hash, write_record

            record = generate_scenario(
                load_preset(args.preset), args.seed, stratum=args.stratum, cache=args.cache
            )
            if args.output:
                write_record(args.output, record.payload())
            env = make_env(generated=record, render_mode="ansi")
            try:
                env.reset()
                result = {
                    "status": record.validation.status,
                    "scenario_hash": scenario_hash(record.scenario),
                    "stratum": record.stratum,
                    "witness_steps": len(record.validation.actions),
                    "search_expansions": record.validation.expansions,
                    "diagnostics": record.diagnostics,
                    "world_debug": env.render(),
                    "output": str(args.output) if args.output else None,
                }
            finally:
                env.close()
        elif args.command == "validate":
            from dataclasses import asdict

            from ather_exploration.worlds.generation import load_generated
            from ather_exploration.worlds.validation import validate_scenario

            record = load_generated(args.scenario)
            result = asdict(validate_scenario(record.scenario, max_expansions=args.max_expansions))
        elif args.command == "build-suite":
            from ather_exploration.worlds.suites import build_development_suite

            result = build_development_suite(
                load_preset(args.preset),
                args.root_seed,
                args.output,
                count=args.count,
                train_starts=args.train_starts,
                attempts_per_world=args.attempts_per_world,
            )
        elif args.command == "evaluate":
            from ather_exploration.evaluation.runner import evaluate_baselines

            result = evaluate_baselines(
                agents=tuple(args.agents),
                presets=tuple(args.presets),
                seeds=tuple(args.seeds),
                action_repeats=args.action_repeats,
                root_seed=args.root_seed,
                output=args.output,
                bank=args.bank,
                split=args.split,
            )
        else:
            from ather_exploration.setup_check import run_setup_check

            result = run_setup_check()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False))


def fixture_rollout(name, actions, seed):
    from ather_exploration.environment.env import make_fixture_env
    from ather_exploration.types import Action

    cases = {case["name"]: case for case in fixture_catalog()["cases"]}
    if name not in cases:
        raise ValueError(f"Unknown fixture: {name}")
    if actions is None:
        chosen = [Action(a) for a in cases[name]["actions"]]
    else:
        chosen = []
        for value in actions:
            try:
                chosen.append(
                    Action[value.upper()] if not value.lstrip("-").isdigit() else Action(int(value))
                )
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"Invalid action {value!r}; use NORTH/SOUTH/EAST/WEST/WAIT or0..4"
                ) from error
    env = make_fixture_env(name, render_mode="ansi")
    try:
        obs, _ = env.reset(seed=seed)
        trace = [
            {
                "tick": 0,
                "world_debug": env.render(),
                "known_floor": int(obs["memory"][2].sum()),
                "reward": 0.0,
            }
        ]
        total = 0.0
        for action in chosen:
            obs, reward, terminated, truncated, info = env.step(action)
            snapshot = env.unwrapped.evaluator_snapshot()
            total += reward
            trace.append(
                {
                    "tick": snapshot.step_count,
                    "action": action.name,
                    "event": info["transition"],
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "collision_stage_debug": env.unwrapped.collision_stage,
                    "known_floor": int(obs["memory"][2].sum()),
                    "visited_cells": int(obs["memory"][5].sum()),
                    "world_debug": env.render(),
                }
            )
            if terminated or truncated:
                break
        return {
            "fixture": name,
            "scope": "G1 fixture execution; world_debug is privileged display",
            "legend": "# wall, . floor, A agent, M monster, p pending POI, P active POI, X death",
            "return": total,
            "trace": trace,
        }
    finally:
        env.close()


if __name__ == "__main__":
    main()
