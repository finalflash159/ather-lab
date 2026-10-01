"""Small development banks; explicit split isolation and resumable construction.

These are generation checks, not the formal experiment suite or training runner.
"""

from dataclasses import replace
from pathlib import Path
from time import perf_counter

from ather_exploration.config import config_hash
from ather_exploration.seeds import derive_seed, stage_rng
from ather_exploration.types import ValidatorStatus
from ather_exploration.worlds.generation import (
    STRATA,
    GeneratedScenario,
    GenerationError,
    generate_scenario,
    load_generated,
    spawn_diagnostics,
)
from ather_exploration.worlds.scenarios import (
    geometry_hash,
    implementation_id,
    read_record,
    world_hash,
    write_record,
)
from ather_exploration.worlds.topology import floor_cells
from ather_exploration.worlds.validation import validate_scenario

SPLITS = ("train", "validation_quick", "validation_selection", "heldout_id")


def _record_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Bank record path must stay inside bank directory")
    return path


def _spawn_pool(primary, count):
    """Return up to count validated starts on one identical world; never resample it."""
    pool = [primary]
    if count == 1:
        return pool
    scenario, config = primary.scenario, primary.config
    candidates = list(floor_cells(scenario.terrain))
    stage_rng(scenario.seed, "development-spawn-pool").shuffle(candidates)
    calls = 0
    for spawn in candidates[: config.budgets.spawn_attempts]:
        if spawn == scenario.spawn:
            continue
        alternate = replace(scenario, spawn=spawn)
        try:
            diagnostics = spawn_diagnostics(alternate, config)
        except ValueError:
            continue
        if config.spawn_weights[STRATA.index(diagnostics["stratum"])] == 0:
            continue
        if calls >= config.budgets.validator_calls_per_reset:
            break
        calls += 1
        validation = validate_scenario(
            alternate, max_expansions=config.budgets.validator_expansions
        )
        if validation.status is ValidatorStatus.VALIDATED:
            pool.append(
                GeneratedScenario(
                    alternate, validation, config, diagnostics["stratum"], diagnostics
                )
            )
        if len(pool) == count:
            break
    return pool


def _weights(pool, config):
    counts = {s: sum(p.stratum == s for p in pool) for s in STRATA}
    mass = sum(w for s, w in zip(STRATA, config.spawn_weights, strict=True) if counts[s])
    return [config.spawn_weights[STRATA.index(p.stratum)] / counts[p.stratum] / mass for p in pool]


def build_development_suite(
    config, root_seed, output, *, count=2, train_starts=1, attempts_per_world=32
):
    """Build count worlds PER split, or raise with a resumable BUILDING manifest.

    Existing records are immutable. Re-running the same request resumes incomplete
    slots; increasing attempts_per_world permits retries after a bounded failure.
    """
    for name, value, maximum in (
        ("count", count, None),
        ("train_starts", train_starts, 8),
        ("attempts_per_world", attempts_per_world, None),
    ):
        if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
            raise ValueError(f"Invalid {name}")
    derive_seed(root_seed, "development")
    root = Path(output)
    manifest_path = root / "manifest.json"
    request = {
        "namespace": "development",
        "root_seed": root_seed,
        "count_per_split": count,
        "train_starts": train_starts,
        "config_hash": config_hash(config, source_revision=implementation_id()),
    }
    if manifest_path.exists():
        manifest = read_record(manifest_path)
        if manifest["request"] != request:
            raise ValueError("Cannot resume bank with a different seed/config/code/request")
    else:
        manifest = {
            "state": "BUILDING",
            "request": request,
            "worlds": [],
            "attempts": {},
            "rejections": [],
        }
    if manifest["state"] not in ("BUILDING", "READY"):
        raise ValueError("Invalid bank state")
    expected_slots = {f"{split}/{i}" for split in SPLITS for i in range(count)}
    slots = [w["slot"] for w in manifest["worlds"]]
    if len(set(slots)) != len(slots) or not set(slots) <= expected_slots:
        raise ValueError("Invalid or duplicate bank slots")
    seen = set()
    for world in manifest["worlds"]:
        if (
            not world["starts"]
            or world["split"] not in SPLITS
            or not world["slot"].startswith(world["split"] + "/")
        ):
            raise ValueError("Invalid bank world entry")
        records = [load_generated(_record_path(root, p)) for p in world["starts"]]
        if any(p.scenario.config_hash != request["config_hash"] for p in records):
            raise ValueError("Bank record config differs from manifest")
        if (
            world["spawn_weights"] != _weights(records, config)
            or world["world_weight"] != 1 / count
        ):
            raise ValueError("Bank sampling weights disagree with config")
        hashed = geometry_hash(records[0].scenario.terrain)
        if hashed != world["geometry_hash"] or hashed in seen:
            raise ValueError("Bank geometry duplicate or mismatch")
        if len({world_hash(p.scenario) for p in records}) != 1:
            raise ValueError("Spawn pool mixes worlds")
        seen.add(hashed)
    for split in SPLITS:
        for index in range(count):
            slot = f"{split}/{index}"
            if any(w["slot"] == slot for w in manifest["worlds"]):
                continue
            target_stratum = STRATA[
                int(
                    stage_rng(root_seed, "development-stratum", split, config.preset, index).choice(
                        4, p=config.spawn_weights
                    )
                )
            ]
            accepted = False
            for attempt in range(manifest["attempts"].get(slot, 0), attempts_per_world):
                seed = derive_seed(root_seed, "development", split, config.preset, index, attempt)
                try:
                    timing = {}
                    primary = generate_scenario(
                        config, seed, stratum=target_stratum, timings=timing
                    )
                    hashed = geometry_hash(primary.scenario.terrain)
                    if hashed in seen:
                        raise ValueError("canonical geometry duplicate")
                except ValueError as error:
                    manifest["attempts"][slot] = attempt + 1
                    manifest["rejections"].append(
                        {"slot": slot, "attempt": attempt, "seed": seed, "reason": str(error)}
                    )
                    write_record(manifest_path, manifest, replace=True)
                    continue
                pool_started = perf_counter()
                pool = _spawn_pool(primary, train_starts if split == "train" else 1)
                timing["pool_seconds"] = perf_counter() - pool_started
                paths = []
                for start_index, record in enumerate(pool):
                    relative = f"{split}/{index}-{attempt}-{start_index}.json"
                    write_record(root / relative, record.payload())
                    paths.append(relative)
                manifest["worlds"].append(
                    {
                        "slot": slot,
                        "split": split,
                        "seed": seed,
                        "geometry_hash": hashed,
                        "starts": paths,
                        "spawn_weights": _weights(pool, config),
                        "world_weight": 1 / count,
                        "timing": timing,
                        "target_stratum": primary.stratum,
                        "target_weights": list(config.spawn_weights),
                        "requested_starts": train_starts if split == "train" else 1,
                        "available_strata": sorted({p.stratum for p in pool}),
                    }
                )
                manifest["attempts"][slot] = attempt + 1
                seen.add(hashed)
                write_record(manifest_path, manifest, replace=True)
                accepted = True
                break
            if not accepted:
                raise GenerationError(
                    f"Development bank incomplete at {slot}; increase attempts_per_world to resume",
                    manifest["attempts"],
                )
    manifest["state"] = "READY"
    write_record(manifest_path, manifest, replace=True)
    return manifest


def sample_bank_start(output, split, rng):
    """Equal world marginal first, then conditional weighted start. Evaluator API."""
    if split not in SPLITS:
        raise ValueError("Unknown split")
    root = Path(output)
    manifest = read_record(root / "manifest.json")
    if manifest["state"] != "READY":
        raise ValueError("Cannot sample an incomplete bank")
    worlds = [w for w in manifest["worlds"] if w["split"] == split]
    if len(worlds) != manifest["request"]["count_per_split"]:
        raise ValueError("Incomplete split")
    world = worlds[int(rng.integers(len(worlds)))]
    path = world["starts"][int(rng.choice(len(world["starts"]), p=world["spawn_weights"]))]
    return load_generated(_record_path(root, path))
