import json
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from ather_exploration.config import EnvConfig, load_preset
from ather_exploration.environment.env import make_env
from ather_exploration.seeds import derive_seed, stage_rng
from ather_exploration.types import ValidatorStatus
from ather_exploration.worlds.generation import (
    STRATA,
    GenerationError,
    check_generated,
    generate_scenario,
    load_generated,
    spawn_diagnostics,
)
from ather_exploration.worlds.scenarios import (
    digest,
    geometry_hash,
    read_record,
    scenario_hash,
    write_record,
)
from ather_exploration.worlds.topology import navigation_graph


@pytest.mark.parametrize("preset", ["small", "medium", "large"])
@pytest.mark.parametrize("seed", [0, 1, 42, 2026])
def test_generated_constraints_and_runtime_witness(preset, seed):
    record = generate_scenario(load_preset(preset), seed)
    check_generated(record)
    s, c = record.scenario, record.config
    assert len(record.diagnostics["rooms"]) == c.num_rooms
    assert all(
        c.room_side_min <= side <= c.room_side_max
        for room in record.diagnostics["rooms"]
        for side in room[2:]
    )
    assert len(set().union(*map(set, s.routes))) == sum(map(len, s.routes))
    env = make_env(generated=record)
    obs, info = env.reset()
    assert info == {} and not obs["local"][3].any()
    assert env.observation_space.contains(obs)
    for action in record.validation.actions:
        obs, _reward, done, truncated, info = env.step(action)
        assert set(info) == {"transition"}
    final = env.unwrapped.evaluator_snapshot()
    assert done and not truncated and final.step_count == s.horizon
    assert final.activated_pois == set(s.pois) and final.end_reason == "budget"
    env.close()


@pytest.mark.parametrize("stratum", STRATA)
def test_requested_stratum_never_silently_changes(stratum):
    r = generate_scenario(load_preset("small"), 42, stratum=stratum)
    assert r.stratum == stratum == r.diagnostics["stratum"]


def test_bounded_unknown_and_impossible_geometry():
    config = load_preset("small")
    budgets = config.budgets.model_copy(
        update={"validator_expansions": 1, "validator_calls_per_reset": 1}
    )
    with pytest.raises(GenerationError, match="validator call budget") as failure:
        generate_scenario(config.model_copy(update={"budgets": budgets}), 42)
    assert failure.value.stats["unknown"] == 1
    impossible = EnvConfig(width=11, height=11, num_rooms=3, room_side_min=5, room_side_max=5)
    with pytest.raises(GenerationError, match="bounded generation") as failure:
        generate_scenario(impossible, 0)
    assert failure.value.stats["geometry_attempts"] == impossible.budgets.geometry_attempts


def test_room_area_does_not_create_navigation_cycle_and_parallel_paths_survive():
    terrain = ("#######", "#.....#", "#.....#", "#.....#", "#######")
    labels = tuple(tuple(0 if cell == "." else -1 for cell in row) for row in terrain)
    assert navigation_graph(terrain, labels)["cycles"] == 0
    # Two separate corridors join the same two rooms.
    terrain = ("#########", "#.......#", "#..###..#", "#.......#", "#########")
    labels = tuple(
        tuple(
            (0 if x <= 2 else 1 if x >= 6 else -1) if cell == "." else -1
            for x, cell in enumerate(row)
        )
        for row in terrain
    )
    graph = navigation_graph(terrain, labels)
    assert graph["cycles"] == 1
    assert len(graph["edges"]) == 2
    assert graph["edges"][0][:2] == graph["edges"][1][:2]


def test_identity_cache_checksum_and_forged_witness(tmp_path):
    config = load_preset("small")
    cold = generate_scenario(config, 42, cache=tmp_path)
    warm = generate_scenario(config, 42, cache=tmp_path)
    assert digest(cold.payload()) == digest(warm.payload())
    path = next(tmp_path.glob("*.json"))
    payload = read_record(path)
    payload["validation"]["actions"] = [4] * config.horizon
    write_record(path, payload, replace=True)
    with pytest.raises(ValueError, match="Witness"):
        load_generated(path)
    data = json.loads(path.read_text())
    data["payload"]["scenario"]["seed"] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        load_generated(path)
    with pytest.raises(ValueError, match="checksum|overwrite"):
        write_record(path, cold.payload())


def test_stale_and_unknown_certificate_rejected(tmp_path):
    record = generate_scenario(load_preset("small"), 42)
    with pytest.raises(ValueError, match="certificate"):
        make_env(
            generated=replace(
                record, validation=replace(record.validation, status=ValidatorStatus.UNKNOWN)
            )
        )
    with pytest.raises(ValueError, match="certificate"):
        make_env(generated=replace(record, scenario=replace(record.scenario, horizon=200)))
    payload = record.payload()
    payload["scenario"]["source_revision"] = "old"
    path = tmp_path / "old.json"
    write_record(path, payload)
    with pytest.raises(ValueError, match="Stale"):
        load_generated(path)


def test_cross_process_python_hash_seed_and_cache_do_not_change_generation(tmp_path):
    expected = generate_scenario(load_preset("small"), 42)
    code = """from ather_exploration.config import load_preset
from ather_exploration.worlds.generation import generate_scenario
from ather_exploration.worlds.scenarios import digest
import sys
print(digest(generate_scenario(load_preset('small'),42,cache=sys.argv[1]).payload()))
"""
    for salt in ("12", "999"):
        result = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)],
            env={**os.environ, "PYTHONHASHSEED": salt, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == digest(expected.payload())


def test_seed_sequence_reset_and_information_boundary(tmp_path):
    a, b = make_env(cache=tmp_path), make_env(cache=tmp_path)
    hashes = []
    for seed in (42, None, None, 42):
        oa, ia = a.reset(seed=seed)
        ob, ib = b.reset(seed=seed)
        assert ia == ib == {}
        assert all(np.array_equal(oa[k], ob[k]) for k in oa)
        assert scenario_hash(a.unwrapped.scenario) == scenario_hash(b.unwrapped.scenario)
        hashes.append(scenario_hash(a.unwrapped.scenario))
    assert hashes[0] == hashes[-1] and len(set(hashes[:3])) == 3
    a.close()
    b.close()


def test_canonical_geometry_and_seed_namespaces():
    terrain = ("#####", "#...#", "##..#", "#####")
    transformed = tuple("".join(row) for row in np.rot90(np.array([list(r) for r in terrain])))
    assert (
        geometry_hash(terrain)
        == geometry_hash(transformed)
        == geometry_hash(tuple(r[::-1] for r in terrain))
    )
    namespaces = ["geometry", "placement", "phase", "spawn", "model", "action", "stats"]
    assert len({derive_seed(42, name) for name in namespaces}) == len(namespaces)
    assert np.array_equal(
        stage_rng(42, "action").integers(100, size=10),
        stage_rng(42, "action").integers(100, size=10),
    )
    with pytest.raises(ValueError):
        derive_seed(True, "geometry")


def test_poi_behind_transparent_monster_cannot_be_admitted():
    from ather_exploration.types import Scenario

    s = Scenario(
        ("#######", "#.....#", "#######"),
        (1, 1),
        ((4, 1),),
        (((3, 1), (2, 1)),),
        (0,),
        8,
        0,
        room_labels=((-1,) * 7, (-1, 0, 0, 0, 0, 0, -1), (-1,) * 7),
    )
    with pytest.raises(ValueError, match="POI visible"):
        spawn_diagnostics(s, load_preset("small"))


def test_gymnasium_checker_procedural_and_timing_is_not_identity():
    from gymnasium.utils.env_checker import check_env

    env = make_env()
    check_env(env.unwrapped, skip_render_check=True)
    env.close()
    timings = {}
    a = generate_scenario(load_preset("small"), 42, timings=timings)
    b = generate_scenario(load_preset("small"), 42)
    assert timings["total_seconds"] >= timings["validation_seconds"] >= 0
    assert digest(a.payload()) == digest(b.payload())


def test_atomic_record_idempotent_and_immutable(tmp_path):
    path = tmp_path / "scenario.json"
    write_record(path, {"version": 1})
    write_record(path, {"version": 1})
    with pytest.raises(ValueError, match="overwrite"):
        write_record(path, {"version": 2})
    assert read_record(path) == {"version": 1}
    assert not list(tmp_path.glob(".pending-*"))


def test_source_identity_tracks_nested_paths_and_survives_relocation(tmp_path, monkeypatch):
    """Moving code to subpackages must not hide changes or collide on basenames."""
    import shutil

    from ather_exploration.worlds import scenarios

    root = tmp_path / "first" / "ather_exploration"
    for folder in ("environment", "worlds", "ui"):
        (root / folder).mkdir(parents=True)
    (root / "environment" / "shared.py").write_text("VALUE = 1\n")
    (root / "ui" / "shared.py").write_text("VALUE = 2\n")
    monkeypatch.setattr(scenarios, "__file__", str(root / "worlds" / "scenarios.py"))
    original = scenarios.implementation_id()
    for relative in ("environment/shared.py", "ui/shared.py"):
        path = root / relative
        before = path.read_text()
        path.write_text(before + "# change\n")
        assert scenarios.implementation_id() != original
        path.write_text(before)
        assert scenarios.implementation_id() == original
    # Relative path changes count even if content does not; generated files don't.
    (root / "ui" / "shared.py").rename(root / "ui" / "renamed.py")
    renamed = scenarios.implementation_id()
    assert renamed != original
    (root / "ui" / "ignored.pyc").write_bytes(b"cache")
    assert scenarios.implementation_id() == renamed
    relocated = tmp_path / "elsewhere" / "ather_exploration"
    shutil.copytree(root, relocated)
    monkeypatch.setattr(scenarios, "__file__", str(relocated / "worlds" / "scenarios.py"))
    assert scenarios.implementation_id() == renamed
