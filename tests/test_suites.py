import numpy as np
import pytest

from ather_exploration.config import load_preset
from ather_exploration.worlds.generation import GenerationError
from ather_exploration.worlds.scenarios import read_record
from ather_exploration.worlds.suites import SPLITS, build_development_suite, sample_bank_start


def test_bank_disjoint_spawn_pools_and_repeat_resume(tmp_path):
    config = load_preset("small")
    manifest = build_development_suite(config, 11, tmp_path, count=2, train_starts=3)
    assert manifest["state"] == "READY"
    assert len(manifest["worlds"]) == 8
    assert len({w["geometry_hash"] for w in manifest["worlds"]}) == 8
    for world in manifest["worlds"]:
        assert sum(world["spawn_weights"]) == pytest.approx(1)
        assert world["world_weight"] == 0.5
        assert 1 <= len(world["starts"]) <= (3 if world["split"] == "train" else 1)
    for split in SPLITS:
        sample = sample_bank_start(tmp_path, split, np.random.default_rng(0))
        assert sample.validation.status == "validated"
    assert build_development_suite(config, 11, tmp_path, count=2, train_starts=3) == manifest
    with pytest.raises(ValueError, match="different"):
        build_development_suite(config, 12, tmp_path, count=2, train_starts=3)


def test_interrupted_build_resumes_and_keeps_partial_records(tmp_path, monkeypatch):
    from ather_exploration.worlds import suites

    original = suites.generate_scenario
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(suites, "generate_scenario", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        build_development_suite(load_preset("small"), 11, tmp_path, count=1)
    partial = read_record(tmp_path / "manifest.json")
    assert partial["state"] == "BUILDING" and len(partial["worlds"]) == 1
    with pytest.raises(ValueError, match="incomplete"):
        sample_bank_start(tmp_path, "train", np.random.default_rng(0))
    monkeypatch.setattr(suites, "generate_scenario", original)
    final = build_development_suite(load_preset("small"), 11, tmp_path, count=1)
    assert final["worlds"][0] == partial["worlds"][0]
    assert final["state"] == "READY" and len(final["worlds"]) == 4


def test_duplicate_geometry_is_logged_not_silently_admitted(tmp_path, monkeypatch):
    from ather_exploration.worlds import suites

    primary = suites.generate_scenario(load_preset("small"), 42)
    monkeypatch.setattr(suites, "generate_scenario", lambda *a, **k: primary)
    with pytest.raises(GenerationError, match="incomplete"):
        build_development_suite(load_preset("small"), 11, tmp_path, count=1, attempts_per_world=2)
    partial = read_record(tmp_path / "manifest.json")
    assert len(partial["worlds"]) == 1 and len(partial["rejections"]) == 2
    assert all(r["reason"] == "canonical geometry duplicate" for r in partial["rejections"])


def test_bank_record_cannot_escape_output_directory(tmp_path):
    from ather_exploration.worlds.suites import _record_path

    with pytest.raises(ValueError, match="inside"):
        _record_path(tmp_path, "../elsewhere.json")
