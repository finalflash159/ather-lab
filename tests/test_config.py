import json

import pytest
from pydantic import ValidationError

from ather_exploration.config import EnvConfig, config_hash, load_config, load_preset


@pytest.mark.parametrize(
    "name,size,rooms,pois,monsters,horizon",
    [
        ("small", 21, 4, 2, 1, 256),
        ("medium", 31, 6, 3, 2, 512),
        ("large", 41, 10, 4, 3, 1024),
    ],
)
def test_presets(name, size, rooms, pois, monsters, horizon):
    c = load_preset(name)
    assert (c.width, c.height, c.num_rooms, c.num_pois, c.num_monsters, c.horizon) == (
        size,
        size,
        rooms,
        pois,
        monsters,
        horizon,
    )
    assert c.observation.radius == 4
    assert c.patrol_period == 2
    assert c.observation.memory_shape == (11, 81, 81)


@pytest.mark.parametrize(
    "updates",
    [
        {"width": True},
        {"width": 21.0},
        {"width": 43},
        {"height": 0},
        {"horizon": 1025},
        {"num_pois": 5},
        {"patrol_period": 0},
        {"seed": -1},
        {"seed": True},
        {"seed": 2**64},
        {"typo": 3},
        {"room_side_min": 10, "room_side_max": 5},
        {"route_length_min": 1},
        {"num_monsters": 0},
        {"observation": {"radius": 0}},
        {"observation": {"map_capacity": 99}},
        {"spawn_weights": [0.1, 0.2, 0.3, 0.5]},
        {"reward": {"death": float("nan")}},
        {"budgets": {"validator_expansions": 0}},
    ],
)
def test_invalid_config_rejected(updates):
    data = load_preset("small").model_dump(mode="json")
    data.update(updates)
    with pytest.raises(ValidationError):
        EnvConfig.model_validate(data)


def test_hash_and_roundtrip(tmp_path):
    c = load_preset("medium")
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(c.model_dump(mode="json")))
    assert load_config(path) == c
    assert config_hash(c) == config_hash(load_config(path))
    assert config_hash(c, source_revision="a") != config_hash(c, source_revision="b")
    changed = c.model_dump(mode="json")
    changed["horizon"] = 513
    assert config_hash(c) != config_hash(EnvConfig.model_validate(changed))


def test_unknown_preset_and_duplicate_yaml(tmp_path):
    with pytest.raises(ValueError, match="preset"):
        load_preset("../medium")
    path = tmp_path / "duplicate.yaml"
    path.write_text("width: 21\nwidth: 31\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_config(path)


def test_nested_config_is_frozen():
    c = load_preset("small")
    with pytest.raises(ValidationError):
        c.reward.area = 2.0


@pytest.mark.parametrize(
    "updates",
    [
        {"corridor_width": True},
        {"room_leaf_margin": 1.0},
        {"observation": {"poi_capacity": 4.0}},
        {"reward": {"area": True}},
    ],
)
def test_literal_and_weight_types_are_strict(updates):
    data = load_preset("small").model_dump(mode="json")
    data.update(updates)
    with pytest.raises(ValidationError):
        EnvConfig.model_validate(data)
