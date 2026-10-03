"""Strict, immutable environment configuration and cwd-independent presets.

G0 only establishes the env/observation/reward contracts. Training and UI
configuration get their own validated models when their steps are implemented.
"""

from __future__ import annotations

import hashlib
import json
import math
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
Weight = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PRESET_NAMES = ("small", "medium", "large")


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="before")
    @classmethod
    def reject_literal_numeric_coercion(cls, values):
        # Pydantic Literal[1] compares equal to True/1.0 even under strict=True.
        if isinstance(values, dict):
            for name, value in values.items():
                if isinstance(value, bool) and not (
                    name in cls.model_fields and cls.model_fields[name].annotation is bool
                ):
                    # Pydantic validators require ValueError to wrap ValidationError.
                    raise ValueError(f"Boolean is not a valid numeric/config value: {name}")
                if (
                    name
                    in {
                        "map_capacity",
                        "horizon_capacity",
                        "poi_capacity",
                        "corridor_width",
                        "room_leaf_margin",
                    }
                    and type(value) is not int
                ):
                    raise ValueError(f"{name} must be an exact integer")
        return values


class Versions(FrozenConfig):
    dynamics: Literal["1"] = "1"
    generator: Literal["1"] = "1"
    sensor: Literal["1"] = "1"
    observation_schema: Literal["1"] = "1"
    reward: Literal["1"] = "1"
    metric: Literal["1"] = "1"
    experiment_protocol: Literal["1"] = "1"
    artifact_schema: Literal["1"] = "1"


class ObservationConfig(FrozenConfig):
    radius: Annotated[int, Field(ge=1, le=40)] = 4
    map_capacity: Literal[41] = 41
    horizon_capacity: Literal[1024] = 1024
    poi_capacity: Literal[4] = 4

    @property
    def local_shape(self) -> tuple[int, int, int]:
        side = 2 * self.radius + 1
        return (6, side, side)

    @property
    def memory_shape(self) -> tuple[int, int, int]:
        side = 2 * self.map_capacity - 1
        return (11, side, side)


class RewardConfig(FrozenConfig):
    area: Weight = 0.01
    discovery: Weight = 0.05
    activation: Weight = 0.50
    death: Weight = 2.0  # Positive magnitude; the death term subtracts it.


class GenerationBudgets(FrozenConfig):
    geometry_attempts: PositiveInt = 32
    placement_attempts: PositiveInt = 16
    phase_attempts: PositiveInt = 8
    spawn_attempts: PositiveInt = 64
    route_expansions: PositiveInt = 20_000
    validator_expansions: PositiveInt = 4_000_000
    validator_calls_per_reset: PositiveInt = 128


class EnvConfig(FrozenConfig):
    preset: Literal["small", "medium", "large", "custom"] = "custom"
    width: Annotated[int, Field(ge=5, le=41)] = 31
    height: Annotated[int, Field(ge=5, le=41)] = 31
    num_rooms: PositiveInt = 6
    room_side_min: Annotated[int, Field(ge=2)] = 5
    room_side_max: Annotated[int, Field(ge=2)] = 9
    num_pois: Annotated[int, Field(ge=1, le=4)] = 3
    num_monsters: PositiveInt = 2
    route_length_min: Annotated[int, Field(ge=2)] = 8
    route_length_max: Annotated[int, Field(ge=2)] = 12
    horizon: Annotated[int, Field(ge=1, le=1024)] = 512
    patrol_period: PositiveInt = 2
    extra_connections_min: NonNegativeInt = 1
    extra_connections_max: NonNegativeInt = 2
    corridor_width: Literal[1] = 1
    room_leaf_margin: Literal[1] = 1
    bsp_aspect_bias: Annotated[float, Field(gt=1, allow_inf_nan=False)] = 1.5
    min_poi_geodesic_distance: PositiveInt = 6
    min_spawn_poi_distance: PositiveInt = 2
    min_spawn_monster_distance: PositiveInt = 2
    spawn_weights: tuple[Weight, Weight, Weight, Weight] = (0.25, 0.25, 0.25, 0.25)
    seed: Annotated[int, Field(ge=0, lt=2**64)] | None = None
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    budgets: GenerationBudgets = Field(default_factory=GenerationBudgets)
    versions: Versions = Field(default_factory=Versions)

    @model_validator(mode="before")
    @classmethod
    def accept_serialized_weights(cls, values):
        # YAML/JSON has lists, not tuples. Convert only this documented sequence;
        # scalar coercions (e.g. true/21.0 -> width) remain forbidden.
        if isinstance(values, dict) and isinstance(values.get("spawn_weights"), list):
            values = dict(values)
            values["spawn_weights"] = tuple(values["spawn_weights"])
        return values

    @model_validator(mode="after")
    def cross_validate(self):
        if self.room_side_min > self.room_side_max:
            raise ValueError("room_side_min must not exceed room_side_max")
        if self.room_side_min > min(self.width, self.height) - 2:
            raise ValueError("Smallest room cannot fit inside map boundaries")
        if self.route_length_min > self.route_length_max:
            raise ValueError("route_length_min must not exceed route_length_max")
        if self.extra_connections_min > self.extra_connections_max:
            raise ValueError("extra_connections_min must not exceed maximum")
        interior = (self.width - 2) * (self.height - 2)
        if self.num_rooms * self.room_side_min**2 > interior:
            raise ValueError("Minimum room area exceeds interior capacity")
        if self.num_monsters * self.route_length_min > interior:
            raise ValueError("Disjoint minimum routes exceed interior capacity")
        if not math.isclose(sum(self.spawn_weights), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("spawn_weights must sum to 1")
        return self


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys rather than silently use the last value."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise TypeError("Configuration keys must be strings")
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _parse(text: str) -> EnvConfig:
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML configuration: {error}") from error
    return EnvConfig.model_validate(data)


def load_config(path: str | Path) -> EnvConfig:
    return _parse(Path(path).expanduser().read_text(encoding="utf-8"))


def load_preset(name: str) -> EnvConfig:
    if name not in PRESET_NAMES:
        raise ValueError(f"Unknown preset {name!r}; choose {PRESET_NAMES}")
    resource = files("ather_exploration").joinpath("resources", "presets", f"{name}.yaml")
    return _parse(resource.read_text(encoding="utf-8"))


def config_hash(config: EnvConfig, *, source_revision: str = "unrecorded") -> str:
    """Deterministic content hash. Formal runners must supply a source revision."""
    payload = {"config": config.model_dump(mode="json"), "source_revision": source_revision}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
