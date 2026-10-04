"""Data boundaries shared by future dynamics, sensor, agent and evaluator.

Coordinates are (x, y), increasing right/down. Immutable scenarios store terrain
as rows of text; the MiniGrid adapter will materialize a Grid from these rows.
No simulator, LOS, reward computation or procedural generation lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from itertools import pairwise
from numbers import Integral
from typing import Any, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

Position = tuple[int, int]
ACTION_DELTAS: tuple[Position, ...] = ((0, -1), (0, 1), (1, 0), (-1, 0), (0, 0))


class Action(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    WAIT = 4


def normalize_action(value: Any) -> Action:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("action must be a scalar integer, not bool/float/array")
    return Action(int(value))


class Motion(StrEnum):
    MOVED = "moved"
    WALL_BLOCKED = "wall_blocked"
    WAITED = "waited"


class EndReason(StrEnum):
    NONE = "none"
    DEATH = "death"
    BUDGET = "budget"


class ValidatorStatus(StrEnum):
    VALIDATED = "validated"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


def _integer(value, name: str, minimum: int = 0, maximum: int | None = None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _position(value):
    if type(value) is not tuple or len(value) != 2 or any(type(v) is not int for v in value):
        raise ValueError("positions must be immutable (int x, int y) tuples")


@dataclass(frozen=True, slots=True)
class Scenario:
    terrain: tuple[str, ...]  # '#' wall, '.' floor. No mutable MiniGrid objects here.
    spawn: Position
    pois: tuple[Position, ...]
    routes: tuple[tuple[Position, ...], ...]
    phases: tuple[int, ...]
    horizon: int
    seed: int
    patrol_period: int = 2
    fixture_only: bool = False
    skill_task: str | None = None
    config_hash: str = ""
    source_revision: str = ""
    schema_version: str = "1"
    room_labels: tuple[tuple[int, ...], ...] = ()
    topology_edges: tuple[tuple[int, int], ...] = ()
    witness_reference: str | None = None

    def __post_init__(self):
        for name in ("config_hash", "source_revision", "schema_version"):
            if type(getattr(self, name)) is not str:
                raise TypeError(f"{name} must be a string")
        if self.witness_reference is not None and type(self.witness_reference) is not str:
            raise TypeError("witness_reference must be a string or None")
        if self.schema_version != "1":
            raise ValueError("Unsupported scenario schema version")
        for name in ("terrain", "pois", "routes", "phases", "room_labels", "topology_edges"):
            if type(getattr(self, name)) is not tuple:
                raise TypeError(f"{name} must be an immutable tuple")
        _integer(self.horizon, "horizon", 1, 1024)
        _integer(self.seed, "seed", 0, 2**64 - 1)
        _integer(self.patrol_period, "patrol_period", 1)
        if type(self.fixture_only) is not bool:
            raise TypeError("fixture_only must be bool")
        if not self.terrain or any(type(row) is not str for row in self.terrain):
            raise ValueError("terrain must contain nonempty string rows")
        width = len(self.terrain[0])
        if not 3 <= width <= 41 or not 3 <= len(self.terrain) <= 41:
            raise ValueError("terrain dimensions must be 3..41")
        if any(len(row) != width or set(row) - {"#", "."} for row in self.terrain):
            raise ValueError("terrain must be rectangular and use only #/.")
        if set(self.terrain[0] + self.terrain[-1]) != {"#"} or any(
            row[0] != "#" or row[-1] != "#" for row in self.terrain
        ):
            raise ValueError("outer boundary must be walls")
        if self.skill_task is not None and self.skill_task not in (
            "P1a",
            "P1b",
            "P2a",
            "P2b",
            "P2c",
            "P3",
            "P3a",
            "P3b",
            "P3c",
            "P4a",
            "P4b",
            "P4c",
        ):
            raise ValueError("Unknown skill task")
        if self.skill_task is not None and self.fixture_only:
            raise ValueError("Skill scenarios are not test fixtures")
        self._require_floor(self.spawn)
        for poi in self.pois:
            self._require_floor(poi)
        if len(set(self.pois)) != len(self.pois) or len(self.pois) > 4:
            raise ValueError("POIs must be unique, at most 4")
        if len(self.routes) != len(self.phases):
            raise ValueError("one phase is required for each route")
        occupied = set()
        for route, phase in zip(self.routes, self.phases, strict=True):
            if type(route) is not tuple or len(route) < 2:
                raise ValueError("route must be an immutable sequence of at least two cells")
            for pos in route:
                self._require_floor(pos)
            if len(set(route)) != len(route) or occupied.intersection(route):
                raise ValueError("routes must be simple and mutually vertex-disjoint")
            if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in pairwise(route)):
                raise ValueError("route cells must be four-neighbor adjacent")
            _integer(phase, "phase", 0, self.patrol_period * 2 * (len(route) - 1) - 1)
            occupied.update(route)
        if self.room_labels:
            if len(self.room_labels) != len(self.terrain):
                raise ValueError("room_labels must match terrain dimensions")
            for row in self.room_labels:
                if (
                    type(row) is not tuple
                    or len(row) != width
                    or any(type(v) is not int for v in row)
                ):
                    raise ValueError("room_labels must be immutable integer rows")
        for edge in self.topology_edges:
            _position(edge)
        if not self.fixture_only and self.skill_task is None and (not self.pois or not self.routes):
            raise ValueError("main-task scenario requires POIs and monsters")

    def _require_floor(self, pos: Position):
        _position(pos)
        x, y = pos
        if not (0 <= y < len(self.terrain) and 0 <= x < len(self.terrain[0])):
            raise ValueError(f"position outside terrain: {pos}")
        if self.terrain[y][x] != ".":
            raise ValueError(f"position must be on floor: {pos}")


@dataclass(slots=True)
class EpisodeState:
    """Privileged mutable state; must never be handed to policy.act()."""

    agent_position: Position
    monster_positions: list[Position]
    activated_pois: set[Position] = field(default_factory=set)
    step_count: int = 0
    alive: bool = True
    done: bool = False
    end_reason: EndReason = EndReason.NONE

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> EpisodeState:
        positions = []
        for route, phase in zip(scenario.routes, scenario.phases, strict=True):
            cycle = route + route[-2:0:-1]
            positions.append(cycle[phase // scenario.patrol_period])
        return cls(agent_position=scenario.spawn, monster_positions=positions)


@dataclass(frozen=True, slots=True)
class PublicTransition:
    action: Action
    motion: Motion
    actual_delta: Position
    new_floor: int = 0
    new_poi: int = 0
    activated: bool = False
    died: bool = False
    end_reason: EndReason = EndReason.NONE

    def __post_init__(self):
        if not isinstance(self.action, Action) or not isinstance(self.motion, Motion):
            raise TypeError("action/motion must use the contract enums")
        if not isinstance(self.end_reason, EndReason):
            raise TypeError("end_reason must use the contract enum")
        _position(self.actual_delta)
        _integer(self.new_floor, "new_floor")
        _integer(self.new_poi, "new_poi", 0, 4)
        if type(self.activated) is not bool or type(self.died) is not bool:
            raise TypeError("activated/died must be bool")
        expected = ACTION_DELTAS[self.action] if self.motion is Motion.MOVED else (0, 0)
        if self.actual_delta != expected:
            raise ValueError("actual_delta disagrees with action/motion")
        if (self.action is Action.WAIT) != (self.motion is Motion.WAITED):
            raise ValueError("WAIT must have waited motion; movement must not")
        if self.died != (self.end_reason is EndReason.DEATH):
            raise ValueError("death event and end_reason must agree")


class PublicObservation(TypedDict):
    local: NDArray[np.uint8]
    memory: NDArray[np.float32]
    state: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class EvaluatorSnapshot:
    """Privileged copy for evaluation/rendering, not an observation feature."""

    scenario: Scenario
    agent_position: Position
    monster_positions: tuple[Position, ...]
    activated_pois: frozenset[Position]
    step_count: int
    end_reason: EndReason


@dataclass(slots=True)
class AgentState:
    """Opaque adapter-owned recurrent/planner state, reset separately per env."""

    recurrent: Any = None
    planner: Any = None
    episode_start: bool = True


class AgentAdapter(Protocol):
    def act(
        self,
        observation: PublicObservation,
        state: AgentState,
        *,
        deterministic: bool,
        action_rng: np.random.Generator,
    ) -> tuple[Action, AgentState]: ...
