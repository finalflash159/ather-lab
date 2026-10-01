"""Finite-horizon reachability proof; witness is privileged and never a policy input."""

from array import array
from dataclasses import dataclass

import numpy as np

from ather_exploration.environment.dynamics import advance, make_grid, monster_positions
from ather_exploration.types import (
    ACTION_DELTAS,
    Action,
    EpisodeState,
    Scenario,
    ValidatorStatus,
    normalize_action,
)
from ather_exploration.worlds.topology import distances, floor_cells


@dataclass(frozen=True, slots=True)
class ValidationResult:
    status: ValidatorStatus
    actions: tuple[int, ...]
    expansions: int
    reason: str
    scenario_hash: str


def replay_witness(scenario: Scenario, actions) -> None:
    if len(actions) != scenario.horizon:
        raise ValueError("Witness must contain exactly H actions")
    state = EpisodeState.from_scenario(scenario)
    grid = make_grid(scenario)
    if state.agent_position in state.monster_positions:
        raise ValueError("Spawn collides at reset")
    for action in actions:
        event, _ = advance(grid, scenario, state, normalize_action(action))
        if event.died:
            raise ValueError("Witness dies during runtime replay")
    if not state.done or not state.alive or set(scenario.pois) != state.activated_pois:
        raise ValueError("Witness did not activate all POIs and survive H")


def validate_scenario(scenario: Scenario, *, max_expansions: int = 1_000_000) -> ValidationResult:
    from ather_exploration.worlds.scenarios import scenario_hash

    if type(max_expansions) is not int or max_expansions <= 0:
        raise ValueError("max_expansions must be a positive integer")
    digest = scenario_hash(scenario)

    def result(status, actions, expanded, reason):
        if status is ValidatorStatus.VALIDATED:
            replay_witness(scenario, actions)
        return ValidationResult(status, tuple(actions), expanded, reason, digest)

    cells = floor_cells(scenario.terrain)
    index = {p: i for i, p in enumerate(cells)}
    n = len(cells)
    schedules = [
        {index[p] for p in monster_positions(scenario, t)} for t in range(scenario.horizon + 1)
    ]
    start = index[scenario.spawn]
    if start in schedules[0]:
        return result(ValidatorStatus.INFEASIBLE, (), 0, "spawn_collision")
    full_mask = (1 << len(scenario.pois)) - 1
    poi_bit = {index[p]: 1 << i for i, p in enumerate(scenario.pois)}
    union = {index[p] for route in scenario.routes for p in route}
    destinations = []
    for x, y in cells:
        destinations.append(
            tuple(index.get((x + dx, y + dy), index[(x, y)]) for dx, dy in ACTION_DELTAS)
        )
    lower = np.zeros((full_mask + 1, n), dtype=np.int32)
    distance_maps = [distances(scenario.terrain, [p]) for p in scenario.pois]
    for mask in range(full_mask + 1):
        remaining = [i for i in range(len(scenario.pois)) if not mask & (1 << i)]
        for pos, p in enumerate(cells):
            lower[mask, pos] = max(
                (distance_maps[i].get(p, scenario.horizon + 1) for i in remaining), default=0
            )
    parents = array("i", [-1])
    actions = array("B", [4])
    frontier = {start: 0}
    expanded = 0

    def witness(node, tail):
        path = []
        while parents[node] != -1:
            path.append(actions[node])
            node = parents[node]
        return tuple(reversed(path)) + (int(Action.WAIT),) * tail

    for tick in range(scenario.horizon + 1):
        following = {}
        for code, node in frontier.items():
            mask, pos = divmod(code, n)
            if mask == full_mask and (tick == scenario.horizon or pos not in union):
                return result(
                    ValidatorStatus.VALIDATED,
                    witness(node, scenario.horizon - tick),
                    expanded,
                    "replayed_witness",
                )
            if tick == scenario.horizon or lower[mask, pos] > scenario.horizon - tick:
                continue
            if expanded >= max_expansions:
                return result(ValidatorStatus.UNKNOWN, (), expanded, "expansion_budget")
            expanded += 1
            unsafe = schedules[tick] | schedules[tick + 1]
            for action, target in enumerate(destinations[pos]):
                if target in unsafe:
                    continue  # collision1 and collision2
                updated = mask | poi_bit.get(target, 0)
                if lower[updated, target] > scenario.horizon - tick - 1:
                    continue
                key = updated * n + target
                if key in following:
                    continue
                following[key] = len(parents)
                parents.append(node)
                actions.append(action)
        frontier = following
        if not frontier:
            break
    return result(ValidatorStatus.INFEASIBLE, (), expanded, "exhausted_reachable_states")
