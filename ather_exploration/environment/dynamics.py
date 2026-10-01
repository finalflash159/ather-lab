"""MiniGrid terrain and the single transition rule shared by env and future validator."""

from ather_exploration.types import (
    ACTION_DELTAS,
    Action,
    EndReason,
    EpisodeState,
    Motion,
    PublicTransition,
    Scenario,
    normalize_action,
)
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall


def make_grid(scenario: Scenario) -> Grid:
    grid = Grid(len(scenario.terrain[0]), len(scenario.terrain))
    for y, row in enumerate(scenario.terrain):
        for x, tile in enumerate(row):
            if tile == "#":
                grid.set(x, y, Wall())
    return grid


def walkable(grid: Grid, position: tuple[int, int]) -> bool:
    x, y = position
    if not (0 <= x < grid.width and 0 <= y < grid.height):
        return False
    cell = grid.get(x, y)
    return cell is None or cell.can_overlap()


def monster_positions(scenario: Scenario, tick: int) -> list[tuple[int, int]]:
    if type(tick) is not int or tick < 0:
        raise ValueError("tick must be a nonnegative integer")
    positions = []
    for route, phase in zip(scenario.routes, scenario.phases, strict=True):
        period = 2 * (len(route) - 1)
        index = ((tick + phase) // scenario.patrol_period) % period
        route_index = index if index < len(route) else period - index
        positions.append(route[route_index])
    return positions


def advance(
    grid: Grid, scenario: Scenario, state: EpisodeState, action: int
) -> tuple[PublicTransition, int | None]:
    """Commit one tick. Collision stage is privileged diagnostics, separate from public event."""
    action = normalize_action(action)  # Reject before any mutation.
    if state.done or state.step_count >= scenario.horizon:
        raise RuntimeError("Episode ended; call reset before step")
    old_x, old_y = state.agent_position
    dx, dy = ACTION_DELTAS[action]
    target = (old_x + dx, old_y + dy)
    if action is Action.WAIT:
        motion = Motion.WAITED
    elif walkable(grid, target):
        motion = Motion.MOVED
    else:
        motion = Motion.WALL_BLOCKED
        target = state.agent_position
    state.agent_position = target
    actual_delta = (target[0] - old_x, target[1] - old_y)
    tick = state.step_count + 1
    collision = None
    activated = False
    if target in state.monster_positions:
        collision = 1  # Keep actual old monster positions; no activation/monster phase.
    else:
        if target in scenario.pois and target not in state.activated_pois:
            state.activated_pois.add(target)
            activated = True
        state.monster_positions = monster_positions(scenario, tick)
        if target in state.monster_positions:
            collision = 2
    state.step_count = tick
    state.alive = collision is None
    state.end_reason = (
        EndReason.DEATH
        if not state.alive
        else EndReason.BUDGET
        if tick == scenario.horizon
        else EndReason.NONE
    )
    state.done = state.end_reason is not EndReason.NONE
    return PublicTransition(
        action,
        motion,
        actual_delta,
        activated=activated,
        died=not state.alive,
        end_reason=state.end_reason,
    ), collision
