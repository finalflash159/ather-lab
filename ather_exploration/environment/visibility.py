"""360-degree disk sensor with exact closed-wall occlusion on a MiniGrid Grid."""

from fractions import Fraction
from functools import lru_cache

import numpy as np

from ather_exploration.types import EpisodeState, Scenario
from minigrid.core.grid import Grid


def _touches_closed_cell(dx: int, dy: int, bx: int, by: int) -> bool:
    # Slab clipping of p(t)=(2*dx*t,2*dy*t), 0<=t<=1, against the closed cell.
    low, high = Fraction(0), Fraction(1)
    for delta, center in ((dx, bx), (dy, by)):
        left, right = 2 * center - 1, 2 * center + 1
        if delta == 0:
            if not left <= 0 <= right:
                return False
        else:
            a, b = Fraction(left, 2 * delta), Fraction(right, 2 * delta)
            low, high = max(low, min(a, b)), min(high, max(a, b))
            if low > high:
                return False
    return True


@lru_cache(maxsize=8192)
def blocker_offsets(dx: int, dy: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (bx, by)
        for by in range(min(0, dy), max(0, dy) + 1)
        for bx in range(min(0, dx), max(0, dx) + 1)
        if (bx, by) not in ((0, 0), (dx, dy)) and _touches_closed_cell(dx, dy, bx, by)
    )


@lru_cache(maxsize=40)
def disk_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    if type(radius) is not int or not 1 <= radius <= 40:
        raise ValueError("radius must be integer 1..40")
    return tuple(
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    )


def visibility_mask(grid: Grid, position: tuple[int, int], radius: int) -> np.ndarray:
    offsets = disk_offsets(radius)
    mask = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=bool)
    x, y = position
    for dx, dy in offsets:
        tx, ty = x + dx, y + dy
        if not (0 <= tx < grid.width and 0 <= ty < grid.height):
            continue
        blocked = False
        for bx, by in blocker_offsets(dx, dy):
            cell = grid.get(x + bx, y + by)
            if cell is not None and cell.type == "wall":
                blocked = True
                break
        mask[dy + radius, dx + radius] = not blocked
    return mask


def sense(grid: Grid, scenario: Scenario, state: EpisodeState, radius: int) -> np.ndarray:
    """Ground truth is used only here to simulate sensing; returns public local channels."""
    mask = visibility_mask(grid, state.agent_position, radius)
    local = np.zeros((6, 2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    local[0] = mask
    x, y = state.agent_position
    pois, monsters = set(scenario.pois), set(state.monster_positions)
    for row, col in np.argwhere(mask):
        position = (x + int(col) - radius, y + int(row) - radius)
        cell = grid.get(*position)
        wall = cell is not None and cell.type == "wall"
        local[1, row, col] = wall
        local[2, row, col] = not wall
        if position in pois:
            local[4 if position in state.activated_pois else 3, row, col] = 1
        local[5, row, col] = position in monsters
    return local
