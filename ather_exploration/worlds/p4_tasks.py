"""Reproducible threat lessons; hidden timing used only by scenario validation."""

from dataclasses import asdict, replace
from functools import lru_cache

import numpy as np

from ather_exploration.types import Scenario, ValidatorStatus
from ather_exploration.worlds.p3_tasks import p3_scenario, terrain_split
from ather_exploration.worlds.scenarios import digest
from ather_exploration.worlds.validation import validate_scenario

TASKS = ("P4a", "P4b", "P4c")


@lru_cache(maxsize=4096)
def _p4_candidate(task, seed):
    if task not in TASKS or type(seed) is not int or seed < 0:
        raise ValueError("Invalid threat task/seed")
    # Matched triples: identical terrain/spawn/POI, only patrol offset changes.
    rng = np.random.default_rng(seed // 3 if task == "P4a" else seed)
    if task == "P4a":
        n = int(rng.choice((9, 11, 13, 15)))
        x, y = int(rng.integers(3, n - 3)), n // 2
        grid = np.full((n, n), "#", dtype="<U1")
        grid[y, 1:-1] = "."
        grid[y - 1 : y + 2, x] = "."
        spawn = (x - int(rng.integers(1, min(x, 4))), y)
        poi = (x + int(rng.integers(1, min(n - x - 1, 4))), y)
        turns = int(rng.integers(4))

        def rotate(point):
            a, b = point
            for _ in range(turns):
                a, b = b, n - 1 - a
            return a, b

        scenario = Scenario(
            tuple("".join(r) for r in np.rot90(grid, turns)),
            rotate(spawn),
            (rotate(poi),),
            (tuple(rotate(p) for p in ((x, y - 1), (x, y), (x, y + 1))),),
            ((0, 2, 4)[seed % 3],),
            128,
            seed,
            skill_task=task,
        )
    else:
        base = p3_scenario("P3b" if task == "P4b" else "P3c", seed % 3_000_000)
        floor = {
            (x, y) for y, row in enumerate(base.terrain) for x, t in enumerate(row) if t == "."
        }
        # Cross the room-side approach perpendicular to the doorway. A patrol
        # along the entire narrow connector can seal it permanently under the
        # two collision checks; that is not a learnable timing task.
        connectors = [p for p in sorted(floor) if base.room_labels[p[1]][p[0]] < 0]
        choices = []
        for x, y in connectors:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cx, cy = x + dx, y + dy
                route = ((cx - dy, cy + dx), (cx, cy), (cx + dy, cy - dx))
                if (
                    all(p in floor and p != base.spawn for p in route)
                    and base.room_labels[cy][cx] >= 0
                    and route not in choices
                ):
                    choices.append(route)
        if not choices:
            raise ValueError("No valid connector patrol")
        rng.shuffle(choices)
        count = 1 if task == "P4b" else 1 + seed % 2
        routes = []
        for route in choices:
            if not any(set(route) & set(old) for old in routes):
                routes.append(route)
            if len(routes) == count:
                break
        if len(routes) != count:
            raise ValueError("Insufficient separate connector patrols")
        scenario = replace(
            base,
            routes=tuple(routes),
            phases=tuple(int(rng.integers(8)) for _ in routes),
            skill_task=task,
        )
    if validate_scenario(scenario, max_expansions=300000).status is not ValidatorStatus.VALIDATED:
        raise ValueError("Threat scenario lacks validated temporal solution")
    return scenario


@lru_cache(maxsize=4096)
def p4_scenario(task, seed):
    if task not in TASKS or type(seed) is not int or seed < 0:
        raise ValueError("Invalid threat task/seed")
    for attempt in range(32):
        try:
            return replace(_p4_candidate(task, seed + attempt * 300003), seed=seed)
        except ValueError:
            if task == "P4a":
                raise
    raise ValueError("No validated threat layout in bounded generation attempts")


def group(scenario):
    return {
        "size": len(scenario.terrain),
        "monster_count": len(scenario.routes),
        "timing_bucket": scenario.seed % 3,
        "poi_count": len(scenario.pois),
    }


def timing_geometry_identity(scenario):
    """Keep matched offsets and D4-equivalent crossing layouts in one split."""
    cells = np.array([[0 if c == "#" else 1 for c in row] for row in scenario.terrain])
    x, y = scenario.spawn
    cells[y, x] += 2
    for x, y in scenario.pois:
        cells[y, x] += 4
    for route in scenario.routes:
        for x, y in route:
            cells[y, x] += 8
    variants = [np.rot90(a, k).tolist() for a in (cells, np.fliplr(cells)) for k in range(4)]
    return min(digest(a) for a in variants)


@lru_cache(maxsize=32)
def p4_pool(task, count, validation=False):
    split = "validation" if validation else "train"
    rows = []
    seen = set()
    for seed in range(100000 if validation else 0, 200000 if validation else 100000):
        try:
            sc = p4_scenario(task, seed)
        except ValueError:
            continue
        payload = asdict(sc)
        payload.pop("seed")
        payload.pop("skill_task")
        identity = digest(payload)
        if task == "P4a":
            selected = (
                "validation" if int(timing_geometry_identity(sc)[:8], 16) % 5 == 0 else "train"
            )
        else:
            selected = terrain_split(sc.terrain)
        if selected != split or identity in seen:
            continue
        seen.add(identity)
        rows.append((seed, identity))
        if len(rows) == count:
            return tuple(rows)
    raise ValueError("Insufficient threat suite; no fallback")
