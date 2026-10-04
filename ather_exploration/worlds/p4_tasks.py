"""Reproducible threat lessons; hidden timing used only by scenario validation."""

from dataclasses import asdict, replace
from functools import lru_cache

import numpy as np

from ather_exploration.types import Scenario, ValidatorStatus
from ather_exploration.worlds.p3_tasks import p3_scenario, terrain_split
from ather_exploration.worlds.scenarios import digest
from ather_exploration.worlds.validation import validate_scenario

TASKS = ("P4a", "P4b", "P4c")
ENCOUNTERS = ("crossing", "yield_alcoves", "bypass")
CROSSING, APPROACH = "doorway_crossing", "room_approach"
P4C_PATTERNS = (
    (CROSSING,),
    (APPROACH,),
    (CROSSING, CROSSING),
    (APPROACH, APPROACH),
    (CROSSING, APPROACH),
)


def _encounter(seed: int, rng: np.random.Generator) -> Scenario:
    """Three navigational decisions, with a safe observation position at reset."""
    family = ENCOUNTERS[(seed // 3) % len(ENCOUNTERS)]
    n = int(rng.choice((13, 15, 17)))
    length = int(rng.integers(3, 6))
    y = n // 2
    grid = np.full((n, n), "#", dtype="<U1")
    grid[y, 1:-1] = "."
    if family == "yield_alcoves":
        x = int(rng.integers(3, n - length - 2))
        route = tuple((x + k, y) for k in range(length))
        # Separate pockets let the learner leave the shared lane and let the
        # monster pass. They are not a parallel corridor around the encounter.
        for k in range(length):
            grid[y + (1 if k % 2 else -1), x + k] = "."
        spawn = (x - 2, y)
        poi = (x + length + 1, y)
    else:
        x = int(rng.integers(4, n - 4))
        offset = int(rng.integers(1, length - 1))
        route = tuple((x, y + k) for k in range(-offset, length - offset))
        for a, b in route:
            grid[b, a] = "."
        spawn = (x - int(rng.integers(2, 4)), y)
        poi = (x + int(rng.integers(2, 4)), y)
        if family == "bypass":
            # A longer, completely safe path competes with the direct crossing.
            top = route[0][1] - 1
            grid[top, x - 2 : x + 3] = "."
            grid[top : y + 1, x - 2] = "."
            grid[top : y + 1, x + 2] = "."
    turns = int(rng.integers(4))
    reflect = bool(rng.integers(2))

    def transform(point):
        a, b = point
        if reflect:
            a = n - 1 - a
        for _ in range(turns):
            a, b = b, n - 1 - a
        return a, b

    transformed = np.fliplr(grid) if reflect else grid
    return Scenario(
        tuple("".join(row) for row in np.rot90(transformed, turns)),
        transform(spawn),
        (transform(poi),),
        (tuple(transform(p) for p in route),),
        ((0, 2, 4)[seed % 3],),
        128,
        seed,
        skill_task="P4a",
    )


@lru_cache(maxsize=4096)
def _p4_candidate(task, seed):
    if task not in TASKS or type(seed) is not int or seed < 0:
        raise ValueError("Invalid threat task/seed")
    # Matched triples: identical terrain/spawn/POI, only patrol offset changes.
    rng = np.random.default_rng(seed // 3 if task == "P4a" else seed)
    if task == "P4a":
        scenario = _encounter(seed, rng)
    else:
        base = p3_scenario("P3b" if task == "P4b" else "P3c", seed % 3_000_000)
        floor = {
            (x, y) for y, row in enumerate(base.terrain) for x, t in enumerate(row) if t == "."
        }
        # Cross the room-side approach perpendicular to the doorway. A patrol
        # along the entire narrow connector can seal it permanently under the
        # two collision checks; that is not a learnable timing task.
        connectors = [p for p in sorted(floor) if base.room_labels[p[1]][p[0]] < 0]
        pattern = (
            ((CROSSING,) if seed % 2 == 0 else (APPROACH,))
            if task == "P4b"
            else P4C_PATTERNS[seed % 5]
        )
        candidates = {}
        for family in dict.fromkeys(pattern):
            choices = []
            for x, y in connectors:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cx, cy = x + dx, y + dy
                    room = base.room_labels[cy][cx] if (cx, cy) in floor else -1
                    # Leave a room-side waiting cell behind the crossing. Every
                    # patrol cell stays inside this room, never along the connector.
                    refuge = (cx + dx, cy + dy)
                    if (
                        room < 0
                        or refuge not in floor
                        or base.room_labels[refuge[1]][refuge[0]] != room
                    ):
                        continue
                    if family == "room_approach":
                        # A patrol shares the direction of travel into the room;
                        # the learner can pass beside it or yield off the lane.
                        for length in (3, 4, 5):
                            route = tuple((cx + dx * k, cy + dy * k) for k in range(1, length + 1))
                            if (
                                all(
                                    p in floor
                                    and p != base.spawn
                                    and base.room_labels[p[1]][p[0]] == room
                                    and any(
                                        (p[0] + dy * side, p[1] - dx * side) in floor
                                        for side in (-1, 1)
                                    )
                                    for p in route
                                )
                                and route not in choices
                            ):
                                choices.append(route)
                        continue
                    for length in (3, 4, 5):
                        for offset in range(1, length - 1):
                            route = tuple(
                                (cx - dy * k, cy + dx * k) for k in range(-offset, length - offset)
                            )
                            if (
                                all(
                                    p in floor
                                    and p != base.spawn
                                    and base.room_labels[p[1]][p[0]] == room
                                    for p in route
                                )
                                and route not in choices
                            ):
                                choices.append(route)
            choices = [route for route in choices if _route_family(base, route) == family]
            rng.shuffle(choices)
            candidates[family] = choices
        routes = []
        for family in pattern:
            for route in candidates[family]:
                if not any(
                    abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1
                    for old in routes
                    for a in route
                    for b in old
                ):
                    routes.append(route)
                    break
            else:
                raise ValueError("Insufficient separate patrols for encounter pattern")
        scenario = replace(
            base,
            routes=tuple(routes),
            phases=tuple(int(rng.integers(4 * (len(route) - 1))) for route in routes),
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
            return replace(_p4_candidate(task, seed + attempt * 6000060), seed=seed)
        except ValueError:
            if task == "P4a":
                raise
    raise ValueError("No validated threat layout in bounded generation attempts")


def _route_family(scenario, route):
    """Classify actual geometry, including layouts produced by bounded retries."""
    dx, dy = route[1][0] - route[0][0], route[1][1] - route[0][1]
    for x, y in route:
        for side in (-1, 1):
            a, b = x + dy * side, y - dx * side
            if (
                0 <= b < len(scenario.terrain)
                and 0 <= a < len(scenario.terrain[b])
                and scenario.terrain[b][a] == "."
                and scenario.room_labels[b][a] < 0
            ):
                return "doorway_crossing"
    return "room_approach"


def group(scenario):
    families = (
        tuple(_route_family(scenario, route) for route in scenario.routes)
        if scenario.room_labels
        else ()
    )
    return {
        "encounter_family": (
            ENCOUNTERS[(scenario.seed // 3) % len(ENCOUNTERS)]
            if scenario.skill_task == "P4a"
            else families[0]
            if len(set(families)) == 1
            else "mixed"
        ),
        "patrol_families": families,
        "patrol_lengths": tuple(len(route) for route in scenario.routes),
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
    strata = (
        [(ENCOUNTERS[i % 3], 3 + (i // 3) % 3, (i // 9 + i % 3 + i // 3) % 3) for i in range(27)]
        if task == "P4a"
        else [
            (family, count)
            for count in ((1,) if task == "P4b" else (1, 2))
            for family in ("doorway_crossing", "room_approach")
        ]
        + ([("mixed", 2)] if task == "P4c" else [])
    )
    buckets = {key: [] for key in strata}
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
        key = (
            (group(sc)["encounter_family"], len(sc.routes[0]), seed % 3)
            if task == "P4a"
            else (group(sc)["encounter_family"], len(sc.routes))
        )
        quota = (count + len(strata) - 1 - strata.index(key)) // len(strata)
        if len(buckets[key]) >= quota:
            continue
        seen.add(identity)
        buckets[key].append((seed, identity))
        if sum(map(len, buckets.values())) == count:
            # Interleave strata so a diagnostic prefix also covers the families.
            return tuple(
                item
                for i in range(max(map(len, buckets.values())))
                for bucket in buckets.values()
                for item in bucket[i : i + 1]
            )
    raise ValueError("Insufficient threat suite; no fallback")
