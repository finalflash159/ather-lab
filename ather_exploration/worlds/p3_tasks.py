"""Multi-room lessons; terrain-only canonical splits shared by all P3 tasks."""

from functools import lru_cache
from itertools import permutations

import numpy as np

from ather_exploration.types import EpisodeState, Scenario
from ather_exploration.worlds.scenarios import digest
from ather_exploration.worlds.topology import distances

SIZES = {"P3a": (11, 13, 15), "P3b": (13, 15, 17, 19), "P3c": (15, 17, 19, 21)}
OOD_SIZES = {"P3a": (12, 14), "P3b": (14, 16, 18), "P3c": (16, 18, 20)}


def terrain_identity(terrain):
    """Rotation/reflection cannot move the same floor plan between splits."""
    grid = np.array([list(row) for row in terrain])
    variants = [
        tuple("".join(row) for row in np.rot90(g, k))
        for g in (grid, np.fliplr(grid))
        for k in range(4)
    ]
    return digest(min(variants))


def terrain_split(terrain):
    bucket = int(terrain_identity(terrain)[:8], 16) % 10
    return "test" if bucket == 0 else "validation" if bucket in (1, 2) else "train"


def _layout(task, rng, n):
    gap = int(rng.integers(1, 3)) if task == "P3c" else 1
    x = int(rng.integers(4, n - gap - 3))
    y = int(rng.integers(4, n - gap - 3))
    left, right = (1, x), (x + gap, n - 1)
    top, bottom = (1, y), (y + gap, n - 1)
    if task == "P3a":
        rooms = [(1, 1, x, n - 1), (x + gap, 1, n - 1, n - 1)]
        links = [(0, 1)]
        topology = "two_rooms"
    elif task == "P3b":
        rooms = [(1, 1, x, y), (1, y + gap, x, n - 1), (x + gap, 1, n - 1, n - 1)]
        links = [(0, 1), (int(rng.integers(2)), 2)]
        topology = "three_rooms"
    else:
        rooms = [(a, c, b, d) for c, d in (top, bottom) for a, b in (left, right)]
        edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
        if rng.random() < 0.5:
            edges.pop(int(rng.integers(4)))
            topology = "chain"
        else:
            topology = "cycle"
        links = edges
    # Independently vary outer room boundaries; keep shared doorway ranges overlapping.
    varied = []
    for a, c, b, d in rooms:
        margin = max(1, (d - c - 2) // 2) if task == "P3a" else 1
        varied.append((a, c + int(rng.integers(margin)), b, d - int(rng.integers(margin))))
    rooms = varied
    grid = np.full((n, n), "#", dtype="<U1")
    room_cells = []
    for a, c, b, d in rooms:
        grid[c:d, a:b] = "."
        room_cells.append([(i, j) for j in range(c, d) for i in range(a, b)])
    for u, v in links:
        a, c, b, d = rooms[u]
        e, g, f, h = rooms[v]
        if b <= e or f <= a:
            row = int(rng.integers(max(c, g), min(d, h)))
            grid[row, min(b, f) - 1 : max(a, e) + 1] = "."
        else:
            col = int(rng.integers(max(a, e), min(b, f)))
            grid[min(d, h) - 1 : max(c, g) + 1, col] = "."
    turns = int(rng.integers(4))

    def rotate(point):
        x, y = point
        for _ in range(turns):
            x, y = y, n - 1 - x
        return x, y

    terrain = tuple("".join(row) for row in np.rot90(grid, turns))
    return terrain, [[rotate(p) for p in room] for room in room_cells], topology


@lru_cache(maxsize=8192)
def p3_scenario(task, seed):
    from ather_exploration.environment.dynamics import make_grid
    from ather_exploration.environment.visibility import sense

    if task not in SIZES or type(seed) is not int or seed < 0:
        raise ValueError("Invalid P3 task/seed")
    rng = np.random.default_rng(seed)
    sizes = OOD_SIZES[task] if seed >= 3_000_000 else SIZES[task]
    n = int(rng.choice(sizes))
    terrain, rooms, _ = _layout(task, rng, n)
    room_labels = np.full((n, n), -1, dtype=np.int16)
    for room_id, cells in enumerate(rooms):
        for x, y in cells:
            room_labels[y, x] = room_id
    room_labels = tuple(tuple(int(label) for label in row) for row in room_labels)
    k = 1 if task == "P3a" else 2 if task == "P3b" else 1 + seed % 2
    for _ in range(2000):
        order = rng.permutation(len(rooms))
        spawn_room = rooms[int(order[0])]
        spawn = spawn_room[int(rng.integers(len(spawn_room)))]
        pois = tuple(rooms[int(r)][int(rng.integers(len(rooms[int(r)])))] for r in order[1 : k + 1])
        scenario = Scenario(
            terrain,
            spawn,
            pois,
            (),
            (),
            256,
            seed,
            skill_task=task,
            room_labels=room_labels,
        )
        local = sense(make_grid(scenario), scenario, EpisodeState.from_scenario(scenario), 4)
        if not local[3].any() and oracle_tour(scenario) <= 192:
            return scenario
    raise ValueError(f"Could not place hidden P3 POIs: {task}/{seed}")


def room_coverage_fractions(scenario, memory):
    """Fraction of each generated room revealed in the public map memory."""
    labels = np.asarray(scenario.room_labels, dtype=np.int16)
    if labels.shape != (len(scenario.terrain), len(scenario.terrain[0])):
        raise ValueError("P3 scenario is missing room labels")
    room_ids = sorted(int(value) for value in np.unique(labels) if value >= 0)
    if not room_ids:
        raise ValueError("P3 scenario has no labeled rooms")
    if memory.ndim != 3 or memory.shape[0] < 3:
        raise ValueError("Expected public memory with walkable-terrain channel")

    center = memory.shape[-1] // 2
    sx, sy = scenario.spawn
    seen_floor = memory[2] > 0
    coverages = []
    for room_id in room_ids:
        ys, xs = np.where(labels == room_id)
        rows = center + ys - sy
        cols = center + xs - sx
        if (
            np.any(rows < 0)
            or np.any(cols < 0)
            or np.any(rows >= memory.shape[1])
            or np.any(cols >= memory.shape[2])
        ):
            raise ValueError("P3 room falls outside the public memory map")
        coverages.append(float(seen_floor[rows, cols].mean()))
    return np.asarray(coverages, dtype=np.float64)


def room_exploration_potential(coverages):
    """Bounded room-balanced potential with larger marginal value for sparse rooms."""
    coverages = np.asarray(coverages, dtype=np.float64)
    if coverages.ndim != 1 or not len(coverages):
        raise ValueError("Room coverage must be a nonempty vector")
    if not np.isfinite(coverages).all() or np.any(coverages < 0) or np.any(coverages > 1):
        raise ValueError("Room coverage values must be finite fractions")
    return float(np.mean(2 * coverages - coverages**2))


def oracle_tour(scenario):
    """Evaluator/generator only: shortest open tour through every POI."""
    nodes = (scenario.spawn, *scenario.pois)
    ds = {p: distances(scenario.terrain, [p]) for p in nodes}
    return min(
        sum(ds[a][b] for a, b in zip((scenario.spawn, *order), order))
        for order in permutations(scenario.pois)
    )


def map_group(scenario):
    rng = np.random.default_rng(scenario.seed)
    sizes = (
        OOD_SIZES[scenario.skill_task] if scenario.seed >= 3_000_000 else SIZES[scenario.skill_task]
    )
    n = int(rng.choice(sizes))
    _, _, topology = _layout(scenario.skill_task, rng, n)
    return {"topology": topology, "size": n, "poi_count": len(scenario.pois)}


@lru_cache(maxsize=64)
def p3_pool(task, count, split="train"):
    if split not in ("train", "validation", "test", "ood"):
        raise ValueError("Unknown P3 split")
    start = {"train": 0, "validation": 100000, "test": 200000, "ood": 3000000}[split]
    sizes = OOD_SIZES[task] if split == "ood" else SIZES[task]
    topologies = (
        ("chain", "cycle") if task == "P3c" else ("two_rooms" if task == "P3a" else "three_rooms",)
    )
    counts = (1, 2) if task == "P3c" else (1 if task == "P3a" else 2,)
    groups = [(n, t, k) for n in sizes for t in topologies for k in counts]
    buckets = {g: [] for g in groups}
    seen = set()
    for seed in range(start, start + 100000):
        sc = p3_scenario(task, seed)
        if terrain_split(sc.terrain) != ("validation" if split == "ood" else split):
            continue
        identity = terrain_identity(sc.terrain)
        if identity in seen:
            continue
        info = map_group(sc)
        group = info["size"], info["topology"], info["poi_count"]
        needed = (count + len(groups) - 1 - groups.index(group)) // len(groups)
        if len(buckets[group]) < needed:
            buckets[group].append((seed, identity))
            seen.add(identity)
        if sum(map(len, buckets.values())) == count:
            return tuple(item for group in groups for item in buckets[group])
    raise ValueError(f"Insufficient unique P3 terrain: {task}/{split}/{count}")
