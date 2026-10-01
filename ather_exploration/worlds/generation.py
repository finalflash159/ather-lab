"""Bounded BSP generation, disjoint patrols and validated stratified starts."""

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from ather_exploration.config import EnvConfig, config_hash
from ather_exploration.environment.dynamics import advance, make_grid
from ather_exploration.environment.visibility import sense
from ather_exploration.seeds import stage_rng
from ather_exploration.types import EpisodeState, Scenario, ValidatorStatus
from ather_exploration.worlds.scenarios import (
    digest,
    geometry_hash,
    implementation_id,
    read_record,
    scenario_from_dict,
    scenario_hash,
    world_hash,
    write_record,
)
from ather_exploration.worlds.topology import distances, floor_cells, navigation_graph, neighbors
from ather_exploration.worlds.validation import ValidationResult, replay_witness, validate_scenario

STRATA = ("room_quiet", "room_threat", "corridor_quiet", "corridor_threat")


class GenerationError(ValueError):
    def __init__(self, reason, stats):
        self.stats = dict(stats)
        super().__init__(f"{reason}; diagnostics={self.stats}")


@dataclass(frozen=True)
class GeneratedScenario:
    scenario: Scenario
    validation: ValidationResult
    config: EnvConfig
    stratum: str
    diagnostics: dict

    def payload(self):
        return {
            "scenario": asdict(self.scenario),
            "validation": asdict(self.validation),
            "config": self.config.model_dump(mode="json"),
            "stratum": self.stratum,
            "diagnostics": self.diagnostics,
            "geometry_hash": geometry_hash(self.scenario.terrain),
            "world_hash": world_hash(self.scenario),
        }


def _geometry(config, rng):
    w, h = config.width, config.height
    minimum = config.room_side_min + 2
    terrain = [["#"] * w for _ in range(h)]
    labels = [[-1] * w for _ in range(h)]
    rooms, edges = [], []

    def capacity(width, height):
        return (width // minimum) * (height // minimum)

    def split(x, y, width, height, count):
        if count == 1:
            rw = int(rng.integers(config.room_side_min, min(config.room_side_max, width - 2) + 1))
            rh = int(rng.integers(config.room_side_min, min(config.room_side_max, height - 2) + 1))
            rx = int(rng.integers(x + 1, x + width - rw))
            ry = int(rng.integers(y + 1, y + height - rh))
            room_id = len(rooms)
            rooms.append((rx, ry, rw, rh))
            for yy in range(ry, ry + rh):
                for xx in range(rx, rx + rw):
                    terrain[yy][xx], labels[yy][xx] = ".", room_id
            return [room_id]
        choices = []
        for axis, extent in ((0, width), (1, height)):
            for cut in range(minimum, extent - minimum + 1):
                a = capacity(cut, height) if axis == 0 else capacity(width, cut)
                b = capacity(width - cut, height) if axis == 0 else capacity(width, height - cut)
                for first in range(1, count):
                    if first <= a and count - first <= b:
                        choices.append((axis, cut, first))
        if not choices:
            raise ValueError("BSP capacity cannot fit exact room count")
        preferred = (
            0
            if width / height > config.bsp_aspect_bias
            else 1
            if height / width > config.bsp_aspect_bias
            else None
        )
        biased = [choice for choice in choices if choice[0] == preferred]
        axis, cut, first = (biased or choices)[int(rng.integers(len(biased or choices)))]
        if axis == 0:
            left = split(x, y, cut, height, first)
            right = split(x + cut, y, width - cut, height, count - first)
        else:
            left = split(x, y, width, cut, first)
            right = split(x, y + cut, width, height - cut, count - first)
        edges.append((int(rng.choice(left)), int(rng.choice(right))))
        return left + right

    # Leaves cover the full extent: their one-cell margin is also the outer wall.
    split(0, 0, w, h, config.num_rooms)
    used = {tuple(sorted(edge)) for edge in edges}
    options = [
        (a, b) for a in range(len(rooms)) for b in range(a + 1, len(rooms)) if (a, b) not in used
    ]
    extra = int(rng.integers(config.extra_connections_min, config.extra_connections_max + 1))
    if extra > len(options):
        raise ValueError("Not enough distinct room pairs for extra connections")
    rng.shuffle(options)
    edges.extend(options[:extra])
    for a, b in edges:
        ax, ay, aw, ah = rooms[a]
        bx, by, bw, bh = rooms[b]
        x, y, tx, ty = ax + aw // 2, ay + ah // 2, bx + bw // 2, by + bh // 2
        axes = (0, 1) if rng.integers(2) else (1, 0)
        for axis in axes:
            while (x if axis == 0 else y) != (tx if axis == 0 else ty):
                terrain[y][x] = "."
                if axis == 0:
                    x += 1 if tx > x else -1
                else:
                    y += 1 if ty > y else -1
            terrain[y][x] = "."
    terrain = tuple("".join(row) for row in terrain)
    labels = tuple(tuple(row) for row in labels)
    if any(
        all(terrain[yy][xx] == "." and labels[yy][xx] < 0 for yy in (y, y + 1) for xx in (x, x + 1))
        for y in range(h - 1)
        for x in range(w - 1)
    ):
        raise ValueError("Parallel touching corridors create a 2x2 floor patch")
    topology = navigation_graph(terrain, labels)
    if topology["cycles"] < 1 or topology["dead_ends"] < 1:
        raise ValueError("Rasterized graph needs a cycle and a dead end")
    return terrain, labels, tuple(edges), {"topology": topology, "rooms": rooms}


def _placement(terrain, config, poi_rng, route_rng):
    floors = floor_cells(terrain)
    available = list(floors)
    poi_rng.shuffle(available)
    pois = []
    for pos in available:
        if all(
            distances(terrain, [poi]).get(pos, 0) >= config.min_poi_geodesic_distance
            for poi in pois
        ):
            pois.append(pos)
            if len(pois) == config.num_pois:
                break
    if len(pois) != config.num_pois:
        raise ValueError("POI separation")
    occupied, routes = set(), []
    expansions = 0
    floor_set = set(floors)
    rng = route_rng
    for _ in range(config.num_monsters):
        length = int(rng.integers(config.route_length_min, config.route_length_max + 1))
        starts = list(floors)
        rng.shuffle(starts)

        def search(path, used, length=length):
            nonlocal expansions
            if len(path) == length:
                return tuple(path)
            expansions += 1
            if expansions > config.budgets.route_expansions:
                raise ValueError("route expansion budget")
            candidates = [
                p for p in neighbors(path[-1], floor_set) if p not in occupied and p not in used
            ]
            rng.shuffle(candidates)
            for pos in candidates:
                found = search([*path, pos], used | {pos})
                if found:
                    return found
            return None

        route = None
        for start in starts:
            if start not in occupied:
                route = search([start], {start})
                if route:
                    break
        if route is None:
            raise ValueError("disjoint route unavailable")
        routes.append(route)
        occupied.update(route)
    if any(
        not any(n not in occupied for p in route for n in neighbors(p, floor_set))
        for route in routes
    ):
        raise ValueError("patrol has no outside refuge adjacency")
    return tuple(pois), tuple(routes)


def spawn_diagnostics(scenario, config):
    """Privileged admission diagnostics; never an action mask for the agent."""
    grid = make_grid(scenario)
    state = EpisodeState.from_scenario(scenario)
    local = sense(grid, scenario, state, config.observation.radius)
    if local[3].any():
        raise ValueError("POI visible at reset")
    if scenario.spawn in scenario.pois:
        raise ValueError("spawn is a POI")
    d = distances(scenario.terrain, [scenario.spawn])
    if min(d.get(p, 0) for p in scenario.pois) < config.min_spawn_poi_distance:
        raise ValueError("spawn too close to POI")
    if min(d.get(p, 0) for p in state.monster_positions) < config.min_spawn_monster_distance:
        raise ValueError("spawn too close to monster")
    alive = 0
    for action in range(5):
        trial = EpisodeState.from_scenario(scenario)
        advance(grid, scenario, trial, action)
        alive += trial.alive
    if not alive:
        raise ValueError("all first actions die")
    room = scenario.room_labels[scenario.spawn[1]][scenario.spawn[0]] >= 0
    stratum = STRATA[(0 if room else 2) + int(local[5].any())]
    # Public conservative certificate: target must be observed floor, not adjacent
    # to ANY current visible monster; unknown target cannot be certified safe.
    radius = config.observation.radius
    safe = 0
    from ather_exploration.types import ACTION_DELTAS

    monsters = [(int(c) - radius, int(r) - radius) for r, c in np.argwhere(local[5])]
    for dx, dy in ACTION_DELTAS:
        row, col = radius + dy, radius + dx
        if local[0, row, col] and local[1, row, col]:
            dx = dy = 0
            row = col = radius
        if (
            local[0, row, col]
            and local[2, row, col]
            and all(abs(dx - mx) + abs(dy - my) > 1 for mx, my in monsters)
        ):
            # A hidden monster one move away from target would also need to be
            # visible to certify safety. Require all walkable neighbors observed.
            known = True
            for nx, ny in ACTION_DELTAS[:4]:
                rr, cc = row + ny, col + nx
                if not (0 <= rr < local.shape[1] and 0 <= cc < local.shape[2] and local[0, rr, cc]):
                    known = False
            safe += known
    px, py = scenario.spawn
    degree = sum((px + dx, py + dy) in d for dx, dy in ACTION_DELTAS[:4])
    return {
        "spawn_position_label": "room" if room else "corridor",
        "spawn_floor_degree": degree,
        "spawn_cell_dead_end": degree == 1,
        "spawn_cell_junction": degree >= 3,
        "spawn_poi_distances": [d[p] for p in scenario.pois],
        "spawn_monster_distances": [d[p] for p in state.monster_positions],
        "poi_position_labels": [
            "room" if scenario.room_labels[y][x] >= 0 else "corridor" for x, y in scenario.pois
        ],
        "route_lengths": [len(route) for route in scenario.routes],
        "stratum": stratum,
        "public_safe_action_count": int(safe),
        "actual_surviving_first_actions": alive,
    }


def generate_scenario(
    config: EnvConfig, seed: int, *, stratum=None, cache=None, timings=None
) -> GeneratedScenario:
    """Generate deterministically; optional timing sink is outside semantic records."""
    started = perf_counter()
    measured = {"validation_seconds": 0.0, "cache_hit": False}
    try:
        return _generate_scenario(config, seed, stratum=stratum, cache=cache, timings=measured)
    finally:
        measured["total_seconds"] = perf_counter() - started
        if timings is not None:
            timings.update(measured)


def _generate_scenario(config, seed, *, stratum=None, cache=None, timings):
    revision = implementation_id()
    identity = config_hash(config, source_revision=revision)
    # Validate seed before cache access and before selecting a stratum.
    chooser = stage_rng(seed, "spawn-stratum")
    if stratum is None:
        stratum = STRATA[int(chooser.choice(4, p=config.spawn_weights))]
    if stratum not in STRATA:
        raise ValueError(f"Unknown spawn stratum: {stratum}")
    key = digest([identity, seed, stratum])
    path = Path(cache) / f"{key}.json" if cache is not None else None
    if path is not None and path.exists():
        result = load_generated(path)
        timings["cache_hit"] = True
        if (
            result.scenario.seed != seed
            or result.stratum != stratum
            or result.scenario.config_hash != identity
        ):
            raise ValueError("Cache identity mismatch")
        return result
    stats = Counter()
    budget = config.budgets
    for gi in range(budget.geometry_attempts):
        stats["geometry_attempts"] += 1
        try:
            terrain, labels, edges, geometry = _geometry(config, stage_rng(seed, "geometry", gi))
        except ValueError as error:
            stats[str(error)] += 1
            continue
        for pi in range(budget.placement_attempts):
            stats["placement_attempts"] += 1
            try:
                pois, routes = _placement(
                    terrain,
                    config,
                    stage_rng(seed, "poi", gi, pi),
                    stage_rng(seed, "route", gi, pi),
                )
            except ValueError as error:
                stats[str(error)] += 1
                continue
            for fi in range(budget.phase_attempts):
                stats["phase_attempts"] += 1
                rng = stage_rng(seed, "phase", gi, pi, fi)
                phases = tuple(
                    int(rng.integers(config.patrol_period * 2 * (len(route) - 1)))
                    for route in routes
                )
                candidates = [
                    p
                    for p in floor_cells(terrain)
                    if (labels[p[1]][p[0]] >= 0) == stratum.startswith("room")
                ]
                stage_rng(seed, "spawn", gi, pi, fi).shuffle(candidates)
                for spawn in candidates[: budget.spawn_attempts]:
                    stats["spawn_attempts"] += 1
                    scenario = Scenario(
                        terrain,
                        spawn,
                        pois,
                        routes,
                        phases,
                        config.horizon,
                        seed,
                        patrol_period=config.patrol_period,
                        config_hash=identity,
                        source_revision=revision,
                        room_labels=labels,
                        topology_edges=edges,
                    )
                    try:
                        diagnostic = spawn_diagnostics(scenario, config)
                    except ValueError as error:
                        stats[str(error)] += 1
                        continue
                    if diagnostic["stratum"] != stratum:
                        stats["stratum mismatch"] += 1
                        continue
                    if stats["validator_calls"] >= budget.validator_calls_per_reset:
                        raise GenerationError("global validator call budget exhausted", stats)
                    stats["validator_calls"] += 1
                    validation_started = perf_counter()
                    validation = validate_scenario(
                        scenario, max_expansions=budget.validator_expansions
                    )
                    timings["validation_seconds"] += perf_counter() - validation_started
                    if validation.status is not ValidatorStatus.VALIDATED:
                        stats[validation.status.value] += 1
                        continue
                    result = GeneratedScenario(
                        scenario,
                        validation,
                        config,
                        stratum,
                        {**dict(stats), **geometry, **diagnostic},
                    )
                    if path is not None:
                        write_record(path, result.payload())
                    return result
    raise GenerationError(f"bounded generation exhausted for {stratum}", stats)


def load_generated(path):
    payload = read_record(path)
    config = EnvConfig.model_validate(payload["config"])
    scenario = scenario_from_dict(payload["scenario"])
    if scenario.source_revision != implementation_id() or scenario.config_hash != config_hash(
        config, source_revision=implementation_id()
    ):
        raise ValueError("Stale scenario: code/config identity mismatch; regenerate explicitly")
    value = payload["validation"]
    validation = ValidationResult(
        ValidatorStatus(value["status"]),
        tuple(value["actions"]),
        value["expansions"],
        value["reason"],
        value["scenario_hash"],
    )
    result = GeneratedScenario(
        scenario, validation, config, payload["stratum"], payload["diagnostics"]
    )
    check_generated(result)
    if payload["geometry_hash"] != geometry_hash(scenario.terrain) or payload[
        "world_hash"
    ] != world_hash(scenario):
        raise ValueError("Scenario hash mismatch")
    return result


def check_generated(result: GeneratedScenario) -> None:
    scenario, config, validation = result.scenario, result.config, result.validation
    if (
        scenario.fixture_only
        or validation.status is not ValidatorStatus.VALIDATED
        or validation.scenario_hash != scenario_hash(scenario)
    ):
        raise ValueError("Main-task admission requires a matching validated certificate")
    if (
        len(scenario.terrain[0]),
        len(scenario.terrain),
        len(scenario.pois),
        len(scenario.routes),
        scenario.horizon,
        scenario.patrol_period,
    ) != (
        config.width,
        config.height,
        config.num_pois,
        config.num_monsters,
        config.horizon,
        config.patrol_period,
    ):
        raise ValueError("Scenario disagrees with config")
    if not scenario.room_labels:
        raise ValueError("Missing room labels")
    if scenario.source_revision != implementation_id() or scenario.config_hash != config_hash(
        config, source_revision=implementation_id()
    ):
        raise ValueError("Stale scenario: code/config identity mismatch")
    labels = scenario.room_labels
    for y, row in enumerate(labels):
        for x, label in enumerate(row):
            if (
                label < -1
                or label >= config.num_rooms
                or (label >= 0 and scenario.terrain[y][x] != ".")
            ):
                raise ValueError("Invalid room label")
    for room_id in range(config.num_rooms):
        cells = [
            (x, y)
            for y, row in enumerate(labels)
            for x, label in enumerate(row)
            if label == room_id
        ]
        if not cells:
            raise ValueError("Missing room")
        xs, ys = zip(*cells, strict=True)
        width, height = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        if len(cells) != width * height or not all(
            config.room_side_min <= side <= config.room_side_max for side in (width, height)
        ):
            raise ValueError("Room bounds disagree with config")
    if any(
        all(
            scenario.terrain[yy][xx] == "." and labels[yy][xx] < 0
            for yy in (y, y + 1)
            for xx in (x, x + 1)
        )
        for y in range(config.height - 1)
        for x in range(config.width - 1)
    ):
        raise ValueError("2x2 corridor patch is not a meaningful loop")
    topology = navigation_graph(scenario.terrain, scenario.room_labels)
    if (
        topology["room_count"] != config.num_rooms
        or topology["cycles"] < 1
        or topology["dead_ends"] < 1
    ):
        raise ValueError("Invalid actual navigation topology")
    for poi in scenario.pois:
        d = distances(scenario.terrain, [poi])
        if any(d[p] < config.min_poi_geodesic_distance for p in scenario.pois if p != poi):
            raise ValueError("POI separation invalid")
    union = set().union(*map(set, scenario.routes))
    floors = set(floor_cells(scenario.terrain))
    for route in scenario.routes:
        if not config.route_length_min <= len(route) <= config.route_length_max:
            raise ValueError("Invalid route length")
        if not any(n not in union for p in route for n in neighbors(p, floors)):
            raise ValueError("Missing route refuge")
    if spawn_diagnostics(scenario, config)["stratum"] != result.stratum:
        raise ValueError("Spawn stratum mismatch")
    replay_witness(scenario, validation.actions)
