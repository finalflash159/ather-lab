"""Diverse patrol geometry, honest public starts, and split isolation; no learning."""

from collections import Counter
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from ather_exploration.environment.dynamics import advance, make_grid
from ather_exploration.types import Action, EpisodeState, ValidatorStatus
from ather_exploration.worlds.p4_tasks import (
    ENCOUNTERS,
    group,
    p4_pool,
    p4_scenario,
    timing_geometry_identity,
)
from ather_exploration.worlds.validation import replay_witness, validate_scenario


@pytest.mark.parametrize("task", ("P4a", "P4b", "P4c"))
def test_full_diverse_pools_safe_starts_and_runtime_witnesses(task):
    pools = [p4_pool(task, 256), p4_pool(task, 64, True)]
    identities = []
    for pool in pools:
        families = Counter()
        lengths = set()
        counts = set()
        strata = Counter()
        split_ids = set()
        for seed, _ in pool:
            scenario = p4_scenario(task, seed)
            diagnostic = group(scenario)
            families[diagnostic["encounter_family"]] += 1
            lengths.update(diagnostic["patrol_lengths"])
            counts.add(len(scenario.routes))
            strata[(diagnostic["encounter_family"], len(scenario.routes))] += 1
            assert diagnostic["timing_bucket"] == seed % 3
            assert scenario.spawn not in {p for route in scenario.routes for p in route}
            # The learner can observe a complete patrol cycle at reset without
            # a guessed escape action. This does not assume hidden phase access.
            state = EpisodeState.from_scenario(scenario)
            grid = make_grid(scenario)
            for _ in range(16):
                event, _ = advance(grid, scenario, state, Action.WAIT)
                assert not event.died
            for route in scenario.routes:
                assert len(route) in (3, 4, 5)
                deltas = {(b[0] - a[0], b[1] - a[1]) for a, b in pairwise(route)}
                assert len(deltas) == 1
            result = validate_scenario(scenario, max_expansions=300000)
            assert result.status is ValidatorStatus.VALIDATED
            replay_witness(scenario, result.actions)
            split_ids.add(timing_geometry_identity(scenario))
            if task != "P4a":
                assert set(diagnostic["patrol_families"]) == (
                    {"doorway_crossing", "room_approach"}
                    if diagnostic["encounter_family"] == "mixed"
                    else {diagnostic["encounter_family"]}
                )
                # No patrol occupies a connector; every patrol has off-lane
                # room floor available rather than sealing a one-cell hallway.
                occupied = {p for route in scenario.routes for p in route}
                for route in scenario.routes:
                    room_ids = {scenario.room_labels[y][x] for x, y in route}
                    assert len(room_ids) == 1 and min(room_ids) >= 0
                    assert any(
                        (x + dx, y + dy) not in occupied and scenario.terrain[y + dy][x + dx] == "."
                        for x, y in route
                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    )
        assert lengths == {3, 4, 5}
        balanced = strata if task == "P4c" else families
        assert max(balanced.values()) - min(balanced.values()) <= 1
        assert set(families) == (
            set(ENCOUNTERS)
            if task == "P4a"
            else {"doorway_crossing", "room_approach", "mixed"}
            if task == "P4c"
            else {"doorway_crossing", "room_approach"}
        )
        if task == "P4c":
            assert counts == {1, 2}
        identities.append(split_ids)
    assert not identities[0] & identities[1]


def test_matched_timing_triples_and_dihedral_identity():
    for base in range(0, 81, 3):
        triple = [p4_scenario("P4a", base + i) for i in range(3)]
        assert len({timing_geometry_identity(sc) for sc in triple}) == 1
        assert {sc.phases for sc in triple} == {(0,), (2,), (4,)}
        assert len({replace(sc, seed=0, phases=(0,)) for sc in triple}) == 1
        sc = triple[0]
        n = len(sc.terrain)
        for mirror in (False, True):
            for turns in range(4):

                def transform(point, mirror=mirror, n=n, turns=turns):
                    x, y = point
                    if mirror:
                        x = n - 1 - x
                    for _ in range(turns):
                        x, y = y, n - 1 - x
                    return x, y

                grid = np.array([list(row) for row in sc.terrain])
                if mirror:
                    grid = np.fliplr(grid)
                altered = replace(
                    sc,
                    terrain=tuple("".join(row) for row in np.rot90(grid, turns)),
                    spawn=transform(sc.spawn),
                    pois=tuple(map(transform, sc.pois)),
                    routes=tuple(tuple(map(transform, route)) for route in sc.routes),
                )
                assert timing_geometry_identity(altered) == timing_geometry_identity(sc)
