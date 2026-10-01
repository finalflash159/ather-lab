from dataclasses import replace
from itertools import product

import pytest

from ather_exploration.fixtures import fixture_scenario
from ather_exploration.types import Scenario, ValidatorStatus
from ather_exploration.worlds.validation import replay_witness, validate_scenario


def exhaustive_reference(s):
    # Independent tiny oracle: deliberately does not call dynamics.advance/schedule.
    def monsters(t):
        out = []
        for route, phase in zip(s.routes, s.phases, strict=True):
            order = list(range(len(route))) + list(range(len(route) - 2, 0, -1))
            out.append(route[order[((t + phase) // s.patrol_period) % len(order)]])
        return out

    states = {(s.spawn, frozenset())}
    for t in range(s.horizon):
        next_states = set()
        for (x, y), visited in states:
            for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0)]:
                p = (x + dx, y + dy)
                if (
                    not (0 <= p[1] < len(s.terrain) and 0 <= p[0] < len(s.terrain[0]))
                    or s.terrain[p[1]][p[0]] == "#"
                ):
                    p = (x, y)
                if p in monsters(t) or p in monsters(t + 1):
                    continue
                next_states.add((p, visited | ({p} if p in s.pois else set())))
        states = next_states
    return any(len(visited) == len(s.pois) for _, visited in states)


def test_unknown_is_not_infeasible_and_witness_replays():
    s = fixture_scenario("poi_revisit")
    assert validate_scenario(s, max_expansions=1).status is ValidatorStatus.UNKNOWN
    result = validate_scenario(s, max_expansions=1000)
    assert result.status is ValidatorStatus.VALIDATED
    assert len(result.actions) == s.horizon
    replay_witness(s, result.actions)
    with pytest.raises(ValueError):
        replay_witness(s, result.actions[:-1])


def test_individually_reachable_not_jointly_reachable():
    s = Scenario(
        ("#######", "#.....#", "#######"), (3, 1), ((1, 1), (5, 1)), (), (), 3, 0, fixture_only=True
    )
    for poi in s.pois:
        assert validate_scenario(replace(s, pois=(poi,))).status is ValidatorStatus.VALIDATED
    assert validate_scenario(s).status is ValidatorStatus.INFEASIBLE


def test_activation_without_survival_is_not_success():
    s = Scenario(
        ("####", "#..#", "####"),
        (1, 1),
        ((1, 1),),
        (((2, 1), (1, 1)),),
        (0,),
        2,
        0,
        patrol_period=2,
        fixture_only=True,
    )
    assert validate_scenario(s).status is ValidatorStatus.INFEASIBLE


@pytest.mark.parametrize(
    "period,phase,horizon,spawn", list(product([1, 2], [0, 1], [1, 2, 4], [(1, 1), (3, 1)]))
)
def test_matches_exhaustive_reference(period, phase, horizon, spawn):
    s = Scenario(
        ("######", "#....#", "#....#", "######"),
        spawn,
        ((4, 1), (1, 2)),
        (((2, 1), (2, 2)),),
        (phase,),
        horizon,
        0,
        patrol_period=period,
        fixture_only=True,
    )
    result = validate_scenario(s, max_expansions=10000)
    assert (result.status is ValidatorStatus.VALIDATED) == exhaustive_reference(s)
    assert result.status is not ValidatorStatus.UNKNOWN
    if result.status is ValidatorStatus.VALIDATED:
        replay_witness(s, result.actions)
