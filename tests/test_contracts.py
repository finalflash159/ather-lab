from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

from ather_exploration.config import load_preset
from ather_exploration.schema import observation_space, validate_observation
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


def test_actions():
    assert [int(a) for a in Action] == list(range(5))
    assert ACTION_DELTAS == ((0, -1), (0, 1), (1, 0), (-1, 0), (0, 0))
    assert normalize_action(np.int64(4)) is Action.WAIT
    for bad in [True, False, 1.0, -1, 5, np.array(1), np.array([1]), "1"]:
        with pytest.raises((ValueError, TypeError)):
            normalize_action(bad)


def test_public_contract_has_no_privileged_fields():
    names = {f.name for f in fields(PublicTransition)}
    assert names == {
        "action",
        "motion",
        "actual_delta",
        "new_floor",
        "new_poi",
        "activated",
        "died",
        "end_reason",
    }


def test_state_instances_do_not_share_mutable_data():
    scenario = Scenario(
        terrain=("#####", "#...#", "#####"),
        spawn=(1, 1),
        pois=((3, 1),),
        routes=(),
        phases=(),
        horizon=3,
        seed=0,
        fixture_only=True,
    )
    a, b = EpisodeState.from_scenario(scenario), EpisodeState.from_scenario(scenario)
    a.activated_pois.add((3, 1))
    assert not b.activated_pois
    with pytest.raises(FrozenInstanceError):
        scenario.spawn = (2, 1)


def test_scenario_rejects_mutable_and_invalid_geometry():
    with pytest.raises((ValueError, TypeError)):
        Scenario(
            terrain=["###", "#.#", "###"],
            spawn=(1, 1),
            pois=(),
            routes=(),
            phases=(),
            horizon=1,
            seed=0,
            fixture_only=True,
        )
    with pytest.raises(ValueError):
        Scenario(
            terrain=("###", "#.#", "###"),
            spawn=(0, 0),
            pois=(),
            routes=(),
            phases=(),
            horizon=1,
            seed=0,
            fixture_only=True,
        )


def test_observation_bounds_dtype_and_shape():
    space = observation_space(load_preset("medium").observation)
    obs = {
        "local": np.zeros((6, 9, 9), dtype=np.uint8),
        "memory": np.zeros((11, 81, 81), dtype=np.float32),
        "state": np.zeros(17, dtype=np.float32),
    }
    obs["state"][8] = -1
    validate_observation(obs, space)
    for key, value in [
        ("state", np.full(17, np.nan, dtype=np.float32)),
        ("local", np.zeros((6, 9, 9), dtype=np.float32)),
        ("memory", np.zeros((11, 41, 41), dtype=np.float32)),
    ]:
        invalid = dict(obs, **{key: value})
        with pytest.raises(ValueError):
            validate_observation(invalid, space)


def test_public_transition_consistency():
    event = PublicTransition(
        Action.EAST, Motion.MOVED, (1, 0), activated=True, died=True, end_reason=EndReason.DEATH
    )
    assert event.activated and event.died  # collision2 can legitimately produce both.
    for updates in [
        {"actual_delta": (0, 1)},
        {"new_floor": -1},
        {"died": False},
        {"motion": Motion.WAITED},
        {"action": 2},
        {"activated": 1},
    ]:
        with pytest.raises((ValueError, TypeError)):
            replace(event, **updates)


@pytest.mark.parametrize(
    "phase,expected",
    [
        (0, (2, 1)),
        (1, (2, 1)),
        (2, (3, 1)),
        (3, (3, 1)),
        (4, (4, 1)),
        (5, (4, 1)),
        (6, (3, 1)),
        (7, (3, 1)),
    ],
)
def test_initial_patrol_phase(phase, expected):
    scenario = Scenario(
        terrain=("######", "#....#", "######"),
        spawn=(1, 1),
        pois=((4, 1),),
        routes=(((2, 1), (3, 1), (4, 1)),),
        phases=(phase,),
        horizon=8,
        seed=0,
        patrol_period=2,
        fixture_only=True,
    )
    assert EpisodeState.from_scenario(scenario).monster_positions == [expected]


def test_scenario_route_invariants():
    scenario = Scenario(
        terrain=("######", "#....#", "######"),
        spawn=(1, 1),
        pois=((4, 1),),
        routes=(((2, 1), (3, 1)),),
        phases=(0,),
        horizon=8,
        seed=0,
        fixture_only=True,
    )
    for updates in [
        {"routes": (((2, 1), (4, 1)),)},
        {"phases": (4,)},
        {"pois": ((4, 1), (4, 1))},
        {"routes": (((2, 1), (3, 1)), ((3, 1), (4, 1))), "phases": (0, 0)},
        {"source_revision": []},
    ]:
        with pytest.raises((ValueError, TypeError)):
            replace(scenario, **updates)
