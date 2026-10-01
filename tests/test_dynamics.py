from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from ather_exploration.environment.dynamics import advance, make_grid, monster_positions
from ather_exploration.fixtures import fixture_scenario
from ather_exploration.types import Action, EndReason, EpisodeState, Motion


@pytest.mark.parametrize(
    "name,stage,activated",
    [
        ("collision1_poi", 1, False),
        ("collision2_poi", 2, True),
        ("edge_swap", 1, False),
        ("death_at_horizon", 1, False),
    ],
)
def test_collision_order(name, stage, activated):
    scenario = fixture_scenario(name)
    state = EpisodeState.from_scenario(scenario)
    event, collision = advance(make_grid(scenario), scenario, state, Action.EAST)
    assert collision == stage and event.activated == activated
    assert state.done and not state.alive and state.end_reason is EndReason.DEATH
    assert state.monster_positions == [(2, 1)]
    assert len(state.activated_pois) == int(activated)


def test_wait_blocked_and_horizon():
    scenario = replace(fixture_scenario("wall_wait"), horizon=2)
    state, grid = EpisodeState.from_scenario(scenario), make_grid(scenario)
    for action, motion in [(Action.NORTH, Motion.WALL_BLOCKED), (Action.WAIT, Motion.WAITED)]:
        event, _ = advance(grid, scenario, state, action)
        assert event.motion is motion and event.actual_delta == (0, 0)
    assert state.step_count == 2 and state.done and state.alive
    assert state.end_reason is EndReason.BUDGET
    before = deepcopy(state)
    with pytest.raises(RuntimeError):
        advance(grid, scenario, state, Action.WAIT)
    assert state == before


@pytest.mark.parametrize("bad", [True, 2.0, -1, 5, np.array(2), "2"])
def test_bad_actions_do_not_mutate(bad):
    scenario = fixture_scenario("wall_wait")
    state = EpisodeState.from_scenario(scenario)
    before = deepcopy(state)
    with pytest.raises((ValueError, TypeError)):
        advance(make_grid(scenario), scenario, state, bad)
    assert state == before


def test_revisit_and_last_poi_do_not_end_episode():
    scenario = fixture_scenario("poi_revisit")
    state, grid = EpisodeState.from_scenario(scenario), make_grid(scenario)
    events = [advance(grid, scenario, state, a)[0] for a in [2, 3, 2]]
    assert [e.activated for e in events] == [True, False, False]
    assert not state.done


@pytest.mark.parametrize("period", [1, 2, 3])
def test_schedule_all_phases_and_no_extra_endpoint_pause(period):
    scenario = replace(fixture_scenario("stale_memory"), patrol_period=period)
    expected = [(3, 1), (4, 1), (5, 1), (4, 1)]
    for phase in range(period * 4):
        s = replace(scenario, phases=(phase,))
        for tick in range(3 * period * 4):
            assert monster_positions(s, tick) == [expected[((tick + phase) // period) % 4]]


def test_monster_can_enter_cell_agent_just_left():
    scenario = replace(fixture_scenario("edge_swap"), spawn=(3, 1), routes=(((2, 1), (3, 1)),))
    state = EpisodeState.from_scenario(scenario)
    event, collision = advance(make_grid(scenario), scenario, state, Action.EAST)
    assert not event.died and collision is None
    assert state.agent_position == (4, 1) and state.monster_positions == [(3, 1)]


@pytest.mark.parametrize(
    "action,position", [(0, (2, 1)), (1, (2, 3)), (2, (3, 2)), (3, (1, 2)), (4, (2, 2))]
)
def test_all_absolute_directions(action, position):
    from ather_exploration.types import Scenario

    s = Scenario(
        ("#####", "#...#", "#...#", "#...#", "#####"), (2, 2), (), (), (), 4, 0, fixture_only=True
    )
    state = EpisodeState.from_scenario(s)
    event, _ = advance(make_grid(s), s, state, action)
    assert state.agent_position == position
    assert event.actual_delta == (position[0] - 2, position[1] - 2)


def test_death_at_h_collision2_and_same_pose_monsters_frozen_at_collision1():
    s = replace(fixture_scenario("collision2_poi"), horizon=1)
    state = EpisodeState.from_scenario(s)
    event, stage = advance(make_grid(s), s, state, 2)
    assert event.activated and event.died and stage == 2 and event.end_reason is EndReason.DEATH
    s = replace(
        fixture_scenario("collision1_poi"),
        terrain=("########", "#......#", "########"),
        routes=(((2, 1), (3, 1)), ((5, 1), (6, 1))),
        phases=(0, 0),
    )
    state = EpisodeState.from_scenario(s)
    before = state.monster_positions.copy()
    advance(make_grid(s), s, state, 2)
    assert state.monster_positions == before  # Stop the whole monster phase, not only collider.
