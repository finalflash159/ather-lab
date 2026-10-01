from dataclasses import replace

import numpy as np
import pytest

from ather_exploration.config import ObservationConfig, RewardConfig
from ather_exploration.environment.dynamics import advance, make_grid
from ather_exploration.environment.memory import PublicMemory
from ather_exploration.environment.reward import RewardTracker
from ather_exploration.environment.visibility import sense
from ather_exploration.fixtures import fixture_scenario
from ather_exploration.types import Action, EpisodeState, Motion, PublicTransition


def test_stale_monster_aging_and_clear_from_public_view():
    scenario = fixture_scenario("stale_memory")
    grid, state = make_grid(scenario), EpisodeState.from_scenario(scenario)
    mem = PublicMemory(ObservationConfig(radius=1))
    mem.reset(sense(grid, scenario, state, 1), scenario.horizon)
    assert mem.observation()["memory"][9, 40, 41] == 1
    for tick, action in enumerate([3, 4, 2], 1):
        event, _ = advance(grid, scenario, state, action)
        mem.update(sense(grid, scenario, state, 1), event)
        obs = mem.observation()
        assert obs["memory"][9, 40, 41] == (tick < 3)
        assert obs["memory"][10, 40, 41] == pytest.approx(
            0 if tick == 3 else np.log1p(tick) / np.log1p(1024)
        )


def test_wait_block_dont_count_visits_and_getters_dont_alias():
    s = fixture_scenario("wall_wait")
    grid, state = make_grid(s), EpisodeState.from_scenario(s)
    mem = PublicMemory(ObservationConfig())
    mem.reset(sense(grid, s, state, 4), s.horizon)
    count = mem.observation()["memory"][6, 40, 40]
    for action in [0, 4]:
        event, _ = advance(grid, s, state, action)
        mem.update(sense(grid, s, state, 4), event)
        assert mem.observation()["memory"][6, 40, 40] == count
    a = mem.observation()
    a["memory"].fill(1)
    a["state"].fill(1)
    a["local"].fill(1)
    assert not np.all(mem.observation()["memory"] == 1)
    assert mem.observation()["state"][16] == 0
    mem.reset(sense(grid, s, EpisodeState.from_scenario(s), 4), s.horizon)
    assert mem.observation()["state"][16] == 1
    assert mem.observation()["memory"][5].sum() == 1


def test_distant_poi_kept_and_reward_identity():
    s = replace(fixture_scenario("poi_revisit"), horizon=8)
    grid, state = make_grid(s), EpisodeState.from_scenario(s)
    mem, tracker = PublicMemory(ObservationConfig(radius=1)), RewardTracker(RewardConfig())
    initial = sense(grid, s, state, 1)
    mem.reset(initial, s.horizon)
    tracker.reset(initial)
    start_seen = len(tracker.seen_floor)
    start_pois = len(tracker.seen_pois)
    total = 0
    for action in [2, 2, 2, 4, 3, 3, 4, 4]:
        event, _ = advance(grid, s, state, action)
        local = sense(grid, s, state, 1)
        event, reward = tracker.update(local, event)
        mem.update(local, event)
        total += reward
    assert total == pytest.approx(
        0.01 * (len(tracker.seen_floor) - start_seen)
        + 0.05 * (len(tracker.seen_pois) - start_pois)
        + 0.5
    )
    assert mem.observation()["memory"][4, 40, 41] == 1
    assert mem.observation()["memory"][0].sum() >= mem.observation()["memory"][8].sum()


def test_terminal_activation_and_penalty():
    rewards = []
    for name in ["collision1_poi", "collision2_poi"]:
        s = fixture_scenario(name)
        grid = make_grid(s)
        state = EpisodeState.from_scenario(s)
        tracker = RewardTracker(RewardConfig())
        tracker.reset(sense(grid, s, state, 4))
        event, _ = advance(grid, s, state, 2)
        enriched, reward = tracker.update(sense(grid, s, state, 4), event)
        assert enriched.died
        rewards.append(reward)
    assert rewards == pytest.approx([-2.0, -1.5])


def test_memory_rejects_overflow_without_mutation():
    local = np.zeros((6, 3, 3), dtype=np.uint8)
    local[0, 1, 1] = local[2, 1, 1] = 1
    memory = PublicMemory(ObservationConfig(radius=1))
    memory.reset(local, 100)
    event = PublicTransition(Action.EAST, Motion.MOVED, (1, 0))
    for _ in range(40):
        memory.update(local, event)
    before = memory.observation()
    with pytest.raises(ValueError, match="capacity"):
        memory.update(local, event)
    for key in before:
        assert np.array_equal(before[key], memory.observation()[key])


def test_reward_uses_only_public_history_and_deduplicates_activation():
    local = np.zeros((6, 3, 3), dtype=np.uint8)
    local[0, 1, 1] = local[2, 1, 1] = local[4, 1, 1] = 1
    tracker = RewardTracker(RewardConfig())
    tracker.reset(local)
    event = PublicTransition(Action.WAIT, Motion.WAITED, (0, 0), activated=True)
    # Initial active POI is already recorded, so repeated activation earns nothing.
    _, reward = tracker.update(local, event)
    assert reward == 0


def test_pending_poi_stays_in_memory_after_leaving_view():
    s = replace(fixture_scenario("poi_revisit"), spawn=(2, 1), pois=((3, 1),))
    grid = make_grid(s)
    state = EpisodeState.from_scenario(s)
    mem = PublicMemory(ObservationConfig(radius=1))
    mem.reset(sense(grid, s, state, 1), s.horizon)
    event, _ = advance(grid, s, state, Action.WEST)
    mem.update(sense(grid, s, state, 1), event)
    obs = mem.observation()
    assert obs["memory"][3, 40, 41] == 1 and obs["memory"][8, 40, 41] == 0
    assert obs["memory"][10, 40, 41] > 0
