import numpy as np

from ather_exploration.agents.baselines import BaselineConfig, FrontierAgent, PublicMap, RandomAgent
from ather_exploration.environment.env import make_fixture_env
from ather_exploration.types import AgentState


def test_random_uniform_all_five_actions_and_reproducible():
    a = RandomAgent()
    state = AgentState()
    rng = np.random.default_rng(42)
    draws = [int(a.act({}, state, deterministic=True, action_rng=rng)[0]) for _ in range(10000)]
    assert all(abs(n - 2000) < 150 for n in np.bincount(draws, minlength=5))
    rng = np.random.default_rng(42)
    assert draws[:20] == [
        int(a.act({}, state, deterministic=False, action_rng=rng)[0]) for _ in range(20)
    ]


def test_same_public_observation_same_action_without_env_reference():
    env = make_fixture_env("corner_occlusion")
    obs, _ = env.reset()
    original = {k: v.copy() for k, v in obs.items()}
    a, b = FrontierAgent(), FrontierAgent()
    aa, sa = a.act(obs, AgentState(), deterministic=True, action_rng=np.random.default_rng(1))
    bb, sb = b.act(
        {k: v.copy() for k, v in obs.items()},
        AgentState(),
        deterministic=True,
        action_rng=np.random.default_rng(999),
    )
    assert aa == bb and sa.planner.diagnostics == sb.planner.diagnostics
    assert all(np.array_equal(obs[k], original[k]) for k in obs)
    assert not hasattr(a, "env") and not hasattr(a, "scenario")
    env.close()


def test_unsafe_wait_is_not_certified_and_stale_is_cost_not_wall():
    env = make_fixture_env("stale_memory")
    obs, _ = env.reset()
    public = PublicMap(obs, BaselineConfig())
    assert not public.safety(4)[0]
    for action in [3, 4]:
        obs, *_ = env.step(action)
    public = PublicMap(obs, BaselineConfig())
    assert public.stale and not public.monsters
    costs, _ = public.paths(public.position)
    assert any(p in costs for p in public.stale)
    env.close()


def test_no_goals_still_returns_action_and_episode_start_resets_state():
    env = make_fixture_env("wall_wait")
    obs, _ = env.reset()
    agent = FrontierAgent()
    state = AgentState()
    rng = np.random.default_rng(0)
    action, state = agent.act(obs, state, deterministic=True, action_rng=rng)
    assert 0 <= action < 5
    old = state.planner
    agent.act(obs, state, deterministic=True, action_rng=rng)
    assert state.planner is not old
    env.close()


def public_room():
    # A public map with a remembered POI beyond the currently visible neighborhood.
    obs = {
        "local": np.zeros((6, 9, 9), np.uint8),
        "memory": np.zeros((11, 81, 81), np.float32),
        "state": np.zeros(17, np.float32),
    }
    m = obs["memory"]
    m[0, 36:45, 36:46] = 1
    m[1, 36:45, 36:46] = 1
    m[1, 37:44, 37:45] = 0
    m[2, 37:44, 37:45] = 1
    m[7, 40, 40] = 1
    m[8, 38:43, 38:43] = 1
    m[3, 40, 44] = 1
    return obs


def test_remembered_poi_target_and_current_monster_wait_then_return():
    obs = public_room()
    agent = FrontierAgent()
    state = AgentState()
    rng = np.random.default_rng(0)
    action, state = agent.act(obs, state, deterministic=True, action_rng=rng)
    assert action == 2 and state.planner.target == (44, 40)
    # Visible threat cuts the next move, while WAIT at current cell is certified.
    obs["memory"][9, 39, 41] = 1
    action, state = agent.act(obs, state, deterministic=True, action_rng=rng)
    assert action == 4 and state.planner.target == (44, 40)
    obs["memory"][9, 39, 41] = 0
    action, state = agent.act(obs, state, deterministic=True, action_rng=rng)
    assert action == 2 and state.planner.target == (44, 40)


def test_unsafe_wait_selects_safe_retreat_and_logs_uncertainty():
    obs = public_room()
    obs["memory"][9, 40, 41] = 1
    agent = FrontierAgent()
    state = AgentState()
    action, state = agent.act(obs, state, deterministic=True, action_rng=np.random.default_rng(0))
    assert action != 4 and not state.planner.diagnostics["uncertain_action"]
    obs["memory"][8].fill(0)
    obs["memory"][8, 40, 40] = 1
    action, state = agent.act(
        obs, AgentState(), deterministic=True, action_rng=np.random.default_rng(0)
    )
    assert 0 <= action < 5 and state.planner.diagnostics["uncertain_action"]


def test_certificate_is_sound_against_actual_two_collision_dynamics():
    from ather_exploration.environment.dynamics import advance, make_grid
    from ather_exploration.environment.env import make_env
    from ather_exploration.types import EpisodeState

    env = make_env()
    obs, _ = env.reset(seed=42)
    rng = np.random.default_rng(17)
    certified = 0
    for _ in range(100):
        snap = env.unwrapped.evaluator_snapshot()
        public = PublicMap(obs, BaselineConfig())
        for action in range(5):
            if public.safety(action)[0]:
                state = EpisodeState(
                    snap.agent_position,
                    list(snap.monster_positions),
                    set(snap.activated_pois),
                    snap.step_count,
                )
                event, _ = advance(make_grid(snap.scenario), snap.scenario, state, action)
                assert not event.died
                certified += 1
        obs, _, done, _, _ = env.step(int(rng.integers(5)))
        if done:
            break
    assert certified > 0
    env.close()


def test_blocked_target_cooldown_preserves_pending_poi():
    obs = public_room()
    agent = FrontierAgent()
    state = AgentState()
    rng = np.random.default_rng(0)
    _, state = agent.act(obs, state, deterministic=True, action_rng=rng)
    obs["state"][6] = 1
    for _ in range(8):
        _, state = agent.act(obs, state, deterministic=True, action_rng=rng)
    assert (44, 40) in state.planner.cooldown_until and obs["memory"][3, 40, 44] == 1
    obs["state"][6] = 0
    for _ in range(8):
        _, state = agent.act(obs, state, deterministic=True, action_rng=rng)
    assert state.planner.target == (44, 40)
