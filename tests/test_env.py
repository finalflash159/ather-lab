from copy import deepcopy
from dataclasses import replace
from functools import partial

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check
from stable_baselines3.common.env_checker import check_env as sb3_check
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ather_exploration.config import ObservationConfig
from ather_exploration.environment.env import ExplorationEnv, make_fixture_env
from ather_exploration.environment.memory import PublicMemoryWrapper
from ather_exploration.fixtures import fixture_scenario
from ather_exploration.types import Scenario
from minigrid.minigrid_env import MiniGridEnv


def same_obs(a, b):
    return all(np.array_equal(a[k], b[k]) for k in a)


def test_checkers_and_spaces():
    env = make_fixture_env("wall_wait")
    assert isinstance(env.unwrapped, MiniGridEnv)
    assert env.action_space.n == 5
    with pytest.warns(UserWarning):
        gym_check(env, skip_render_check=True)
    # SB3 heuristically calls 3D symbolic maps 'images'. Bounds0..1 are deliberate.
    with pytest.warns(UserWarning):
        sb3_check(env)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert set(obs) == {"local", "memory", "state"}
    assert env.unwrapped.grid.get(0, 0).type == "wall"
    env.close()


@pytest.mark.parametrize(
    "name,action,reward,activated",
    [
        ("collision1_poi", 2, -2.0, False),
        ("collision2_poi", 2, -1.5, True),
        ("horizon_alive", 4, 0.0, False),
        ("death_at_horizon", 2, -2.0, False),
    ],
)
def test_terminal_final_observation_and_flags(name, action, reward, activated):
    env = make_fixture_env(name)
    env.reset(seed=123)
    obs, r, terminated, truncated, info = env.step(action)
    assert r == pytest.approx(reward) and terminated and not truncated
    assert bool(obs["state"][14]) == activated
    assert obs["state"][16] == 0
    assert info["transition"]["activated"] == activated
    assert set(info) == {"transition"}
    with pytest.raises(RuntimeError):
        env.step(4)
    terminal = {k: v.copy() for k, v in obs.items()}
    reset_obs, _ = env.reset()
    assert same_obs(terminal, obs) and reset_obs["state"][16] == 1
    assert not any(np.shares_memory(obs[k], reset_obs[k]) for k in obs)
    env.close()


def test_getters_and_render_never_advance_or_alias():
    env = make_fixture_env("stale_memory", render_mode="rgb_array")
    env.reset(seed=8)
    snapshot = env.unwrapped.evaluator_snapshot()
    rng = deepcopy(env.unwrapped.np_random.bit_generator.state)
    obs = env.gen_obs()
    for _ in range(3):
        assert env.render().dtype == np.uint8
        assert same_obs(obs, env.gen_obs())
    assert snapshot == env.unwrapped.evaluator_snapshot()
    assert rng == env.unwrapped.np_random.bit_generator.state
    obs["memory"].fill(1)
    assert not np.all(env.gen_obs()["memory"] == 1)
    env.close()


@pytest.mark.parametrize("vec_class", [DummyVecEnv, SubprocVecEnv])
def test_autoreset_only_one_env_and_terminal_copy(vec_class):
    kwargs = {"start_method": "spawn"} if vec_class is SubprocVecEnv else {}
    vec = vec_class(
        [
            partial(make_fixture_env, "wall_wait", horizon=1),
            partial(make_fixture_env, "wall_wait", horizon=3),
        ],
        **kwargs,
    )
    try:
        vec.seed(123)
        vec.reset()
        obs, _reward, dones, infos = vec.step(np.array([4, 4]))
        assert dones.tolist() == [True, False]
        assert obs["state"][:, 16].tolist() == [1, 0]
        assert obs["state"][1, 10] == pytest.approx(2 / 1024)
        terminal = infos[0]["terminal_observation"]
        assert terminal["state"][10] == 0 and terminal["state"][16] == 0
        assert not infos[0]["TimeLimit.truncated"]
        saved = {k: v.copy() for k, v in terminal.items()}
        vec.step(np.array([4, 4]))
        assert same_obs(saved, terminal)
    finally:
        vec.close()


def test_external_truncation_keeps_actual_final_observation():
    vec = DummyVecEnv(
        [lambda: gym.wrappers.TimeLimit(make_fixture_env("wall_wait"), max_episode_steps=2)]
    )
    try:
        vec.reset()
        vec.step(np.array([4]))
        obs, _, done, info = vec.step(np.array([4]))
        assert done[0] and info[0]["TimeLimit.truncated"]
        assert info[0]["terminal_observation"]["state"][10] == pytest.approx(6 / 1024)
        assert obs["state"][0, 16] == 1
    finally:
        vec.close()


def test_equal_public_history_hidden_world_different_rewards_same():
    a = Scenario(
        ("#########", "#.......#", "#.......#", "#.......#", "#########"),
        (1, 1),
        ((7, 3),),
        (),
        (),
        4,
        0,
        fixture_only=True,
    )
    b = replace(
        a, pois=((6, 3),), terrain=("#########", "#.......#", "#.....#.#", "#.......#", "#########")
    )
    config = ObservationConfig(radius=1)
    envs = [
        PublicMemoryWrapper(ExplorationEnv(s, observation_config=config), config, s.horizon)
        for s in [a, b]
    ]
    try:
        observations = [env.reset(seed=7)[0] for env in envs]
        assert same_obs(*observations)
        for action in [2, 3, 4, 4]:
            first, second = [env.step(action) for env in envs]
            assert same_obs(first[0], second[0]) and first[1:] == second[1:]
    finally:
        for env in envs:
            env.close()


def test_main_task_cannot_silently_use_unvalidated_fixture_path():
    s = replace(fixture_scenario("entities_transparent"), fixture_only=False)
    with pytest.raises(ValueError, match="validated"):
        ExplorationEnv(s)


def test_reset_seed_and_invalid_seed_do_not_mutate():
    env = make_fixture_env("poi_revisit")
    first = env.reset(seed=10)[0]
    step = env.step(2)
    assert same_obs(first, env.reset(seed=10)[0])
    assert same_obs(step[0], env.step(2)[0])
    before = env.unwrapped.evaluator_snapshot()
    with pytest.raises(ValueError):
        env.reset(seed=True)
    assert env.unwrapped.evaluator_snapshot() == before
    env.close()


def test_terminal_sensor_still_awards_new_floor_and_discovery():
    scenario = replace(fixture_scenario("collision1_poi"), pois=((3, 1),))
    config = ObservationConfig(radius=1)
    env = PublicMemoryWrapper(
        ExplorationEnv(scenario, observation_config=config), config, scenario.horizon
    )
    try:
        env.reset()
        obs, reward, done, _, info = env.step(2)
        event = info["transition"]
        assert done and event["new_floor"] == 1 and event["new_poi"] == 1
        assert not event["activated"] and reward == pytest.approx(-1.94)
        assert obs["memory"][3, 40, 42] == 1
    finally:
        env.close()


def test_recurrent_predict_resets_only_done_env():
    import torch
    from sb3_contrib import RecurrentPPO

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    vec = DummyVecEnv(
        [
            partial(make_fixture_env, "wall_wait", horizon=1),
            partial(make_fixture_env, "wall_wait", horizon=3),
        ]
    )
    try:
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            vec,
            device="cpu",
            seed=0,
            n_steps=2,
            batch_size=4,
            policy_kwargs={"lstm_hidden_size": 8, "net_arch": {"pi": [8], "vf": [8]}},
        )
        vec.reset()
        obs, _, dones, _ = vec.step(np.array([4, 4]))
        dirty = (
            np.full((1, 2, 8), 2.0, dtype=np.float32),
            np.full((1, 2, 8), 2.0, dtype=np.float32),
        )
        zeros = tuple(np.zeros_like(s) for s in dirty)
        _, mixed = model.predict(obs, state=dirty, episode_start=dones, deterministic=True)
        _, fresh = model.predict(
            obs, state=zeros, episode_start=np.ones(2, dtype=bool), deterministic=True
        )
        for m, f in zip(mixed, fresh, strict=True):
            assert np.allclose(m[:, 0], f[:, 0])
        assert not np.allclose(mixed[1][:, 1], fresh[1][:, 1])
    finally:
        vec.close()
        torch.set_num_threads(threads)
