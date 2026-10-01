"""G4 checks explicitly forbid learn/train/optimizer steps: no training in this suite."""

import numpy as np
import pytest
import torch

from ather_exploration.agents.learning import build_model
from ather_exploration.environment.env import make_fixture_env
from ather_exploration.training.checkpoints import inspect_checkpoint, load_agent, save_checkpoint
from ather_exploration.training.config import TrainingConfig
from ather_exploration.training.curriculum import Curriculum
from ather_exploration.types import AgentState


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    def forbidden(*args, **kwargs):
        raise AssertionError("User prohibited executing training")

    for cls in (PPO, RecurrentPPO):
        monkeypatch.setattr(cls, "learn", forbidden)
        monkeypatch.setattr(cls, "train", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    torch.set_num_threads(1)


def config(method="ppo"):
    return TrainingConfig(
        method=method,
        banks={k: "/not-loaded-in-this-test" for k in ("small", "medium", "large")},
        n_steps=8,
        batch_size=8,
        total_timesteps=16,
    )


@pytest.mark.parametrize("method", ["ppo", "recurrent_ppo"])
def test_encoder_inference_save_load_no_learning(method, tmp_path):
    env = make_fixture_env("poi_revisit", radius=4)
    try:
        model = build_model(config(method), env)
        obs, _ = env.reset()
        tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            encoded = model.policy.features_extractor(tensor)
        assert encoded.shape == (1, 256) and torch.isfinite(encoded).all()
        original = {k: v.clone() for k, v in model.policy.state_dict().items()}
        path = tmp_path / "checkpoint"
        save_checkpoint(
            model,
            path,
            config(method),
            {"workers": [], "curriculum": {}, "elapsed_seconds": 0, "env_steps": 0},
            {},
        )
        agent = load_agent(path, env.observation_space)
        a, state = agent.act(
            obs, AgentState(), deterministic=True, action_rng=np.random.default_rng(0)
        )
        assert 0 <= a < 5 and not state.episode_start
        b, _ = agent.act(obs, AgentState(), deterministic=True, action_rng=np.random.default_rng(9))
        assert a == b
        if method == "recurrent_ppo":
            assert state.recurrent[0].shape == (1, 1, 256)
            state.episode_start = True
            reset, _ = agent.act(
                obs, state, deterministic=True, action_rng=np.random.default_rng(0)
            )
            assert reset == a
        for key, value in model.policy.state_dict().items():
            assert torch.equal(value, original[key])
        other = make_fixture_env("poi_revisit", radius=3)
        try:
            with pytest.raises(ValueError, match="schema"):
                load_agent(path, other.observation_space)
        finally:
            other.close()
        (path / "model.zip").write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="checksum"):
            inspect_checkpoint(path)
    finally:
        env.close()


def test_curriculum_forced_and_two_disjoint_mastery_windows():
    direct = Curriculum(False)
    direct.advance(0, 100)
    assert direct.stage == 2
    cur = Curriculum(True)
    cur.advance(25, 100)
    assert cur.stage == 1
    cur.advance(75, 100)
    assert cur.stage == 2
    assert cur.transitions[-1]["reason"] == "budget_forced"
    cur = Curriculum(True)
    episodes = [
        {
            "status": "completed",
            "group": ("small", "medium", "large")[i % 3],
            "metadata": {"stage": 0, "component": "near", "fallback": False},
            "metrics": {"survival": 1},
            "final_counts": {"activated": 1},
        }
        for i in range(128)
    ]
    cur.advance(0, 100000, episodes)
    assert cur.stage == 0 and cur.passed_windows == 1
    cur.advance(0, 100000, episodes)
    assert cur.stage == 1
    assert cur.transitions[-1]["reason"] == "mastered"


def test_config_rejects_partial_rollout_budget_and_missing_group():
    with pytest.raises(ValueError):
        config().model_validate({**config().model_dump(), "total_timesteps": 17})
    with pytest.raises(ValueError):
        TrainingConfig(banks={"small": "x"})


@pytest.fixture(scope="module")
def banks(tmp_path_factory):
    from ather_exploration.config import load_preset
    from ather_exploration.worlds.suites import build_development_suite

    root = tmp_path_factory.mktemp("g4-banks")
    paths = {}
    for group in ("small", "medium", "large"):
        path = root / group
        build_development_suite(load_preset(group), 301, path, count=1, train_starts=3)
        paths[group] = str(path)
    return paths


def test_bank_worker_state_and_verified_replay_without_model(banks, tmp_path):
    from ather_exploration.evaluation.replay import verify_replay
    from ather_exploration.training.environments import TrainingEnv
    from ather_exploration.worlds.scenarios import implementation_id, write_record

    env = TrainingEnv(banks, 5, 0, trace_every=1)
    other = TrainingEnv(banks, 5, 0)
    try:
        env.set_stage(0)
        env.reset()
        for _ in range(3):
            _, _, terminated, truncated, _ = env.step(4)
            if terminated or truncated:
                break
        state = env.checkpoint_state()
        assert state["unfinished"] is None or state["unfinished"]["status"] == "cancelled"
        # Saving a checkpoint does not finalize/mutate active metrics.
        if env.active:
            env.step(4)
        other.restore(state)
        obs_a, _ = env.reset()
        obs_b, _ = other.reset()
        assert all(np.array_equal(obs_a[k], obs_b[k]) for k in obs_a)
        for result, trace in env.drain():
            path = tmp_path / f"{result['episode_id']}.json"
            write_record(
                path, {"schema": "g4-replay-v1", "source_revision": implementation_id(), **trace}
            )
            assert verify_replay(path)["status"] == "verified"
    finally:
        env.close()
        other.close()


def test_sampler_preserves_group_world_marginal_and_target_fallback(banks):
    from ather_exploration.training.curriculum import WorldBank

    bank = WorldBank(banks)
    for group in bank.worlds:
        for world in bank.worlds[group]["train"]:
            world["near"] = []
    rng = np.random.default_rng(8)
    counts = dict.fromkeys(banks, 0)
    fallbacks = 0
    for _ in range(900):
        _, meta = bank.sample(rng, 0)
        counts[meta["group"]] += 1
        fallbacks += meta["fallback"]
        assert meta["world_index"] == 0
    assert all(240 < n < 360 for n in counts.values())
    assert 600 < fallbacks < 750


def test_animation_phase_order_does_not_advance_environment():
    from dataclasses import replace

    from ather_exploration.ui.rendering import animation_positions

    env = make_fixture_env("collision2_poi")
    try:
        env.reset()
        before = env.unwrapped.evaluator_snapshot()
        env.step(2)
        after = env.unwrapped.evaluator_snapshot()
        agent, monsters = animation_positions(before, after, 0.25, 2)
        assert monsters == before.monster_positions
        assert before.agent_position[0] < agent[0] < after.agent_position[0]
        agent, monsters = animation_positions(before, after, 0.75, 2)
        assert agent == after.agent_position
        collision1 = replace(after, monster_positions=before.monster_positions)
        assert animation_positions(before, collision1, 0.75, 1)[1] == before.monster_positions
        assert env.unwrapped.evaluator_snapshot() == after
    finally:
        env.close()


def test_initialized_policy_evaluation_and_replay_cli_no_training(banks, tmp_path):
    from ather_exploration.evaluation.learned import evaluate_checkpoint
    from ather_exploration.evaluation.replay import verify_replay
    from ather_exploration.worlds.scenarios import read_record

    env = make_fixture_env("poi_revisit", radius=4)
    try:
        model = build_model(config(), env)
        path = tmp_path / "init-checkpoint"
        save_checkpoint(model, path, config(), {}, {})
        summary = evaluate_checkpoint(path, banks["small"], tmp_path / "evaluation")
        assert summary["completed"] > 0
        assert read_record(tmp_path / "evaluation/status.json")["state"] == "READY"
        assert verify_replay(tmp_path / "evaluation/replays/0.json")["status"] == "verified"
    finally:
        env.close()


@pytest.mark.parametrize(
    "name,recurrent,curriculum",
    [
        ("ppo", False, False),
        ("recurrent_ppo", True, False),
        ("ppo_curriculum", False, True),
        ("recurrent_ppo_curriculum", True, True),
    ],
)
def test_method_names_select_architecture_and_curriculum(name, recurrent, curriculum):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    from ather_exploration.agents.learning import algorithm

    cfg = config(name)
    assert cfg.recurrent is recurrent
    assert cfg.curriculum_enabled is curriculum
    assert algorithm(cfg.method) is (RecurrentPPO if recurrent else PPO)
    assert cfg.model_dump()["method"] == name


@pytest.mark.parametrize(
    "old,new",
    [
        ("A", "ppo"),
        ("B", "recurrent_ppo"),
        ("C", "ppo_curriculum"),
        ("D", "recurrent_ppo_curriculum"),
    ],
)
def test_legacy_method_config_normalizes(old, new):
    assert config(old).model_dump()["method"] == new


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="Unknown method"):
        config("ppo_typo")


def test_checkpoint_dropdown_restarts_same_map_and_keeps_selection(tmp_path, monkeypatch):
    import time

    import pygame
    import pygame_gui

    from ather_exploration.ui.app import Viewer
    from ather_exploration.ui.session import SessionSpec

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    env = make_fixture_env("poi_revisit", radius=4)
    model = build_model(config(), env)
    for step in (256, 1024):
        model.num_timesteps = step
        save_checkpoint(
            model,
            tmp_path / "checkpoints" / f"step_{step}",
            config(),
            {"workers": [], "curriculum": {}, "elapsed_seconds": 0, "env_steps": step},
            {},
        )
    env.close()
    app = Viewer(
        SessionSpec(fixture="poi_revisit", agent="checkpoint", seed=42), checkpoint_dir=tmp_path
    )

    def finish_generation():
        deadline = time.monotonic() + 10
        while app.controller.busy:
            app.update(0.01)
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert not app.controller.error, app.controller.error

    try:
        finish_generation()
        assert list(app.checkpoints) == ["step_256", "step_1024"]
        assert app.spec.checkpoint.endswith("step_1024")
        app.handle(
            pygame.event.Event(
                pygame_gui.UI_DROP_DOWN_MENU_CHANGED,
                ui_element=app.widgets["checkpoint"],
                text="step_256",
            )
        )
        finish_generation()
        assert app.spec.seed == 42
        assert app.controller.frame["row"]["t"] == 0
        assert app.controller.frame["checkpoint"].endswith("step_256")
        app.new_map()
        finish_generation()
        assert app.spec.checkpoint.endswith("step_256")
        app.draw()
    finally:
        app.close()
