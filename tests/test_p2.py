"""P2 reward, diagnosis and transfer regression checks. No learning/optimizer steps."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from ather_exploration.training.config import P2RewardConfig, SkillConfig, TrainingConfig
from ather_exploration.training.skill_curriculum import SkillController
from ather_exploration.worlds.skill_tasks import configured_skill_env


@pytest.fixture(autouse=True)
def no_learning(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not perform optimizer updates")

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    torch.set_num_threads(1)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_p2_invalid_cost(value):
    with pytest.raises(ValueError):
        P2RewardConfig(step_cost=value)


def test_p2_wait_block_activation_and_repeat_reward():
    skills = SkillConfig(enabled=True)
    env = configured_skill_env("P1a", 100438, skills, phase="P2a")
    env.reset()
    for action in (4, 0, 0):  # spawn(1,1), blocked north
        _, reward, _, _, info = env.step(action)
        assert reward == pytest.approx(-0.005 if action == 4 else -0.025)
        assert info["skill"]["reward_components"]["area"] == 0
    env.close()
    env = configured_skill_env("P1a", 100020, skills, phase="P2b")
    env.reset()
    _, reward, done, _, info = env.step(0)
    assert done and info["skill"]["success"]
    assert info["skill"]["source_task"] == "P1a"
    assert info["skill"]["phase"] == "P2b"
    assert info["skill"]["reward_components"]["activation"] == 0.5
    assert info["skill"]["reward_components"]["discovery"] == 0  # visible at reset
    assert reward == pytest.approx(sum(info["skill"]["reward_components"].values()))
    env.close()


def test_p2_shortest_route_discovers_then_activates_once():
    from ather_exploration.types import ACTION_DELTAS
    from ather_exploration.worlds.topology import distances

    env = configured_skill_env("P2b", 17, SkillConfig(enabled=True))
    obs, _ = env.reset()
    assert not obs["local"][3].any()
    sc = env.unwrapped.scenario
    ds = distances(sc.terrain, [sc.pois[0]])
    pos = sc.spawn
    discovery = activation = cost = 0
    while pos != sc.pois[0]:
        action = next(
            a
            for a, (x, y) in enumerate(ACTION_DELTAS[:4])
            if ds.get((pos[0] + x, pos[1] + y), 999) < ds[pos]
        )
        _, reward, term, trunc, info = env.step(action)
        rc = info["skill"]["reward_components"]
        assert reward == pytest.approx(sum(rc.values()))
        discovery += rc["discovery"]
        activation += rc["activation"]
        cost += rc["step_cost"]
        dx, dy = ACTION_DELTAS[action]
        pos = pos[0] + dx, pos[1] + dy
        if not (term or trunc):
            # Waiting never reclaims area/discovery reward.
            _, wait_reward, _, _, _ = env.step(4)
            assert wait_reward == pytest.approx(-0.005)
    assert discovery == pytest.approx(0.05)
    assert activation == pytest.approx(0.5)
    assert cost == pytest.approx(-0.005 * ds[sc.spawn])
    env.close()


def test_p2_worker_easy_map_uses_active_reward(monkeypatch):
    from ather_exploration.training.skill_environments import SkillTrainingEnv

    monkeypatch.setattr(SkillController, "mixture", lambda self: [("P1b", 1.0)])
    cfg = TrainingConfig(
        banks={g: "unused" for g in ("small", "medium", "large")},
        skills=SkillConfig(enabled=True, train_count=8, stop_after="P2"),
    )
    env = SkillTrainingEnv(cfg)
    env.set_controller(asdict(SkillController(index=2)))
    env.reset()
    for _ in range(96):
        _, reward, _, _, _ = env.step(4)
        assert reward == pytest.approx(-0.005)
    row = env.drain()[0]
    assert row["task"] == "P2a" and row["source_task"] == "P1b"
    assert row["return"] == pytest.approx(-0.48)
    assert row["reward_components"]["step_cost"] == pytest.approx(-0.48)
    env.close()


def observation(floors=1, seen=False, blocked=False):
    memory = np.zeros((11, 5, 5))
    memory[2].flat[:floors] = 1
    memory[3, 0, 0] = seen
    state = np.zeros(17)
    state[6] = blocked
    return {"memory": memory, "state": state}


def test_search_metrics_missing_and_discovery_boundary():
    from ather_exploration.evaluation.skills import SearchDiagnostics

    d = SearchDiagnostics(observation(), 10)
    d.update(1, observation(3, blocked=True), {"activated": False})
    d.update(2, observation(4, seen=True, blocked=True), {"activated": False})
    d.update(3, observation(5, seen=True), {"activated": True})
    row = d.row()
    assert row["first_poi_seen_step"] == 2
    assert row["steps_seen_to_activation"] == 1
    assert row["coverage_gain_before_seen"] == pytest.approx(0.3)
    assert row["coverage_gain_after_seen"] == pytest.approx(0.1)
    assert row["longest_wall_block_streak"] == 2
    assert row["activated_if_seen"] == 1
    unseen = SearchDiagnostics(observation(), 10).row()
    assert unseen["first_poi_seen_step"] is None
    assert unseen["steps_seen_to_activation"] is None
    assert unseen["activated_if_seen"] is None
    visible = SearchDiagnostics(observation(seen=True), 10).row()
    assert visible["first_poi_seen_step"] == 0
    assert visible["activated_if_seen"] == 0
    assert visible["steps_seen_to_activation"] is None


def test_evaluator_null_means_and_frozen_weights(monkeypatch):
    import ather_exploration.evaluation.skills as evaluation

    class WaitAgent:
        def __init__(self, *a):
            pass

        def act(self, obs, state, **kw):
            return 4, state

    monkeypatch.setattr(evaluation, "LearnedAgent", WaitAgent)
    monkeypatch.setattr(evaluation, "skill_pool", lambda *a: [(17, "fixture")])
    cfg = TrainingConfig(
        banks={g: "unused" for g in ("small", "medium", "large")}, skills=SkillConfig(enabled=True)
    )
    result = evaluation.evaluate_skill(None, "P2b", cfg)
    for mode in result["summary"].values():
        assert mode["first_poi_seen_step"] is None
        assert mode["first_poi_seen_step_count"] == 0
        assert mode["activated_if_seen"] is None
        assert mode["timeout"] == 1
        assert mode["reward_step_cost"] == pytest.approx(-0.48)
    assert not result["passed"]
    json.dumps(result, allow_nan=False)


@pytest.fixture
def parent_checkpoint(tmp_path):
    from ather_exploration.agents.learning import build_model
    from ather_exploration.training.checkpoints import P1_TRANSFER_SOURCE, save_checkpoint
    from ather_exploration.training.skill_environments import SkillTrainingEnv, skill_identity

    cfg = TrainingConfig(
        total_timesteps=4063232,
        banks={g: "unused" for g in ("small", "medium", "large")},
        skills=SkillConfig(enabled=True, train_count=8, validation_count=4, stop_after="P1"),
    )
    env = SkillTrainingEnv(cfg)
    model = build_model(cfg, env)
    model.num_timesteps, model._n_updates = 98304, 12
    # Synthetic optimizer slots, created directly: no learn() or optimizer.step().
    param = next(model.policy.parameters())
    model.policy.optimizer.state[param] = {
        "step": torch.tensor(2.0),
        "exp_avg": torch.zeros_like(param),
        "exp_avg_sq": torch.ones_like(param) * 0.1,
    }
    state = {
        "state": "PHASE_COMPLETED",
        "env_steps": 98304,
        "viewer_task": "P1b",
        "skill_controller": asdict(SkillController(index=2, phase_start=98304, family_start=98304)),
        "workers": [env.checkpoint_state()],
    }
    path = tmp_path / "parent"
    save_checkpoint(model, path, cfg, state, skill_identity(cfg))
    # Recreate a legacy metadata envelope in the temporary fixture only.
    meta = json.loads((path / "metadata.json").read_text())
    meta["source_revision"] = P1_TRANSFER_SOURCE
    meta["curriculum_protocol"] = "active-phase-v1"
    del meta["config"]["skills"]["p2_reward"]
    rewrite(path, "metadata.json", meta)
    dest = cfg.model_copy(update={"skills": cfg.skills.model_copy(update={"stop_after": "P2"})})
    yield path, dest, model
    env.close()


def rewrite(path, name, payload):
    (path / name).write_text(json.dumps(payload))
    hashes = json.loads((path / "checksums.json").read_text())
    hashes[name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    (path / "checksums.json").write_text(json.dumps(hashes))


def test_transfer_preserves_model_optimizer_schedule_and_parent(parent_checkpoint):
    from ather_exploration.training.checkpoints import inspect_checkpoint
    from ather_exploration.training.skill_environments import SkillTrainingEnv
    from ather_exploration.training.skill_transfer import prepare_p1_transfer

    path, cfg, original = parent_checkpoint
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(ValueError, match="source revision"):
        inspect_checkpoint(path)
    env = SkillTrainingEnv(cfg)
    loaded, state, provenance = prepare_p1_transfer(path, cfg, env)
    for k, v in original.policy.state_dict().items():
        assert torch.equal(loaded.policy.state_dict()[k], v)
    old_opt = original.policy.optimizer.state_dict()
    new_opt = loaded.policy.optimizer.state_dict()
    assert old_opt["param_groups"] == new_opt["param_groups"]
    for key in old_opt["state"]:
        for field, value in old_opt["state"][key].items():
            assert torch.equal(value, new_opt["state"][key][field])
    assert loaded.num_timesteps == 98304
    assert loaded.lr_schedule(0.8) == original.lr_schedule(0.8)
    assert state["skill_controller"]["index"] == 2
    assert provenance["changes"]["p2_reward"]["from"]["step_cost"] == 0
    assert provenance["changes"]["p2_reward"]["to"]["step_cost"] == 0.005
    assert before == {p.name: p.read_bytes() for p in path.iterdir()}
    env.close()


@pytest.mark.parametrize(
    "fault", ["source", "schema", "controller", "weights", "config", "architecture"]
)
def test_transfer_rejects_incompatible_parent(parent_checkpoint, fault):
    from ather_exploration.training.skill_environments import SkillTrainingEnv
    from ather_exploration.training.skill_transfer import prepare_p1_transfer

    path, cfg, _ = parent_checkpoint
    meta = json.loads((path / "metadata.json").read_text())
    if fault == "source":
        meta["source_revision"] = "unreviewed"
    elif fault == "schema":
        meta["schema"]["actions"].reverse()
    elif fault == "controller":
        meta["viewer_task"] = "P1a"
    elif fault == "weights":
        (path / "model.zip").write_bytes(b"corrupt")
    elif fault == "config":
        cfg = cfg.model_copy(update={"gamma": 0.9})
    elif fault == "architecture":
        # Actual serialized policy mismatch, with a valid integrity envelope.
        from stable_baselines3 import PPO

        model = PPO.load(path / "model.zip", device="cpu")
        model.policy.action_net = torch.nn.Linear(64, 6)
        model.save(path / "model.zip")
        hashes = json.loads((path / "checksums.json").read_text())
        hashes["model.zip"] = hashlib.sha256((path / "model.zip").read_bytes()).hexdigest()
        (path / "checksums.json").write_text(json.dumps(hashes))
    rewrite(path, "metadata.json", meta)
    env = SkillTrainingEnv(cfg)
    with pytest.raises((ValueError, RuntimeError)):
        prepare_p1_transfer(path, cfg, env)
    env.close()


def test_transfer_runner_and_p2_ui_lifecycle(parent_checkpoint, tmp_path, monkeypatch):
    from stable_baselines3 import PPO

    from ather_exploration.training.checkpoints import inspect_checkpoint
    from ather_exploration.training.skill_runner import SkillCallback, run_skill_training
    from ather_exploration.ui.session import EpisodeSession, SessionSpec

    path, cfg, _ = parent_checkpoint
    evaluated = []

    def evaluate(self, task):
        evaluated.append(task)
        return {
            "passed": True,
            "task": task,
            "summary": {
                "deterministic": {
                    "success": 1.0,
                    "efficiency": 1.0,
                    "approach_efficiency": 1.0,
                    "coverage": 1.0,
                    "coverage_auc": 1.0,
                }
            },
        }

    monkeypatch.setattr(SkillCallback, "evaluate", evaluate)

    def simulate_boundaries(self, *, callback, **kwargs):
        assert self.num_timesteps == 98304
        assert kwargs["reset_num_timesteps"] is False
        assert kwargs["total_timesteps"] == cfg.total_timesteps - 98304
        callback.init_callback(self)
        self._last_obs = self.env.reset()
        for step in (131072, 147456, 180224, 196608, 229376, 245760):
            self.num_timesteps = step
            callback.boundary()
        return self

    monkeypatch.setattr(PPO, "learn", simulate_boundaries)
    root = tmp_path / "p2"
    result = run_skill_training(
        cfg, root, resume=path, continue_curriculum=True, transfer_p1_to_p2=True
    )
    assert result["state"] == "PHASE_COMPLETED"
    assert evaluated == ["P2a", "P2a", "P2b", "P2b", "P2c", "P2c"]
    cp = root / "checkpoints/step_245760"
    _, meta = inspect_checkpoint(cp)
    assert meta["viewer_task"] == "P2c"
    assert meta["skill_controller"]["index"] == 5
    assert meta["transfer"]["parent_env_steps"] == 98304
    assert (root / "transfer.json").exists()
    from ather_exploration.worlds.scenarios import read_record

    best = read_record(root / "best.json")["by_task"]
    assert set(best) == {"P2a", "P2b", "P2c"}
    for record in best.values():
        assert record["run_id"] == root.name
        assert (root / record["checkpoint"] / "READY").exists()
    session = EpisodeSession(SessionSpec(agent="checkpoint", checkpoint=str(cp), seed=17))
    try:
        assert session.frame()["phase"] == "P2c"
        assert session.env.step_cost == 0.005
        session.step()
        assert session.total_reward == pytest.approx(session.metrics.steps[-1]["learning_reward"])
    finally:
        session.env.close()


def test_p2_four_subprocess_workers_share_phase_reward():
    from functools import partial

    from stable_baselines3.common.vec_env import SubprocVecEnv

    from ather_exploration.training.skill_environments import SkillTrainingEnv

    cfg = TrainingConfig(
        n_envs=4,
        vec_backend="subproc",
        banks={g: "unused" for g in ("small", "medium", "large")},
        skills=SkillConfig(enabled=True, train_count=8, stop_after="P2"),
    )
    env = SubprocVecEnv([partial(SkillTrainingEnv, cfg, i) for i in range(4)], start_method="spawn")
    try:
        env.env_method("set_controller", asdict(SkillController(index=2)))
        obs = env.reset()
        assert obs["memory"].shape[0] == 4
        for _ in range(96):
            _, reward, done, infos = env.step(np.array([4] * 4))
            np.testing.assert_allclose(reward, -0.005)
            assert all(info["skill"]["phase"] == "P2a" for info in infos)
        assert done.all()
        for rows in env.env_method("drain"):
            assert len(rows) == 1
            assert rows[0]["reward_components"]["step_cost"] == pytest.approx(-0.48)
    finally:
        env.close()


def test_p2_streak_minimum_and_independent_budgets():
    c = SkillController(index=2, phase_start=98304, family_start=98304)
    assert not c.observe(True, 114688)
    assert c.history[-1]["streak"] == 1 and not c.history[-1]["eligible"]
    assert c.observe(True, 131072)
    assert c.task == "P2b"
    assert not c.observe(False, 147456)
    assert c.passed == 0
    assert not c.observe(False, 360448)  # family cap no longer ends P2b prematurely
    assert not c.failed
    assert not c.observe(False, 393216)
    assert c.failed


def test_p2c_activation_continues_and_reward_is_once():
    env = configured_skill_env(
        "P1a", 100020, SkillConfig(enabled=True, p2c_horizon=32), phase="P2c"
    )
    env.reset()
    _, reward, term, trunc, info = env.step(0)
    assert not term and not trunc
    assert info["skill"]["reward_components"]["activation"] == 0.5
    for action in (1, 0):
        _, reward, term, trunc, info = env.step(action)
        assert info["skill"]["reward_components"]["activation"] == 0
        assert reward == pytest.approx(sum(info["skill"]["reward_components"].values()))
    for _ in range(29):
        _, _, term, trunc, info = env.step(4)
    assert (term or trunc) and info["skill"]["success"]
    env.close()


def test_p2b_latched_discovery_suppresses_only_later_area():
    from ather_exploration.types import ACTION_DELTAS
    from ather_exploration.worlds.topology import distances

    env = configured_skill_env("P2b", 17, SkillConfig(enabled=True))
    obs, _ = env.reset()
    sc = env.unwrapped.scenario
    ds = distances(sc.terrain, [sc.pois[0]])
    pos = sc.spawn
    seen = False
    while pos != sc.pois[0]:
        a = next(
            a
            for a, (dx, dy) in enumerate(ACTION_DELTAS[:4])
            if ds.get((pos[0] + dx, pos[1] + dy), 999) < ds[pos]
        )
        obs, r, term, _trunc, info = env.step(a)
        rc = info["skill"]["reward_components"]
        if seen:
            assert rc["area"] == 0
        else:
            assert rc["area"] == pytest.approx(info["transition"]["new_floor"] * 0.01)
        seen = seen or bool(obs["memory"][3:5].any())
        assert env.poi_seen == seen
        pos = tuple(x + y for x, y in zip(pos, ACTION_DELTAS[a], strict=True))
        assert r == pytest.approx(sum(rc.values()))
    assert seen and term
    env.close()


def test_p2c_generator_hidden_and_disjoint_pool():
    from ather_exploration.worlds.skill_tasks import skill_pool

    for seed, _ in skill_pool("P2c", 8):
        env = configured_skill_env("P2c", seed, SkillConfig(enabled=True))
        obs, _ = env.reset()
        assert not obs["memory"][3:5].any()
        assert len(env.unwrapped.scenario.pois) == 1
        assert not env.unwrapped.scenario.routes
        assert env.unwrapped.scenario.horizon == 192
        env.close()
    assert not ({h for _, h in skill_pool("P2c", 8)} & {h for _, h in skill_pool("P2c", 4, True)})


def test_p2b_p2c_geometry_uses_same_content_split():
    from ather_exploration.worlds.skill_tasks import skill_pool

    for validation in (False, True):
        assert skill_pool("P2b", 8, validation) == skill_pool("P2c", 8, validation)


def test_task_best_prioritizes_gate_and_current_objective():
    from ather_exploration.training.skill_runner import task_score

    def result(task, passed, efficiency, approach, coverage):
        return {
            "task": task,
            "passed": passed,
            "summary": {
                "deterministic": {
                    "success": 1.0,
                    "efficiency": efficiency,
                    "approach_efficiency": approach,
                    "coverage": coverage,
                    "coverage_auc": coverage,
                    "wall_block": 0.0,
                }
            },
        }

    assert task_score(result("P2a", True, 0.7, 0.1, 0.1)) > task_score(
        result("P2a", False, 0.9, 0.9, 0.9)
    )
    assert task_score(result("P2a", True, 0.9, 0.1, 0.1)) > task_score(
        result("P2a", True, 0.7, 0.9, 0.9)
    )
    assert task_score(result("P2b", True, 0.1, 0.9, 0.1)) > task_score(
        result("P2b", True, 0.9, 0.7, 0.9)
    )
    assert task_score(result("P2c", True, 0.1, 0.1, 0.9)) > task_score(
        result("P2c", True, 0.9, 0.9, 0.7)
    )


@pytest.mark.parametrize("phase", ["P2a", "P2b", "P2c"])
def test_ui_skill_reward_accounting_through_full_episode(monkeypatch, phase):
    import ather_exploration.ui.session as ui
    from ather_exploration.types import ACTION_DELTAS
    from ather_exploration.worlds.topology import distances

    env = configured_skill_env(phase, 42, SkillConfig(enabled=True))
    monkeypatch.setattr(ui, "make_fixture_env", lambda _: env)
    s = ui.EpisodeSession(ui.SessionSpec(fixture="skill", agent="manual"))
    sc = env.unwrapped.scenario
    ds = distances(sc.terrain, [sc.pois[0]])
    pos = sc.spawn
    try:
        while pos != sc.pois[0]:
            a = next(
                a
                for a, (dx, dy) in enumerate(ACTION_DELTAS[:4])
                if ds.get((pos[0] + dx, pos[1] + dy), 999) < ds[pos]
            )
            frame = s.step(a)
            assert frame["row"]["learning_reward"] == pytest.approx(
                sum(frame["row"]["skill_reward_components"].values())
            )
            pos = tuple(x + y for x, y in zip(pos, ACTION_DELTAS[a], strict=True))
        while not s.done:
            s.step(4)
        assert s.result["skill_success"]
    finally:
        s.close()
