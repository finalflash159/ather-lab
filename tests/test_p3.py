"""Multi-room curriculum invariants; no learning is executed."""

from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
import torch

from ather_exploration.evaluation.p3 import ExplorationDiagnostics, add_p3_gates
from ather_exploration.training.config import SkillConfig, read_training_config
from ather_exploration.training.skill_curriculum import STAGES, SkillController
from ather_exploration.types import ACTION_DELTAS
from ather_exploration.worlds.p3_tasks import (
    map_group,
    oracle_tour,
    p3_pool,
    p3_scenario,
    terrain_identity,
    terrain_split,
)
from ather_exploration.worlds.skill_tasks import configured_skill_env
from ather_exploration.worlds.topology import distances


@pytest.fixture(autouse=True)
def no_learning(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No optimizer steps in these checks")

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    torch.set_num_threads(1)


@pytest.mark.parametrize("task", ["P3a", "P3b", "P3c"])
def test_full_pools_disjoint_connected_hidden_and_feasible(task):
    pools = {
        split: p3_pool(task, count, split)
        for split, count in [("train", 256), ("validation", 64), ("ood", 64)]
    }
    sets = [{identity for _, identity in records} for records in pools.values()]
    assert sum(map(len, sets)) == len(set.union(*sets))
    for split, records in pools.items():
        for seed, identity in records:
            sc = p3_scenario(task, seed)
            assert sc == p3_scenario(task, seed)
            assert terrain_identity(sc.terrain) == identity
            assert terrain_split(sc.terrain) == ("validation" if split == "ood" else split)
            floor_count = sum(row.count(".") for row in sc.terrain)
            assert len(distances(sc.terrain, [sc.spawn])) == floor_count
            assert oracle_tour(sc) <= 192
            assert not sc.routes and len(set(sc.pois)) == len(sc.pois)
            env = configured_skill_env(task, seed, SkillConfig(enabled=True))
            obs, _ = env.reset()
            assert not obs["local"][3].any()
            assert env.observation_space.contains(obs)
            env.close()
    groups = [map_group(p3_scenario(task, s)) for s, _ in pools["validation"]]
    assert len({g["size"] for g in groups}) >= 3
    if task == "P3c":
        assert {g["topology"] for g in groups} == {"chain", "cycle"}
        assert {g["poi_count"] for g in groups} == {1, 2}


def test_split_invariant_under_rotations_reflections():
    sc = p3_scenario("P3b", 7)
    a = np.array([list(r) for r in sc.terrain])
    for grid in (a, np.fliplr(a)):
        for k in range(4):
            assert terrain_identity(
                tuple("".join(r) for r in np.rot90(grid, k))
            ) == terrain_identity(sc.terrain)


def test_rewards_pois_once_continue_and_public_diagnostics():
    env = configured_skill_env("P3b", 4, SkillConfig(enabled=True))
    obs, _ = env.reset()
    sc = env.unwrapped.scenario
    diag = ExplorationDiagnostics(sc, obs)
    # Evaluator-only shortest tour exercises both activations, never supplies hints to policy.
    ds = {p: distances(sc.terrain, [p]) for p in (sc.spawn, *sc.pois)}
    tour = min(
        permutations(sc.pois), key=lambda ps: sum(ds[a][b] for a, b in zip((sc.spawn, *ps), ps))
    )
    position = sc.spawn
    tick = 0
    activation = discovery = 0
    for goal in tour:
        while position != goal:
            action = next(
                a
                for a, (dx, dy) in enumerate(ACTION_DELTAS[:4])
                if ds[goal].get((position[0] + dx, position[1] + dy), 999) < ds[goal][position]
            )
            obs, reward, term, trunc, info = env.step(action)
            tick += 1
            assert not term and not trunc
            assert reward == pytest.approx(sum(info["skill"]["reward_components"].values()))
            activation += info["skill"]["reward_components"]["activation"]
            discovery += info["skill"]["reward_components"]["discovery"]
            position = env.unwrapped.evaluator_snapshot().agent_position
            diag.update(
                tick,
                obs,
                env.unwrapped.evaluator_snapshot().activated_pois,
                progress=bool(info["transition"]["new_floor"] or info["transition"]["activated"]),
            )
    assert activation == 1.0 and discovery == pytest.approx(0.1)
    row = diag.row()
    assert row["all_pois_activated_step"] == tick
    assert all(r["activated_step"] >= r["seen_step"] for r in row["poi_events"])
    while tick < 256:
        obs, reward, term, trunc, info = env.step(4)
        tick += 1
        assert reward == 0 and term == (tick == 256)
    assert term and not trunc and info["skill"]["success"]
    env.close()


def test_wait_valid_wall_cost_and_mixed_geometry_objective():
    env = configured_skill_env("P1a", 100438, SkillConfig(enabled=True), phase="P3a")
    env.reset()
    _, r, _, _, _ = env.step(4)
    assert r == 0
    _, r, _, _, info = env.step(0)
    assert r == pytest.approx(-0.02)
    assert info["skill"]["source_task"] == "P1a" and info["skill"]["phase"] == "P3a"
    assert env.unwrapped.scenario.horizon == 256
    env.close()


def test_exhausted_world_does_not_count_as_stuck():
    sc = p3_scenario("P3a", 0)
    obs = {"memory": np.zeros((11, 81, 81))}
    diag = ExplorationDiagnostics(sc, obs)
    for t in range(1, 11):
        diag.update(t, obs, frozenset(), progress=False)
    assert diag.row()["unfinished_no_progress_streak"] == 10
    floors = sum(r.count(".") for r in sc.terrain)
    obs["memory"][2].flat[:floors] = 1
    for t in range(11, 100):
        diag.update(t, obs, frozenset(sc.pois), progress=False)
    assert diag.row()["unfinished_no_progress_streak"] == 10
    assert diag.row()["unfinished_steps"] == 10


def test_controller_minimum_caps_and_stop_boundary():
    c = SkillController(index=5, phase_start=376832, family_start=376832)
    assert c.mixture() == [("P3a", 0.8), ("P2c", 0.2)]
    for stage in ("P3a", "P3b", "P3c"):
        start = c.phase_start
        assert c.task == stage
        assert not c.observe(True, start + 16384)
        assert not c.observe(True, start + 32768)
        assert not c.observe(True, start + 49152)
        assert c.observe(True, start + 65536)
    assert c.task == "P4a"
    failed = SkillController(index=7, phase_start=100, family_start=0)
    assert not failed.observe(False, 100 + 524288) and failed.failed


def test_joint_gate_cannot_pass_by_separate_successful_subsets():
    cfg = read_training_config("ather_exploration/resources/training/skills_p3.yaml")
    rows = []
    for det in (True, False):
        for i in range(4):
            rows.append(
                {
                    "deterministic": det,
                    "size": 11,
                    "topology": "two_rooms",
                    "poi_count": 1,
                    "success": 1 if i < 3 else 0,
                    "coverage": 0.5 if i < 3 else 1,
                    "coverage_auc": 0.6,
                    "joint_success": 0,
                    "wall_block": 0,
                    "unfinished_no_progress_streak": 0,
                    "unfinished_no_progress_fraction": 0,
                    "all_pois_activated_step": None,
                    "coverage_after_first_activation": None,
                    "poi_events": [],
                }
            )
    summary = {
        m: {"coverage": 0.8, "coverage_auc": 0.6, "wall_block": 0}
        for m in ("deterministic", "stochastic")
    }
    checks = []

    def check(name, value, threshold, maximum=False):
        ok = value <= threshold if maximum else value >= threshold
        checks.append((name, ok))
        return ok

    passed, _ = add_p3_gates("P3a", rows, summary, check, cfg)
    assert not passed and ("deterministic/joint_success", False) in checks


def test_real_completed_parent_transfer_if_present():
    from ather_exploration.training.checkpoints import inspect_checkpoint
    from ather_exploration.training.skill_environments import SkillTrainingEnv
    from ather_exploration.training.skill_transfer import prepare_p2_transfer

    parent = Path("artifacts/modal/skills-p2-02-resume-01/checkpoints/step_376832")
    if not parent.exists():
        pytest.skip("Local trained checkpoint unavailable")
    cfg = read_training_config("ather_exploration/resources/training/experiments/skills_p3.yaml")
    env = SkillTrainingEnv(cfg)
    try:
        before = (parent / "checksums.json").read_bytes()
        model, state, provenance = prepare_p2_transfer(parent, cfg, env)
        assert model.num_timesteps == 376832 and model._n_updates == 1370
        assert STAGES[state["skill_controller"]["index"]] == "P3a"
        assert provenance["protocol"] == "p2-to-p3-multiroom-v1"
        assert before == (parent / "checksums.json").read_bytes()
        bad = cfg.model_copy(update={"gamma": 0.9})
        with pytest.raises(ValueError, match="config mismatch"):
            prepare_p2_transfer(parent, bad, env)
        with pytest.raises(ValueError, match="source revision"):
            inspect_checkpoint(parent)
    finally:
        env.close()


@pytest.mark.parametrize("task", ["P3a", "P3b", "P3c"])
def test_viewer_loads_new_tasks_and_accounts_reward(task, tmp_path):
    from ather_exploration.agents.learning import build_model
    from ather_exploration.training.checkpoints import save_checkpoint
    from ather_exploration.training.skill_environments import SkillTrainingEnv, skill_identity
    from ather_exploration.ui.session import EpisodeSession, SessionSpec

    cfg = read_training_config("ather_exploration/resources/training/skills_p3.yaml")
    env = SkillTrainingEnv(cfg)
    try:
        model = build_model(cfg, env)
        checkpoint = tmp_path / "checkpoint"
        save_checkpoint(
            model,
            checkpoint,
            cfg,
            {"skill_controller": {"index": STAGES.index(task)}, "viewer_task": task},
            skill_identity(cfg),
        )
        session = EpisodeSession(
            SessionSpec(agent="checkpoint", checkpoint=str(checkpoint), seed=42)
        )
        try:
            assert session.env.phase == task
            assert session.env.unwrapped.scenario.horizon == 256
            for _ in range(8):
                session.step()
        finally:
            session.close()
    finally:
        env.close()
