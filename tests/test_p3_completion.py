"""Room-completion reward, replay, and P3 boundary resume contracts."""

from pathlib import Path

import numpy as np
import pytest
import torch

from ather_exploration.evaluation.p3 import ExplorationDiagnostics
from ather_exploration.training.config import read_training_config
from ather_exploration.training.p3_completion import (
    BANDS,
    P3PrefixArchive,
    prefix_quality,
)
from ather_exploration.worlds.p3_tasks import (
    p3_scenario,
    room_coverage_fractions,
    room_exploration_potential,
)
from ather_exploration.worlds.skill_tasks import configured_skill_env


@pytest.fixture(autouse=True)
def prohibit_learning(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("P3 completion checks must not train")

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    torch.set_num_threads(1)


def test_p3_scenarios_label_rooms_without_changing_observation():
    for task, expected in (("P3a", 2), ("P3b", 3), ("P3c", 4)):
        scenario = p3_scenario(task, 41)
        assert len(scenario.room_labels) == len(scenario.terrain)
        labels = {value for row in scenario.room_labels for value in row if value >= 0}
        assert labels == set(range(expected))
        assert all(
            scenario.room_labels[y][x] == -1
            for y, row in enumerate(scenario.terrain)
            for x, tile in enumerate(row)
            if tile == "#"
        )


def test_room_potential_rewards_underexplored_rooms_more():
    nearly_empty = room_exploration_potential(np.array([0.1, 0.8]))
    after_progress = room_exploration_potential(np.array([0.2, 0.8]))
    nearly_full = room_exploration_potential(np.array([0.8, 0.8]))
    after_nearly_full = room_exploration_potential(np.array([0.9, 0.8]))
    assert after_progress - nearly_empty > after_nearly_full - nearly_full
    assert room_exploration_potential(np.ones(4)) == pytest.approx(1.0)


def test_room_coverage_uses_public_memory_and_excludes_corridors():
    scenario = p3_scenario("P3b", 41)
    memory = np.zeros((11, 81, 81), dtype=np.float32)
    center = memory.shape[-1] // 2
    sx, sy = scenario.spawn
    for y, row in enumerate(scenario.room_labels):
        for x, label in enumerate(row):
            if label == 0:
                memory[2, center + y - sy, center + x - sx] = 1
    coverage = room_coverage_fractions(scenario, memory)
    assert len(coverage) == 3
    assert coverage[0] == pytest.approx(1.0)
    assert coverage[1:] == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize("phase", ["P3b", "P3c"])
def test_room_bonus_is_positive_on_room_progress_and_absent_in_p3a(phase):
    config = read_training_config("ather_exploration/resources/training/skills_p3_completion.yaml")
    assert config.skills.p3_reward.room_exploration == pytest.approx(1.0)
    env = configured_skill_env(phase, 41, config.skills)
    try:
        observation, _ = env.reset()
        path = _path_to_unseen_room(env, observation)
        received = []
        for action in path:
            observation, _, terminated, truncated, info = env.step(action)
            assert not terminated and not truncated
            received.append(info["skill"]["reward_components"]["room_exploration"])
        assert any(value > 0 for value in received)
    finally:
        env.close()

    p3a = configured_skill_env("P3a", 41, config.skills)
    try:
        p3a.reset()
        _, _, _, _, info = p3a.step(4)
        assert info["skill"]["reward_components"]["room_exploration"] == 0
    finally:
        p3a.close()


def test_p3b_c_coverage_gate_measures_least_explored_room():
    for task in ("P3b", "P3c"):
        scenario = p3_scenario(task, 41)
        memory = np.zeros((11, 81, 81), dtype=np.float32)
        center = memory.shape[-1] // 2
        sx, sy = scenario.spawn
        # Reveal almost all cells in every room except the final room.
        labels = np.asarray(scenario.room_labels)
        for room_id in range(max(map(max, scenario.room_labels)) + 1):
            ys, xs = np.where(labels == room_id)
            if room_id != max(map(max, scenario.room_labels)):
                for x, y in zip(xs, ys, strict=True):
                    memory[2, center + y - sy, center + x - sx] = 1
        observation = {"memory": memory}
        diagnostic = ExplorationDiagnostics(scenario, observation)
        assert diagnostic.coverage == 0
        assert diagnostic.row()["tile_coverage"] > diagnostic.coverage


def test_prefix_archive_prioritizes_suffix_progress_and_is_phase_local():
    assert BANDS == ((0, 1), (2, 4), (5, 8))
    assert prefix_quality([0.1, 0.8, 0.8], [0.4, 0.8, 0.8], 0, 0, 2) > 0
    assert prefix_quality([0.1, 0.8, 0.8], [0.1, 0.8, 0.8], 0, 0, 2) == 0
    config = read_training_config("ather_exploration/resources/training/skills_p3_completion.yaml")
    archive = P3PrefixArchive(config.p3_resume, np.random.default_rng(3))
    archive.offer(
        "P3b",
        {"task": "P3b", "seed": 10, "band": 0, "quality": 0.4, "actions": [4]},
    )
    item, requested = archive.choose("P3b")
    assert requested and item["seed"] == 10
    assert archive.choose("P3c")[0] is None
    state = archive.state()
    restored = P3PrefixArchive(config.p3_resume, np.random.default_rng(99))
    restored.restore(state)
    assert restored.state() == state


def test_productive_prefix_is_captured_scored_and_replayed():
    config = read_training_config("ather_exploration/resources/training/skills_p3_completion.yaml")
    archive = P3PrefixArchive(
        config.p3_resume.model_copy(update={"restart_probability": 0.5}),
        np.random.default_rng(3),
    )
    env = configured_skill_env("P3b", 41, config.skills, phase="P3b")
    try:
        observation, _ = env.reset()
        actions = [4] * 16
        for action in actions:
            observation, _, terminated, truncated, _ = env.step(action)
            assert not terminated and not truncated
        candidate = archive.capture(
            "P3b",
            41,
            actions,
            observation,
            env.unwrapped.scenario,
            len(env.unwrapped.evaluator_snapshot().activated_pois),
            set(),
        )
        assert candidate is not None
        candidate["source_task"] = "P3b"
        assert candidate["band"] == 1

        restarted = configured_skill_env("P3b", 41, config.skills, phase="P3b")
        try:
            observation, _ = archive.replay(restarted, candidate)
            for action in _path_to_unseen_room(restarted, observation):
                observation, _, terminated, truncated, _ = restarted.step(action)
                assert not terminated and not truncated
            qualities = archive.complete(
                "P3b",
                [candidate],
                observation,
                restarted.unwrapped.scenario,
                len(restarted.unwrapped.evaluator_snapshot().activated_pois),
            )
            assert qualities and qualities[0] > 0
        finally:
            restarted.close()

        archive.level = 1
        replay_item, requested = archive.choose("P3b")
        assert requested and replay_item["source_task"] == "P3b"
        assert replay_item["band"] == 1
        assert archive.choose("P3c")[0] is None
    finally:
        env.close()


def test_p3c_starts_with_its_own_easy_prefix_level():
    from ather_exploration.training.skill_curriculum import SkillController

    controller = SkillController(index=6, phase_start=0, restart_level=2, restart_results=[True])
    assert not controller.observe(True, 65536)
    assert controller.observe(True, 81920)
    assert controller.task == "P3c"
    assert controller.restart_level == 0
    assert controller.restart_results == []


def test_p3_resume_preflight_accepts_only_the_passed_p3a_boundary():
    from ather_exploration.training.p3_completion import check_p3_resume

    parent = Path("artifacts/modal/skills-p3-frontier-01/checkpoints/step_737280")
    if not parent.exists():
        pytest.skip("P3a boundary checkpoint is not present in this checkout")
    config = read_training_config("ather_exploration/resources/training/skills_p3_completion.yaml")
    result = check_p3_resume(parent, config)
    assert result["status"] == "valid"
    assert result["learning_executed"] is False
    assert result["transfer"]["parent_env_steps"] == 737280
    assert result["phase"] == "P3b"
    bad = config.model_copy(update={"gamma": 0.9})
    with pytest.raises(ValueError, match="config mismatch"):
        check_p3_resume(parent, bad)


def test_mid_p3_completion_checkpoint_can_resume_without_learning(tmp_path):
    import copy
    import json
    from dataclasses import asdict

    from stable_baselines3 import PPO

    from ather_exploration.training.checkpoints import save_checkpoint
    from ather_exploration.training.p3_completion import prepare_p3_resume
    from ather_exploration.training.skill_curriculum import SkillController
    from ather_exploration.training.skill_environments import SkillTrainingEnv, skill_identity

    source = Path("artifacts/modal/skills-p3-frontier-01/checkpoints/step_737280")
    if not source.exists():
        pytest.skip("P3a boundary checkpoint is not present in this checkout")
    config = read_training_config("ather_exploration/resources/training/skills_p3_completion.yaml")
    old_state = json.loads((source / "runner_state.json").read_text())
    controller = copy.deepcopy(old_state["skill_controller"])
    step = 737280 + config.n_envs * config.n_steps
    source_revision = old_state.get("transfer")
    env = SkillTrainingEnv(config, worker=0)
    try:
        model = PPO.load(source / "model.zip", env=env, device="cpu")
        model.num_timesteps = step
        model._n_updates += 1
        checkpoint_state = {
            "env_steps": step,
            "state": "RUNNING",
            "viewer_task": "P3b",
            "skill_controller": asdict(SkillController(**controller)),
            "workers": [env.checkpoint_state() for _ in range(config.n_envs)],
            "transfer": {
                **(source_revision or {}),
                "protocol": "p3a-boundary-room-completion-v1",
            },
        }
        checkpoint = tmp_path / "mid-run" / "checkpoints" / f"step_{step}"
        save_checkpoint(model, checkpoint, config, checkpoint_state, skill_identity(config))
        restored, state, audit = prepare_p3_resume(checkpoint, config, env)
        assert restored.num_timesteps == step
        assert state["env_steps"] == step
        assert audit["resume_kind"] == "p3_completion_checkpoint"
        assert audit["resume_rng_checkpoint"] == str(checkpoint.resolve())
    finally:
        env.close()


def _path_to_unseen_room(env, observation):
    scenario = env.unwrapped.scenario
    fractions = room_coverage_fractions(scenario, observation["memory"])
    target_room = int(np.argmin(fractions))
    labels = np.asarray(scenario.room_labels)
    target_cells = {
        (x, y)
        for y, x in zip(*np.where(labels == target_room), strict=True)
        if observation["memory"][2, 40 + y - scenario.spawn[1], 40 + x - scenario.spawn[0]] == 0
    }
    start = scenario.spawn
    queue = [start]
    previous = {start: None}
    for position in queue:
        if position in target_cells:
            break
        x, y = position
        for nxt in ((x, y - 1), (x, y + 1), (x + 1, y), (x - 1, y)):
            nx, ny = nxt
            if (
                nxt not in previous
                and 0 <= ny < len(scenario.terrain)
                and 0 <= nx < len(scenario.terrain[0])
                and scenario.terrain[ny][nx] == "."
            ):
                previous[nxt] = position
                queue.append(nxt)
    target = next((position for position in queue if position in target_cells), None)
    assert target is not None
    positions = []
    while target != start:
        positions.append(target)
        target = previous[target]
    positions.reverse()
    actions = []
    current = start
    for x, y in positions:
        actions.append(
            {(0, -1): 0, (0, 1): 1, (1, 0): 2, (-1, 0): 3}[(x - current[0], y - current[1])]
        )
        current = (x, y)
    return actions
