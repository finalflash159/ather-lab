"""Threat curriculum contracts, including real checkpoint migration; no learning."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from ather_exploration.training.config import read_training_config
from ather_exploration.training.skill_curriculum import SkillController
from ather_exploration.types import ValidatorStatus
from ather_exploration.worlds.p4_tasks import p4_pool, p4_scenario
from ather_exploration.worlds.skill_tasks import configured_skill_env
from ather_exploration.worlds.validation import replay_witness, validate_scenario


@pytest.fixture
def config():
    return read_training_config("ather_exploration/resources/training/skills_p4.yaml")


@pytest.fixture(autouse=True)
def no_learning(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No training authorized by these tests")

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    torch.set_num_threads(1)


@pytest.mark.parametrize("task", ["P4a", "P4b", "P4c"])
def test_full_suites_reproduce_and_have_runtime_witness(task):
    train = p4_pool(task, 256)
    val = p4_pool(task, 64, True)
    assert not {i for _, i in train} & {i for _, i in val}
    for seed, _ in (*train, *val):
        sc = p4_scenario(task, seed)
        assert sc == p4_scenario(task, seed)
        result = validate_scenario(sc, max_expansions=300000)
        assert result.status is ValidatorStatus.VALIDATED
        replay_witness(sc, result.actions)
    if task == "P4c":
        assert {len(p4_scenario(task, seed).routes) for seed, _ in val} == {1, 2}


def test_timing_triples_share_geometry():
    for base in range(0, 30, 3):
        scenarios = [p4_scenario("P4a", base + i) for i in range(3)]
        normalized = [replace(s, seed=0, phases=(0,)) for s in scenarios]
        assert normalized[0] == normalized[1] == normalized[2]
        assert {s.phases for s in scenarios} == {(0,), (2,), (4,)}


def test_uploaded_suite_contains_complete_p4_scenarios(config, tmp_path):
    from ather_exploration.worlds.scenarios import read_record
    from ather_exploration.worlds.skill_tasks import build_skill_suite

    config = config.model_copy(
        update={
            "skills": config.skills.model_copy(update={"train_count": 3, "validation_count": 3})
        }
    )
    root = tmp_path / "suite"
    manifest = build_skill_suite(config, root)
    assert manifest["state"] == "READY"
    for task in ("P4a", "P4b", "P4c"):
        for split in ("train", "validation"):
            for seed, identity in manifest["tasks"][task][split]:
                record = read_record(root / task / split / f"{seed}.json")
                assert record["identity"] == identity
                assert record["scenario"]["routes"]
                assert record["scenario"]["phases"]
                assert record["scenario"]["seed"] == seed


@pytest.mark.parametrize("task", ["P4a", "P4b", "P4c", "P3c"])
def test_public_history_rewards_and_episode_completion(task, config):
    env = configured_skill_env(task, 0, config.skills)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    assert not obs["memory"][12:].any()
    scenario = env.unwrapped.scenario
    result = validate_scenario(scenario)
    assert result.status is ValidatorStatus.VALIDATED
    done = False
    for tick, action in enumerate(result.actions, 1):
        previous = np.stack((obs["memory"][9] * obs["memory"][8], obs["memory"][8]))
        obs, reward, term, trunc, info = env.step(action)
        np.testing.assert_array_equal(obs["memory"][12:], previous)
        assert env.observation_space.contains(obs)
        assert reward == pytest.approx(sum(info["skill"]["reward_components"].values()))
        if term or trunc:
            done = True
            assert info["skill"]["success"]
            assert not info["transition"]["died"]
            assert task == "P4a" or tick == scenario.horizon
            break
    assert done
    reset, _ = env.reset()
    assert not reset["memory"][12:].any()
    env.close()


def test_controller_requires_minimum_and_two_passes_for_each_stage():
    c = SkillController(index=8, p4_enabled=True, phase_start=1638400, family_start=1638400)
    for task in ("P4a", "P4b", "P4c"):
        assert c.task == task
        start = c.phase_start
        assert not c.observe(True, start + 16384)
        assert not c.observe(True, start + 32768)
        assert c.observe(True, start + 65536)
    assert c.task == "P5a"
    assert c.family_start == c.phase_start
    legacy = SkillController(index=10)
    assert legacy.task == "P5a"


def test_reject_impossible_minimum(config):
    from ather_exploration.training.config import TrainingConfig

    payload = config.model_dump(mode="json")
    payload["skills"]["p4"].update(minimum=131072, task_budget=65536)
    with pytest.raises(ValueError, match="P4 budgets"):
        TrainingConfig.model_validate(payload)


@pytest.mark.parametrize("task", ["P4a", "P4b", "P4c"])
def test_evaluator_witness_and_idle_have_distinct_success(task, config, monkeypatch):
    import ather_exploration.evaluation.skills as evaluation

    monkeypatch.setattr(evaluation, "skill_pool", lambda *a, **k: ((0, "fixture"),))
    witness = validate_scenario(p4_scenario(task, 0)).actions

    class Witness:
        def act(self, obs, state, **kwargs):
            index = state.recurrent or 0
            state.recurrent = index + 1
            return witness[index], state

    class Idle:
        def act(self, obs, state, **kwargs):
            return 4, state

    result = evaluation.evaluate_skill(None, task, config, agent=Witness())
    for mode in ("deterministic", "stochastic"):
        assert result["summary"][mode]["success"] == 1
        if task != "P4a":
            assert result["summary"][mode]["survival"] == 1
    idle = evaluation.evaluate_skill(None, task, config, agent=Idle())
    assert not idle["passed"]
    assert idle["summary"]["deterministic"]["success"] == 0
    assert idle["summary"]["deterministic"]["joint_success"] == 0
    # A full-map witness proves reachability, never supplies labels to the policy.


def test_worker_retention_never_labels_monster_maps(config):
    from ather_exploration.training.skill_environments import SkillTrainingEnv

    env = SkillTrainingEnv(config)
    env.set_controller(asdict(SkillController(index=8, p4_enabled=True)))
    sources = set()
    for _ in range(24):
        env.reset()
        _, _, _, _, info = env.step(4)
        sources.add(env.task)
        assert info["teaching_safe_source"] == (env.task == "P3c")
        if env.task == "P3c":
            assert env.env.get_wrapper_attr("phase") == "P3c"
            assert not env.env.unwrapped.scenario.routes
    assert sources == {"P3c", "P4a"}
    env.close()


def test_real_completed_parent_forward_equivalence(config, tmp_path):
    from ather_exploration.agents.route_ppo import RoutePPO
    from ather_exploration.training.checkpoints import save_checkpoint
    from ather_exploration.training.p4_transfer import prepare_p4
    from ather_exploration.training.skill_environments import SkillTrainingEnv, skill_identity
    from ather_exploration.ui.session import EpisodeSession, SessionSpec

    parent = Path("artifacts/modal/skills-p3c-teaching-01/checkpoints/step_1638400")
    if not parent.exists():
        pytest.skip("Local audited checkpoint unavailable")
    config = config.model_copy(update={"n_envs": 1, "batch_size": 256})
    env = SkillTrainingEnv(config)
    model, state, audit = prepare_p4(parent, config, env)
    old = RoutePPO.load(parent / "model.zip", device="cpu")
    env.set_controller(state["skill_controller"])
    obs, _ = env.reset()
    before = {**obs, "memory": obs["memory"][:12]}
    with torch.no_grad():
        a, _ = model.policy.obs_to_tensor(obs)
        b, _ = old.policy.obs_to_tensor(before)
        torch.testing.assert_close(
            model.policy.get_distribution(a).distribution.probs,
            old.policy.get_distribution(b).distribution.probs,
        )
        torch.testing.assert_close(model.policy.predict_values(a), old.policy.predict_values(b))
    assert not model.policy.optimizer.state
    assert model.num_timesteps == 1638400
    assert len(model.route_memory) == 0
    assert state["skill_controller"]["p4_enabled"]
    assert audit["optimizer"].startswith("fresh")
    state["workers"] = [env.checkpoint_state()]
    state["viewer_task"] = "P4a"
    child = tmp_path / "migrated"
    save_checkpoint(model, child, config, state, skill_identity(config))
    reloaded, resumed_state, _ = prepare_p4(child, config, env)
    assert resumed_state["skill_controller"] == state["skill_controller"]
    for name, value in model.policy.state_dict().items():
        torch.testing.assert_close(value, reloaded.policy.state_dict()[name])
    session = EpisodeSession(SessionSpec(agent="checkpoint", checkpoint=str(child)))
    try:
        for _ in range(8):
            if session.done:
                break
            frame = session.step()
            assert frame["observation"]["memory"].shape[0] == 14
    finally:
        session.close()
    env.close()
