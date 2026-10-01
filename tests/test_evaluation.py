import json

import pytest

from ather_exploration.environment.env import make_fixture_env
from ather_exploration.evaluation.metrics import recompute_progress_metrics
from ather_exploration.evaluation.runner import (
    evaluate_baselines,
    evaluate_episode,
    read_evaluation,
)
from ather_exploration.worlds.scenarios import read_record


def test_raw_evidence_roundtrip_and_recomputed_counts(tmp_path):
    output = tmp_path / "run"
    result = evaluate_baselines(seeds=(42,), action_repeats=2, output=output)
    assert result["episodes"] == 3 and result["status"] == "pass"
    records, steps, summary = read_evaluation(output)
    assert result == summary
    for record in records:
        trace = [r for r in steps if r["episode_id"] == record["episode_id"]]
        rebuilt = recompute_progress_metrics(
            trace, horizon=record["H"], floor_count=record["F"], poi_count=record["K"]
        )
        for key in ("coverage", "coverage_gain", "coverage_auc", "activation_auc"):
            assert rebuilt[key] == pytest.approx(record["metrics"][key])
        assert rebuilt["return"] == pytest.approx(record["return"])
    with pytest.raises(FileExistsError):
        evaluate_baselines(output=output)
    path = output / "steps.jsonl"
    path.write_text(path.read_text() + "{}\n")
    with pytest.raises(ValueError, match="checksum"):
        read_evaluation(output)


def test_exception_becomes_explicit_failed_episode():
    class FailingAgent:
        name = "failure-test"

        def act(self, *args, **kwargs):
            raise RuntimeError("test failure")

    env = make_fixture_env("poi_revisit")
    result, steps = evaluate_episode(env, FailingAgent(), episode_id="failed", group="small")
    assert result["status"] == "failed" and result["metrics"]["survival"] is None
    assert "test failure" in result["failure"] and len(steps) == 1
    env.close()


def test_failed_generation_has_explicit_trial_manifest(tmp_path, monkeypatch):
    from ather_exploration.evaluation import runner as evaluation

    def fail(*args, **kwargs):
        raise ValueError("bad candidate")

    monkeypatch.setattr(evaluation, "generate_scenario", fail)
    with pytest.raises(ValueError, match="candidate"):
        evaluate_baselines(output=tmp_path / "failed")
    assert read_record(tmp_path / "failed" / "manifest.json")["state"] == "FAILED"


def test_deterministic_baseline_trace_repeats(tmp_path):
    for name in ("a", "b"):
        evaluate_baselines(agents=("frontier",), seeds=(1,), output=tmp_path / name)
    a, sa, _ = read_evaluation(tmp_path / "a")
    b, sb, _ = read_evaluation(tmp_path / "b")
    assert sa == sb
    a[0].pop("timing")
    b[0].pop("timing")
    assert a == b


def test_bank_evaluation_and_cli(tmp_path):
    import subprocess
    import sys

    from ather_exploration.config import load_preset
    from ather_exploration.worlds.suites import build_development_suite

    bank = tmp_path / "bank"
    build_development_suite(load_preset("small"), 11, bank, count=1)
    result = evaluate_baselines(agents=("random",), bank=bank, action_repeats=1)
    assert result["scenarios"] == 1 and result["episodes"] == 1
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "ather_exploration",
            "evaluate",
            "--agents",
            "random",
            "--action-repeats",
            "1",
            "--seeds",
            "42",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(cli.stdout)["status"] == "pass"


def test_external_truncation_is_logged_and_not_a_death():
    from gymnasium.wrappers import TimeLimit

    from ather_exploration.types import Action

    class WaitingAgent:
        name = "wait-test"

        def act(self, obs, state, **kwargs):
            return Action.WAIT, state

    env = TimeLimit(make_fixture_env("poi_revisit"), max_episode_steps=1)
    result, steps = evaluate_episode(env, WaitingAgent(), episode_id="truncated", group="small")
    assert result["status"] == "cancelled" and result["metrics"]["survival"] is None
    assert result["cancellation_reason"] == "external_truncation"
    assert steps[-1]["truncated"] and not steps[-1]["terminated"]
    env.close()
