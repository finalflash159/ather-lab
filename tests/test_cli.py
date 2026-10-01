import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "args",
    [["config", "--preset", "medium"], ["schema"], ["fixtures", "--name", "corner_occlusion"]],
)
def test_editable_cli_outside_project(tmp_path, args):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", "-m", "ather_exploration", *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert isinstance(json.loads(result.stdout), dict)


def test_unimplemented_command_and_bad_config_report_errors(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("horizon: true\n")
    for args in [["train"], ["config", "--file", str(path)]]:
        result = subprocess.run(
            [sys.executable, "-m", "ather_exploration", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
        assert "error:" in result.stderr
        assert "Traceback" not in result.stderr


def test_rollout_executes_fixture_instead_of_echoing_expected():
    result = subprocess.run(
        [sys.executable, "-m", "ather_exploration", "rollout", "--fixture", "collision2_poi"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data["trace"][-1]["event"]["activated"]
    assert data["trace"][-1]["event"]["died"]
    assert data["trace"][-1]["collision_stage_debug"] == 2
    assert data["return"] == -1.5


def test_g2_cli_generate_validate_and_suite(tmp_path):
    import json
    import subprocess
    import sys

    def run(*args):
        result = subprocess.run(
            [sys.executable, "-m", "ather_exploration", *map(str, args)],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    path = tmp_path / "scenario.json"
    generated = run("generate", "--preset", "small", "--seed", 42, "--output", path)
    assert generated["status"] == "validated" and generated["witness_steps"] == 256
    validated = run("validate", "--scenario", path)
    assert validated["status"] == "validated" and len(validated["actions"]) == 256
    suite = run(
        "build-suite",
        "--root-seed",
        11,
        "--count",
        1,
        "--train-starts",
        2,
        "--output",
        tmp_path / "bank",
    )
    assert suite["state"] == "READY" and len(suite["worlds"]) == 4
