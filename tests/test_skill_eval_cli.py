"""Skill evaluation CLI exposes the supported post-training task set."""

import subprocess
import sys


def test_evaluate_skills_help_includes_p4_tasks():
    result = subprocess.run(
        [sys.executable, "-m", "ather_exploration", "evaluate-skills", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "P4a" in result.stdout
    assert "P4b" in result.stdout
    assert "P4c" in result.stdout
