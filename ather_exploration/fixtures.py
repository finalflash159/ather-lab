"""Hand-specified examples, deliberately separate from accepted training data."""

import json
from importlib.resources import files

from ather_exploration.types import Scenario


def fixture_catalog() -> dict:
    path = files("ather_exploration").joinpath("resources", "fixtures", "catalog.json")
    return json.loads(path.read_text(encoding="utf-8"))


def fixture_scenario(name: str) -> Scenario:
    cases = {case["name"]: case for case in fixture_catalog()["cases"]}
    if name not in cases:
        raise ValueError(f"Unknown fixture: {name}")
    data = cases[name]["scenario"].copy()
    data["terrain"] = tuple(data["terrain"])
    data["spawn"] = tuple(data["spawn"])
    data["pois"] = tuple(tuple(pos) for pos in data["pois"])
    data["routes"] = tuple(tuple(tuple(pos) for pos in route) for route in data["routes"])
    data["phases"] = tuple(data["phases"])
    return Scenario(**data)


def require_main_task_scenario(scenario: Scenario) -> None:
    """Admission guard only; S07 still must produce and replay a valid witness."""
    if scenario.fixture_only:
        raise ValueError("Test fixture must not enter training/evaluation suites")
