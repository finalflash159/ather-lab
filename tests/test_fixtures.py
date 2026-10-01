import pytest

from ather_exploration.fixtures import fixture_catalog, fixture_scenario, require_main_task_scenario
from ather_exploration.types import EpisodeState, normalize_action


@pytest.mark.parametrize("case", fixture_catalog()["cases"], ids=lambda case: case["name"])
def test_fixtures_are_well_formed_and_isolated(case):
    scenario = fixture_scenario(case["name"])
    assert case["expected"] and case["purpose"]
    assert len(case["actions"]) <= scenario.horizon
    for action in case["actions"]:
        normalize_action(action)
    EpisodeState.from_scenario(scenario)
    with pytest.raises(ValueError, match="fixture"):
        require_main_task_scenario(scenario)
