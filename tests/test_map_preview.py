"""Dataset preview uses real pools and never invokes an agent."""

import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pygame_gui
import pytest

from ather_exploration.training.config import read_training_config
from ather_exploration.ui.map_preview import TASKS, MapPreview, load_map
from ather_exploration.worlds.skill_tasks import skill_pool


@pytest.fixture
def config():
    cfg = read_training_config("ather_exploration/resources/training/skills_p4.yaml")
    return cfg.model_copy(
        update={"skills": cfg.skills.model_copy(update={"train_count": 3, "validation_count": 3})}
    )


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("split", ["train", "validation"])
def test_exact_pool(config, task, split):
    record = load_map(config, task, split, 1)
    seed, identity = skill_pool(task, 3, split == "validation", p4=True)[1]
    assert record.scenario.seed == seed
    assert record.identity == identity
    assert load_map(config, task, split, 1) == record
    assert record.snapshot(10).agent_position == record.scenario.spawn
    assert not record.snapshot(10).activated_pois


def test_invalid_index(config):
    with pytest.raises(ValueError, match="index"):
        load_map(config, "P1a", "train", 3)


def test_ui_switch_next_reset_and_render(config, tmp_path):
    viewer = MapPreview(config)

    def ready():
        until = time.monotonic() + 15
        while viewer.future is not None and time.monotonic() < until:
            viewer.update(0.03)
            time.sleep(0.01)
        assert viewer.future is None and not viewer.error

    def button(name):
        viewer.event(
            pygame.event.Event(pygame_gui.UI_BUTTON_PRESSED, ui_element=viewer.widgets[name])
        )

    try:
        ready()
        button("next")
        ready()
        assert viewer.record.index == 1
        button("step")
        assert viewer.tick == 1
        button("reset")
        assert viewer.tick == 0 and viewer.record.index == 1
        for name, value in (("task", "P4c"), ("split", "validation")):
            viewer.event(
                pygame.event.Event(
                    pygame_gui.UI_DROP_DOWN_MENU_CHANGED,
                    ui_element=viewer.widgets[name],
                    text=value,
                )
            )
            ready()
        assert viewer.record.task == "P4c" and viewer.record.split == "validation"
        viewer.draw()
        pygame.image.save(viewer.screen, tmp_path / "preview.png")
    finally:
        viewer.close()
        pygame.quit()
