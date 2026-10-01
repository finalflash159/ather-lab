"""The viewer must preserve semantics, public boundaries, and tick ownership."""

import time
from threading import Event

import numpy as np
import pytest

from ather_exploration.environment.env import make_fixture_env
from ather_exploration.evaluation.metrics import observation_hash
from ather_exploration.fixtures import fixture_catalog
from ather_exploration.ui.session import EpisodeSession, SessionController, SessionSpec
from ather_exploration.worlds.scenarios import read_record


def drain(controller):
    deadline = time.monotonic() + 10
    while controller.busy:
        assert time.monotonic() < deadline, "UI worker did not finish"
        controller.poll()
        time.sleep(0.001)
    assert not controller.error, controller.error


@pytest.fixture
def display(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    pygame.init()
    yield pygame
    pygame.quit()


@pytest.mark.parametrize("name", ["collision2_poi", "poi_revisit", "corner_occlusion"])
def test_ui_actions_and_repeated_render_match_headless(name, display, tmp_path):
    from ather_exploration.ui.rendering import public_surface, world_surface

    session = EpisodeSession(SessionSpec(fixture=name))
    env = make_fixture_env(name)
    try:
        obs, _ = env.reset(seed=42)
        case = next(c for c in fixture_catalog()["cases"] if c["name"] == name)
        for action in case["actions"]:
            before = session.frame()
            digest = observation_hash(before["observation"])
            for _ in range(3):
                world_surface(before["snapshot"], routes=True)
                public_surface(before["observation"], view="local")
                public_surface(before["observation"], view="memory")
            assert observation_hash(session.obs) == digest
            frame = session.step(action)
            obs, reward, term, trunc, _info = env.step(action)
            assert observation_hash(obs) == observation_hash(frame["observation"])
            assert frame["row"]["reward"] == reward
            assert frame["snapshot"] == env.unwrapped.evaluator_snapshot()
            if term or trunc:
                break
        session.export(tmp_path / "trace.json")
        record = read_record(tmp_path / "trace.json")
        assert record["steps"][-1]["observation_hash"] == observation_hash(obs)
        assert record["scope"] == "debug_not_formal_evaluation"
    finally:
        session.close()
        env.close()


def test_public_renderer_has_no_world_input_and_hides_stale_position(display):
    from ather_exploration.ui.rendering import public_surface

    session = EpisodeSession(SessionSpec(fixture="poi_revisit"))
    try:
        obs = {k: v.copy() for k, v in session.obs.items()}
        first = display.image.tobytes(public_surface(obs), "RGB")
        # Changing only the privileged monster state cannot change a public panel.
        session.env.unwrapped._state.monster_positions = [(1, 1)]
        assert display.image.tobytes(public_surface(obs), "RGB") == first
        # Detached frame arrays cannot change the simulation's memory.
        frame = session.frame()
        frame["observation"]["memory"][:] = 0
        assert np.any(session.obs["memory"])
    finally:
        session.close()


def test_random_ui_reset_reproduces_trace_without_render_rng(display):
    from ather_exploration.ui.rendering import world_surface

    traces = []
    for renders in (0, 4):
        session = EpisodeSession(SessionSpec(fixture="poi_revisit", agent="random", action_seed=16))
        try:
            while not session.done:
                for _ in range(renders):
                    world_surface(session.frame()["snapshot"])
                session.step()
            traces.append(session.metrics.steps)
        finally:
            session.close()
    assert traces[0] == traces[1]


def test_controller_stale_generation_cancel_and_error_recovery(monkeypatch):
    from ather_exploration.ui import session as ui_session

    started, release = Event(), Event()
    original = ui_session.EpisodeSession

    def slow(spec):
        if spec.seed == 1:
            started.set()
            assert release.wait(5)
        return original(spec)

    monkeypatch.setattr(ui_session, "EpisodeSession", slow)
    controller = SessionController()
    try:
        controller.generate(SessionSpec(seed=1, fixture="poi_revisit"))
        assert started.wait(2)
        controller.cancel()
        controller.generate(SessionSpec(seed=2, fixture="collision2_poi"))
        release.set()
        drain(controller)
        assert controller.frame["spec"].seed == 2
        controller.step(2)
        drain(controller)
        assert controller.frame["done"] and controller.state == "ended"
        controller.generate(SessionSpec(fixture="does-not-exist"))
        deadline = time.monotonic() + 5
        while controller.busy and time.monotonic() < deadline:
            controller.poll()
            time.sleep(0.001)
        assert controller.state == "error" and controller.error
        controller.generate(SessionSpec(fixture="poi_revisit"))
        drain(controller)
        assert controller.state == "paused"
    finally:
        release.set()
        controller.close()


def test_viewer_autoplays_and_only_has_reset_new_controls(display):
    import pygame_gui

    from ather_exploration.ui.app import Viewer

    app = Viewer(SessionSpec(fixture="poi_revisit", agent="random"))
    try:
        assert set(app.widgets) == {"new", "reset"}
        deadline = time.monotonic() + 5
        while app.controller.busy:
            app.update(0.01)
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert app.controller.running
        app.draw()
        app.next_step = 0
        app.update(0.01)
        drain(app.controller)
        assert app.controller.frame["row"]["t"] == 1
        app.handle(
            display.event.Event(pygame_gui.UI_BUTTON_PRESSED, ui_element=app.widgets["reset"])
        )
        while app.controller.busy:
            app.update(0.01)
            time.sleep(0.001)
        assert app.controller.frame["row"]["t"] == 0
        assert app.controller.running
        app.handle(display.event.Event(display.VIDEORESIZE, w=1000, h=700))
        app.draw()
    finally:
        app.close()


def test_viewer_rejects_manual(display):
    from ather_exploration.ui.app import Viewer

    with pytest.raises(ValueError, match="automatically"):
        Viewer(SessionSpec(agent="manual"))


def test_headless_session_does_not_import_ui_or_open_display():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from ather_exploration.ui.session import EpisodeSession, SessionSpec
s = EpisodeSession(SessionSpec(fixture='collision2_poi'))
s.step(2)
assert 'ather_exploration.ui.app' not in sys.modules
import pygame
assert not pygame.display.get_init()
s.close()
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
