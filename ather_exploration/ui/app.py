"""Autoplay viewer. Configuration belongs to the CLI; metrics belong to W&B."""

import os
import re
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
import pygame_gui
from pygame_gui.elements import UIButton, UIDropDownMenu

from ather_exploration.ui.rendering import (
    BG,
    BLUE,
    GREEN,
    INK,
    MUTED,
    PANEL,
    RED,
    fit,
    public_surface,
    world_surface,
)
from ather_exploration.ui.session import SessionController, SessionSpec


def checkpoint_options(directory):
    """List immutable local checkpoints numerically; selection verifies their contents."""
    root = Path(directory).expanduser().resolve()
    if (root / "checkpoints").is_dir():
        root = root / "checkpoints"
    paths = sorted(
        (
            p
            for p in root.glob("step_*")
            if re.fullmatch(r"step_[0-9]+", p.name)
            and (p / "READY").is_file()
            and not p.is_symlink()
        ),
        key=lambda p: int(p.name.removeprefix("step_")),
    )
    if not paths:
        raise ValueError("No READY checkpoints in checkpoint directory")
    import json

    result = {}
    for p in paths:
        # Labels are metadata only; selection still runs checksum/source verification.
        metadata = json.loads((p / "metadata.json").read_text())
        phase = metadata.get("viewer_task")
        label = f"{phase} | {p.name}" if phase else p.name
        result[label] = str(p)
    return result


class Viewer:
    def __init__(self, spec, *, fps=8, checkpoint_dir=None):
        if spec.agent == "manual":
            raise ValueError("Viewer plays automatically; select an agent/replay through CLI")
        if not 1 <= fps <= 60:
            raise ValueError("fps must be 1..60")
        self.checkpoints = {}
        if checkpoint_dir is not None:
            if (
                spec.checkpoint
                or spec.agent not in ("checkpoint", "ppo", "recurrent")
                or spec.replay
            ):
                raise ValueError(
                    "Checkpoint directory requires a learned agent and no explicit checkpoint/replay"
                )
            self.checkpoints = checkpoint_options(checkpoint_dir)
            spec = replace(spec, checkpoint=next(reversed(self.checkpoints.values())))
        if spec.checkpoint:
            from ather_exploration.training.checkpoints import inspect_checkpoint

            path, _ = inspect_checkpoint(spec.checkpoint, inference=True)
            spec = replace(spec, checkpoint=str(path))
        pygame.init()
        pygame.display.set_caption("Ather | Agent viewer")
        self.screen = pygame.display.set_mode((1200, 760), pygame.RESIZABLE)
        self.manager = pygame_gui.UIManager(self.screen.get_size())
        self.font = pygame.font.Font(None, 24)
        self.title = pygame.font.Font(None, 32)
        self.controller = SessionController()
        self.spec, self.fps = spec, fps
        self.next_step = 0.0
        self.restart_at = None
        self.previous = None
        self.changed_at = 0.0
        self.widgets = {
            "new": UIButton(pygame.Rect(20, 62, 150, 36), "New map", self.manager),
            "reset": UIButton(pygame.Rect(182, 62, 190, 36), "Reset same map", self.manager),
        }
        if spec.fixture or spec.replay:
            self.widgets["new"].disable()
        if self.checkpoints:
            self.widgets["checkpoint"] = UIDropDownMenu(
                list(self.checkpoints),
                next(k for k, v in self.checkpoints.items() if v == spec.checkpoint),
                pygame.Rect(390, 62, 280, 36),
                self.manager,
            )
        self.generate(spec)

    def generate(self, spec):
        # The selected checkpoint stays fixed until the user changes the dropdown.
        self.spec = spec
        self.previous = None
        self.restart_at = None
        self.controller.generate(spec)

    def new_map(self):
        if self.spec.fixture or self.spec.replay:
            self.generate(self.spec)
        else:
            self.generate(
                replace(
                    self.spec, seed=(self.spec.seed + 1) % 2**32, checkpoint=self.spec.checkpoint
                )
            )

    def handle(self, event):
        if event.type == pygame.VIDEORESIZE:
            size = (max(900, event.w), max(620, event.h))
            self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
            self.manager.set_window_resolution(size)
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.widgets["new"]:
                self.new_map()
            elif event.ui_element == self.widgets["reset"]:
                frame = self.controller.frame
                checkpoint = frame.get("checkpoint") if frame else self.spec.checkpoint
                self.generate(replace(self.spec, checkpoint=checkpoint or self.spec.checkpoint))
        if (
            event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED
            and event.ui_element == self.widgets.get("checkpoint")
        ):
            # Restart the same map/action seed; never mix two policies in an episode.
            self.generate(replace(self.spec, checkpoint=self.checkpoints[event.text]))
        self.manager.process_events(event)

    def update(self, dt):
        now = time.monotonic()
        old = self.controller.frame
        generating = self.controller.state == "generating"
        if self.controller.poll():
            frame = self.controller.frame
            if frame and old and frame["row"]["t"] == old["row"]["t"] + 1:
                self.previous = old["snapshot"]
                self.changed_at = now
            else:
                self.previous = None
            if frame and generating:
                self.controller.start()
            if frame and frame["done"]:
                self.restart_at = now + 2
            self.next_step = now + 1 / self.fps
        if self.restart_at is not None and now >= self.restart_at:
            self.new_map()
        if self.controller.running and now >= self.next_step:
            self.controller.step()
            self.next_step = now + 1 / self.fps
        self.manager.update(dt)

    def text(self, value, x, y, color=INK, title=False):
        self.screen.blit(
            (self.title if title else self.font).render(str(value), True, color), (x, y)
        )

    def draw(self):
        self.screen.fill(BG)
        width, height = self.screen.get_size()
        self.text("ATHER / AGENT VIEWER", 20, 20, title=True)
        if not self.checkpoints:
            self.text("Autoplay • Offline checkpoint viewer", 390, 72, MUTED)
        frame = self.controller.frame
        status = self.controller.error or self.controller.state
        self.text(
            f"{(self.controller.frame or {}).get('phase') or self.spec.preset} | seed {self.spec.seed} | {self.spec.agent} | {status}",
            20,
            115,
            RED if self.controller.error else BLUE,
        )
        panels = [
            pygame.Rect(20, 165, (width - 60) // 2, height - 245),
            pygame.Rect(40 + (width - 60) // 2, 165, (width - 60) // 2, height - 245),
        ]
        for rect, label in zip(panels, ("World (viewer only)", "Agent memory"), strict=True):
            pygame.draw.rect(self.screen, PANEL, rect, border_radius=10)
            self.text(label, rect.x + 12, rect.y + 12)
        if frame:
            progress = min(1, (time.monotonic() - self.changed_at) / (0.8 / self.fps))
            surfaces = [
                world_surface(
                    frame["snapshot"],
                    previous=self.previous,
                    progress=progress,
                    collision_stage=frame["row"]["collision_stage"],
                ),
                public_surface(frame["observation"], view="memory"),
            ]
            for surface, rect in zip(surfaces, panels, strict=True):
                fit(
                    surface,
                    self.screen,
                    pygame.Rect(rect.x + 12, rect.y + 44, rect.w - 24, rect.h - 56),
                )
            checkpoint = (
                Path(frame["checkpoint"]).name if frame.get("checkpoint") else "baseline/replay"
            )
            self.text(
                f"Step {frame['row']['t']} | Return {frame['total_reward']:.2f} | {checkpoint}",
                20,
                height - 65,
                GREEN,
            )
        self.text(
            "Blue: agent   Red: monster   Gold: POI   Green: activated | Checkpoint preview, not live training",
            20,
            height - 35,
            MUTED,
        )
        self.manager.draw_ui(self.screen)
        pygame.display.flip()

    def close(self):
        self.controller.close()
        pygame.quit()


def run_ui(spec: SessionSpec, *, fps=8, checkpoint_dir=None):
    app = Viewer(spec, fps=fps, checkpoint_dir=checkpoint_dir)
    clock = pygame.time.Clock()
    try:
        active = True
        while active:
            dt = min(clock.tick(60) / 1000, 0.1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    active = False
                else:
                    app.handle(event)
            app.update(dt)
            app.draw()
    finally:
        app.close()
