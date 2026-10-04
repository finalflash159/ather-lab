"""Privileged dataset browser, independent of policies and episode simulation."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ather_exploration.environment.dynamics import monster_positions
from ather_exploration.types import EndReason, EvaluatorSnapshot, Scenario
from ather_exploration.worlds.skill_tasks import configured_skill_env, skill_pool

TASKS = ("P1a", "P1b", "P2a", "P2b", "P2c", "P3a", "P3b", "P3c", "P4a", "P4b", "P4c")


@dataclass(frozen=True)
class PreviewMap:
    scenario: Scenario
    task: str
    split: str
    index: int
    count: int
    identity: str

    def snapshot(self, tick=0):
        # Patrol timeline only: spawn remains a marker, no agent/environment step.
        return EvaluatorSnapshot(
            self.scenario,
            self.scenario.spawn,
            tuple(monster_positions(self.scenario, tick)),
            frozenset(),
            tick,
            EndReason.NONE,
        )


def load_map(config, task, split, index):
    if task not in TASKS or split not in ("train", "validation"):
        raise ValueError("Select a supported task and train/validation split")
    if task == "P4c" and not config.skills.p4.enabled:
        raise ValueError("P4c requires a config with skills.p4.enabled")
    count = config.skills.train_count if split == "train" else config.skills.validation_count
    if not 0 <= index < count:
        raise ValueError(f"Map index must be 0..{count - 1}")
    seed, identity = skill_pool(task, count, split == "validation", p4=config.skills.p4.enabled)[
        index
    ]
    env = configured_skill_env(task, seed, config.skills, phase=task)
    try:
        scenario = env.unwrapped.scenario
    finally:
        env.close()
    return PreviewMap(scenario, task, split, index, count, identity)


class MapPreview:
    def __init__(self, config, task="P4a", split="train", index=0):
        import pygame
        import pygame_gui
        from pygame_gui.elements import UIButton, UIDropDownMenu, UITextEntryLine

        self.config = config
        self.task, self.split, self.index = task, split, index
        pygame.init()
        pygame.display.set_caption("Ather | Dataset map preview (no policy)")
        self.screen = pygame.display.set_mode((1100, 850), pygame.RESIZABLE)
        self.manager = pygame_gui.UIManager(self.screen.get_size())
        self.font = pygame.font.Font(None, 23)
        tasks = list(TASKS if config.skills.p4.enabled else TASKS[:-1])
        self.widgets = {
            "task": UIDropDownMenu(tasks, task, pygame.Rect(16, 44, 140, 34), self.manager),
            "split": UIDropDownMenu(
                ["train", "validation"], split, pygame.Rect(168, 44, 160, 34), self.manager
            ),
            "index": UITextEntryLine(pygame.Rect(340, 44, 80, 34), self.manager),
        }
        self.widgets["index"].set_text(str(index))
        for name, label, x, width in (
            ("go", "Go", 432, 60),
            ("prev", "Previous", 504, 100),
            ("next", "Next map", 616, 100),
            ("reset", "Reset", 728, 85),
            ("play", "Play patrol", 825, 120),
            ("step", "Tick +1", 957, 100),
        ):
            self.widgets[name] = UIButton(pygame.Rect(x, 44, width, 34), label, self.manager)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.record = None
        self.tick, self.playing, self.elapsed = 0, False, 0.0
        self.error = ""
        self.request()

    def request(self):
        if self.future is not None:
            return
        self.tick, self.playing, self.elapsed = 0, False, 0.0
        self.widgets["play"].set_text("Play patrol")
        self.error = ""
        self.future = self.executor.submit(load_map, self.config, self.task, self.split, self.index)
        for widget in self.widgets.values():
            widget.disable()

    def update(self, dt):
        if self.future is not None and self.future.done():
            try:
                self.record = self.future.result()
            except (ValueError, RuntimeError, OSError) as error:
                self.record = None
                self.error = str(error)
            self.future = None
            for widget in self.widgets.values():
                widget.enable()
        if self.playing and self.record:
            self.elapsed += dt
            while self.elapsed >= 0.25:
                self.elapsed -= 0.25
                self.tick = (self.tick + 1) % (self.record.scenario.horizon + 1)
        self.manager.update(dt)

    def event(self, event):
        import pygame_gui

        if self.future is not None:
            self.manager.process_events(event)
            return
        if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.widgets["task"]:
                self.task = event.text
            elif event.ui_element == self.widgets["split"]:
                self.split = event.text
            self.index = 0
            self.widgets["index"].set_text("0")
            self.request()
        elif event.type == pygame_gui.UI_BUTTON_PRESSED:
            name = next((k for k, v in self.widgets.items() if v == event.ui_element), None)
            if name in ("go", "prev", "next"):
                count = (
                    self.config.skills.train_count
                    if self.split == "train"
                    else self.config.skills.validation_count
                )
                try:
                    self.index = (
                        int(self.widgets["index"].get_text())
                        if name == "go"
                        else (self.index + (1 if name == "next" else -1)) % count
                    )
                    if not 0 <= self.index < count:
                        raise ValueError()
                except ValueError:
                    self.error = f"Index must be an integer in 0..{count - 1}"
                else:
                    self.widgets["index"].set_text(str(self.index))
                    self.request()
            elif name == "reset":
                self.tick, self.playing = 0, False
                self.widgets["play"].set_text("Play patrol")
            elif name == "play" and self.record:
                self.playing = not self.playing
                self.widgets["play"].set_text("Pause patrol" if self.playing else "Play patrol")
            elif name == "step" and self.record:
                self.tick = (self.tick + 1) % (self.record.scenario.horizon + 1)
        self.manager.process_events(event)

    def draw(self):
        import pygame

        from ather_exploration.ui.rendering import BG, INK, RED, fit, world_surface

        self.screen.fill(BG)
        lines = [
            "DATASET PREVIEW | Full map and hidden routes | No policy, no training",
            "Task                  Split                     Index (0-based)",
        ]
        for text, y in zip(lines, (8, 84), strict=True):
            self.screen.blit(self.font.render(text, True, INK), (16, y))
        if self.future is not None:
            message = "Loading exact dataset pool..."
        elif self.error:
            message = self.error
        elif self.record:
            r, s = self.record, self.record.scenario
            message = f"{r.task} | {r.split} {r.index}/{r.count - 1} | seed {s.seed} | {len(s.terrain)}x{len(s.terrain[0])} | POI {len(s.pois)} | monsters {len(s.routes)} | tick {self.tick}/{s.horizon}"
        else:
            message = "No map"
        self.screen.blit(self.font.render(message, True, RED if self.error else INK), (16, 112))
        if self.record and self.future is None:
            fit(
                world_surface(self.record.snapshot(self.tick), routes=True),
                self.screen,
                pygame.Rect(16, 150, self.screen.get_width() - 32, self.screen.get_height() - 215),
            )
        self.screen.blit(
            self.font.render(
                "Blue: spawn marker | Gold: POI | Red: monster | Patrol playback does not simulate agent actions/collisions",
                True,
                INK,
            ),
            (16, self.screen.get_height() - 42),
        )
        self.manager.draw_ui(self.screen)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def run(self):
        import pygame

        clock = pygame.time.Clock()
        try:
            running = True
            while running:
                dt = clock.tick(30) / 1000
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.VIDEORESIZE:
                        self.screen = pygame.display.set_mode(
                            (max(1080, event.w), max(600, event.h)), pygame.RESIZABLE
                        )
                        self.manager.set_window_resolution(self.screen.get_size())
                    else:
                        self.event(event)
                self.update(dt)
                self.draw()
                pygame.display.flip()
        finally:
            self.close()
            pygame.quit()
