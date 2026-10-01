"""G0 dependency smoke on vanilla MiniGrid, never a custom-task training run."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from pathlib import Path


def run_setup_check() -> dict:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import gymnasium as gym
    import numpy as np
    import pygame
    import pygame_gui
    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401

    import minigrid
    from ather_exploration.config import load_preset
    from ather_exploration.fixtures import fixture_catalog, fixture_scenario
    from ather_exploration.schema import observation_space
    from minigrid.wrappers import ImgObsWrapper

    distributions = [
        "ather-exploration",
        "gymnasium",
        "numpy",
        "torch",
        "stable-baselines3",
        "sb3-contrib",
        "pygame-ce",
        "pygame_gui",
        "tensorboard",
        "pydantic",
        "PyYAML",
    ]
    versions = {name: importlib.metadata.version(name) for name in distributions}
    try:
        importlib.metadata.version("pygame")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("pygame and pygame-ce distributions conflict; keep only pygame-ce")

    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    checks = {}
    try:
        env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
        try:
            obs, _ = env.reset(seed=1234)
            image = obs["image"].copy()
            frame = env.render()
            assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
            again, _ = env.reset(seed=1234)
            assert np.array_equal(image, again["image"])
            checks["vanilla_minigrid_reset_render"] = {"frame_shape": list(frame.shape)}
        finally:
            env.close()

        for algorithm in (PPO, RecurrentPPO):
            env = gym.wrappers.FlattenObservation(ImgObsWrapper(gym.make("MiniGrid-Empty-5x5-v0")))
            try:
                policy = "MlpLstmPolicy" if algorithm is RecurrentPPO else "MlpPolicy"
                model = algorithm(
                    policy,
                    env,
                    n_steps=8,
                    batch_size=8,
                    n_epochs=1,
                    device="cpu",
                    seed=1234,
                    verbose=0,
                )
                obs, _ = env.reset(seed=1234)
                action, _ = model.predict(obs, deterministic=True)
                assert env.action_space.contains(action)
                _, reward, _, _, _ = env.step(int(action))
                assert np.isfinite(reward)
                checks[algorithm.__name__] = {"initialize_predict_step": "pass", "device": "cpu"}
            finally:
                env.close()

        pygame.init()
        try:
            screen = pygame.display.set_mode((320, 240))
            manager = pygame_gui.UIManager((320, 240))
            pygame_gui.elements.UIButton(pygame.Rect(10, 10, 100, 35), "G0 smoke", manager)
            manager.update(1 / 60)
            manager.draw_ui(screen)
            checks["pygame_gui_dummy_draw"] = "pass"
        finally:
            pygame.quit()
        for name in ("small", "medium", "large"):
            observation_space(load_preset(name).observation)
        for case in fixture_catalog()["cases"]:
            fixture_scenario(case["name"])
        checks["bundled_presets_and_fixtures"] = "pass"
    finally:
        torch.set_num_threads(original_threads)
    return {
        "gate": "G0 dependency smoke",
        "status": "pass",
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "minigrid_version": minigrid.__version__,
        "minigrid_source": str(Path(minigrid.__file__).resolve()),
        "versions": versions,
        "checks": checks,
        "scope": "Vanilla dependency integration only. No custom environment, learn(), or UI app.",
    }
