"""MiniGrid runtime for fixtures and G2 validated procedural scenarios."""

from dataclasses import asdict, replace
from typing import ClassVar

import gymnasium as gym
import numpy as np

from ather_exploration.config import EnvConfig, ObservationConfig, RewardConfig, load_preset
from ather_exploration.environment.dynamics import advance, make_grid
from ather_exploration.environment.memory import PublicMemoryWrapper
from ather_exploration.environment.reward import RewardTracker
from ather_exploration.environment.visibility import disk_offsets, sense, visibility_mask
from ather_exploration.fixtures import fixture_catalog, fixture_scenario
from ather_exploration.types import Action, EpisodeState, EvaluatorSnapshot, Scenario
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv


class ExplorationEnv(MiniGridEnv):
    """Core emits local observation/public events; PublicMemoryWrapper adds history.

    Main-task scenarios require a validated record; procedural resets produce one.
    """

    metadata: ClassVar[dict] = {"render_modes": ["ansi", "rgb_array"], "render_fps": 5}

    def __init__(
        self,
        scenario: Scenario | None = None,
        *,
        env_config: EnvConfig | None = None,
        generated=None,
        cache=None,
        observation_config: ObservationConfig | None = None,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ):
        from ather_exploration.worlds.generation import check_generated

        if generated is not None:
            if scenario is not None or env_config is not None:
                raise ValueError("Provide generated record alone")
            check_generated(generated)
            scenario, env_config = generated.scenario, generated.config
        if scenario is not None and not scenario.fixture_only and generated is None:
            raise ValueError("Main-task scenario requires a validated generated record")
        if scenario is None and env_config is None:
            raise ValueError("Provide a fixture, generated record, or env_config")
        self.procedural_config = env_config if scenario is None else None
        self.generated = generated
        self.cache = cache
        self._initial_seed_used = False
        if render_mode not in (None, "ansi", "rgb_array"):
            raise ValueError(
                "Environment supports ansi/rgb_array; interactive human UI belongs to S15"
            )
        self.scenario = scenario
        self.sensor_config = observation_config or (
            env_config.observation if env_config else ObservationConfig()
        )
        self.reward_config = reward_config or (env_config.reward if env_config else RewardConfig())
        self._state = None
        self._collision_stage = None
        self._reward_tracker = RewardTracker(self.reward_config)
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: "Explore and activate POIs"),
            width=len(scenario.terrain[0]) if scenario else env_config.width,
            height=len(scenario.terrain) if scenario else env_config.height,
            max_steps=scenario.horizon if scenario else env_config.horizon,
            agent_view_size=2 * self.sensor_config.radius + 1,
            render_mode=render_mode,
            highlight=False,
        )
        self.actions = Action
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Dict(
            {"local": gym.spaces.Box(0, 1, self.sensor_config.local_shape, np.uint8)}
        )
        w = self.reward_config
        self.reward_range = (
            -w.death,
            w.area * len(disk_offsets(self.sensor_config.radius))
            + w.discovery * self.sensor_config.poi_capacity
            + w.activation,
        )

    def _gen_grid(self, width, height):
        if self.procedural_config is not None:
            from ather_exploration.worlds.generation import generate_scenario

            self.generated = generate_scenario(
                self.procedural_config, self._generation_seed, cache=self.cache
            )
            self.scenario = self.generated.scenario
        self.grid = make_grid(self.scenario)
        self._state = EpisodeState.from_scenario(self.scenario)
        self._collision_stage = None
        self._sync_minigrid_state()

    def _sync_minigrid_state(self):
        self.agent_pos = self._state.agent_position
        self.agent_dir = 0  # Display convention only, never affects actions or sensor.
        self.step_count = self._state.step_count

    def reset(self, *, seed=None, options=None):
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an integer 0..2**64-1 or None")
        if options:
            raise ValueError("reset does not accept scenario/config overrides")
        if self.procedural_config is not None:
            if seed is None and not self._initial_seed_used:
                seed = self.procedural_config.seed
            gym.Env.reset(self, seed=seed)
            self._generation_seed = (
                seed if seed is not None else int(self.np_random.bit_generator.random_raw())
            )
            self._initial_seed_used = True
        obs, info = super().reset(seed=seed, options=options)
        self._reward_tracker.reset(obs["local"])
        return obs, info

    def step(self, action):
        self._require_reset()
        event, self._collision_stage = advance(self.grid, self.scenario, self._state, action)
        self._sync_minigrid_state()
        obs = self.gen_obs()  # Actual final state, including early collision1 phase.
        event, reward = self._reward_tracker.update(obs["local"], event)
        return obs, reward, self._state.done, False, {"transition": asdict(event)}

    def _require_reset(self):
        if self._state is None:
            raise gym.error.ResetNeeded("Call reset before accessing the episode")

    def gen_obs(self):
        self._require_reset()
        return {"local": sense(self.grid, self.scenario, self._state, self.sensor_config.radius)}

    def evaluator_snapshot(self) -> EvaluatorSnapshot:
        self._require_reset()
        return EvaluatorSnapshot(
            self.scenario,
            self._state.agent_position,
            tuple(self._state.monster_positions),
            frozenset(self._state.activated_pois),
            self._state.step_count,
            self._state.end_reason,
        )

    @property
    def collision_stage(self):
        """Privileged debug diagnostic; never included in policy observation/info."""
        return self._collision_stage

    def gen_obs_grid(self, agent_view_size=None):
        raise NotImplementedError("Directional MiniGrid image helpers do not apply; use gen_obs")

    def get_pov_render(self, tile_size):
        raise NotImplementedError("Agent-view UI belongs to S15; policy uses symbolic gen_obs")

    def get_full_render(self, highlight, tile_size):
        return self.get_frame(highlight=highlight, tile_size=tile_size)

    def get_frame(self, highlight=False, tile_size=16, agent_pov=False):
        self._require_reset()
        if agent_pov:
            raise NotImplementedError("Agent-view UI belongs to S15")
        if type(tile_size) is not int or tile_size < 8:
            raise ValueError("tile_size must be an integer >=8")
        highlighted = np.zeros((self.width, self.height), dtype=bool)
        if highlight:
            mask = visibility_mask(self.grid, self.agent_pos, self.sensor_config.radius)
            radius = self.sensor_config.radius
            for row, col in np.argwhere(mask):
                highlighted[self.agent_pos[0] + col - radius, self.agent_pos[1] + row - radius] = (
                    True
                )
        # MiniGrid renders terrain; sidecars render independently so POI is not overwritten.
        frame = self.grid.render(tile_size, (-1, -1), 0, highlight_mask=highlighted)
        for x, y in self.scenario.pois:
            color = (60, 200, 100) if (x, y) in self._state.activated_pois else (230, 190, 50)
            tile = frame[y * tile_size : (y + 1) * tile_size, x * tile_size : (x + 1) * tile_size]
            tile[2:4, 2:-2] = color
            tile[-4:-2, 2:-2] = color
            tile[2:-2, 2:4] = color
            tile[2:-2, -4:-2] = color
        for x, y in self._state.monster_positions:
            frame[
                y * tile_size + tile_size // 3 : y * tile_size + 2 * tile_size // 3,
                x * tile_size + tile_size // 3 : x * tile_size + 2 * tile_size // 3,
            ] = (230, 70, 50)
        x, y = self._state.agent_position
        frame[
            y * tile_size + 1 : y * tile_size + max(3, tile_size // 3),
            x * tile_size + 1 : x * tile_size + max(3, tile_size // 3),
        ] = (60, 160, 255)
        return frame

    def render(self):
        self._require_reset()
        if self.render_mode == "rgb_array":
            return self.get_frame()
        if self.render_mode == "ansi":
            rows = [list(row) for row in self.scenario.terrain]
            for x, y in self.scenario.pois:
                rows[y][x] = "P" if (x, y) in self._state.activated_pois else "p"
            for x, y in self._state.monster_positions:
                rows[y][x] = "M"
            x, y = self._state.agent_position
            rows[y][x] = "A" if self._state.alive else "X"
            return "\n".join("".join(row) for row in rows)
        return None


def make_fixture_env(
    name: str,
    *,
    radius: int | None = None,
    horizon: int | None = None,
    render_mode: str | None = None,
) -> PublicMemoryWrapper:
    """G1 test/learning factory. Seed never changes the named fixture's geometry."""
    scenario = fixture_scenario(name)
    if horizon is not None:
        scenario = replace(scenario, horizon=horizon)
    if radius is None:
        radius = next(c["radius"] for c in fixture_catalog()["cases"] if c["name"] == name)
    config = ObservationConfig(radius=radius)
    core = ExplorationEnv(scenario, observation_config=config, render_mode=render_mode)
    return PublicMemoryWrapper(core, config, scenario.horizon)


def make_env(config=None, *, cache=None, generated=None, render_mode=None):
    """Main-task factory. Procedural resets and immutable bank starts share G1 dynamics."""
    if generated is not None:
        if config is not None or cache is not None:
            raise ValueError("A fixed generated record cannot also have procedural config/cache")
        core = ExplorationEnv(generated=generated, render_mode=render_mode)
        config = generated.config
    else:
        config = config or load_preset("small")
        core = ExplorationEnv(env_config=config, cache=cache, render_mode=render_mode)
    return PublicMemoryWrapper(core, config.observation, config.horizon)
