"""Version-1 policy tensor contract. This does not generate observations."""

import gymnasium as gym
import numpy as np

from ather_exploration.config import ObservationConfig
from ather_exploration.types import PublicObservation

LOCAL_CHANNELS = (
    "visible_now",
    "wall",
    "walkable_terrain",
    "poi_pending",
    "poi_activated",
    "monster_now",
)
MEMORY_CHANNELS = (
    "seen",
    "wall",
    "walkable_terrain",
    "poi_pending",
    "poi_activated",
    "visited",
    "visit_count_scaled",
    "agent_here",
    "visible_now",
    "monster_at_last_observation",
    "observation_age_scaled",
)
STATE_FIELDS = (
    "previous_north",
    "previous_south",
    "previous_east",
    "previous_west",
    "previous_wait",
    "moved",
    "wall_blocked",
    "waited",
    "delta_x",
    "delta_y",
    "remaining_steps_scaled",
    "episode_horizon_scaled",
    "new_floor_scaled",
    "new_poi_scaled",
    "activated",
    "died",
    "episode_start",
)


def observation_space(config: ObservationConfig) -> gym.spaces.Dict:
    low = np.zeros(len(STATE_FIELDS), dtype=np.float32)
    low[8:10] = -1
    return gym.spaces.Dict(
        {
            "local": gym.spaces.Box(0, 1, config.local_shape, np.uint8),
            "memory": gym.spaces.Box(0, 1, config.memory_shape, np.float32),
            "state": gym.spaces.Box(low, np.ones(len(STATE_FIELDS), dtype=np.float32)),
        }
    )


def validate_observation(observation: PublicObservation, space: gym.spaces.Dict) -> None:
    """Check exact dtype as well as bounds; Gym Box permits some dtype casts."""
    if set(observation) != set(space.spaces):
        raise ValueError("observation keys must be local, memory, state")
    for key, box in space.spaces.items():
        value = observation[key]
        if not isinstance(value, np.ndarray) or value.dtype != box.dtype or not box.contains(value):
            raise ValueError(
                f"invalid {key}: expected dtype={box.dtype}, shape={box.shape}, finite bounds"
            )
