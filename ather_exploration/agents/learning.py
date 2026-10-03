"""Shared symbolic encoder and SB3 adapters; construction does not train."""

import numpy as np
import torch
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from ather_exploration.schema import LOCAL_CHANNELS, MEMORY_CHANNELS, STATE_FIELDS
from ather_exploration.training.config import normalize_method
from ather_exploration.types import Action


class ExplorationEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=256)
        local = observation_space["local"].shape
        memory = observation_space["memory"].shape
        if (
            local[0] != 6
            or memory not in ((11, 81, 81), (12, 81, 81), (14, 81, 81))
            or observation_space["state"].shape != (17,)
        ):
            raise ValueError("Incompatible symbolic observation shapes")
        self.local = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * local[1] * local[2], 128),
            nn.ReLU(),
        )
        self.memory = nn.Sequential(
            nn.Conv2d(memory[0], 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, 256),
            nn.ReLU(),
        )
        self.state = nn.Sequential(nn.Linear(17, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(448, 256), nn.ReLU())

    def forward(self, observations):
        return self.fusion(
            torch.cat(
                (
                    self.local(observations["local"].float()),
                    self.memory(observations["memory"].float()),
                    self.state(observations["state"].float()),
                ),
                dim=1,
            )
        )


def schema_signature(space):
    return {
        "version": 3
        if space["memory"].shape[0] == 14
        else 2
        if space["memory"].shape[0] == 12
        else 1,
        "channels": {
            "local": list(LOCAL_CHANNELS),
            "memory": list(MEMORY_CHANNELS)
            + (["frontier"] if space["memory"].shape[0] >= 12 else [])
            + (
                ["previous_monster_visible", "previous_visibility"]
                if space["memory"].shape[0] == 14
                else []
            ),
            "state": list(STATE_FIELDS),
        },
        "shapes": {k: list(v.shape) for k, v in space.spaces.items()},
        "dtypes": {k: str(v.dtype) for k, v in space.spaces.items()},
        "actions": ["NORTH", "SOUTH", "EAST", "WEST", "WAIT"],
        "normalize_images": False,
    }


def algorithm(method):
    method = normalize_method(method)
    return RecurrentPPO if method in ("recurrent_ppo", "recurrent_ppo_curriculum") else PPO


def build_model(config, env):
    recurrent = config.recurrent
    policy_kwargs = {
        "features_extractor_class": ExplorationEncoder,
        "normalize_images": False,
        "ortho_init": True,
        "activation_fn": nn.ReLU,
        "net_arch": {
            "pi": [64] if recurrent else [256, 64],
            "vf": [64] if recurrent else [256, 64],
        },
    }
    if recurrent:
        policy_kwargs.update(
            lstm_hidden_size=256,
            n_lstm_layers=1,
            shared_lstm=False,
            enable_critic_lstm=True,
            share_features_extractor=True,
        )
    # SB3 supplies progress against the full budget, including restored num_timesteps.
    schedule = LinearSchedule(config.learning_rate, config.final_learning_rate)
    model_class = (
        MaskablePPO
        if config.skills.enabled and config.skills.wall_mask
        else algorithm(config.method)
    )
    if config.p4_transfer:
        from ather_exploration.agents.route_ppo import RoutePPO

        model_class = RoutePPO
    return model_class(
        "MultiInputLstmPolicy" if recurrent else "MultiInputPolicy",
        env,
        learning_rate=schedule,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=0.2,
        target_kl=0.03,
        ent_coef=config.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs=policy_kwargs,
        seed=config.seed,
        device=config.device,
        verbose=0,
    )


class LinearSchedule:
    def __init__(self, initial, final):
        self.initial, self.final = initial, final

    def __call__(self, progress_remaining):
        return self.final + (self.initial - self.final) * max(0.0, min(1.0, progress_remaining))


class LearnedAgent:
    def __init__(self, model, metadata):
        self.model, self.metadata = model, metadata
        self.name = "recurrent" if isinstance(model, RecurrentPPO) else "ppo"
        self.model.policy.set_training_mode(False)

    def act(self, observation, state, *, deterministic, action_rng):
        # Stochastic sampling uses the caller RNG, never global Torch RNG shared by training.
        with torch.no_grad():
            tensor, _ = self.model.policy.obs_to_tensor(observation)
            if isinstance(self.model, RecurrentPPO):
                if state.recurrent is None:
                    shape = self.model.policy.lstm_hidden_state_shape
                    state.recurrent = (np.zeros(shape, np.float32), np.zeros(shape, np.float32))
                hidden = tuple(
                    torch.as_tensor(v, device=self.model.device) for v in state.recurrent
                )
                distribution, next_state = self.model.policy.get_distribution(
                    tensor,
                    hidden,
                    torch.as_tensor(
                        [state.episode_start], device=self.model.device, dtype=torch.float32
                    ),
                )
                state.recurrent = tuple(v.cpu().numpy() for v in next_state)
            elif isinstance(self.model, MaskablePPO):
                from ather_exploration.worlds.skill_tasks import wall_mask

                distribution = self.model.policy.get_distribution(
                    tensor, action_masks=wall_mask(observation)
                )
            else:
                distribution = self.model.policy.get_distribution(tensor)
            probabilities = distribution.distribution.probs.cpu().numpy()[0].astype(float)
        if not np.isfinite(probabilities).all():
            raise ValueError("Nonfinite policy probabilities")
        action = (
            int(probabilities.argmax())
            if deterministic
            else int(action_rng.choice(5, p=probabilities / probabilities.sum()))
        )
        state.episode_start = False
        return Action(action), state
