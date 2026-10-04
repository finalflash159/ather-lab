"""Public-threat-conditioned exploration, consistent in sampling and PPO likelihoods."""

import torch
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


class ThreatPolicy(MultiInputActorCriticPolicy):
    def __init__(
        self, *args, threat_exploration=0.1, threat_mode="mixture", threat_temperature=1.0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.threat_exploration = threat_exploration
        self.threat_mode = threat_mode
        self.threat_temperature = threat_temperature
        if threat_mode not in ("mixture", "temperature") or threat_temperature < 1:
            raise ValueError("Invalid threat distribution")
        if not self.share_features_extractor:
            raise ValueError("Threat policy requires the shared symbolic encoder")

    def _distribution(self, obs, latent):
        if self.threat_mode == "temperature":
            logits = self.action_net(latent)
            active = threat_present(obs)
            temperature = torch.where(active, self.threat_temperature, 1.0)
            return CategoricalDistribution(self.action_space.n).proba_distribution(
                logits / temperature[:, None]
            )
        distribution = self._get_action_dist_from_latent(latent)
        # Only CURRENT visible monsters; stale memory cannot trigger exploration.
        memory = obs["memory"]
        visible = ((memory[:, 9] * memory[:, 8]).flatten(1).sum(1) > 0).float()[:, None]
        epsilon = self.threat_exploration * visible
        probabilities = distribution.distribution.probs
        mixed = (1 - epsilon) * probabilities + epsilon / self.action_space.n
        return CategoricalDistribution(self.action_space.n).proba_distribution(
            mixed.clamp_min(1e-30).log()
        )

    def get_distribution(self, obs):
        latent = self.mlp_extractor.forward_actor(self.extract_features(obs))
        return self._distribution(obs, latent)

    def forward(self, obs, deterministic=False):
        actor, critic = self.mlp_extractor(self.extract_features(obs))
        distribution = self._distribution(obs, actor)
        actions = distribution.get_actions(deterministic=deterministic)
        return (
            actions.reshape((-1, *self.action_space.shape)),
            self.value_net(critic),
            distribution.log_prob(actions),
        )

    def evaluate_actions(self, obs, actions):
        actor, critic = self.mlp_extractor(self.extract_features(obs))
        distribution = self._distribution(obs, actor)
        return self.value_net(critic), distribution.log_prob(actions), distribution.entropy()


def threat_present(obs):
    """Current or two-frame recent public sightings, never stale hidden routes."""
    memory = obs["memory"]
    present = (memory[:, 9] * memory[:, 8]).flatten(1).any(1)
    for channel in range(12, memory.shape[1], 2):
        present = present | memory[:, channel].flatten(1).any(1)
    return present
