"""Transition quotas, one-time actor calibration and source-aware P4 telemetry."""

from collections import defaultdict

import numpy as np
import torch


def worker_source(task, worker, n_envs):
    """Fixed workers preserve transition shares despite unequal episode lengths.

    P4b/c allocate 75% of transitions to the active threat task, 12.5% to the
    preceding threat skill, and 12.5% to P3c retention. P4a has no preceding
    threat task, so it retains its existing 75/25 split.
    """
    if task not in ("P4a", "P4b", "P4c") or n_envs < 8 or n_envs % 8:
        raise ValueError("Threat quota requires a P4 task and a multiple of eight workers")
    if not 0 <= worker < n_envs:
        raise ValueError("Worker outside quota")
    unit = n_envs // 8
    if task == "P4a":
        return "P3c" if worker >= 6 * unit else task
    if worker >= 7 * unit:
        return "P3c"
    if worker >= 6 * unit:
        return "P4a" if task == "P4b" else "P4b"
    return task


def p4a_worker_family(worker, n_envs):
    """Equal transition quotas within the twelve P4a workers."""
    if n_envs < 8 or n_envs % 8 or not 0 <= worker < 3 * n_envs // 4:
        raise ValueError("P4a family quota requires 8N workers and a P4a worker")
    return ("crossing", "bypass", "yield_alcoves")[worker // (n_envs // 4)]


def distribution_summary(logits):
    probs = logits.softmax(-1)
    entropy = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
    return {
        "entropy_mean": float(entropy.mean()),
        "wait_mean": float(probs[:, 4].mean()),
        "wait_p10": float(torch.quantile(probs[:, 4], 0.1)),
        "max_probability_p95": float(torch.quantile(probs.max(-1).values, 0.95)),
    }


def calibrate_actor(model, config):
    """Choose the least softening meeting declared TRAIN-only initialization bounds.

    No optimization, labels, validation, hidden monster state or inference helper.
    Scaling final actor weights and bias preserves greedy ordering and the critic.
    This is initialization hygiene, not a learned avoidance policy.
    """
    from ather_exploration.worlds.scenarios import digest
    from ather_exploration.worlds.skill_tasks import configured_skill_env, skill_pool

    pool = skill_pool("P4a", config.skills.train_count, p4=True)
    policy = model.policy
    mode = policy.training
    policy.set_training_mode(False)
    logits = []
    try:
        with torch.no_grad():
            for seed, _ in pool:
                env = configured_skill_env("P4a", seed, config.skills)
                try:
                    obs, _ = env.reset()
                    # Include public history while observing the patrol, not just reset frames.
                    for _ in range(3):
                        tensor, _ = policy.obs_to_tensor(obs)
                        logits.append(policy.get_distribution(tensor).distribution.logits.cpu())
                        obs, _, term, trunc, _ = env.step(4)
                        if term or trunc:
                            break
                finally:
                    env.close()
            original = torch.cat(logits)
            if not torch.isfinite(original).all():
                raise ValueError("Nonfinite transfer logits")
            candidates = []
            for power in range(17):
                scale = 2.0**-power
                summary = distribution_summary(original * scale)
                candidates.append({"scale": scale, **summary})
                if (
                    summary["entropy_mean"] >= 0.8
                    and summary["wait_p10"] >= 0.01
                    and summary["max_probability_p95"] <= 0.95
                ):
                    break
            else:
                raise ValueError("Cannot calibrate transfer actor within declared scale range")
            policy.action_net.weight.mul_(scale)
            policy.action_net.bias.mul_(scale)
    finally:
        policy.set_training_mode(mode)
    return {
        "split": "train",
        "pool_identity": digest(pool),
        "observations": len(original),
        "before": distribution_summary(original),
        "after": summary,
        "scale": scale,
        "candidates": candidates,
        "rule": "largest dyadic scale with entropy_mean>=0.8, WAIT_p10>=0.01, max_prob_p95<=0.95",
    }


class SourceTelemetry:
    """Count every transition, including unfinished episodes; drain per rollout."""

    def __init__(self):
        self.counts = defaultdict(lambda: np.zeros(5, dtype=np.int64))
        self.family_counts = defaultdict(lambda: np.zeros(5, dtype=np.int64))
        self.entropies = defaultdict(list)

    def observe(self, model, locals_, calls):
        infos = locals_["infos"]
        actions = np.asarray(locals_["actions"]).reshape(-1)
        for info, action in zip(infos, actions, strict=True):
            self.counts[info["source_task"]][int(action)] += 1
            if info.get("encounter_family"):
                self.family_counts[info["encounter_family"]][int(action)] += 1
        if calls % 16 == 0:
            with torch.no_grad():
                obs, _ = model.policy.obs_to_tensor(model._last_obs)
                entropy = model.policy.get_distribution(obs).distribution.entropy().cpu().numpy()
            for info, value in zip(infos, entropy, strict=True):
                self.entropies[info["source_task"]].append(float(value))

    def drain(self):
        total = sum(int(counts.sum()) for counts in self.counts.values())
        result = {}
        for source, counts in self.counts.items():
            steps = int(counts.sum())
            prefix = f"skill_train/{source}/"
            result[prefix + "transitions"] = steps
            result[prefix + "transition_fraction"] = steps / total
            for name, count in zip(("north", "south", "east", "west", "wait"), counts, strict=True):
                result[prefix + f"action_{name}_fraction"] = float(count / steps)
            if self.entropies[source]:
                result[prefix + "policy_entropy"] = float(np.mean(self.entropies[source]))
        for family, counts in self.family_counts.items():
            steps = int(counts.sum())
            prefix = f"skill_train/P4a/{family}/"
            result[prefix + "transitions"] = steps
            result[prefix + "transition_fraction"] = steps / total
            result[prefix + "action_wait_fraction"] = float(counts[4] / steps)
        self.counts.clear()
        self.family_counts.clear()
        self.entropies.clear()
        return result


def update_threat_exploration(model, config):
    """Change mixture only at update boundaries; checkpoint constructor tracks it."""
    p4 = config.skills.p4
    if p4.timing:
        return 0.0  # Temperature stays calibrated; no blind time-based annealing.
    progress = max(0, model.num_timesteps - 1638400) / p4.exploration_decay_steps
    epsilon = p4.threat_exploration * max(0.0, 1.0 - progress)
    model.policy.threat_exploration = epsilon
    model.policy_kwargs["threat_exploration"] = epsilon
    return epsilon


def calibrate_threat_temperature(model, config):
    """Train-only public sightings; preserve every network parameter and safe policy."""
    from ather_exploration.agents.threat_policy import threat_present
    from ather_exploration.training.threat_lessons import lesson_seeds
    from ather_exploration.worlds.skill_tasks import configured_skill_env

    policy = model.policy
    logits = []
    seeds = lesson_seeds(config.skills.train_count, 0)
    with torch.no_grad():
        for seed in seeds:
            env = configured_skill_env("P4a", seed, config.skills, threat_lesson=0)
            try:
                obs, _ = env.reset()
                for _ in range(4):
                    tensor, _ = policy.obs_to_tensor(obs)
                    if bool(threat_present(tensor)[0]):
                        latent = policy.mlp_extractor.forward_actor(policy.extract_features(tensor))
                        logits.append(policy.action_net(latent).cpu())
                    obs, _, term, trunc, _ = env.step(4)
                    if term or trunc:
                        break
            finally:
                env.close()
    if not logits:
        raise ValueError("No public threats for temperature calibration")
    logits = torch.cat(logits)
    for power in range(9):
        temperature = float(2**power)
        summary = distribution_summary(logits / temperature)
        if summary["wait_p10"] >= 0.02 and summary["max_probability_p95"] <= 0.95:
            break
    else:
        raise ValueError("Cannot calibrate public threat temperature")
    policy.threat_temperature = temperature
    model.policy_kwargs["threat_temperature"] = temperature
    return {
        "split": "train",
        "temperature": temperature,
        "observations": len(logits),
        "before": distribution_summary(logits),
        "after": summary,
        "annealing": "none; fixed across rollout, evaluation, inference and resume",
    }
