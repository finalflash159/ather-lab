"""Frozen-parent distillation on P3 train states, with guarded auxiliary updates."""

import copy
import hashlib

import numpy as np
import torch

from ather_exploration.training.route_teaching import RouteMemory, tensors


class PolicyMemory(RouteMemory):
    """Parent probabilities depend on all inputs, not only the route map."""

    @staticmethod
    def observation_digest(observation):
        digest = hashlib.sha256()
        for key in sorted(observation):
            value = observation[key]
            digest.update(key.encode())
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.tobytes())
        return digest.digest()


class ThreatRetention:
    """Serializable CPU teacher and bounded, exact public-observation reservoirs."""

    def __init__(self, parent_policy, seed, train_seeds):
        self.teacher = copy.deepcopy(parent_policy).cpu()
        self.teacher.optimizer = None
        self.teacher.set_training_mode(False)
        self.teacher.requires_grad_(False)
        self.memory = PolicyMemory(per_map=8, seed=seed)
        self.guard = PolicyMemory(per_map=4, seed=seed + 1)
        self.train_seeds = tuple(train_seeds)
        self.initialized = False
        self.batches = 4
        self.max_kl = 0.01

    def targets(self, observations):
        channels = self.teacher.observation_space["memory"].shape[0]
        batch = {
            key: torch.as_tensor(np.stack([o[key] for o in observations]), device="cpu")
            for key in observations[0]
        }
        batch["memory"] = batch["memory"][:, :channels]
        with torch.no_grad():
            return self.teacher.get_distribution(batch).distribution.probs.cpu().numpy()

    def collect(self, observations, infos, calls):
        if calls % 16:
            return
        selected = [i for i, info in enumerate(infos) if info.get("teaching_safe_source")]
        if selected:
            obs = [{key: value[i] for key, value in observations.items()} for i in selected]
            labels = self.targets(obs)
            for i, observation, target in zip(selected, obs, labels, strict=True):
                seed = infos[i]["teaching_seed"]
                if seed not in self.train_seeds or infos[i]["source_task"] != "P3c":
                    raise ValueError("Retention only accepts P3c train observations")
                self.memory.offer(observation, target, ("P3c", seed), "learner")
        for i, info in enumerate(infos):
            if info["source_task"].startswith("P4"):
                obs = {key: value[i] for key, value in observations.items()}
                # Bounded per worker reservoir. Labels are unused by the guard.
                self.guard.offer(obs, np.zeros(5, np.float32), ("P4", i), "learner")


def initialize_retention(retention, config):
    """Parent trajectories on 16 spread-out TRAIN maps; inference only."""
    from ather_exploration.worlds.skill_tasks import configured_skill_env

    if retention.initialized:
        return
    seeds = retention.train_seeds[:: max(1, len(retention.train_seeds) // 16)][:16]
    envs, observations = [], []
    try:
        for seed in seeds:
            env = configured_skill_env("P3c", seed, config.skills, phase="P3c")
            envs.append(env)
            observations.append(env.reset()[0])
        active = list(range(len(envs)))
        for tick in range(config.skills.p3_horizon):
            if not active:
                break
            probabilities = retention.targets([observations[i] for i in active])
            next_active = []
            for i, target in zip(active, probabilities, strict=True):
                if tick % 8 == 0:
                    retention.memory.offer(observations[i], target, ("P3c", seeds[i]), "teacher")
                observations[i], _, term, trunc, _ = envs[i].step(int(target.argmax()))
                retention.memory.collection_steps += 1
                if not (term or trunc):
                    next_active.append(i)
            active = next_active
        retention.initialized = True
    finally:
        for env in envs:
            env.close()


def divergence(target, current):
    return (target * (target.clamp_min(1e-12).log() - current.clamp_min(1e-12).log())).sum(1)


def retain(model):
    """Post-PPO KL distillation; reject the whole auxiliary round if P4 drifts.

    The reference is the CURRENT post-PPO policy, not the unsafe P3 parent.
    On rejection both policy parameters and optimizer state are restored.
    """
    retention, policy = model.threat_retention, model.policy
    result = {"accepted": 0, "rejected": 0, "kl": 0.0, "loss": 0.0}
    guard_rows = retention.guard.sample(64)
    if not len(retention.memory) or guard_rows is None:
        return result
    guard, _ = tensors(guard_rows, model.device)
    weights = copy.deepcopy(policy.state_dict())
    optimizer = copy.deepcopy(policy.optimizer.state_dict())
    with torch.no_grad():
        reference = policy.get_distribution(guard).distribution.probs.detach().clone()
    try:
        for _ in range(retention.batches):
            obs, target = tensors(retention.memory.sample(32), model.device)
            current = policy.get_distribution(obs).distribution.probs
            loss = divergence(target, current).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite retention loss")
            policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm)
            policy.optimizer.step()
            with torch.no_grad():
                current = policy.get_distribution(guard).distribution.probs
                kl = divergence(reference, current).mean()
            if not torch.isfinite(kl) or kl > retention.max_kl:
                policy.load_state_dict(weights)
                policy.optimizer.load_state_dict(optimizer)
                result.update(accepted=0, rejected=1)
                break
            result.update(accepted=result["accepted"] + 1, kl=float(kl), loss=float(loss.detach()))
        else:
            retention.memory.aux_updates += result["accepted"]
    except BaseException:
        policy.load_state_dict(weights)
        policy.optimizer.load_state_dict(optimizer)
        raise
    finally:
        policy.optimizer.zero_grad(set_to_none=True)
    return result
