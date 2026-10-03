"""Persistent public-observation teaching, separate from PPO's rollout objective."""

import copy
import hashlib
import io

import numpy as np
import torch

from ather_exploration.training.public_route import public_route, route_loss


class RouteMemory:
    """Per-map reservoirs for teacher and learner; compressed exact observations.

    Full buckets still consider later candidates. This object, including its RNG,
    is saved inside the model checkpoint. No validation observations are admitted
    by collection callers.
    """

    def __init__(self, per_map=64, seed=0):
        self.per_map = per_map
        self.rng = np.random.default_rng(seed)
        self.buckets = {}
        self.counts = {}
        self.collection_steps = 0
        self.initialized = False
        self.aux_updates = 0

    def offer(self, observation, labels, source, origin):
        if origin not in ("teacher", "learner"):
            raise ValueError("Unknown teaching origin")
        key = (origin, *source)
        if key not in self.buckets and len(self.buckets) >= 1024:
            raise ValueError("Teaching memory map capacity exceeded")
        bucket = self.buckets.setdefault(key, [])
        digest = hashlib.sha256(observation["memory"].tobytes()).digest()
        if any(item[0] == digest for item in bucket):
            return
        self.counts[key] = self.counts.get(key, 0) + 1
        index = (
            len(bucket) if len(bucket) < self.per_map else int(self.rng.integers(self.counts[key]))
        )
        if index >= self.per_map:
            return
        stream = io.BytesIO()
        np.savez_compressed(stream, **observation, labels=labels)
        item = (digest, stream.getvalue())
        if index == len(bucket):
            bucket.append(item)
        else:
            bucket[index] = item

    def sample(self, count):
        groups = {
            origin: [v for k, v in self.buckets.items() if k[0] == origin and v]
            for origin in ("teacher", "learner")
        }
        origins = [k for k, v in groups.items() if v]
        if not origins:
            return None
        rows = []
        for i in range(count):
            buckets = groups[origins[i % len(origins)]]
            bucket = buckets[int(self.rng.integers(len(buckets)))]
            payload = bucket[int(self.rng.integers(len(bucket)))][1]
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                rows.append({k: data[k].copy() for k in data.files})
        return rows

    def __len__(self):
        return sum(map(len, self.buckets.values()))


def tensors(rows, device):
    obs = {
        key: torch.as_tensor(np.stack([r[key] for r in rows]), device=device)
        for key in rows[0]
        if key != "labels"
    }
    labels = torch.as_tensor(np.stack([r["labels"] for r in rows]), device=device)
    return obs, labels


def teach(model, batches=8, max_kl=0.03):
    """Auxiliary-only steps, with post-update KL and exact optimizer rollback.

    A frozen reference is sampled before this round. The KL cap covers all its
    states; an additional agreement penalty protects correct reference actions.
    PPO's optimizer state is preserved and participates in rollback.
    """
    memory = model.route_memory
    rows = memory.sample(128)
    if rows is None:
        return {"accepted": 0, "rejected": 0, "kl": 0.0, "loss": 0.0}
    reference, labels = tensors(rows, model.device)
    policy = model.policy
    with torch.no_grad():
        old = policy.get_distribution(reference).distribution.probs.detach().clone()
        agree = labels[torch.arange(len(labels), device=model.device), old.argmax(1)]
    result = {"accepted": 0, "rejected": 0, "kl": 0.0, "loss": 0.0}
    for _ in range(batches):
        observations, targets = tensors(memory.sample(64), model.device)
        weights = copy.deepcopy(policy.state_dict())
        optimizer = copy.deepcopy(policy.optimizer.state_dict())
        logits = policy.get_distribution(observations).distribution.logits
        loss = route_loss(logits, targets)
        current = policy.get_distribution(reference).distribution.probs
        divergence = (old * (old.clamp_min(1e-12).log() - current.clamp_min(1e-12).log())).sum(1)
        if agree.any():
            loss = loss + 0.1 * divergence[agree].mean()
        policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm)
        policy.optimizer.step()
        try:
            with torch.no_grad():
                current = policy.get_distribution(reference).distribution.probs
                kl = (
                    (old * (old.clamp_min(1e-12).log() - current.clamp_min(1e-12).log()))
                    .sum(1)
                    .mean()
                )
        except (ValueError, RuntimeError):
            policy.load_state_dict(weights)
            policy.optimizer.load_state_dict(optimizer)
            policy.optimizer.zero_grad(set_to_none=True)
            raise
        if not torch.isfinite(kl) or not torch.isfinite(loss) or kl > max_kl:
            policy.load_state_dict(weights)
            policy.optimizer.load_state_dict(optimizer)
            policy.optimizer.zero_grad(set_to_none=True)
            result["rejected"] += 1
            break
        result.update(kl=float(kl), loss=float(loss.detach()))
        result["accepted"] += 1
        memory.aux_updates += 1
    return result


def check_teaching_policy(model, config, seeds):
    """Autonomous TRAIN holdout episodes; no labels or optimizer calls."""
    from ather_exploration.worlds.p3_tasks import room_coverage_fractions
    from ather_exploration.worlds.skill_tasks import configured_skill_env

    successes, joint, auc = [], [], []
    for seed in seeds:
        env = configured_skill_env("P3c", seed, config.skills, phase="P3c")
        try:
            obs, _ = env.reset()
            history = []
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(int(action))
                history.append(
                    float(room_coverage_fractions(env.unwrapped.scenario, obs["memory"]).min())
                )
                if terminated or truncated:
                    success = bool(info["skill"]["success"])
                    successes.append(success)
                    joint.append(success and history[-1] >= config.skills.p3_gates.room_coverage)
                    auc.append(float(np.mean(history)))
                    break
        finally:
            env.close()
    return {
        "episodes": len(seeds),
        "success": float(np.mean(successes)),
        "joint": float(np.mean(joint)),
        "auc": float(np.mean(auc)),
    }


def initialize_teaching(model, config):
    """Teacher trajectories on stratified TRAIN seeds; no PPO buffer insertion."""
    from ather_exploration.training.recovery import probe_seeds
    from ather_exploration.worlds.skill_tasks import configured_skill_env

    memory = model.route_memory
    if memory.initialized:
        return
    seeds = probe_seeds(config, "P3c")
    # The last stratum-balanced round is excluded from teacher collection.
    holdout = seeds[-16:]
    seeds = seeds[:-16]
    before = check_teaching_policy(model, config, holdout)
    for seed in seeds:
        env = configured_skill_env("P3c", seed, config.skills, phase="P3c")
        try:
            obs, _ = env.reset()
            for tick in range(config.skills.p3_horizon):
                route = public_route(obs)
                if route is None:
                    break
                if tick % 2 == 0:
                    memory.offer(obs, route["actions"], ("P3c", seed), "teacher")
                obs, _, terminated, truncated, _ = env.step(
                    int(np.flatnonzero(route["actions"])[0])
                )
                memory.collection_steps += 1
                if terminated or truncated:
                    break
        finally:
            env.close()
    for epoch in range(10):
        result = teach(model, batches=32)
        print(f"[teaching] warmup={epoch + 1}/10 samples={len(memory)} {result}", flush=True)
    after = check_teaching_policy(model, config, holdout)
    memory.warmup_check = {"split": "train", "seeds": holdout, "before": before, "after": after}
    print(f"[teaching] autonomous TRAIN holdout: {memory.warmup_check}", flush=True)
    memory.initialized = True
