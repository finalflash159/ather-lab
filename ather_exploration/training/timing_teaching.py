"""Balanced public timing supervision, persisted with guarded optimizer updates."""

import copy
import io

import numpy as np
import torch

from ather_exploration.training.public_route import route_loss
from ather_exploration.training.public_timing import (
    public_p4_timing,
    public_p4a_timing,
    public_timing,
)
from ather_exploration.training.route_teaching import tensors
from ather_exploration.training.threat_retention import PolicyMemory, divergence


class TimingTeaching:
    def __init__(self, config):
        from ather_exploration.training.threat_lessons import lesson_seeds
        from ather_exploration.worlds.p4_tasks import group, p4_scenario
        from ather_exploration.worlds.skill_tasks import skill_pool

        self.memories = {
            k: PolicyMemory(per_map=8, seed=config.seed + i) for i, k in enumerate(("wait", "go"))
        }
        self.seeds = {
            task: tuple(s for s, _ in skill_pool(task, config.skills.train_count, p4=True))
            for task in ("P4a", "P4b", "P4c")
        }
        # Do not teach from the geometries reserved for lesson probes.
        self.seeds["P4a"] = lesson_seeds(config.skills.train_count, 3)
        self.balanced_families = config.skills.p4.balanced_families
        self.active_task = "P4a"
        self.family_by_seed = (
            {
                seed: group(p4_scenario("P4a", seed))["encounter_family"]
                for seed in self.seeds["P4a"]
            }
            if self.balanced_families
            else {}
        )
        self.batches = config.skills.p4.timing_batches
        self.coefficient = config.skills.p4.timing_coefficient
        self.max_kl = config.skills.p4.retention_max_kl

    def label(self, observation, task):
        if task == "P4a" and self.balanced_families:
            return public_p4a_timing(observation)
        if task in ("P4b", "P4c"):
            return public_p4_timing(observation)
        return public_timing(observation)

    def offer(self, observation, task, seed, origin="learner"):
        if task not in self.seeds or seed not in self.seeds[task]:
            raise ValueError("Timing labels require train-only seeds")
        label = self.label(observation, task)
        if label:
            # Repeated corridor/pocket loops are legal observations, but poor
            # standalone demonstrations. Successful bootstrap trajectories
            # are stored separately; online labels stop after four visits.
            visited_too_often = False
            if self.balanced_families and task == "P4a" and origin == "learner":
                memory = observation["memory"]
                visited_too_often = bool(
                    (memory[6][memory[7] > 0] > np.log1p(4) / np.log1p(1025)).any()
                )
            if not visited_too_often:
                self.memories[label["kind"]].offer(
                    observation, label["actions"], (task, seed), origin
                )
        return label

    def collect(self, observations, infos):
        for i, info in enumerate(infos):
            task = info["source_task"]
            if task in self.seeds:
                # Full P4a replay may include lesson holdout geometry; never label it.
                if info["teaching_seed"] not in self.seeds[task]:
                    continue
                self.offer({k: v[i] for k, v in observations.items()}, task, info["teaching_seed"])

    def sample(self, count):
        if self.balanced_families and self.active_task == "P4a":
            # Balance P4a encounter families as well as GO/WAIT. Never draw
            # future-task bootstrap or validation samples at this stage.
            groups = {}
            for kind, memory in self.memories.items():
                for key, bucket in memory.buckets.items():
                    if key[1] == "P4a" and bucket:
                        family = self.family_by_seed[key[2]]
                        groups.setdefault((family, kind), []).append((memory, bucket))
            families = sorted({family for family, _ in groups})
            if not families:
                self.last_sample_counts = {}
                return None
            rows = []
            for index in range(count):
                family = families[index % len(families)]
                kinds = [kind for kind in ("go", "wait") if (family, kind) in groups]
                kind = kinds[(index // len(families)) % len(kinds)]
                options = groups[(family, kind)]
                memory = self.memories[kind]
                _, bucket = options[int(memory.rng.integers(len(options)))]
                payload = bucket[int(memory.rng.integers(len(bucket)))][1]
                with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                    rows.append({key: data[key].copy() for key in data.files})
            self.last_sample_counts = {"P4a": len(rows)}
            return rows
        current = self.active_task
        previous = {"P4b": "P4a", "P4c": "P4b"}.get(current)
        weights = {current: 0.8}
        if previous:
            weights[previous] = 0.2
        rows = []
        sample_counts = {}
        for memory in self.memories.values():
            available = {}
            for key, bucket in memory.buckets.items():
                if key[0] in ("teacher", "learner") and key[1] in weights and bucket:
                    available.setdefault(key[1], {}).setdefault(key[0], []).append(bucket)
            tasks = [task for task in weights if task in available]
            if not tasks:
                continue
            probabilities = np.asarray([weights[task] for task in tasks], dtype=float)
            probabilities /= probabilities.sum()
            for _ in range(count // 2):
                task = tasks[int(memory.rng.choice(len(tasks), p=probabilities))]
                origins = list(available[task])
                origin = origins[int(memory.rng.integers(len(origins)))]
                buckets = available[task][origin]
                bucket = buckets[int(memory.rng.integers(len(buckets)))]
                payload = bucket[int(memory.rng.integers(len(bucket)))][1]
                with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                    rows.append({key: data[key].copy() for key in data.files})
                sample_counts[task] = sample_counts.get(task, 0) + 1
        self.last_sample_counts = sample_counts
        return rows or None

    def p4a_family_counts(self):
        if not self.balanced_families:
            return {}
        counts = {family: 0 for family in ("crossing", "bypass", "yield_alcoves")}
        for memory in self.memories.values():
            for key, bucket in memory.buckets.items():
                if key[1] == "P4a":
                    counts[self.family_by_seed[key[2]]] += len(bucket)
        return {f"p4a_{family}_samples": value for family, value in counts.items()}


def initialize_timing(model, config):
    """Public-helper trajectories on train maps; retain successful labels only."""
    from ather_exploration.training.public_route import public_route
    from ather_exploration.training.threat_lessons import family_train_seeds, lesson_seeds
    from ather_exploration.worlds.skill_tasks import configured_skill_env

    teacher = model.timing_teaching
    if teacher.balanced_families:
        teacher.bootstrap_success = {
            family: 0 for family in ("crossing", "bypass", "yield_alcoves")
        }
        for family in teacher.bootstrap_success:
            seeds = family_train_seeds(config.skills.train_count, family)
            seeds = seeds[:: max(1, len(seeds) // 16)][:16]
            for index, seed in enumerate(seeds):
                lesson = (
                    0
                    if family == "crossing" and index % 2 == 0
                    else 2
                    if family == "yield_alcoves" and index % 2 == 0
                    else 3
                )
                env = configured_skill_env("P4a", seed, config.skills, threat_lesson=lesson)
                try:
                    obs, _ = env.reset()
                    candidates = []
                    for _ in range(128):
                        label = teacher.label(obs, "P4a")
                        if label:
                            candidates.append(
                                ({key: value.copy() for key, value in obs.items()}, label)
                            )
                        route = public_route(obs) if label is None else None
                        action = (
                            int(np.flatnonzero(label["actions"])[0])
                            if label
                            else int(np.flatnonzero(route["actions"])[0])
                            if route
                            else 4
                        )
                        obs, _, term, trunc, info = env.step(action)
                        if term or trunc:
                            if info["skill"]["success"]:
                                teacher.bootstrap_success[family] += 1
                                for observation, candidate in candidates:
                                    teacher.memories[candidate["kind"]].offer(
                                        observation, candidate["actions"], ("P4a", seed), "teacher"
                                    )
                            break
                finally:
                    env.close()
        return
    for task in teacher.seeds:
        seeds = lesson_seeds(config.skills.train_count, 0) if task == "P4a" else teacher.seeds[task]
        seeds = seeds[:: max(1, len(seeds) // 16)][:16]
        for seed in seeds:
            env = configured_skill_env(
                task, seed, config.skills, threat_lesson=0 if task == "P4a" else None
            )
            try:
                obs, _ = env.reset()
                for _ in range(64):
                    label = teacher.offer(obs, task, seed, "teacher")
                    action = (
                        int(np.flatnonzero(label["actions"])[0])
                        if label
                        else int(model.predict(obs, deterministic=True)[0])
                    )
                    obs, _, term, trunc, _ = env.step(action)
                    if term or trunc:
                        break
            finally:
                env.close()


def _timing_label_diagnostics(logits, labels):
    """Measure policy support for each safe timing label on train-only rows.

    The target mass is the total softmax probability assigned to the allowed
    action set. The margin is its logit-space log-odds against disallowed
    actions, so a positive delta means the policy shifted probability toward
    the helper's WAIT or safe-GO label.
    """
    if logits.shape != labels.shape or logits.ndim != 2:
        raise ValueError("Timing logits and labels must have matching batch/action shapes")
    if not bool(labels.any(dim=1).all()) or bool(labels.all(dim=1).any()):
        raise ValueError("Timing labels must contain both allowed and disallowed actions")
    log_normalizer = torch.logsumexp(logits, dim=1)
    target_log_mass = torch.logsumexp(logits.masked_fill(~labels, -torch.inf), dim=1)
    other_log_mass = torch.logsumexp(logits.masked_fill(labels, -torch.inf), dim=1)
    target_mass = torch.exp(target_log_mass - log_normalizer)
    margin = target_log_mass - other_log_mass
    wait_rows = labels[:, 4]
    result = {}
    for kind, selected in (("wait", wait_rows), ("go", ~wait_rows)):
        count = int(selected.sum().item())
        result[f"{kind}_label_count"] = count
        result[f"{kind}_target_mass"] = float(target_mass[selected].mean().item()) if count else 0.0
        result[f"{kind}_logit_margin"] = float(margin[selected].mean().item()) if count else 0.0
    return result


def teach_timing(model):
    teacher, policy = model.timing_teaching, model.policy
    result = {"accepted": 0, "rejected": 0, "loss": 0.0, "p3_kl": 0.0}
    active_task = getattr(teacher, "active_task", "P4a")
    if active_task == "P4a":
        result.update(teacher.p4a_family_counts())
    else:
        result["active_task"] = active_task
    guard_rows = model.threat_retention.memory.sample(64)
    if not guard_rows:
        return result
    probe_rows = teacher.sample(64)
    if not probe_rows:
        return result
    sample_counts = dict(getattr(teacher, "last_sample_counts", {}))
    guard, _ = tensors(guard_rows, model.device)
    probe, probe_labels = tensors(probe_rows, model.device)
    with torch.no_grad():
        reference = policy.get_distribution(guard).distribution.probs.detach().clone()
        before = _timing_label_diagnostics(
            policy.get_distribution(probe).distribution.logits, probe_labels
        )
    round_weights = copy.deepcopy(policy.state_dict())
    round_optimizer = copy.deepcopy(policy.optimizer.state_dict())
    try:
        for batch_index in range(teacher.batches):
            rows = probe_rows if batch_index == 0 else teacher.sample(64)
            if not rows:
                break
            if batch_index:
                for task, count in getattr(teacher, "last_sample_counts", {}).items():
                    sample_counts[task] = sample_counts.get(task, 0) + count
            weights = copy.deepcopy(policy.state_dict())
            optimizer = copy.deepcopy(policy.optimizer.state_dict())
            obs, labels = tensors(rows, model.device)
            loss = teacher.coefficient * route_loss(
                policy.get_distribution(obs).distribution.logits, labels
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite timing loss")
            policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm)
            policy.optimizer.step()
            with torch.no_grad():
                kl = divergence(reference, policy.get_distribution(guard).distribution.probs).mean()
            if not torch.isfinite(kl) or kl > teacher.max_kl:
                # Keep earlier minibatches that stayed inside the frozen-P3 KL
                # bound; roll back only the update that crossed it.
                policy.load_state_dict(weights)
                policy.optimizer.load_state_dict(optimizer)
                result["rejected"] = 1
                break
            result.update(
                accepted=result["accepted"] + 1, loss=float(loss.detach()), p3_kl=float(kl)
            )
    except BaseException:
        policy.load_state_dict(round_weights)
        policy.optimizer.load_state_dict(round_optimizer)
        raise
    finally:
        policy.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        after = _timing_label_diagnostics(
            policy.get_distribution(probe).distribution.logits, probe_labels
        )
    for kind in ("wait", "go"):
        for metric in ("target_mass", "logit_margin"):
            result[f"{kind}_{metric}_before"] = before[f"{kind}_{metric}"]
            result[f"{kind}_{metric}_after"] = after[f"{kind}_{metric}"]
            result[f"{kind}_{metric}_delta"] = (
                after[f"{kind}_{metric}"] - before[f"{kind}_{metric}"]
            )
        result[f"{kind}_label_count"] = before[f"{kind}_label_count"]
    result.update({f"{k}_samples": len(v) for k, v in teacher.memories.items()})
    if active_task in ("P4b", "P4c"):
        active = sample_counts.get(active_task, 0)
        previous = sample_counts.get({"P4b": "P4a", "P4c": "P4b"}[active_task], 0)
        total = active + previous
        result["active_task_sample_fraction"] = active / total if total else 0.0
        result["previous_task_sample_fraction"] = previous / total if total else 0.0
    return result
