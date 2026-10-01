"""Privileged evaluation counts, public reward audit, aggregation and selection.

Never passed to an agent. Raw t=0 and committed-step records permit recomputation.
"""

import hashlib
import math
from collections import Counter
from functools import cmp_to_key
from statistics import mean

import numpy as np

from ather_exploration.types import EndReason, PublicTransition
from ather_exploration.worlds.scenarios import scenario_hash

THRESHOLDS = (0.5, 0.75, 0.9, 1.0)
ID_GROUPS = ("small", "medium", "large")


def observation_hash(obs):
    h = hashlib.sha256()
    for name in ("local", "memory", "state"):
        value = np.ascontiguousarray(obs[name])
        h.update(name.encode())
        h.update(str(value.shape).encode())
        h.update(value.dtype.str.encode())
        h.update(value.tobytes())
    return h.hexdigest()


def ratio(n, d):
    return n / d if d else None


class EpisodeMetrics:
    def __init__(self, snapshot, observation, reward_config, *, episode_id, group, metadata=None):
        if snapshot.step_count != 0:
            raise ValueError("Metrics must start at reset")
        self.scenario = snapshot.scenario
        self.F = sum(row.count(".") for row in self.scenario.terrain)
        self.K = len(self.scenario.pois)
        self.H = self.scenario.horizon
        self.episode_id = episode_id
        self.group = group
        self.metadata = metadata or {}
        self.reward_config = reward_config
        self.counts = Counter()
        self.reward_terms = Counter()
        self.milestones = {f"coverage_{int(q * 100)}": None for q in THRESHOLDS}
        self.milestones.update(
            first_discovery=None, first_activation=None, all_pois=None, first_encounter=None
        )
        self.steps = []
        self._positions = [snapshot.agent_position]
        self._progress = []
        self._moves = []
        self._gap = 0
        self._longest = 0
        self._threat = False
        self._finalized = False
        self._snapshot = snapshot
        self._record(snapshot, observation, None, 0.0, None)

    def _record(self, snapshot, obs, event, reward, collision):
        t = snapshot.step_count
        seen = int(obs["memory"][2].sum())
        visited = int(obs["memory"][5].sum())
        discovered = int(obs["memory"][3].sum() + obs["memory"][4].sum())
        activated = len(snapshot.activated_pois)
        coverage = seen / self.F
        activation = ratio(activated, self.K)
        threat = bool(obs["local"][5].any())
        values = {
            "seen": seen,
            "visited": visited,
            "discovered": discovered,
            "activated": activated,
        }
        for q in THRESHOLDS:
            key = f"coverage_{int(q * 100)}"
            if self.milestones[key] is None and coverage >= q:
                self.milestones[key] = t
        for key, condition in (
            ("first_discovery", discovered > 0),
            ("first_activation", activated > 0),
            ("all_pois", self.K > 0 and activated == self.K),
            ("first_encounter", threat),
        ):
            if condition and self.milestones[key] is None:
                self.milestones[key] = t
        terms = {}
        cycle = None
        progress = False
        motion = None
        if event is not None:
            previous = self.steps[-1]
            progress = bool(event.new_floor or event.new_poi or event.activated)
            moved = event.motion == "moved"
            motion = event.motion.value
            if (
                seen - previous["counts"]["seen"] != event.new_floor
                or discovered - previous["counts"]["discovered"] != event.new_poi
                or activated - previous["counts"]["activated"] != int(event.activated)
            ):
                raise ValueError("Evaluator counts disagree with public events")
            w = self.reward_config
            terms = {
                "area": w.area * event.new_floor,
                "discovery": w.discovery * event.new_poi,
                "activation": w.activation * int(event.activated),
                "death": -w.death * int(event.died),
            }
            if not math.isfinite(reward) or not math.isclose(
                sum(terms.values()), reward, abs_tol=1e-7
            ):
                raise ValueError("Reward terms do not reconstruct runtime reward")
            self.reward_terms.update(terms)
            self.counts["wait"] += event.action == 4
            self.counts["move_attempts"] += event.action != 4
            self.counts["blocked"] += event.motion == "wall_blocked"
            self.counts["moves"] += moved
            self.counts["revisits"] += moved and snapshot.agent_position in self._positions
            self.counts["backtracks"] += bool(
                moved
                and self._moves
                and self._moves[-1]
                and len(self._positions) >= 2
                and snapshot.agent_position == self._positions[-2]
            )
            self.counts["threat_decisions"] += self._threat
            self._positions.append(snapshot.agent_position)
            self._progress.append(progress)
            self._moves.append(moved)
            self._gap = 0 if progress else self._gap + 1
            self._longest = max(self._longest, self._gap)
            for window in (16, 32):
                self.counts[f"stagnation_{window}"] += len(self._progress) >= window and not any(
                    self._progress[-window:]
                )
            for period in range(2, 9):
                if (
                    t >= 2 * period
                    and not any(self._progress[-2 * period :])
                    and any(self._moves[-2 * period :])
                    and self._positions[-period:] == self._positions[-2 * period : -period]
                ):
                    cycle = period
                    break
            completed_before = previous["coverage"] == 1 and (
                self.K == 0 or previous["activation"] == 1
            )
            segment = "after_completion" if completed_before else "before_completion"
            self.counts[f"actions_{segment}"] += 1
            self.counts[f"cycles_{segment}"] += cycle is not None
            self.counts["cycles"] += cycle is not None
        self._threat = threat
        self._snapshot = snapshot
        record = {
            "episode_id": self.episode_id,
            "t": t,
            "t_over_H": t / self.H,
            "action": int(event.action) if event else None,
            "motion": motion,
            "actual_delta": list(event.actual_delta) if event else [0, 0],
            "events": {
                "new_floor": event.new_floor,
                "new_poi": event.new_poi,
                "activated": event.activated,
                "died": event.died,
            }
            if event
            else None,
            "reward": reward,
            "reward_terms": terms,
            "counts": values,
            "coverage": coverage,
            "visited": visited / self.F,
            "discovery": ratio(discovered, self.K),
            "activation": activation,
            "alive": snapshot.end_reason != EndReason.DEATH,
            "end_reason": snapshot.end_reason.value,
            "terminated": snapshot.end_reason != EndReason.NONE,
            "truncated": False,
            "threat_visible": threat,
            "progress": progress,
            "suspected_cycle_period": cycle,
            "stagnation_gap": self._gap,
            "collision_stage": collision,
            "observation_hash": observation_hash(obs),
        }
        self.steps.append(record)
        return record

    def update(self, snapshot, observation, reward, event, collision_stage=None):
        if self._finalized or self._snapshot.end_reason != EndReason.NONE:
            raise ValueError("Metrics episode already ended")
        if (
            snapshot.scenario != self.scenario
            or snapshot.step_count != self._snapshot.step_count + 1
        ):
            raise ValueError("Metrics need consecutive steps of one scenario")
        if isinstance(event, dict):
            event = PublicTransition(**event)
        return self._record(snapshot, observation, event, reward, collision_stage)

    def finish(self, *, cancelled=False, failure=None):
        self._finalized = True
        last = self.steps[-1]
        T = last["t"]
        completed = last["terminated"] and not cancelled and failure is None
        status = "failed" if failure is not None else "completed" if completed else "cancelled"
        survival = int(last["end_reason"] == "budget" and T == self.H) if completed else None
        all_pois = int(last["counts"]["activated"] == self.K) if self.K else None
        coverage_auc = (
            (sum(r["coverage"] for r in self.steps[1:]) + (self.H - T) * last["coverage"]) / self.H
            if completed
            else None
        )
        activation_auc = (
            (sum(r["activation"] for r in self.steps[1:]) + (self.H - T) * last["activation"])
            / self.H
            if completed and self.K
            else None
        )
        initial = self.steps[0]
        metrics = {
            "coverage": last["coverage"],
            "initial_coverage": initial["coverage"],
            "coverage_gain": last["coverage"] - initial["coverage"],
            "visited": last["visited"],
            "discovery": last["discovery"],
            "activation": last["activation"],
            "survival": survival,
            "coverage_auc": coverage_auc,
            "coverage_gain_auc": coverage_auc - initial["coverage"] if completed else None,
            "activation_auc": activation_auc,
            "new_floor_per_budget_step": (last["counts"]["seen"] - initial["counts"]["seen"])
            / self.H,
            "all_pois_activated": all_pois,
            "all_pois_and_survived_H": survival * all_pois if completed and self.K else None,
            "wall_block_rate": ratio(self.counts["blocked"], self.counts["move_attempts"]),
            "wait_fraction": ratio(self.counts["wait"], T),
            "revisit_ratio": ratio(self.counts["revisits"], self.counts["moves"]),
            "backtrack_count": self.counts["backtracks"],
            "longest_stagnation": self._longest,
            "cycle_rate": ratio(self.counts["cycles"], T),
            "threat_exposure": ratio(self.counts["threat_decisions"], T),
            "terminal_threat_visible": last["threat_visible"],
            "death": int(last["end_reason"] == "death"),
            "q": 0.6 * last["coverage"] + 0.4 * last["activation"] if self.K else None,
        }
        for q in THRESHOLDS:
            metrics[f"joint_success_{int(q * 100)}"] = (
                int(bool(survival and all_pois and last["coverage"] >= q))
                if completed and self.K
                else None
            )
        for window in (16, 32):
            metrics[f"stagnation_rate_{window}"] = ratio(
                self.counts[f"stagnation_{window}"], max(0, T - window + 1)
            )
        for segment in ("before_completion", "after_completion"):
            metrics[f"cycle_rate_{segment}"] = ratio(
                self.counts[f"cycles_{segment}"], self.counts[f"actions_{segment}"]
            )
        return {
            "schema_version": "1",
            "metric_version": "1",
            "episode_id": self.episode_id,
            "group": self.group,
            "status": status,
            "failure": failure,
            "scenario_hash": scenario_hash(self.scenario),
            "scenario_seed": self.scenario.seed,
            "config_hash": self.scenario.config_hash,
            "source_revision": self.scenario.source_revision,
            "H": self.H,
            "T": T,
            "F": self.F,
            "K": self.K,
            "end_reason": last["end_reason"],
            "collision_stage": last["collision_stage"],
            "death_tick": T if last["end_reason"] == "death" else None,
            "initial_counts": initial["counts"],
            "final_counts": last["counts"],
            "metrics": metrics,
            "milestones": dict(self.milestones),
            "behavior_counts": dict(self.counts),
            "reward_terms": dict(self.reward_terms),
            "return": sum(self.reward_terms.values()),
            "metadata": self.metadata,
        }


def aggregate_episodes(records):
    ids = [r["episode_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate episode ID")
    complete = [r for r in records if r["status"] == "completed"]
    groups = {}
    for group in sorted({r["group"] for r in complete}):
        rows = [r for r in complete if r["group"] == group]
        metrics = {}
        counts = {}
        for key in rows[0]["metrics"]:
            values = [r["metrics"][key] for r in rows if r["metrics"][key] is not None]
            metrics[key] = mean(values) if values else None
            counts[key] = len(values)
        milestones = {}
        for key in rows[0]["milestones"]:
            times = [r["milestones"][key] for r in rows if r["milestones"][key] is not None]
            milestones[key] = {
                "attainment_rate": len(times) / len(rows),
                "conditional_mean_tick": mean(times) if times else None,
            }
        groups[group] = {
            "episodes": len(rows),
            "metrics": metrics,
            "valid_metric_counts": counts,
            "milestones": milestones,
        }
    macro = {}
    if groups:
        for key in next(iter(groups.values()))["metrics"]:
            values = [g["metrics"][key] for g in groups.values() if g["metrics"][key] is not None]
            macro[key] = mean(values) if values else None
    return {
        "requested": len(records),
        "completed": len(complete),
        "cancelled": sum(r["status"] == "cancelled" for r in records),
        "failed": sum(r["status"] == "failed" for r in records),
        "groups": groups,
        "macro": macro,
    }


def select_checkpoint(candidates, *, threshold=0.9, tolerance=1e-9):
    if not candidates:
        raise ValueError("No checkpoint candidates")
    if (
        not 0 <= threshold <= 1
        or not math.isfinite(threshold)
        or not math.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("Invalid selection threshold/tolerance")
    if len({c["checkpoint_id"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate checkpoint ID")
    prepared = []
    for c in candidates:
        if (
            type(c["step"]) is not int
            or c["step"] < 0
            or not isinstance(c["checkpoint_id"], str)
            or not c["checkpoint_id"]
        ):
            raise ValueError("Invalid checkpoint ID/step")
        if not set(ID_GROUPS) <= set(c["groups"]):
            raise ValueError("Selection needs all three ID groups")
        groups = [c["groups"][g] for g in ID_GROUPS]
        for group in groups:
            if any(
                not math.isfinite(group[k]) or not 0 <= group[k] <= 1
                for k in ("survival", "q", "coverage_auc")
            ):
                raise ValueError("Invalid checkpoint metrics")
        prepared.append(
            {
                **c,
                "eligible": all(g["survival"] >= threshold for g in groups),
                "worst_survival": min(g["survival"] for g in groups),
                "q": mean(g["q"] for g in groups),
                "auc": mean(g["coverage_auc"] for g in groups),
            }
        )
    eligible = [c for c in prepared if c["eligible"]]
    pool = eligible or prepared
    order = ("q", "auc", "worst_survival") if eligible else ("worst_survival", "q", "auc")

    def compare(a, b):
        for key in order:
            if abs(a[key] - b[key]) > tolerance:
                return -1 if a[key] > b[key] else 1
        if a["step"] != b["step"]:
            return -1 if a["step"] < b["step"] else 1
        return (a["checkpoint_id"] > b["checkpoint_id"]) - (a["checkpoint_id"] < b["checkpoint_id"])

    ranked = sorted(pool, key=cmp_to_key(compare))
    return {
        "checkpoint_id": ranked[0]["checkpoint_id"],
        "gate_failed": not bool(eligible),
        "ranked_ids": [c["checkpoint_id"] for c in ranked],
        "threshold": threshold,
    }


def recompute_progress_metrics(steps, *, horizon, floor_count, poi_count):
    """Independent arithmetic audit from raw step counts, without env/model access."""
    if not steps or [s["t"] for s in steps] != list(range(len(steps))):
        raise ValueError("Need consecutive raw steps including reset")
    if len({s["episode_id"] for s in steps}) != 1:
        raise ValueError("Mixed episode traces")
    last = steps[-1]
    t = last["t"]
    if t > horizon or not last["terminated"]:
        raise ValueError("Progress AUC needs a completed trace within H")
    coverage = [s["counts"]["seen"] / floor_count for s in steps]
    activation = [s["counts"]["activated"] / poi_count for s in steps] if poi_count else None
    return {
        "coverage": coverage[-1],
        "coverage_gain": coverage[-1] - coverage[0],
        "coverage_auc": (sum(coverage[1:]) + (horizon - t) * coverage[-1]) / horizon,
        "activation_auc": (sum(activation[1:]) + (horizon - t) * activation[-1]) / horizon
        if activation
        else None,
        "return": sum(s["reward"] for s in steps),
    }


def checkpoint_candidate(checkpoint_id, step, episodes):
    """Bridge raw evaluator output to the selection contract; reject partial trials."""
    summary = aggregate_episodes(episodes)
    if summary["cancelled"] or summary["failed"] or not set(ID_GROUPS) <= set(summary["groups"]):
        raise ValueError("Checkpoint selection requires complete evaluation of all ID groups")
    return {
        "checkpoint_id": checkpoint_id,
        "step": step,
        "groups": {
            g: {k: summary["groups"][g]["metrics"][k] for k in ("survival", "q", "coverage_auc")}
            for g in ID_GROUPS
        },
    }
