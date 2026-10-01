"""World-preserving conditional spawn curriculum and serializable mastery state."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ather_exploration.agents.baselines import BaselineConfig, PublicMap
from ather_exploration.environment.env import make_env
from ather_exploration.worlds.generation import load_generated
from ather_exploration.worlds.scenarios import digest, geometry_hash, read_record, world_hash
from ather_exploration.worlds.suites import SPLITS, _record_path
from ather_exploration.worlds.topology import distances


@dataclass
class Curriculum:
    enabled: bool
    stage: int = 0
    window: list = field(default_factory=list)
    passed_windows: int = 0
    transitions: list = field(default_factory=list)

    def advance(self, steps, budget, episodes=()):
        if not self.enabled:
            self.stage = 2
            return
        reason = None
        for episode in episodes:
            meta = episode["metadata"]
            if (
                episode["status"] != "completed"
                or meta["component"] == "target"
                or meta["stage"] != self.stage
                or meta["fallback"]
            ):
                continue
            self.window.append(
                (
                    episode["group"],
                    episode["metrics"]["survival"],
                    int(episode["final_counts"]["activated"] > 0),
                )
            )
            if len(self.window) == 128:
                success = (
                    {x[0] for x in self.window} == {"small", "medium", "large"}
                    and np.mean([x[1] for x in self.window]) >= 0.9
                    and np.mean([x[2] for x in self.window]) >= 0.6
                )
                self.passed_windows = self.passed_windows + 1 if success else 0
                self.window = []
                if self.passed_windows >= 2:
                    reason = "mastered"
                    break
        forced = 2 if steps >= 0.75 * budget else 1 if steps >= 0.25 * budget else 0
        target = max(forced, self.stage + int(reason is not None))
        target = min(2, target)
        if target > self.stage:
            self.transitions.append(
                {
                    "from": self.stage,
                    "to": target,
                    "steps": steps,
                    "reason": reason if reason and target == self.stage + 1 else "budget_forced",
                }
            )
            self.stage = target
            self.window = []
            self.passed_windows = 0


class WorldBank:
    """Validated immutable bank, equal group/world marginal; never resample a world for ease."""

    def __init__(self, paths):
        self.worlds = {}
        self.identities = {}
        self.schema = None
        seen = set()
        for group in ("small", "medium", "large"):
            root = Path(paths[group])
            manifest = read_record(root / "manifest.json")
            if manifest["state"] != "READY":
                raise ValueError("Training bank is not READY")
            self.identities[group] = digest(manifest)
            groups = {split: [] for split in SPLITS}
            for world in manifest["worlds"]:
                split = world["split"]
                if split not in groups:
                    raise ValueError("Unknown bank split")
                records = [load_generated(_record_path(root, p)) for p in world["starts"]]
                if not records or any(r.config.preset != group for r in records):
                    raise ValueError("Bank group mismatch or empty starts")
                if any(
                    r.scenario.config_hash != manifest["request"]["config_hash"] for r in records
                ):
                    raise ValueError("Bank manifest/config hash mismatch")
                if len({world_hash(r.scenario) for r in records}) != 1:
                    raise ValueError("Spawn pool changes world geometry/routes/phases")
                identity = geometry_hash(records[0].scenario.terrain)
                if identity in seen:
                    raise ValueError("Duplicate geometry across bank worlds/splits")
                seen.add(identity)
                weights = np.asarray(world["spawn_weights"], dtype=float)
                if (
                    weights.shape != (len(records),)
                    or not np.isfinite(weights).all()
                    or np.any(weights < 0)
                    or not np.isclose(weights.sum(), 1)
                ):
                    raise ValueError("Invalid conditional spawn weights")
                signature = records[0].config.observation.model_dump()
                if self.schema is not None and signature != self.schema:
                    raise ValueError("Banks must share the observation schema")
                self.schema = signature
                near, middle = [], []
                if split == "train":
                    rank = sorted(
                        range(len(records)),
                        key=lambda i: (
                            min(
                                distances(records[i].scenario.terrain, [records[i].scenario.spawn])[
                                    p
                                ]
                                for p in records[i].scenario.pois
                            ),
                            records[i].scenario.spawn[1],
                            records[i].scenario.spawn[0],
                        ),
                    )
                    # Stable thirds by rank; insufficient pools intentionally fall back.
                    third = len(rank) // 3
                    middle = [i for i in rank[third : 2 * third] if weights[i] > 0]
                    for i in rank[:third]:
                        env = make_env(generated=records[i])
                        try:
                            obs, _ = env.reset()
                            public = PublicMap(obs, BaselineConfig())
                            moves = sum(
                                public.safety(a)[0] and public.destination(a) != public.position
                                for a in range(4)
                            )
                            if moves >= 2 and weights[i] > 0:
                                near.append(i)
                        finally:
                            env.close()
                groups[split].append(
                    {"records": records, "weights": weights, "near": near, "middle": middle}
                )
            expected = manifest["request"]["count_per_split"]
            if any(len(items) != expected for items in groups.values()):
                raise ValueError("Incomplete bank split")
            self.worlds[group] = groups

    def sample(self, rng, stage):
        group = ("small", "medium", "large")[int(rng.integers(3))]
        pool = self.worlds[group]["train"]
        index = int(rng.integers(len(pool)))
        world = pool[index]
        components, weights = (
            (("target", "near"), (0.25, 0.75))
            if stage == 0
            else (
                (("target", "middle", "near"), (0.5, 0.35, 0.15))
                if stage == 1
                else (("target",), (1.0,))
            )
        )
        component = str(rng.choice(components, p=weights))
        subset = world.get(component, [])
        fallback = component != "target" and not subset
        if component == "target" or fallback:
            chosen = int(rng.choice(len(world["records"]), p=world["weights"]))
        else:
            probabilities = world["weights"][subset]
            chosen = int(rng.choice(subset, p=probabilities / probabilities.sum()))
        return world["records"][chosen], {
            "group": group,
            "world_index": index,
            "start_index": chosen,
            "stage": stage,
            "component": component,
            "fallback": fallback,
        }
