"""Multi-POI diagnostics and gates. Privileged coordinates never enter policy input."""

import numpy as np


class ExplorationDiagnostics:
    def __init__(self, scenario, observation):
        self.scenario = scenario
        self.floors = sum(row.count(".") for row in scenario.terrain)
        self.seen = {p: None for p in scenario.pois}
        self.activated = {p: None for p in scenario.pois}
        self.first_activation_coverage = None
        self.coverage = 0.0
        self.tile_coverage = 0.0
        self.room_coverage_min = 0.0
        self.room_coverages = np.zeros(0, dtype=np.float64)
        self.milestones = {str(x): None for x in (0.5, 0.75, 0.9, 1.0)}
        self.streak = self.longest = self.unfinished_steps = self.idle_steps = 0
        self.update(0, observation, frozenset(), progress=True)

    def update(self, step, observation, activated, *, progress):
        memory = observation["memory"]
        center = memory.shape[-1] // 2
        sx, sy = self.scenario.spawn
        for p in self.seen:
            x, y = p
            if memory[3:5, center + y - sy, center + x - sx].any() and self.seen[p] is None:
                self.seen[p] = step
            if p in activated and self.activated[p] is None:
                self.activated[p] = step
        self.tile_coverage = float(memory[2].sum()) / self.floors
        if self.scenario.skill_task in ("P3b", "P3c") and self.scenario.room_labels:
            from ather_exploration.worlds.p3_tasks import room_coverage_fractions

            self.room_coverages = room_coverage_fractions(self.scenario, memory)
            self.room_coverage_min = float(self.room_coverages.min())
            self.coverage = self.room_coverage_min
        else:
            self.coverage = self.tile_coverage
            self.room_coverage_min = self.tile_coverage
        if activated and self.first_activation_coverage is None:
            self.first_activation_coverage = self.coverage
        for threshold in self.milestones:
            if self.coverage >= float(threshold) - 1e-9 and self.milestones[threshold] is None:
                self.milestones[threshold] = step
        unfinished = self.coverage < 1.0 - 1e-9 or len(activated) < len(self.activated)
        if step:
            self.unfinished_steps += int(unfinished)
            self.idle_steps += int(unfinished and not progress)
            self.streak = self.streak + 1 if unfinished and not progress else 0
            self.longest = max(self.longest, self.streak)

    def row(self):
        events = [
            {
                "poi": list(p),
                "seen_step": self.seen[p],
                "activated_step": self.activated[p],
                "seen_to_activation": self.activated[p] - self.seen[p]
                if self.activated[p] is not None and self.seen[p] is not None
                else None,
            }
            for p in self.seen
        ]
        times = list(self.activated.values())
        return {
            "poi_events": events,
            "activated_if_seen": (
                sum(p["activated_step"] is not None for p in events if p["seen_step"] is not None)
                / sum(p["seen_step"] is not None for p in events)
                if any(p["seen_step"] is not None for p in events)
                else None
            ),
            "all_pois_activated_step": max(times) if all(t is not None for t in times) else None,
            "coverage_after_first_activation": self.coverage - self.first_activation_coverage
            if self.first_activation_coverage is not None
            else None,
            "coverage_milestones": self.milestones.copy(),
            "tile_coverage": self.tile_coverage,
            "room_coverage_min": self.room_coverage_min,
            "unfinished_no_progress_streak": self.longest,
            "unfinished_no_progress_fraction": self.idle_steps / max(1, self.unfinished_steps),
            "unfinished_steps": self.unfinished_steps,
        }


def add_p3_gates(task, rows, summary, check, config):
    gates = config.skills.p3_gates
    coverage_threshold = gates.room_coverage if task in ("P3b", "P3c") else gates.coverage
    coverage_auc_threshold = (
        gates.room_coverage_auc if task in ("P3b", "P3c") else gates.coverage_auc
    )
    passed = True
    for mode, deterministic in [("deterministic", True), ("stochastic", False)]:
        subset = [r for r in rows if r["deterministic"] == deterministic]
        events = [p for row in subset for p in row["poi_events"] if p["seen_step"] is not None]
        summary[mode]["activated_if_seen"] = (
            sum(p["activated_step"] is not None for p in events) / len(events) if events else None
        )
        summary[mode]["activated_if_seen_count"] = len(events)
        for key in (
            "joint_success",
            "unfinished_no_progress_streak",
            "unfinished_no_progress_fraction",
        ):
            summary[mode][key] = float(np.mean([r[key] for r in subset]))
        for key in ("all_pois_activated_step", "coverage_after_first_activation"):
            values = [r[key] for r in subset if r[key] is not None]
            summary[mode][key] = float(np.mean(values)) if values else None
            summary[mode][key + "_count"] = len(values)
    d = summary["deterministic"]
    passed &= check("deterministic/coverage", d["coverage"], coverage_threshold)
    passed &= check("deterministic/coverage_auc", d["coverage_auc"], coverage_auc_threshold)
    passed &= check("deterministic/joint_success", d["joint_success"], gates.joint_success)
    passed &= check("deterministic/wall_block", d["wall_block"], gates.wall_block, maximum=True)
    groups = {}
    for field in ("size", "topology", "poi_count"):
        for value in sorted({r[field] for r in rows}):
            subset = [r for r in rows if r["deterministic"] and r[field] == value]
            key = f"{field}/{value}"
            groups[key] = {
                "count": len(subset),
                **{
                    m: float(np.mean([r[m] for r in subset]))
                    for m in ("success", "coverage", "coverage_auc", "joint_success", "wall_block")
                },
            }
            passed &= check(key + "/success", groups[key]["success"], gates.subgroup_success)
    return bool(passed), groups
