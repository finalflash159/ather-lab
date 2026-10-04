"""Serializable mastery controller; no training side effects."""

from dataclasses import dataclass, field

STAGES = ("P1a", "P1b", "P2a", "P2b", "P2c", "P3a", "P3b", "P3c", "P4a", "P4b", "P5a", "P5b")
CAPS = (163840, 163840, 262144, 262144, 262144, 524288, 524288, 524288, 1048576, 1048576)
MINIMUM = (32768, 32768, 32768, 32768, 32768, 65536, 65536, 65536, 65536, 65536)


@dataclass
class SkillController:
    index: int = 0
    phase_start: int = 0
    family_start: int = 0
    passed: int = 0
    failed: bool = False
    eval_steps: int = 0
    history: list = field(default_factory=list)
    best: dict = field(default_factory=dict)
    restart_level: int = 0
    restart_results: list = field(default_factory=list)
    p2_task_budget: int = 262144
    best_by_task: dict = field(default_factory=dict)
    p3_minimum: int = 65536
    p3_task_budget: int = 524288
    recovery_p3b_budget: int | None = None
    recovery_p3c_budget: int | None = None
    threat_level: int = 0
    threat_streak: int = 0
    threat_level_start: int = 1638400
    p4_enabled: bool = False
    p4_task_budget: int = 1048576
    p4_minimum: int = 65536
    p4_force_advance: bool = False

    def observe_restart(self, level, progress, window, rate):
        """Public novelty mastery; separate from validation gate and task promotion."""
        if level != self.restart_level:
            return False
        self.restart_results.append(bool(progress))
        del self.restart_results[:-window]
        if (
            len(self.restart_results) == window
            and sum(self.restart_results) / window >= rate
            and self.restart_level < 2
        ):
            self.restart_level += 1
            self.restart_results.clear()
            return True
        return False

    @property
    def minimum(self):
        if self.p4_enabled and self.task.startswith("P4"):
            return self.p4_minimum
        return self.p3_minimum if self.task.startswith("P3") else MINIMUM[self.index]

    @property
    def budget(self):
        if self.p4_enabled and self.task.startswith("P4"):
            return self.p4_task_budget
        if self.task == "P3c" and self.recovery_p3c_budget is not None:
            return self.recovery_p3c_budget
        if self.task == "P3b" and self.recovery_p3b_budget is not None:
            return self.recovery_p3b_budget
        if self.task.startswith("P3"):
            return self.p3_task_budget
        return self.p2_task_budget if self.task.startswith("P2") else CAPS[self.index]

    @property
    def task(self):
        stages = (*STAGES[:10], "P4c", *STAGES[10:]) if self.p4_enabled else STAGES
        return stages[self.index]

    @property
    def stage(self):
        return self.index

    def observe(self, passed, steps):
        if self.task.startswith("P5"):
            return False
        elapsed = steps - self.phase_start
        self.passed = self.passed + 1 if passed else 0
        self.history.append(
            {
                "task": self.task,
                "steps": steps,
                "passed": bool(passed),
                "elapsed": elapsed,
                "minimum": self.minimum,
                "eligible": elapsed >= self.minimum,
                "streak": self.passed,
            }
        )
        if self.passed >= 2 and elapsed >= self.minimum:
            old = self.task[:2]
            self.index += 1
            self.phase_start = steps
            if self.task[:2] != old:
                self.family_start = steps
            if old == "P3" and self.task == "P3c":
                # P3c owns a fresh prefix bank and starts from the easy restart band.
                self.restart_level = 0
                self.restart_results.clear()
            self.passed = 0
            return True
        budget = self.budget
        self.failed = (
            elapsed
            if self.task.startswith(("P2", "P3")) or self.p4_enabled and self.task.startswith("P4")
            else steps - self.family_start
        ) >= budget
        if self.p4_force_advance and self.task == "P4b" and self.failed:
            self.history[-1]["forced_advance"] = "budget_exhausted_after_gate_failure"
            self.index += 1
            self.phase_start = steps
            self.family_start = steps
            self.passed = 0
            self.failed = False
            return True
        return False

    def mixture(self):
        """Geometry sources only; all episodes use the active phase rules."""
        if self.p4_enabled and self.task.startswith("P4"):
            return {
                "P4a": [("P4a", 0.8), ("P3c", 0.2)],
                "P4b": [("P4b", 0.7), ("P4a", 0.1), ("P3c", 0.2)],
                "P4c": [("P4c", 0.7), ("P4b", 0.1), ("P3c", 0.2)],
            }[self.task]
        return {
            "P1a": [("P1a", 1.0)],
            "P1b": [("P1b", 0.75), ("P1a", 0.25)],
            "P2a": [("P2a", 0.8), ("P1b", 0.2)],
            "P2b": [("P2b", 1.0)],
            "P2c": [("P2c", 1.0)],
            "P3a": [("P3a", 0.8), ("P2c", 0.2)],
            "P3b": [("P3b", 0.8), ("P3a", 0.2)],
            "P3c": [("P3c", 0.8), ("P3b", 0.2)],
            "P4a": [("P4a", 0.8), ("P3c", 0.15), ("P2b", 0.05)],
            "P4b": [("P4b", 0.64), ("P4a", 0.16), ("P3c", 0.15), ("P2b", 0.05)],
            "P5a": [("target", 0.8), ("P4b", 0.1), ("P3c", 0.1)],
            "P5b": [("target", 1.0)],
        }[self.task]
