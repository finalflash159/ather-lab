"""Serializable mastery controller; no training side effects."""

from dataclasses import dataclass, field

STAGES = ("P1a", "P1b", "P2a", "P2b", "P3", "P4a", "P4b", "P5a", "P5b")
CAPS = (131072, 131072, 262144, 262144, 524288, 1048576, 1048576)
MINIMUM = (32768, 32768, 32768, 32768, 65536, 65536, 65536)


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

    @property
    def task(self):
        return STAGES[self.index]

    @property
    def stage(self):
        return self.index

    def observe(self, passed, steps):
        if self.index >= 7:
            return False
        elapsed = steps - self.phase_start
        self.passed = self.passed + 1 if passed and elapsed >= MINIMUM[self.index] else 0
        self.history.append({"task": self.task, "steps": steps, "passed": bool(passed)})
        if self.passed >= 2:
            old = self.task[:2]
            self.index += 1
            self.phase_start = steps
            if self.task[:2] != old:
                self.family_start = steps
            self.passed = 0
            return True
        self.failed = steps - self.family_start >= CAPS[self.index]
        return False

    def mixture(self):
        return {
            "P1a": [("P1a", 1.0)],
            "P1b": [("P1b", 0.75), ("P1a", 0.25)],
            "P2a": [("P2a", 0.56), ("P2b", 0.24), ("P1b", 0.20)],
            "P2b": [("P2a", 0.24), ("P2b", 0.56), ("P1b", 0.20)],
            "P3": [("P3", 0.8), ("P2b", 0.1), ("P1b", 0.1)],
            "P4a": [("P4a", 0.8), ("P3", 0.15), ("P2b", 0.05)],
            "P4b": [("P4b", 0.64), ("P4a", 0.16), ("P3", 0.15), ("P2b", 0.05)],
            "P5a": [("target", 0.8), ("P4b", 0.1), ("P3", 0.1)],
            "P5b": [("target", 1.0)],
        }[self.task]
