"""Public-history novelty and the four reward terms; no hidden world inputs."""

from dataclasses import replace

import numpy as np

from ather_exploration.config import RewardConfig
from ather_exploration.environment.public_state import public_cells
from ather_exploration.types import PublicTransition


class RewardTracker:
    def __init__(self, config: RewardConfig):
        self.config = config
        self._ready = False

    def reset(self, local: np.ndarray) -> None:
        cells = public_cells(local, (0, 0))
        self.position = (0, 0)
        self.seen_floor = {p for p, c in cells.items() if c[1]}
        self.seen_pois = {p for p, c in cells.items() if c[2] or c[3]}
        self.activated_pois = {p for p, c in cells.items() if c[3]}
        self._ready = True  # Initial view establishes history; reset has no reward.

    def update(self, local: np.ndarray, event: PublicTransition) -> tuple[PublicTransition, float]:
        if not self._ready:
            raise RuntimeError("RewardTracker must be reset first")
        position = tuple(a + b for a, b in zip(self.position, event.actual_delta, strict=True))
        cells = public_cells(local, position)
        floor = {p for p, c in cells.items() if c[1]}
        pois = {p for p, c in cells.items() if c[2] or c[3]}
        new_floor, new_poi = len(floor - self.seen_floor), len(pois - self.seen_pois)
        activated = event.activated and position not in self.activated_pois
        self.position = position
        self.seen_floor.update(floor)
        self.seen_pois.update(pois)
        if activated:
            self.activated_pois.add(position)
        event = replace(event, new_floor=new_floor, new_poi=new_poi, activated=activated)
        w = self.config
        reward = (
            w.area * new_floor
            + w.discovery * new_poi
            + w.activation * activated
            - w.death * event.died
        )
        return event, float(reward)
