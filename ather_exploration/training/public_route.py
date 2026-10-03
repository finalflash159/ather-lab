"""Stateless shortest-route labels derived exclusively from public observations."""

from collections import deque

import numpy as np

from ather_exploration.environment.frontier import frontier_mask

# Actions use (row, column) offsets, in the public Action enumeration order.
DIRECTIONS = ((-1, 0), (1, 0), (0, 1), (0, -1))


def public_route(observation):
    """Return all shortest first actions and reachable goal coordinates, or None.

    Unknown cells never become traversable. A zero-distance frontier cannot
    teach a move, so only positive-distance targets contribute route labels.
    """
    memory = observation["memory"]
    floor = (memory[0] > 0) & (memory[2] > 0) & (memory[1] == 0)
    positions = np.argwhere(memory[7] > 0)
    if len(positions) != 1:
        raise ValueError("Public route requires exactly one agent cell")
    start = tuple(positions[0])
    distances = {start: 0}
    first = {start: set()}
    queue = deque([start])
    height, width = floor.shape
    while queue:
        row, col = queue.popleft()
        for action, (dr, dc) in enumerate(DIRECTIONS):
            target = row + dr, col + dc
            r, c = target
            if not (0 <= r < height and 0 <= c < width and floor[r, c]):
                continue
            distance = distances[row, col] + 1
            paths = {action} if (row, col) == start else first[row, col]
            if target not in distances:
                distances[target] = distance
                first[target] = set(paths)
                queue.append(target)
            elif distances[target] == distance:
                first[target].update(paths)
    for kind, mask in (("poi", memory[3] > 0), ("frontier", frontier_mask(memory) > 0)):
        goals = [tuple(p) for p in np.argwhere(mask) if distances.get(tuple(p), 0) > 0]
        if not goals:
            continue
        nearest = min(distances[p] for p in goals)
        good = np.zeros(5, dtype=bool)
        for point in goals:
            if distances[point] == nearest:
                good[list(first[point])] = True
        return {"kind": kind, "actions": good, "goals": goals, "distance": nearest}
    return None


def route_loss(logits, good_actions):
    """Stable -log probability mass of the set, with no arbitrary tie breaker."""
    import torch

    if not bool(good_actions.any(dim=1).all()):
        raise ValueError("Empty route label")
    log_probs = torch.log_softmax(logits, dim=-1)
    return -torch.logsumexp(log_probs.masked_fill(~good_actions, -torch.inf), dim=-1).mean()
