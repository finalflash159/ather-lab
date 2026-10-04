"""Conservative timing labels from public observations only, never hidden patrols.

Known dynamics: a monster moves at most one cardinal cell, once every two ticks.
Its next direction (including reversals/turns) is deliberately not assumed known.
"""

from collections import deque
from heapq import heappop, heappush

import numpy as np

from ather_exploration.environment.frontier import frontier_mask
from ather_exploration.training.public_route import DIRECTIONS, public_route

VISIT_COUNT_CAPACITY = 1025
STAGNATION_VISITS = 4
ROUTE_VISIT_PENALTY = 0.35
ROUTE_LOOP_PRESSURE = 1.5
FRONTIER_INFORMATION_VALUE = 4.0


def public_timing(observation, *, allow_consecutive_wait=True):
    memory = observation["memory"]
    visible = memory[8] > 0
    current = {tuple(p) for p in np.argwhere((memory[9] > 0) & visible)}
    if not current:
        return None
    positions = np.argwhere(memory[7] > 0)
    if len(positions) != 1:
        return None
    start = tuple(positions[0])
    if min(abs(r - start[0]) + abs(c - start[1]) for r, c in current) > 4:
        return None
    floor = (memory[0] > 0) & (memory[2] > 0) & (memory[1] == 0)
    height, width = floor.shape

    def inside(p):
        return 0 <= p[0] < height and 0 <= p[1] < width

    def neighbors(p):
        return [(p[0] + dr, p[1] + dc) for dr, dc in DIRECTIONS]

    previous = set()
    previous_visible = np.zeros_like(visible)
    if len(memory) >= 14:
        previous = {tuple(p) for p in np.argwhere(memory[12] > 0)}
        previous_visible = memory[13] > 0
    dangerous = set(current)  # Agent can collide BEFORE the monster moves.
    for position in current:
        area = [position, *neighbors(position)]
        candidates = [p for p in previous if p in area]
        # A unique observed movement implies a stationary next tick. Require a
        # single monster in both frames, and complete prior local visibility so
        # an unseen replacement cannot be mistaken for the same individual.
        moved = (
            len(current) == len(previous) == 1
            and len(candidates) == 1
            and candidates[0] != position
            and all(inside(p) and previous_visible[p] for p in area)
        )
        if not moved:
            dangerous.update(p for p in neighbors(position) if inside(p) and memory[1][p] == 0)
    safe = np.zeros(5, dtype=bool)
    for action, destination in enumerate([*neighbors(start), start]):
        if not inside(destination) or not floor[destination] or destination in dangerous:
            continue
        # Rule out a monster arriving from an unobserved adjacent cell. Visible
        # walls also exclude arrivals; unknown terrain never certifies safety.
        if all(
            inside(p) and (visible[p] or memory[1][p] > 0)
            for p in [destination, *neighbors(destination)]
        ):
            safe[action] = True
    route = public_route(observation)
    if route is None:
        return None
    forward = route["actions"] & safe
    if forward.any():
        return {"kind": "go", "actions": forward}
    previous_wait = bool(observation["state"][4]) if "state" in observation else False
    if safe[4] and (allow_consecutive_wait or not previous_wait):
        wait = np.zeros(5, dtype=bool)
        wait[4] = True
        return {"kind": "wait", "actions": wait}
    # Retreat from a shared lane if waiting is unsafe. No hidden shortest path.
    safe[4] = False
    return {"kind": "go", "actions": safe} if safe.any() else None


def public_p4_timing(observation):
    """P4b/c timing labels that never teach consecutive WAIT actions.

    A single safe WAIT remains a valid label. If it did not open a safe route,
    the next state is unlabeled or receives a safe retreat/route action instead
    of reinforcing an unbounded wait loop.
    """
    return public_timing(observation, allow_consecutive_wait=False)


def _route_visit_cost(visits, point):
    """Bound the cost of entering a repeatedly visited public cell."""
    return ROUTE_VISIT_PENALTY * min(int(visits[point]), 8)


def _known_distances(floor, goal):
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        row, col = queue.popleft()
        for dr, dc in DIRECTIONS:
            target = row + dr, col + dc
            if (
                0 <= target[0] < floor.shape[0]
                and 0 <= target[1] < floor.shape[1]
                and floor[target]
                and target not in distances
            ):
                distances[target] = distances[row, col] + 1
                queue.append(target)
    return distances


def _weighted_distances(floor, visits, goal):
    """Dijkstra distances to a goal, charging for every cell on the route."""
    distances = {goal: 0.0}
    queue = [(0.0, goal)]
    height, width = floor.shape
    while queue:
        cost, current = heappop(queue)
        if cost != distances[current]:
            continue
        step_cost = 1.0 + _route_visit_cost(visits, current)
        for dr, dc in DIRECTIONS:
            previous = current[0] + dr, current[1] + dc
            if not (0 <= previous[0] < height and 0 <= previous[1] < width and floor[previous]):
                continue
            candidate = cost + step_cost
            if candidate < distances.get(previous, float("inf")):
                distances[previous] = candidate
                heappush(queue, (candidate, previous))
    return distances


def _first_action_costs(floor, visits, start, goal, safe_actions, weighted):
    """Return reachable safe directional first steps and their route costs."""
    distances = (
        _weighted_distances(floor, visits, goal) if weighted else _known_distances(floor, goal)
    )
    if start not in distances:
        return {}
    costs = {}
    for action, (dr, dc) in enumerate(DIRECTIONS):
        if not safe_actions[action]:
            continue
        target = start[0] + dr, start[1] + dc
        if target not in distances:
            continue
        edge_cost = 1.0 + (_route_visit_cost(visits, target) if weighted else 0.0)
        costs[action] = edge_cost + distances[target]
    return costs


def _frontier_information(memory, point):
    """Count unknown cardinal neighbors available to a known-floor frontier."""
    unknown = memory[0] <= 0
    height, width = unknown.shape
    count = 0
    for dr, dc in DIRECTIONS:
        row, col = point[0] + dr, point[1] + dc
        if 0 <= row < height and 0 <= col < width and unknown[row, col]:
            count += 1
    return count


def _visit_counts(memory):
    """Decode the public log-scaled visit channel into approximate visit counts."""
    return np.rint(np.expm1(np.clip(memory[6], 0.0, 1.0) * np.log1p(VISIT_COUNT_CAPACITY))).astype(
        np.int32
    )


def _shelter_action(observation):
    """One-step public risk plus route-level revisit-aware public planning.

    This is a teaching proposal, not an inference override or a safety guarantee.
    A short-lived last sighting stops a hidden monster from being treated as gone.
    """
    memory = observation["memory"]
    floor = (memory[0] > 0) & (memory[2] > 0) & (memory[1] == 0)
    agent = np.argwhere(memory[7] > 0)
    if len(agent) != 1:
        return None
    start = tuple(agent[0])
    current = [tuple(p) for p in np.argwhere((memory[9] > 0) & (memory[8] > 0))]
    prior = [tuple(p) for p in np.argwhere(memory[12] > 0)]
    prior2 = [tuple(p) for p in np.argwhere(memory[14] > 0)] if len(memory) >= 16 else []
    axis = None
    if len(current) == len(prior) == 1 and current[0] != prior[0]:
        axis = 0 if current[0][0] != prior[0][0] else 1
    elif len(current) == len(prior2) == 1 and current[0] != prior2[0]:
        axis = 0 if current[0][0] != prior2[0][0] else 1
    elif len(current) == 1:
        row, col = current[0]
        horizontal = bool(
            0 < col < floor.shape[1] - 1 and floor[row, col - 1] and floor[row, col + 1]
        )
        vertical = bool(
            0 < row < floor.shape[0] - 1 and floor[row - 1, col] and floor[row + 1, col]
        )
        if horizontal != vertical:
            axis = 1 if horizontal else 0

    danger = set(current)
    for point in current:
        moved = len(prior) == 1 and prior[0] != point
        if not moved:
            for dr, dc in DIRECTIONS:
                if (axis == 0 and dc) or (axis == 1 and dr):
                    continue
                target = point[0] + dr, point[1] + dc
                if (
                    0 <= target[0] < floor.shape[0]
                    and 0 <= target[1] < floor.shape[1]
                    and floor[target]
                ):
                    danger.add(target)
    if not current:
        # Public memory stores log-scaled age. 0.27 is between five and six
        # unseen ticks at the fixed 1024-tick observation capacity.
        for point in map(tuple, np.argwhere(memory[9] > 0)):
            if float(memory[10][point]) < 0.27:
                danger.add(point)
                for dr, dc in DIRECTIONS:
                    target = point[0] + dr, point[1] + dc
                    if (
                        0 <= target[0] < floor.shape[0]
                        and 0 <= target[1] < floor.shape[1]
                        and floor[target]
                    ):
                        danger.add(target)

    visit_counts = _visit_counts(memory)
    stagnant = visit_counts[start] >= STAGNATION_VISITS
    safe = np.zeros(5, dtype=bool)
    for action, (dr, dc) in enumerate((*DIRECTIONS, (0, 0))):
        target = start[0] + dr, start[1] + dc
        safe[action] = (
            0 <= target[0] < floor.shape[0]
            and 0 <= target[1] < floor.shape[1]
            and floor[target]
            and target not in danger
        )

    def best_action(costs):
        if not costs:
            return None

        # Keep the existing perpendicular step preference for a visible,
        # constrained corridor, but only as a tie-breaker between route costs.
        def rank(item):
            action, cost = item
            dr, dc = DIRECTIONS[action]
            tactical = 0.0
            if current and axis is not None:
                monster = current[0]
                perpendicular = (axis == 1 and dr != 0) or (axis == 0 and dc != 0)
                close = abs(start[0] - monster[0]) + abs(start[1] - monster[1]) <= 3
                if perpendicular and close:
                    tactical = 0.2
            return (
                -cost + tactical,
                -_route_visit_cost(visit_counts, (start[0] + dr, start[1] + dc)),
                -action,
            )

        return max(costs.items(), key=rank)[0]

    pois = [tuple(p) for p in np.argwhere((memory[3] > 0) & floor)]
    if pois:
        base_routes = {goal: _known_distances(floor, goal) for goal in pois}
        reachable = [goal for goal, distances in base_routes.items() if start in distances]
        if not reachable:
            return 4 if safe[4] else None
        weighted_plans = {
            goal: _first_action_costs(floor, visit_counts, start, goal, safe, weighted=True)
            for goal in reachable
        }
        candidates = [
            (cost, goal, action)
            for goal, costs in weighted_plans.items()
            for action, cost in costs.items()
        ]
        plan = min(candidates, default=None)
        if plan is None:
            return 4 if safe[4] else None
        _, selected_goal, weighted_action = plan
        weighted_route = _weighted_distances(floor, visit_counts, selected_goal)
        route_loop = (
            weighted_route[start] - base_routes[selected_goal][start] >= ROUTE_LOOP_PRESSURE
        )
        if stagnant or route_loop:
            return weighted_action

        # Preserve the established timing lesson: when the public shortest route
        # is blocked but waiting is safe, teach WAIT instead of an arbitrary detour.
        route = public_route(observation)
        if route is not None and route["kind"] == "poi":
            forward = route["actions"][:4] & safe[:4]
            if forward.any():
                return min(
                    np.flatnonzero(forward),
                    key=lambda action: (
                        _route_visit_cost(
                            visit_counts,
                            (start[0] + DIRECTIONS[action][0], start[1] + DIRECTIONS[action][1]),
                        ),
                        action,
                    ),
                )
        if safe[4]:
            return 4
        return weighted_action

    frontiers = [
        tuple(p)
        for p in np.argwhere(frontier_mask(memory) > 0)
        if tuple(p) != start and floor[tuple(p)]
    ]
    if not frontiers:
        return 4 if safe[4] else None
    choices = []
    for goal in frontiers:
        base_distances = _known_distances(floor, goal)
        if start not in base_distances:
            continue
        weighted_distances = _weighted_distances(floor, visit_counts, goal)
        pressure = weighted_distances[start] - base_distances[start]
        weighted_costs = _first_action_costs(floor, visit_counts, start, goal, safe, weighted=True)
        plain_costs = _first_action_costs(floor, visit_counts, start, goal, safe, weighted=False)
        if pressure < ROUTE_LOOP_PRESSURE and not stagnant:
            if not plain_costs:
                continue
            # Do not replace a safe timing WAIT with an arbitrary detour unless
            # public visit history says the current route itself is looping.
            action = best_action(plain_costs)
            route_cost = weighted_distances[start]
        else:
            if not weighted_costs:
                continue
            action = best_action(weighted_costs)
            route_cost = weighted_costs[action]
        information = _frontier_information(memory, goal)
        score = FRONTIER_INFORMATION_VALUE * information - route_cost
        choices.append((score, -route_cost, -goal[0], -goal[1], action))
    if not choices:
        return 4 if safe[4] else None
    return max(choices)[-1]


def public_p4a_timing(observation):
    """Use a public corridor/refuge hint; retain old labels elsewhere.

    No scenario family, route, patrol phase or validator state enters this function.
    """
    memory = observation["memory"]
    if len(memory) < 16:
        return public_timing(observation)
    floor = (memory[0] > 0) & (memory[2] > 0) & (memory[1] == 0)
    agent = np.argwhere(memory[7] > 0)
    if len(agent) != 1:
        return None
    start = tuple(agent[0])
    monsters = [tuple(p) for p in np.argwhere((memory[9] > 0) & (memory[8] > 0))]
    if not monsters:
        monsters = [tuple(p) for p in np.argwhere(memory[9] > 0)]
    aligned = False
    for row, col in monsters:
        if not (0 < row < floor.shape[0] - 1 and 0 < col < floor.shape[1] - 1):
            continue
        horizontal = bool(floor[row, col - 1] and floor[row, col + 1])
        vertical = bool(floor[row - 1, col] and floor[row + 1, col])
        aligned |= (horizontal and not vertical and abs(start[0] - row) <= 1) or (
            vertical and not horizontal and abs(start[1] - col) <= 1
        )
    if not aligned:
        return public_timing(observation)
    action = _shelter_action(observation)
    if action is None:
        return None
    mask = np.zeros(5, dtype=bool)
    mask[action] = True
    return {"kind": "wait" if action == 4 else "go", "actions": mask}
