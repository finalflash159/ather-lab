"""Static floor distances and room-contracted navigation graph on the raster map."""

from collections import deque

DIRECTIONS = ((0, -1), (0, 1), (1, 0), (-1, 0))


def floor_cells(terrain):
    return tuple(
        (x, y) for y, row in enumerate(terrain) for x, tile in enumerate(row) if tile == "."
    )


def neighbors(position, floor):
    x, y = position
    return tuple((x + dx, y + dy) for dx, dy in DIRECTIONS if (x + dx, y + dy) in floor)


def distances(terrain, starts):
    floor = set(floor_cells(terrain))
    queue = deque(starts)
    result = {p: 0 for p in starts}
    if not set(result) <= floor:
        raise ValueError("Distance sources must be on floor")
    while queue:
        p = queue.popleft()
        for q in neighbors(p, floor):
            if q not in result:
                result[q] = result[p] + 1
                queue.append(q)
    return result


def navigation_graph(terrain, room_labels):
    floor = set(floor_cells(terrain))
    if not floor:
        raise ValueError("Empty floor")
    if len(distances(terrain, [min(floor)])) != len(floor):
        raise ValueError("Disconnected floor")
    corridors = {p for p in floor if room_labels[p[1]][p[0]] < 0}
    # Adjacent cells in a single broad room doorway form one portal, not a fake loop.
    parent = {p: p for p in corridors}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    touching = {
        p: {room_labels[y][x] for x, y in neighbors(p, floor) if room_labels[y][x] >= 0}
        for p in corridors
    }
    for p in sorted(corridors):
        for q in neighbors(p, corridors):
            if touching[p] & touching[q]:
                parent[find(q)] = find(p)

    def node(p):
        x, y = p
        label = room_labels[y][x]
        return ("room", label) if label >= 0 else ("corridor", *find(p))

    graph = {node(p): set() for p in floor}
    for p in sorted(floor):
        for q in neighbors(p, floor):
            a, b = node(p), node(q)
            if a != b:
                graph[a].add(b)
                graph[b].add(a)
    edge_count = sum(map(len, graph.values())) // 2
    # Preserve separate paths as separate edges while suppressing degree-two corridors.
    anchors = {n for n, adj in graph.items() if n[0] == "room" or len(adj) != 2}
    if not anchors:
        anchors = {min(graph)}
    seen = set()
    chains = []
    for start in sorted(anchors):
        for nxt in sorted(graph[start]):
            edge = frozenset((start, nxt))
            if edge in seen:
                continue
            seen.add(edge)
            previous, current = start, nxt
            length = 1
            while current not in anchors:
                following = next(n for n in sorted(graph[current]) if n != previous)
                seen.add(frozenset((current, following)))
                previous, current = current, following
                length += 1
            chains.append((start, current, length))
    return {
        "floor_count": len(floor),
        "connected_components": 1,
        "room_count": len({room_labels[y][x] for x, y in floor if room_labels[y][x] >= 0}),
        "cycles": edge_count - len(graph) + 1,
        "dead_ends": sum(len(v) == 1 for v in graph.values()),
        "junctions": sum(len(v) >= 3 for v in graph.values()),
        "edges": tuple(chains),
        "corridor_lengths": tuple(edge[2] for edge in chains),
    }
