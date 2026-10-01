"""Read-only symbolic rendering. Public panels cannot access a world snapshot."""

import math

import numpy as np
import pygame

BG = (13, 19, 29)
PANEL = (23, 32, 45)
INK = (226, 234, 246)
MUTED = (148, 167, 190)
BLUE = (88, 179, 255)
RED = (255, 108, 99)
GOLD = (246, 204, 91)
GREEN = (90, 219, 161)


def tile(
    surface,
    rect,
    *,
    seen=True,
    wall=False,
    visible=True,
    pending=False,
    active=False,
    monster=False,
    stale=False,
    agent=False,
    dead=False,
):
    """Layered, distinct silhouettes; POI border survives actor overlap."""
    x, y, w, h = rect
    color = (32, 42, 57) if wall else (65, 83, 105)
    if not seen:
        color = (10, 15, 23)
    elif not visible:
        color = tuple(int(c * 0.55) for c in color)
    pygame.draw.rect(surface, color, rect)
    pygame.draw.rect(surface, BG, rect, 1)
    if not seen:
        return
    if wall and w >= 12:
        pygame.draw.line(surface, (64, 78, 96), (x + 2, y + 2), (x + w - 3, y + 2))
    if pending or active:
        gate = pygame.Rect(x + w * 0.1, y + h * 0.1, w * 0.8, h * 0.8)
        pygame.draw.rect(surface, GREEN if active else GOLD, gate, max(1, w // 12), border_radius=2)
        if active:
            pygame.draw.lines(
                surface,
                GREEN,
                False,
                [
                    (x + w * 0.12, y + h * 0.7),
                    (x + w * 0.3, y + h * 0.88),
                    (x + w * 0.65, y + h * 0.55),
                ],
                max(1, w // 12),
            )
    if monster:
        points = [
            (x + w * 0.22, y + h * 0.7),
            (x + w * 0.2, y + h * 0.22),
            (x + w * 0.4, y + h * 0.35),
            (x + w * 0.6, y + h * 0.35),
            (x + w * 0.8, y + h * 0.22),
            (x + w * 0.78, y + h * 0.7),
        ]
        pygame.draw.polygon(
            surface, (155, 113, 120) if stale else RED, points, max(1, w // 14) if stale else 0
        )
        if not stale and w >= 12:
            for dx in (0.38, 0.62):
                pygame.draw.circle(
                    surface, BG, (int(x + w * dx), int(y + h * 0.5)), max(1, w // 18)
                )
    if agent:
        center = (int(x + w * 0.52), int(y + h * 0.55))
        pygame.draw.circle(surface, BLUE, center, max(2, int(w * 0.23)))
        pygame.draw.circle(surface, INK, center, max(2, int(w * 0.23)), max(1, w // 18))
        if dead:
            pygame.draw.line(
                surface,
                RED,
                (x + w * 0.3, y + h * 0.3),
                (x + w * 0.75, y + h * 0.8),
                max(2, w // 10),
            )


def world_surface(
    snapshot, cell=20, routes=False, *, previous=None, progress=1.0, collision_stage=None
):
    scenario = snapshot.scenario
    surface = pygame.Surface((len(scenario.terrain[0]) * cell, len(scenario.terrain) * cell))
    pois, monsters = set(scenario.pois), set(snapshot.monster_positions)
    animate = previous is not None and progress < 1
    for y, line in enumerate(scenario.terrain):
        for x, terrain in enumerate(line):
            p = (x, y)
            tile(
                surface,
                (x * cell, y * cell, cell, cell),
                wall=terrain == "#",
                pending=p in pois and p not in snapshot.activated_pois,
                active=p in snapshot.activated_pois,
                monster=p in monsters and not animate,
                agent=p == snapshot.agent_position and not animate,
                dead=snapshot.end_reason == "death",
            )
    if animate:
        agent, threats = animation_positions(previous, snapshot, progress, collision_stage)
        for x, y in threats:
            pygame.draw.polygon(
                surface,
                RED,
                [
                    (int((x + dx) * cell), int((y + dy) * cell))
                    for dx, dy in (
                        (0.22, 0.7),
                        (0.2, 0.22),
                        (0.4, 0.35),
                        (0.6, 0.35),
                        (0.8, 0.22),
                        (0.78, 0.7),
                    )
                ],
            )
        center = (int((agent[0] + 0.52) * cell), int((agent[1] + 0.55) * cell))
        pygame.draw.circle(surface, BLUE, center, max(2, int(cell * 0.23)))
        pygame.draw.circle(surface, INK, center, max(2, int(cell * 0.23)), 1)
    if routes:
        for route in scenario.routes:
            pygame.draw.lines(
                surface,
                (182, 111, 100),
                False,
                [((x + 0.5) * cell, (y + 0.5) * cell) for x, y in route],
                1,
            )
    return surface


def public_surface(observation, *, view="memory", cell=20, horizon_capacity=1024):
    """Only public arrays; unknown layout and hidden dynamic actors are unavailable."""
    if view == "local":
        m = observation["local"]
        rows, cols = m.shape[1:]
        bounds = (0, rows, 0, cols)
    else:
        m = observation["memory"]
        occupied = np.argwhere((m[0] > 0.5) | (m[7] > 0.5))
        lo = np.maximum(occupied.min(axis=0) - 1, 0)
        hi = np.minimum(occupied.max(axis=0) + 2, m.shape[1:])
        bounds = (*map(int, (lo[0], hi[0], lo[1], hi[1])),)
    y0, y1, x0, x1 = bounds
    surface = pygame.Surface(((x1 - x0) * cell, (y1 - y0) * cell))
    font = pygame.font.Font(None, max(12, cell // 2))
    for y in range(y0, y1):
        for x in range(x0, x1):
            local = view == "local"
            visible = bool(m[0 if local else 8, y, x])
            stale = not local and bool(m[9, y, x]) and not visible
            rect = ((x - x0) * cell, (y - y0) * cell, cell, cell)
            tile(
                surface,
                rect,
                seen=bool(m[0, y, x]),
                wall=bool(m[1, y, x]),
                visible=visible,
                pending=bool(m[3, y, x]),
                active=bool(m[4, y, x]),
                monster=bool(m[5 if local else 9, y, x]),
                stale=stale,
                agent=(y == rows // 2 and x == cols // 2) if local else bool(m[7, y, x]),
            )
            if stale and cell >= 18:
                age = round(math.expm1(float(m[10, y, x]) * math.log1p(horizon_capacity)))
                surface.blit(font.render(str(age), True, INK), (rect[0] + 1, rect[1] + 1))
    return surface


def fit(surface, target, rect):
    scale = min(rect.width / surface.get_width(), rect.height / surface.get_height())
    size = (max(1, int(surface.get_width() * scale)), max(1, int(surface.get_height() * scale)))
    target.blit(
        pygame.transform.scale(surface, size),
        (rect.centerx - size[0] // 2, rect.centery - size[1] // 2),
    )


def animation_positions(previous, current, progress, collision_stage=None):
    """Display-only interpolation of committed snapshots: agent then monsters."""
    progress = max(0.0, min(1.0, progress))
    agent_phase = progress if collision_stage == 1 else min(1.0, 2 * progress)
    monster_phase = 0.0 if collision_stage == 1 else max(0.0, 2 * progress - 1)

    def lerp(a, b, t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    return (
        lerp(previous.agent_position, current.agent_position, agent_phase),
        tuple(
            lerp(a, b, monster_phase)
            for a, b in zip(previous.monster_positions, current.monster_positions, strict=True)
        ),
    )
