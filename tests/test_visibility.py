from dataclasses import replace

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from ather_exploration.environment.dynamics import make_grid
from ather_exploration.environment.visibility import sense, visibility_mask
from ather_exploration.fixtures import fixture_scenario
from ather_exploration.types import EpisodeState, Scenario


def reference_hits_wall(start, target, cell):
    # Independent integer orientation test against four edges of a closed square.
    a, b = tuple(2 * v for v in start), tuple(2 * v for v in target)
    x, y = (2 * v for v in cell)
    corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1)]

    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def intersects(c, d):
        if max(min(a[0], b[0]), min(c[0], d[0])) > min(max(a[0], b[0]), max(c[0], d[0])):
            return False
        if max(min(a[1], b[1]), min(c[1], d[1])) > min(max(a[1], b[1]), max(c[1], d[1])):
            return False
        return orient(a, b, c) * orient(a, b, d) <= 0 and orient(c, d, a) * orient(c, d, b) <= 0

    return any(intersects(corners[i], corners[(i + 1) % 4]) for i in range(4))


def test_corner_requires_step_south_and_entities_do_not_occlude():
    s = fixture_scenario("corner_occlusion")
    state, grid = EpisodeState.from_scenario(s), make_grid(s)
    assert sense(grid, s, state, 4)[3].sum() == 0
    state.agent_position = (1, 2)
    assert sense(grid, s, state, 4)[3].sum() == 1
    s = fixture_scenario("entities_transparent")
    local = sense(make_grid(s), s, EpisodeState.from_scenario(s), 4)
    assert local[3].sum() == local[5].sum() == 1


def test_target_wall_visible_and_multilayer_cell():
    s = fixture_scenario("collision1_poi")
    local = sense(make_grid(s), s, EpisodeState.from_scenario(s), 4)
    assert local[0, 3, 4] == local[1, 3, 4] == 1
    assert local[2, 4, 5] == local[3, 4, 5] == local[5, 4, 5] == 1
    assert local[:, 0, 0].sum() == 0  # outside world is unknown, not a synthetic wall


def test_disk_has_49_cells_and_rotation_reflection():
    terrain = ("###########",) + ("#.........#",) * 9 + ("###########",)
    s = Scenario(terrain, (5, 5), (), (), (), 8, 0, fixture_only=True)
    assert visibility_mask(make_grid(s), (5, 5), 4).sum() == 49
    rows = list(terrain)
    rows[4] = "#...#.....#"
    s = replace(s, terrain=tuple(rows))
    mask = visibility_mask(make_grid(s), (5, 5), 4)
    terrain_array = np.array([list(row) for row in s.terrain])
    for transformed, expected in [
        (np.rot90(terrain_array), np.rot90(mask)),
        (np.fliplr(terrain_array), np.fliplr(mask)),
    ]:
        r = replace(s, terrain=tuple("".join(row) for row in transformed))
        assert np.array_equal(visibility_mask(make_grid(r), (5, 5), 4), expected)


@given(st.lists(st.booleans(), min_size=25, max_size=25))
@settings(max_examples=35, deadline=None, database=None)
def test_fov_matches_independent_reference(walls):
    rows = ["#######"]
    for y in range(5):
        rows.append("#" + "".join("#" if walls[5 * y + x] else "." for x in range(5)) + "#")
    rows.append("#######")
    rows[3] = rows[3][:3] + "." + rows[3][4:]
    s = Scenario(tuple(rows), (3, 3), (), (), (), 8, 0, fixture_only=True)
    actual = visibility_mask(make_grid(s), s.spawn, 4)
    for y in range(7):
        for x in range(7):
            target = (x, y)
            expected = (x - 3) ** 2 + (y - 3) ** 2 <= 16 and not any(
                reference_hits_wall(s.spawn, target, (wx, wy))
                for wy, row in enumerate(rows)
                for wx, tile in enumerate(row)
                if tile == "#" and (wx, wy) not in (s.spawn, target)
            )
            assert bool(actual[y - 3 + 4, x - 3 + 4]) == expected
