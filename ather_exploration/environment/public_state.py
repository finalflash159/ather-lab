"""Decode only public sensor channels into relative cells; no world/env access."""

import numpy as np


def public_cells(local: np.ndarray, position: tuple[int, int]) -> dict:
    if (
        not isinstance(local, np.ndarray)
        or local.dtype != np.uint8
        or local.ndim != 3
        or local.shape[0] != 6
        or local.shape[1] != local.shape[2]
        or local.shape[1] < 3
        or local.shape[1] % 2 != 1
        or np.any(local > 1)
    ):
        raise ValueError("local must be binary uint8 (6, odd side, odd side)")
    visible = local[0].astype(bool)
    if np.any(local[1:, ~visible]) or np.any(local[1, visible] + local[2, visible] != 1):
        raise ValueError("Invisible cells must be empty; visible terrain must be wall or floor")
    if np.any((local[3] | local[4] | local[5]) & (1 - local[2])) or np.any(local[3] & local[4]):
        raise ValueError("Entities require floor; POI cannot be both pending and active")
    radius = local.shape[1] // 2
    return {
        (position[0] + int(col) - radius, position[1] + int(row) - radius): tuple(
            int(v) for v in local[1:, row, col]
        )
        for row, col in np.argwhere(visible)
    }
