from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class StateScore:
    state_id: Hashable
    kappa: float
    confidence: float
    semantic_tag: str = ""


@dataclass(frozen=True)
class StateSplit:
    bridge: tuple[Hashable, ...]
    garden: tuple[Hashable, ...]
    middle: tuple[Hashable, ...]
    bridge_threshold: float
    garden_threshold: float
    bridge_tie_band: tuple[Hashable, ...] = ()
    garden_tie_band: tuple[Hashable, ...] = ()


def _finite_scores(states: Iterable[StateScore], field: str) -> list[tuple[Hashable, float]]:
    rows = []
    for item in states:
        value = float(getattr(item, field))
        if np.isfinite(value):
            rows.append((item.state_id, value))
    return rows


def top_bottom_split(
    states: Sequence[StateScore],
    *,
    field: str,
    fraction: float = 0.2,
) -> StateSplit:
    """Split states by top/bottom score fraction and expose boundary tie bands."""
    if not 0.0 < fraction < 0.5:
        raise ValueError("fraction must be between 0 and 0.5")
    rows = _finite_scores(states, field)
    if not rows:
        return StateSplit((), (), (), float("nan"), float("nan"))
    rows_sorted = sorted(rows, key=lambda row: (row[1], repr(row[0])))
    n_edge = max(1, int(np.floor(len(rows_sorted) * fraction)))
    garden_rows = rows_sorted[:n_edge]
    bridge_rows = rows_sorted[-n_edge:]
    garden_threshold = float(garden_rows[-1][1])
    bridge_threshold = float(bridge_rows[0][1])
    garden = tuple(row[0] for row in garden_rows)
    bridge = tuple(row[0] for row in bridge_rows)
    garden_tie_band = tuple(row[0] for row in rows_sorted if row[1] <= garden_threshold)
    bridge_tie_band = tuple(row[0] for row in rows_sorted if row[1] >= bridge_threshold)
    edge_ids = set(garden) | set(bridge)
    middle = tuple(row[0] for row in rows_sorted if row[0] not in edge_ids)
    return StateSplit(
        bridge=bridge,
        garden=garden,
        middle=middle,
        bridge_threshold=bridge_threshold,
        garden_threshold=garden_threshold,
        bridge_tie_band=bridge_tie_band,
        garden_tie_band=garden_tie_band,
    )


def split_overlap(left: Sequence[Hashable], right: Sequence[Hashable]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return float(len(left_set & right_set) / min(len(left_set), len(right_set)))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_from_scores(states: Sequence[StateScore]) -> float:
    rows = [
        (float(item.kappa), float(item.confidence))
        for item in states
        if np.isfinite(float(item.kappa)) and np.isfinite(float(item.confidence))
    ]
    if len(rows) < 2:
        return float("nan")
    arr = np.array(rows, dtype=np.float64)
    x = _rank(arr[:, 0])
    y = _rank(arr[:, 1])
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(x, y) / denom)
