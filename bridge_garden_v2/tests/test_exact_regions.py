from __future__ import annotations

import numpy as np

from bridge_garden_v2.exact_regions import (
    StateScore,
    spearman_from_scores,
    split_overlap,
    top_bottom_split,
)


def test_top_bottom_split_uses_state_level_top_bottom_20_percent() -> None:
    states = [
        StateScore(state_id=("seq", idx), kappa=float(idx), confidence=float(10 - idx))
        for idx in range(10)
    ]
    split = top_bottom_split(states, field="kappa", fraction=0.2)
    assert split.garden == (("seq", 0), ("seq", 1))
    assert split.bridge == (("seq", 8), ("seq", 9))
    assert len(split.middle) == 6
    assert split.garden_threshold == 1.0
    assert split.bridge_threshold == 8.0
    assert split.garden_tie_band == split.garden
    assert split.bridge_tie_band == split.bridge


def test_top_bottom_split_reports_boundary_tie_bands() -> None:
    states = [
        StateScore(state_id=idx, kappa=float(idx), confidence=score)
        for idx, score in enumerate([0.1, 0.1, 0.1, 0.4, 0.8, 0.9])
    ]
    split = top_bottom_split(states, field="confidence", fraction=0.2)
    assert split.garden == (0,)
    assert split.garden_tie_band == (0, 1, 2)
    assert split.bridge == (5,)
    assert split.bridge_tie_band == (5,)


def test_split_overlap_uses_smaller_set_denominator() -> None:
    assert split_overlap([1, 2, 3], [2, 3, 4]) == 2 / 3
    assert split_overlap([], []) == 1.0
    assert split_overlap([], [1]) == 0.0


def test_spearman_from_scores_detects_monotonic_relation() -> None:
    positive = [
        StateScore(state_id=idx, kappa=float(idx), confidence=float(idx))
        for idx in range(8)
    ]
    negative = [
        StateScore(state_id=idx, kappa=float(idx), confidence=float(8 - idx))
        for idx in range(8)
    ]
    assert np.isclose(spearman_from_scores(positive), 1.0)
    assert np.isclose(spearman_from_scores(negative), -1.0)


def test_spearman_handles_ties() -> None:
    states = [
        StateScore(state_id=0, kappa=1.0, confidence=1.0),
        StateScore(state_id=1, kappa=1.0, confidence=1.0),
        StateScore(state_id=2, kappa=2.0, confidence=2.0),
        StateScore(state_id=3, kappa=3.0, confidence=3.0),
    ]
    assert spearman_from_scores(states) > 0.9
