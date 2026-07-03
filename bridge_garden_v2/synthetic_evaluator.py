from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import random
from typing import Hashable, Sequence

from .exact_oracle import ExactKappaStateResult, ExactOracle, StudentLoss, exact_kappa_for_state
from .exact_regions import StateScore, StateSplit, spearman_from_scores, split_overlap, top_bottom_split


@dataclass(frozen=True)
class SyntheticEvaluation:
    scores: tuple[StateScore, ...]
    kappa_split: StateSplit
    confidence_split: StateSplit
    confidence_rho: float
    bridge_overlap: float
    garden_overlap: float
    bridge_tie_aware_overlap: float
    garden_tie_aware_overlap: float
    top20_semantic_counts: dict[str, int]
    bottom20_semantic_counts: dict[str, int]
    kappa_results: tuple[ExactKappaStateResult, ...]


def collect_clean_path_states(oracle: ExactOracle, sample_count: int) -> list[Hashable]:
    """Collect one deterministic clean analysis path per scripted sample.

    The chosen path follows the teacher mode at each state. This is for smoke and
    deterministic analysis-set construction; full experiments can replace it with
    sampled sampled analysis sequences.
    """
    states: list[Hashable] = []
    for sample_id in range(sample_count):
        if not hasattr(oracle, "initial_state"):
            raise TypeError("oracle must expose initial_state(sample_id)")
        state = oracle.initial_state(sample_id)  # type: ignore[attr-defined]
        while not oracle.is_terminal(state):
            states.append(state)
            dist = oracle.next_dist(state)
            if not dist:
                break
            action_id = max(dist, key=lambda token_id: (dist[token_id], -int(token_id)))
            state = oracle.step(state, int(action_id))
    return states


def collect_sampled_path_states(oracle: ExactOracle, sample_ids: list[int], *, seed: int = 0) -> list[Hashable]:
    """Collect states from teacher-sampled analysis paths under fixed seeds."""
    rng = random.Random(seed)
    states: list[Hashable] = []
    for sample_id in sample_ids:
        if not hasattr(oracle, "initial_state"):
            raise TypeError("oracle must expose initial_state(sample_id)")
        state = oracle.initial_state(int(sample_id))  # type: ignore[attr-defined]
        while not oracle.is_terminal(state):
            states.append(state)
            dist = oracle.next_dist(state)
            if not dist:
                break
            state = oracle.step(state, _sample_from_dist(dist, rng))
    return states


def evaluate_oracle_states(
    oracle: ExactOracle,
    student: StudentLoss,
    states: list[Hashable],
    *,
    exclude_semantic_tags: set[str] | None = None,
    continuation: str = "expectation",
) -> SyntheticEvaluation:
    if exclude_semantic_tags:
        states = [state for state in states if _semantic_tag(oracle, state) not in exclude_semantic_tags]
    kappa_results = tuple(
        exact_kappa_for_state(oracle, student, state, continuation=continuation)
        for state in states
    )
    return synthetic_evaluation_from_results(oracle, states, kappa_results)


def synthetic_evaluation_from_results(
    oracle: ExactOracle,
    states: list[Hashable],
    kappa_results: Sequence[ExactKappaStateResult],
) -> SyntheticEvaluation:
    scores = tuple(
        StateScore(
            state_id=state,
            kappa=result.state_kappa,
            confidence=result.teacher_confidence,
            semantic_tag=_semantic_tag(oracle, state),
        )
        for state, result in zip(states, kappa_results)
    )
    kappa_split = top_bottom_split(scores, field="kappa", fraction=0.2)
    confidence_split = top_bottom_split(scores, field="confidence", fraction=0.2)
    semantic_by_state = {score.state_id: score.semantic_tag for score in scores}
    top_counts = Counter(semantic_by_state[state_id] for state_id in kappa_split.bridge)
    bottom_counts = Counter(semantic_by_state[state_id] for state_id in kappa_split.garden)
    return SyntheticEvaluation(
        scores=scores,
        kappa_split=kappa_split,
        confidence_split=confidence_split,
        confidence_rho=spearman_from_scores(scores),
        bridge_overlap=split_overlap(kappa_split.bridge, confidence_split.bridge),
        garden_overlap=split_overlap(kappa_split.garden, confidence_split.garden),
        bridge_tie_aware_overlap=split_overlap(kappa_split.bridge, confidence_split.bridge_tie_band),
        garden_tie_aware_overlap=split_overlap(kappa_split.garden, confidence_split.garden_tie_band),
        top20_semantic_counts=dict(top_counts),
        bottom20_semantic_counts=dict(bottom_counts),
        kappa_results=kappa_results,
    )


def estimate_teacher_support_size(
    oracle: ExactOracle,
    state: Hashable,
    *,
    horizon: int | None = None,
    cap: int = 1_000_000,
) -> int:
    """Estimate exact continuation support size with early capping.

    Counts terminal paths under the teacher distribution from `state`. This is a
    check guard for exact κ feasibility; it ignores student loss cost, so it
    is a lower-level structural check.
    """
    start_horizon = oracle.remaining_horizon(state) if horizon is None else horizon

    @lru_cache(maxsize=None)
    def _count(cached_state: Hashable, steps_left: int) -> int:
        if steps_left <= 0 or oracle.is_terminal(cached_state):
            return 1
        dist = oracle.next_dist(cached_state)
        if not dist:
            return 1
        total = 0
        for action_id, prob in dist.items():
            if prob <= 0.0:
                continue
            total += _count(oracle.step(cached_state, int(action_id)), steps_left - 1)
            if total > cap:
                return cap + 1
        return total

    return _count(state, int(start_horizon))


def positions_from_state_ids(state_ids: list[Hashable] | tuple[Hashable, ...]) -> set[int]:
    positions = set()
    for state_id in state_ids:
        if not hasattr(state_id, "position"):
            raise TypeError("state id does not expose a position")
        positions.add(int(getattr(state_id, "position")))
    return positions


def _semantic_tag(oracle: ExactOracle, state: Hashable) -> str:
    if hasattr(oracle, "semantic_tag"):
        return str(oracle.semantic_tag(state))  # type: ignore[attr-defined]
    return ""


def _sample_from_dist(dist, rng: random.Random) -> int:
    total = float(sum(dist.values()))
    if total <= 0.0:
        raise ValueError("cannot sample from empty distribution")
    threshold = rng.random() * total
    cdf = 0.0
    last = None
    for token_id, prob in sorted(dist.items()):
        last = int(token_id)
        cdf += float(prob)
        if threshold <= cdf:
            return int(token_id)
    if last is None:
        raise ValueError("cannot sample from empty distribution")
    return last
