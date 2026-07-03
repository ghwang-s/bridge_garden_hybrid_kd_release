from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Hashable, Mapping, Protocol, Sequence

import numpy as np


class ExactOracle(Protocol):
    """Finite-state oracle teacher used by the synthetic path."""

    eval_token_ids: tuple[int, ...]
    pad_id: int
    bos_id: int
    eos_id: int

    def next_dist(self, state: Hashable) -> Mapping[int, float]:
        """Return the exact teacher next-token distribution at state."""

    def step(self, state: Hashable, action_id: int) -> Hashable:
        """Apply one action, including off-policy overrides, and return next state."""

    def is_terminal(self, state: Hashable) -> bool:
        """Return whether no further loss should be accumulated from this state."""

    def remaining_horizon(self, state: Hashable) -> int:
        """Return the exact remaining horizon for full-continuation evaluation."""

    def eval_token_ids_for_state(self, state: Hashable) -> tuple[int, ...]:
        """Return the optional state-local intervention set for metrics."""


class StudentLoss(Protocol):
    def loss_at_state(self, state: Hashable, oracle: ExactOracle) -> float:
        """Return downstream per-state loss, usually KL(pi_T(.|s) || pi_theta(.|s))."""


@dataclass(frozen=True)
class ExactKappaStateResult:
    state: Hashable
    action_ids: tuple[int, ...]
    q_values: tuple[float, ...]
    qbar: float
    action_kappa: tuple[float, ...]
    state_kappa: float
    teacher_confidence: float
    teacher_max_prob: float
    computation_stats: Mapping[str, Any] | None = None


def teacher_sharpness(next_dist: Mapping[int, float], valid_token_count: int) -> float:
    probs = np.array([p for p in next_dist.values() if p > 0.0], dtype=np.float64)
    if probs.size == 0:
        return 0.0
    if valid_token_count <= 1:
        return 1.0
    entropy = float(-(probs * np.log(probs)).sum())
    return float(1.0 - entropy / np.log(valid_token_count))


def _normalized_dist(dist: Mapping[int, float]) -> dict[int, float]:
    total = float(sum(dist.values()))
    if total <= 0.0:
        return {}
    return {int(k): float(v) / total for k, v in dist.items() if float(v) > 0.0}


def exact_loss_to_go(
    oracle: ExactOracle,
    student: StudentLoss,
    state: Hashable,
    horizon: int | None = None,
    max_expansions: int | None = None,
    continuation: str = "expectation",
) -> float:
    """Exact expected teacher-continuation loss-to-go from a state."""
    start_horizon = oracle.remaining_horizon(state) if horizon is None else horizon
    if continuation not in {"mode", "expectation"}:
        raise ValueError("continuation must be 'mode' or 'expectation'")
    if continuation == "mode":
        current = state
        total = 0.0
        for _ in range(int(start_horizon)):
            if oracle.is_terminal(current):
                break
            total += float(student.loss_at_state(current, oracle))
            dist = _normalized_dist(oracle.next_dist(current))
            if not dist:
                break
            mode_action = max(dist, key=lambda action_id: (dist[action_id], -int(action_id)))
            current = oracle.step(current, int(mode_action))
        return float(total)

    expansions = 0

    @lru_cache(maxsize=None)
    def _dp(cached_state: Hashable, steps_left: int) -> float:
        nonlocal expansions
        expansions += 1
        if max_expansions is not None and expansions > max_expansions:
            raise RuntimeError(
                f"exact continuation exceeded max_expansions={max_expansions}; "
                "the oracle support is too large for the current exact setup"
            )
        if steps_left <= 0 or oracle.is_terminal(cached_state):
            return 0.0
        loss = float(student.loss_at_state(cached_state, oracle))
        dist = _normalized_dist(oracle.next_dist(cached_state))
        if not dist:
            return loss
        future = 0.0
        for action_id, prob in dist.items():
            future += prob * _dp(oracle.step(cached_state, action_id), steps_left - 1)
        return loss + future

    return float(_dp(state, int(start_horizon)))


def exact_kappa_for_state(
    oracle: ExactOracle,
    student: StudentLoss,
    state: Hashable,
    *,
    horizon: int | None = None,
    max_expansions_per_action: int | None = None,
    continuation: str = "expectation",
) -> ExactKappaStateResult:
    """Compute paper-aligned action-level and state-level κ for one state.

    All tokens in `oracle.eval_token_ids` are evaluated. `Qbar` uses the exact
    teacher distribution at `state`; actions with zero teacher probability still
    receive `κ(a|s)=Q(s,a)-Qbar` and contribute to `κ(s)`.
    """
    action_ids = tuple(int(a) for a in oracle.eval_token_ids)
    if oracle.pad_id in action_ids or oracle.bos_id in action_ids:
        raise ValueError("eval_token_ids must exclude PAD and BOS")

    dist = _normalized_dist(oracle.next_dist(state))
    computation_stats: Mapping[str, Any] | None = None
    next_states = tuple(oracle.step(state, action_id) for action_id in action_ids)
    if next_states and len(set(next_states)) == 1:
        q_arr = np.zeros(len(action_ids), dtype=np.float64)
        computation_stats = {
            "constant_next_state_shortcut": True,
            "action_count": len(action_ids),
            "expanded_state_total": 0,
            "max_frontier_states": 1,
            "max_active_states": 0,
        }
    elif continuation == "expectation" and horizon is None:
        q_arr, computation_stats = _batched_expectation_q_values(
            oracle=oracle,
            student=student,
            state=state,
            action_ids=action_ids,
            max_expansions=max_expansions_per_action,
        )
    else:
        q_values = []
        for action_id in action_ids:
            next_state = oracle.step(state, action_id)
            q_values.append(
                exact_loss_to_go(
                    oracle,
                    student,
                    next_state,
                    horizon=horizon,
                    max_expansions=max_expansions_per_action,
                    continuation=continuation,
                )
            )
        q_arr = np.array(q_values, dtype=np.float64)
    weights = np.array([dist.get(action_id, 0.0) for action_id in action_ids], dtype=np.float64)
    qbar = float(np.dot(weights, q_arr))
    action_kappa = q_arr - qbar
    return ExactKappaStateResult(
        state=state,
        action_ids=action_ids,
        q_values=tuple(float(x) for x in q_arr),
        qbar=qbar,
        action_kappa=tuple(float(x) for x in action_kappa),
        state_kappa=float(action_kappa.sum()),
        teacher_confidence=teacher_sharpness(dist, valid_token_count=max(1, len(dist))),
        teacher_max_prob=float(max(dist.values())) if dist else 0.0,
        computation_stats=computation_stats,
    )


def _batched_expectation_q_values(
    *,
    oracle: ExactOracle,
    student: StudentLoss,
    state: Hashable,
    action_ids: Sequence[int],
    max_expansions: int | None,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    if _supports_shared_prefix_tree(state):
        return _shared_prefix_tree_q_values(
            oracle=oracle,
            student=student,
            state=state,
            action_ids=action_ids,
            max_expansions=max_expansions,
        )
    return _prefix_frontier_q_values(
        oracle=oracle,
        student=student,
        state=state,
        action_ids=action_ids,
        max_expansions=max_expansions,
    )


def _prefix_frontier_q_values(
    *,
    oracle: ExactOracle,
    student: StudentLoss,
    state: Hashable,
    action_ids: Sequence[int],
    max_expansions: int | None,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    q_values = np.zeros(len(action_ids), dtype=np.float64)
    frontier: dict[Hashable, np.ndarray] = {}
    for idx, action_id in enumerate(action_ids):
        next_state = oracle.step(state, int(action_id))
        weights = frontier.get(next_state)
        if weights is None:
            weights = np.zeros(len(action_ids), dtype=np.float64)
            frontier[next_state] = weights
        weights[idx] += 1.0

    expansions = 0
    iterations = 0
    max_frontier_states = len(frontier)
    max_active_states = 0
    active_state_total = 0
    while frontier:
        iterations += 1
        max_frontier_states = max(max_frontier_states, len(frontier))
        active_states = [
            current for current in frontier
            if not oracle.is_terminal(current) and oracle.remaining_horizon(current) > 0
        ]
        if not active_states:
            break
        max_active_states = max(max_active_states, len(active_states))
        active_state_total += len(active_states)
        expansions += len(active_states)
        if max_expansions is not None and expansions > max_expansions:
            raise RuntimeError(
                f"exact continuation exceeded max_expansions={max_expansions}; "
                "the oracle support is too large for the current exact setup"
            )
        losses = _loss_many(student, active_states, oracle)
        next_frontier: dict[Hashable, np.ndarray] = {}
        for current, loss_value in zip(active_states, losses):
            weights = frontier[current]
            q_values += weights * float(loss_value)
            dist = _normalized_dist(oracle.next_dist(current))
            if not dist:
                continue
            for action_id, prob in dist.items():
                next_state = oracle.step(current, int(action_id))
                next_weights = next_frontier.get(next_state)
                if next_weights is None:
                    next_weights = np.zeros(len(action_ids), dtype=np.float64)
                    next_frontier[next_state] = next_weights
                next_weights += weights * float(prob)
        frontier = next_frontier
    return q_values, {
        "shared_prefix_tree": False,
        "action_count": len(action_ids),
        "iterations": iterations,
        "expanded_state_total": active_state_total,
        "max_frontier_states": max_frontier_states,
        "max_active_states": max_active_states,
    }


def _shared_prefix_tree_q_values(
    *,
    oracle: ExactOracle,
    student: StudentLoss,
    state: Hashable,
    action_ids: Sequence[int],
    max_expansions: int | None,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Compute Q values while sharing prefix-equivalent teacher trees.

    Scripted synthetic oracles make teacher transitions depend on structural
    state fields such as sample id, position, status, and style, while the
    neural loss still depends on the full token prefix.  The full-state frontier
    keyed by full states therefore expands many copies of the same teacher tree
    when global V_eval actions only differ in the token prefix.  This routine
    shares the teacher tree and batches the prefix-specific loss rows without
    changing the exact Q values.
    """
    base_prefix = _prefix_tuple_for_shared_tree(state)
    action_count = len(action_ids)
    q_values = np.zeros(action_count, dtype=np.float64)
    root_tokens = np.zeros(action_count, dtype=np.int64)
    template = state

    frontier: dict[tuple[tuple[Any, ...], tuple[int, ...]], np.ndarray] = {}
    for idx, action_id in enumerate(action_ids):
        next_state = oracle.step(state, int(action_id))
        appended = _single_appended_token(state, next_state)
        root_tokens[idx] = appended
        key = (_structural_state_key(next_state), ())
        weights = frontier.get(key)
        if weights is None:
            weights = np.zeros(action_count, dtype=np.float64)
            frontier[key] = weights
        weights[idx] += 1.0

    iterations = 0
    expanded_state_total = 0
    abstract_expanded_state_total = 0
    max_frontier_states = _prefix_unique_count(frontier, root_tokens)
    max_active_states = 0
    max_abstract_frontier_nodes = len(frontier)
    max_abstract_active_nodes = 0

    while frontier:
        iterations += 1
        max_frontier_states = max(max_frontier_states, _prefix_unique_count(frontier, root_tokens))
        max_abstract_frontier_nodes = max(max_abstract_frontier_nodes, len(frontier))

        active_items: list[tuple[tuple[Any, ...], tuple[int, ...], np.ndarray, Hashable]] = []
        for (struct_key, suffix), weights in frontier.items():
            first_idx = _first_nonzero_index(weights)
            if first_idx is None:
                continue
            prefix = base_prefix + (int(root_tokens[first_idx]),) + suffix
            representative = _state_from_shared_key(template, struct_key, prefix)
            if oracle.is_terminal(representative) or oracle.remaining_horizon(representative) <= 0:
                continue
            active_items.append((struct_key, suffix, weights, representative))

        if not active_items:
            break

        prefix_active_count = sum(
            len(_prefix_groups(weights, root_tokens, suffix))
            for _, suffix, weights, _ in active_items
        )
        abstract_active_count = len(active_items)
        max_active_states = max(max_active_states, prefix_active_count)
        max_abstract_active_nodes = max(max_abstract_active_nodes, abstract_active_count)
        expanded_state_total += prefix_active_count
        abstract_expanded_state_total += abstract_active_count
        if max_expansions is not None and expanded_state_total > max_expansions:
            raise RuntimeError(
                f"exact continuation exceeded max_expansions={max_expansions}; "
                "the oracle support is too large for the current exact setup"
            )

        loss_states: list[Hashable] = []
        loss_refs: list[tuple[np.ndarray, tuple[int, ...]]] = []
        for struct_key, suffix, weights, _representative in active_items:
            for root_token, grouped_suffix, indices in _prefix_groups(weights, root_tokens, suffix):
                prefix = base_prefix + (root_token,) + grouped_suffix
                loss_states.append(_state_from_shared_key(template, struct_key, prefix))
                loss_refs.append((weights, indices))
        losses = _loss_many(student, loss_states, oracle)
        for (weights, indices), loss_value in zip(loss_refs, losses):
            index_array = np.array(indices, dtype=np.int64)
            q_values[index_array] += weights[index_array] * float(loss_value)

        next_frontier: dict[tuple[tuple[Any, ...], tuple[int, ...]], np.ndarray] = {}
        for _struct_key, suffix, weights, representative in active_items:
            dist = _normalized_dist(oracle.next_dist(representative))
            if not dist:
                continue
            for action_id, prob in dist.items():
                next_state = oracle.step(representative, int(action_id))
                appended = _single_appended_token(representative, next_state)
                key = (_structural_state_key(next_state), suffix + (appended,))
                next_weights = next_frontier.get(key)
                if next_weights is None:
                    next_weights = np.zeros(action_count, dtype=np.float64)
                    next_frontier[key] = next_weights
                next_weights += weights * float(prob)
        frontier = next_frontier

    return q_values, {
        "shared_prefix_tree": True,
        "action_count": action_count,
        "iterations": iterations,
        "expanded_state_total": expanded_state_total,
        "max_frontier_states": max_frontier_states,
        "max_active_states": max_active_states,
        "abstract_expanded_state_total": abstract_expanded_state_total,
        "abstract_max_frontier_nodes": max_abstract_frontier_nodes,
        "abstract_max_active_nodes": max_abstract_active_nodes,
    }


def _loss_many(student: StudentLoss, states: list[Hashable], oracle: ExactOracle) -> list[float]:
    if hasattr(student, "loss_many"):
        values = student.loss_many(states, oracle)  # type: ignore[attr-defined]
        return [float(value) for value in values]
    return [float(student.loss_at_state(state, oracle)) for state in states]


def _supports_shared_prefix_tree(state: Hashable) -> bool:
    try:
        _prefix_tuple_for_shared_tree(state)
        _structural_state_key(state)
        replace(state, prefix=getattr(state, "prefix"))
    except Exception:
        return False
    return True


def _prefix_tuple_for_shared_tree(state: Hashable) -> tuple[int, ...]:
    prefix = getattr(state, "prefix", None)
    if prefix is None:
        raise TypeError("state does not expose prefix")
    return tuple(int(token_id) for token_id in prefix)


def _structural_state_key(state: Hashable) -> tuple[Any, ...]:
    return (
        int(getattr(state, "sample_id")),
        int(getattr(state, "position")),
        str(getattr(state, "status")),
        int(getattr(state, "style", 0)),
    )


def _state_from_shared_key(template: Hashable, key: tuple[Any, ...], prefix: tuple[int, ...]) -> Hashable:
    sample_id, position, status, style = key
    return replace(
        template,
        sample_id=int(sample_id),
        position=int(position),
        status=str(status),
        prefix=tuple(int(token_id) for token_id in prefix),
        style=int(style),
    )


def _single_appended_token(before: Hashable, after: Hashable) -> int:
    before_prefix = _prefix_tuple_for_shared_tree(before)
    after_prefix = _prefix_tuple_for_shared_tree(after)
    if len(after_prefix) != len(before_prefix) + 1:
        raise ValueError("expected oracle.step to append exactly one token")
    return int(after_prefix[-1])


def _first_nonzero_index(weights: np.ndarray) -> int | None:
    indices = np.flatnonzero(weights)
    if len(indices) == 0:
        return None
    return int(indices[0])


def _nonzero_count(weights: np.ndarray) -> int:
    return int(np.count_nonzero(weights))


def _prefix_groups(
    weights: np.ndarray,
    root_tokens: np.ndarray,
    suffix: tuple[int, ...],
) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]:
    groups: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for idx in np.flatnonzero(weights):
        key = (int(root_tokens[int(idx)]), suffix)
        groups.setdefault(key, []).append(int(idx))
    return tuple((root_token, grouped_suffix, tuple(indices)) for (root_token, grouped_suffix), indices in groups.items())


def _prefix_unique_count(
    frontier: Mapping[tuple[tuple[Any, ...], tuple[int, ...]], np.ndarray],
    root_tokens: np.ndarray,
) -> int:
    return sum(len(_prefix_groups(weights, root_tokens, suffix)) for (_struct_key, suffix), weights in frontier.items())
