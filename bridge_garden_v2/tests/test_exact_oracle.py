from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

import numpy as np

from bridge_garden_v2.exact_oracle import exact_kappa_for_state, exact_loss_to_go, teacher_sharpness


@dataclass(frozen=True)
class ToyState:
    step: int
    branch: str = "clean"


class ToyOracle:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    eval_token_ids = (2, 3, 4, 5)

    def next_dist(self, state: Hashable) -> Mapping[int, float]:
        state = _as_state(state)
        if state.step >= 2:
            return {2: 1.0}
        if state.step == 0:
            return {3: 0.8, 4: 0.2}
        if state.branch == "clean":
            return {5: 1.0}
        return {4: 0.5, 5: 0.5}

    def step(self, state: Hashable, action_id: int) -> Hashable:
        state = _as_state(state)
        if action_id == 2:
            return ToyState(step=3, branch=state.branch)
        if state.step == 0:
            return ToyState(step=1, branch="clean" if action_id == 3 else "bad")
        return ToyState(step=state.step + 1, branch=state.branch)

    def is_terminal(self, state: Hashable) -> bool:
        return _as_state(state).step >= 3

    def remaining_horizon(self, state: Hashable) -> int:
        return max(0, 3 - _as_state(state).step)


class ToyStudent:
    def loss_at_state(self, state: Hashable, oracle: ToyOracle) -> float:
        state = _as_state(state)
        return 0.1 if state.branch == "clean" else 2.0


def _as_state(state: Hashable) -> ToyState:
    assert isinstance(state, ToyState)
    return state


def test_teacher_sharpness_bounds() -> None:
    assert teacher_sharpness({}, 1) == 0.0
    assert teacher_sharpness({3: 1.0}, 4) == 1.0
    uniform = teacher_sharpness({2: 0.25, 3: 0.25, 4: 0.25, 5: 0.25}, 4)
    assert np.isclose(uniform, 0.0)


def test_exact_loss_to_go_uses_teacher_expectation() -> None:
    oracle = ToyOracle()
    student = ToyStudent()
    clean = exact_loss_to_go(oracle, student, ToyState(step=1, branch="clean"), continuation="expectation")
    bad = exact_loss_to_go(oracle, student, ToyState(step=1, branch="bad"), continuation="expectation")
    assert np.isclose(clean, 0.2)
    assert np.isclose(bad, 4.0)


def test_exact_loss_to_go_mode_follows_teacher_mode_path() -> None:
    oracle = ToyOracle()
    student = ToyStudent()
    value = exact_loss_to_go(oracle, student, ToyState(step=0), continuation="mode")
    assert np.isclose(value, 0.3)


def test_exact_kappa_uses_full_eval_vocab_and_teacher_weighted_qbar() -> None:
    oracle = ToyOracle()
    student = ToyStudent()
    result = exact_kappa_for_state(oracle, student, ToyState(step=0))
    assert result.action_ids == (2, 3, 4, 5)
    assert len(result.q_values) == 4
    expected_qbar = 0.8 * result.q_values[1] + 0.2 * result.q_values[2]
    assert np.isclose(result.qbar, expected_qbar)
    assert np.isclose(result.state_kappa, sum(result.action_kappa))
    assert np.isclose(result.teacher_confidence, teacher_sharpness({3: 0.8, 4: 0.2}, 2))


def test_exact_kappa_rejects_pad_or_bos_eval_vocab() -> None:
    class BadOracle(ToyOracle):
        eval_token_ids = (0, 2, 3)

    try:
        exact_kappa_for_state(BadOracle(), ToyStudent(), ToyState(step=0))
    except ValueError as exc:
        assert "exclude PAD and BOS" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_exact_loss_to_go_has_expansion_guard() -> None:
    oracle = ToyOracle()
    student = ToyStudent()
    try:
        exact_loss_to_go(oracle, student, ToyState(step=0), max_expansions=1, continuation="expectation")
    except RuntimeError as exc:
        assert "max_expansions" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
