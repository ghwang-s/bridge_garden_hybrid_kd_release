from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence


@dataclass(frozen=True)
class ScriptStep:
    dist: tuple[tuple[int, float], ...]
    semantic_tag: str
    expected_role: str
    violation_dist: tuple[tuple[int, float], ...] = ()
    off_policy_violates: bool = True
    clean_token_ids: tuple[int, ...] = ()
    eval_token_ids: tuple[int, ...] = ()
    style_dists: tuple[tuple[tuple[int, float], ...], ...] = ()
    style_token_ids: tuple[tuple[int, ...], ...] = ()
    style_canonical_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ScriptState:
    sample_id: int
    position: int
    status: str
    prefix: tuple[int, ...]
    style: int = 0


class ScriptedOracle:
    """Exact finite-state oracle over pre-built probabilistic scripts."""

    def __init__(
        self,
        *,
        scripts: Sequence[Sequence[ScriptStep]],
        eval_token_ids: Sequence[int],
        pad_id: int,
        bos_id: int,
        eos_id: int,
    ) -> None:
        self.scripts = tuple(tuple(step for step in script) for script in scripts)
        self.eval_token_ids = tuple(int(x) for x in eval_token_ids)
        self.pad_id = int(pad_id)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)

    def initial_state(self, sample_id: int) -> ScriptState:
        return ScriptState(sample_id=int(sample_id), position=0, status="clean", prefix=(self.bos_id,), style=0)

    def next_dist(self, state: Hashable) -> Mapping[int, float]:
        state = self._as_state(state)
        if self.is_terminal(state):
            return {}
        step = self.scripts[state.sample_id][state.position]
        if state.status == "clean" or not step.violation_dist:
            if state.status == "clean" and step.style_dists:
                style = max(0, min(int(state.style), len(step.style_dists) - 1))
                return dict(step.style_dists[style])
            return dict(step.dist)
        return dict(step.violation_dist)

    def eval_token_ids_for_state(self, state: Hashable) -> tuple[int, ...]:
        state = self._as_state(state)
        if self.is_terminal(state):
            return ()
        step = self.scripts[state.sample_id][state.position]
        if step.eval_token_ids:
            return tuple(int(x) for x in step.eval_token_ids)
        return self.eval_token_ids

    def step(self, state: Hashable, action_id: int) -> Hashable:
        state = self._as_state(state)
        if self.is_terminal(state):
            return state
        dist = self.next_dist(state)
        next_status = state.status
        next_action_id = int(action_id)
        next_style = int(state.style)
        script_step = self.scripts[state.sample_id][state.position]
        if state.status == "clean":
            if script_step.clean_token_ids:
                if int(action_id) not in set(script_step.clean_token_ids):
                    if script_step.off_policy_violates:
                        next_status = "violation"
                    else:
                        next_action_id = _canonical_style_token(script_step, next_style, int(script_step.clean_token_ids[0]))
                elif not script_step.off_policy_violates:
                    next_style = _style_after_action(script_step, int(action_id), next_style)
            elif script_step.off_policy_violates and not _keeps_clean(script_step, dist, int(action_id)):
                next_status = "violation"
        return ScriptState(
            sample_id=state.sample_id,
            position=state.position + 1,
            status=next_status,
            prefix=state.prefix + (next_action_id,),
            style=next_style,
        )

    def is_terminal(self, state: Hashable) -> bool:
        state = self._as_state(state)
        return state.position >= len(self.scripts[state.sample_id])

    def remaining_horizon(self, state: Hashable) -> int:
        state = self._as_state(state)
        return max(0, len(self.scripts[state.sample_id]) - state.position)

    def semantic_tag(self, state: Hashable) -> str:
        state = self._as_state(state)
        if self.is_terminal(state):
            return "terminal"
        return self.scripts[state.sample_id][state.position].semantic_tag

    def expected_role(self, state: Hashable) -> str:
        state = self._as_state(state)
        if self.is_terminal(state):
            return "terminal"
        return self.scripts[state.sample_id][state.position].expected_role

    @staticmethod
    def _as_state(state: Hashable) -> ScriptState:
        if not isinstance(state, ScriptState):
            raise TypeError(f"expected ScriptState, got {type(state)!r}")
        return state


def _keeps_clean(step: ScriptStep, dist: Mapping[int, float], action_id: int) -> bool:
    if step.clean_token_ids:
        return int(action_id) in set(step.clean_token_ids)
    return float(dist.get(int(action_id), 0.0)) > 0.0


def _style_after_action(step: ScriptStep, action_id: int, current_style: int) -> int:
    for style, token_ids in enumerate(step.style_token_ids):
        if int(action_id) in set(token_ids):
            return int(style)
    return int(current_style)


def _canonical_style_token(step: ScriptStep, style: int, fallback: int) -> int:
    if not step.style_canonical_token_ids:
        return int(fallback)
    style = max(0, min(int(style), len(step.style_canonical_token_ids) - 1))
    return int(step.style_canonical_token_ids[style])
