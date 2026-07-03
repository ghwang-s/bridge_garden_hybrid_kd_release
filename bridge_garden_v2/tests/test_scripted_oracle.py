from __future__ import annotations

from typing import Hashable

from bridge_garden_v2.exact_oracle import exact_kappa_for_state
from bridge_garden_v2.scripted_oracle import ScriptedOracle, ScriptStep


def _oracle() -> ScriptedOracle:
    return ScriptedOracle(
        scripts=[
            [
                ScriptStep(dist=((3, 0.9), (4, 0.1)), semantic_tag="operator", expected_role="high_risk", violation_dist=((5, 1.0),)),
                ScriptStep(dist=((5, 1.0),), semantic_tag="return_semantics", expected_role="high_risk", off_policy_violates=False),
            ]
        ],
        eval_token_ids=(2, 3, 4, 5),
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )


def test_scripted_oracle_clean_transition_keeps_clean_status() -> None:
    oracle = _oracle()
    state = oracle.initial_state(0)
    next_state = oracle.step(state, 3)
    assert next_state.status == "clean"
    assert next_state.prefix == (1, 3)
    assert oracle.next_dist(next_state) == {5: 1.0}


def test_scripted_oracle_can_mark_flexible_step_as_off_policy_tolerant() -> None:
    oracle = _oracle()
    state = oracle.step(oracle.initial_state(0), 3)
    next_state = oracle.step(state, 4)
    assert next_state.status == "clean"


def test_scripted_oracle_clean_token_ids_override_teacher_support() -> None:
    oracle = ScriptedOracle(
        scripts=[
            [
                ScriptStep(
                    dist=((3, 0.5), (4, 0.5)),
                    semantic_tag="equivalent_form",
                    expected_role="flexible",
                    violation_dist=((5, 1.0),),
                    clean_token_ids=(3,),
                ),
                ScriptStep(
                    dist=((2, 1.0),),
                    semantic_tag="return_semantics",
                    expected_role="high_risk",
                    violation_dist=((5, 1.0),),
                ),
            ]
        ],
        eval_token_ids=(2, 3, 4, 5),
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )
    state = oracle.initial_state(0)

    assert oracle.step(state, 3).status == "clean"

    unsupported_equivalent = oracle.step(state, 4)
    assert unsupported_equivalent.status == "violation"
    assert oracle.next_dist(unsupported_equivalent) == {5: 1.0}


def test_scripted_oracle_style_conditioned_flexible_state_is_prefix_inferable() -> None:
    oracle = ScriptedOracle(
        scripts=[
            [
                ScriptStep(
                    dist=((3, 0.6), (4, 0.4), (6, 0.0), (7, 0.0)),
                    semantic_tag="equivalent_form",
                    expected_role="flexible",
                    off_policy_violates=False,
                    clean_token_ids=(3, 4, 6, 7),
                    eval_token_ids=(3, 4, 6, 7),
                    style_dists=(
                        ((3, 0.6), (4, 0.3), (6, 0.07), (7, 0.03)),
                        ((6, 0.6), (7, 0.3), (3, 0.07), (4, 0.03)),
                    ),
                    style_token_ids=((3, 4), (6, 7)),
                    style_canonical_token_ids=(3, 6),
                ),
                ScriptStep(
                    dist=((8, 1.0),),
                    semantic_tag="equivalent_form",
                    expected_role="flexible",
                    off_policy_violates=False,
                    style_dists=(((8, 1.0),), ((9, 1.0),)),
                ),
            ]
        ],
        eval_token_ids=(2, 3, 4, 5, 6, 7, 8, 9),
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )
    state = oracle.initial_state(0)
    assert state.style == 0
    assert oracle.next_dist(state)[3] == 0.6

    style1_state = oracle.step(state, 7)
    assert style1_state.status == "clean"
    assert style1_state.style == 1
    assert style1_state.prefix == (1, 7)
    assert oracle.next_dist(style1_state) == {9: 1.0}

    unsupported_state = oracle.step(state, 5)
    assert unsupported_state.status == "clean"
    assert unsupported_state.style == 0
    assert unsupported_state.prefix == (1, 3)


def test_supported_flexible_tokens_keep_prefix_sensitive_q_values() -> None:
    oracle = ScriptedOracle(
        scripts=[
            [
                ScriptStep(
                    dist=((3, 0.6), (6, 0.4)),
                    semantic_tag="equivalent_form",
                    expected_role="flexible",
                    off_policy_violates=False,
                    clean_token_ids=(3, 6),
                    eval_token_ids=(3, 6),
                    style_token_ids=((3,), (6,)),
                    style_canonical_token_ids=(3, 6),
                ),
                ScriptStep(
                    dist=((2, 1.0),),
                    semantic_tag="terminal",
                    expected_role="terminal",
                    off_policy_violates=False,
                ),
            ]
        ],
        eval_token_ids=(2, 3, 6),
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )

    class PrefixSensitiveLoss:
        def loss_at_state(self, state: Hashable, _oracle: ScriptedOracle) -> float:
            return float(getattr(state, "prefix")[-1])

    state = oracle.initial_state(0)
    result = exact_kappa_for_state(oracle, PrefixSensitiveLoss(), state, horizon=1)
    q_by_action = dict(zip(result.action_ids, result.q_values))

    assert oracle.step(state, 6).prefix == (1, 6)
    assert q_by_action[3] != q_by_action[6]


def test_scripted_oracle_off_policy_override_enters_violation_branch() -> None:
    oracle = _oracle()
    state = oracle.initial_state(0)
    next_state = oracle.step(state, 5)
    assert next_state.status == "violation"
    assert next_state.prefix == (1, 5)
    assert oracle.next_dist(next_state) == {5: 1.0}


def test_scripted_oracle_reports_semantic_tag_and_role() -> None:
    oracle = _oracle()
    state = oracle.initial_state(0)
    assert oracle.semantic_tag(state) == "operator"
    assert oracle.expected_role(state) == "high_risk"
    terminal = oracle.step(oracle.step(state, 3), 5)
    assert oracle.is_terminal(terminal)
    assert oracle.remaining_horizon(terminal) == 0
