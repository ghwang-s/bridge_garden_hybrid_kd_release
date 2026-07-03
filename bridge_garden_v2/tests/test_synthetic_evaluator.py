from __future__ import annotations

from typing import Hashable

from bridge_garden_v2.synthetic_evaluator import (
    collect_clean_path_states,
    estimate_teacher_support_size,
    evaluate_oracle_states,
    positions_from_state_ids,
)
from bridge_garden_v2.scripted_oracle import ScriptedOracle, ScriptStep


class RoleConditionedStudent:
    def loss_at_state(self, state: Hashable, oracle: ScriptedOracle) -> float:
        if getattr(state, "status") == "violation":
            return 4.0
        tag = oracle.semantic_tag(state)
        return 0.1 if tag == "semantic_risk" else 0.05


def _scripted_smoke_oracle() -> ScriptedOracle:
    scripts = []
    for _ in range(5):
        scripts.append(
            [
                ScriptStep(
                    dist=((3, 0.95), (4, 0.05)),
                    semantic_tag="semantic_risk",
                    expected_role="high_risk",
                    violation_dist=((6, 1.0),),
                ),
                ScriptStep(
                    dist=((5, 0.25), (6, 0.25), (7, 0.25), (8, 0.25)),
                    semantic_tag="equivalent_form",
                    expected_role="flexible",
                    violation_dist=((6, 1.0),),
                    off_policy_violates=False,
                ),
                ScriptStep(
                    dist=((3, 0.9), (4, 0.1)),
                    semantic_tag="semantic_risk",
                    expected_role="high_risk",
                    violation_dist=((6, 1.0),),
                ),
                ScriptStep(
                    dist=((5, 0.34), (6, 0.33), (7, 0.33)),
                    semantic_tag="equivalent_form",
                    expected_role="flexible",
                    violation_dist=((6, 1.0),),
                    off_policy_violates=False,
                ),
            ]
        )
    return ScriptedOracle(
        scripts=scripts,
        eval_token_ids=(2, 3, 4, 5, 6, 7, 8),
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )


def test_collect_clean_path_states_gets_each_nonterminal_state() -> None:
    oracle = _scripted_smoke_oracle()
    states = collect_clean_path_states(oracle, sample_count=5)
    assert len(states) == 20
    assert all(getattr(state, "status") == "clean" for state in states)


def test_evaluate_oracle_states_outputs_kappa_and_confidence_splits() -> None:
    oracle = _scripted_smoke_oracle()
    states = collect_clean_path_states(oracle, sample_count=5)
    result = evaluate_oracle_states(oracle, RoleConditionedStudent(), states, exclude_semantic_tags={"terminal"})
    assert len(result.scores) == 20
    assert len(result.kappa_split.bridge) == 4
    assert len(result.kappa_split.garden) == 4
    assert len(result.confidence_split.bridge) == 4
    assert len(result.confidence_split.garden) == 4
    assert result.confidence_rho > 0.5
    assert result.bridge_overlap >= 0.5
    assert result.bridge_tie_aware_overlap >= result.bridge_overlap
    assert result.garden_tie_aware_overlap >= result.garden_overlap
    assert sum(result.top20_semantic_counts.values()) == 4
    assert sum(result.bottom20_semantic_counts.values()) == 4


def test_estimate_teacher_support_size_caps_exploding_support() -> None:
    oracle = _scripted_smoke_oracle()
    state = oracle.initial_state(0)
    assert estimate_teacher_support_size(oracle, state, horizon=2, cap=100) == 8
    assert estimate_teacher_support_size(oracle, state, horizon=4, cap=10) == 11


def test_off_policy_tolerant_flexible_step_repairs_unregistered_global_token() -> None:
    oracle = ScriptedOracle(
        scripts=[
            [
                ScriptStep(
                    dist=((5, 0.5), (6, 0.5)),
                    semantic_tag="equivalent_form",
                    expected_role="flexible",
                    off_policy_violates=False,
                    clean_token_ids=(5, 6),
                    eval_token_ids=(5, 6),
                ),
                ScriptStep(
                    dist=((5, 1.0),),
                    semantic_tag="terminalish",
                    expected_role="excluded",
                    violation_dist=((8, 1.0),),
                    off_policy_violates=False,
                ),
            ]
        ],
        eval_token_ids=(2, 3, 4, 5, 6, 7, 8),
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )
    state = oracle.initial_state(0)
    assert getattr(oracle.step(state, 5), "status") == "clean"
    repaired = oracle.step(state, 7)
    assert getattr(repaired, "status") == "clean"
    assert getattr(repaired, "prefix") == (1, 5)


def test_positions_from_state_ids_extracts_positions() -> None:
    oracle = _scripted_smoke_oracle()
    states = collect_clean_path_states(oracle, sample_count=1)
    assert positions_from_state_ids(tuple(states[:3])) == {0, 1, 2}
