from __future__ import annotations

import numpy as np
import torch

from bridge_garden_v2.exact_oracle import ExactKappaStateResult
from bridge_garden_v2.oracle_eval import summarize_oracle_eb, summarize_oracle_eb_by_positions
from bridge_garden_v2.oracle_eval import summarize_oracle_expected_intervention
from bridge_garden_v2.oracle_eval import summarize_oracle_sampled_rollout_statuses
from bridge_garden_v2.tests.test_oracle_student import PrefixAgnosticModel
from bridge_garden_v2.scripted_oracle import ScriptedOracle, ScriptState, ScriptStep


def test_summarize_oracle_eb_zero_when_student_matches_every_state() -> None:
    oracle = ScriptedOracle(
        scripts=[[
            ScriptStep(dist=((2, 0.7), (3, 0.3)), semantic_tag="x", expected_role="x"),
            ScriptStep(dist=((2, 0.7), (3, 0.3)), semantic_tag="x", expected_role="x"),
        ]],
        eval_token_ids=(2, 3),
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )
    logits = torch.log(torch.tensor([1e-8, 1e-8, 0.7, 0.3]))
    model = PrefixAgnosticModel(logits)
    summary = summarize_oracle_eb(
        oracle=oracle,
        model=model,
        vocab_size=4,
        sample_ids=[0],
        device=torch.device("cpu"),
    )
    assert summary.teacher_forced_kl < 1e-6
    assert summary.rollout_kl < 1e-6
    assert abs(summary.exposure_bias) < 1e-6
    assert summary.teacher_forced_steps == 2


def test_summarize_oracle_eb_runs_with_mismatched_student() -> None:
    oracle = ScriptedOracle(
        scripts=[[
            ScriptStep(dist=((2, 1.0),), semantic_tag="x", expected_role="x"),
            ScriptStep(dist=((3, 1.0),), semantic_tag="x", expected_role="x"),
        ]],
        eval_token_ids=(2, 3),
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )
    model = PrefixAgnosticModel(torch.tensor([0.0, 0.0, -2.0, 2.0]))
    summary = summarize_oracle_eb(
        oracle=oracle,
        model=model,
        vocab_size=4,
        sample_ids=[0],
        max_steps=2,
        device=torch.device("cpu"),
    )
    assert summary.teacher_forced_kl > 0.0
    assert summary.rollout_steps == 2


def test_sampled_rollout_status_metric_counts_off_support() -> None:
    oracle = ScriptedOracle(
        scripts=[[
            ScriptStep(dist=((2, 1.0),), semantic_tag="branch_guard", expected_role="high"),
            ScriptStep(dist=((3, 1.0),), semantic_tag="return_semantics", expected_role="high"),
        ]],
        eval_token_ids=(2, 3, 4),
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )
    model = PrefixAgnosticModel(torch.tensor([-100.0, -100.0, -100.0, -100.0, 100.0]))
    summary = summarize_oracle_sampled_rollout_statuses(
        oracle=oracle,
        model=model,
        sample_ids=[0],
        rollout_repeats=3,
        device=torch.device("cpu"),
    )
    assert summary["off_teacher_support_count"] == 6
    assert summary["first_violation_count"] == 3
    assert summary["first_violation_top"] == [
        {"position": 0, "semantic_tag": "branch_guard", "count": 3}
    ]
    assert summary["transition_counts"]["clean->violation"] == 3


def test_summarize_oracle_eb_by_positions_reports_regions() -> None:
    oracle = ScriptedOracle(
        scripts=[[
            ScriptStep(dist=((2, 1.0),), semantic_tag="bridge", expected_role="high"),
            ScriptStep(dist=((3, 1.0),), semantic_tag="garden", expected_role="low"),
        ]],
        eval_token_ids=(2, 3),
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )
    model = PrefixAgnosticModel(torch.tensor([0.0, 0.0, 2.0, -2.0]))
    summary = summarize_oracle_eb_by_positions(
        oracle=oracle,
        model=model,
        vocab_size=4,
        sample_ids=[0],
        bridge_positions={0},
        garden_positions={1},
        device=torch.device("cpu"),
    )
    assert summary.overall.teacher_forced_steps == 2
    assert summary.bridge_teacher_forced_kl >= 0.0
    assert summary.garden_teacher_forced_kl >= 0.0


def test_expected_intervention_uses_unconditional_student_mass() -> None:
    state = ScriptState(sample_id=0, position=0, status="clean", prefix=(1,))
    result = ExactKappaStateResult(
        state=state,
        action_ids=(2, 3),
        q_values=(1.0, 3.0),
        qbar=1.4,
        action_kappa=(-0.4, 1.6),
        state_kappa=1.2,
        teacher_confidence=0.5,
        teacher_max_prob=0.8,
    )
    model = PrefixAgnosticModel(torch.log(torch.tensor([0.2, 0.1, 0.4, 0.3]).clamp(min=1e-8)))
    summary = summarize_oracle_expected_intervention(
        model=model,
        kappa_results=[result],
        bridge_states=[state],
        garden_states=[],
        device=torch.device("cpu"),
    )
    expected_student_loss = 0.4 * 1.0 + 0.3 * 3.0 + 0.3 * 3.0
    assert np.isclose(summary.overall.student_expected_loss, expected_student_loss)
    assert np.isclose(summary.overall.mean_student_eval_mass, 0.7)
