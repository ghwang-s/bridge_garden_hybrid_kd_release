from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch

from bridge_garden_v2.synthetic_domains import FLEXIBLE_SPREAD_LOGIT_RANGE
from bridge_garden_v2.synthetic_domains import build_synthetic_domain
from scripts.check_synthetic_exact_cost import _enforce_paper_config as enforce_exact_check
from scripts.check_synthetic_oracle_cost import _enforce_paper_config as enforce_oracle_check
from scripts.run_synthetic_mini_pipeline import _clean_total_risk_summary
from scripts.run_synthetic_mini_pipeline import _enforce_paper_config as enforce_runner
from scripts.run_synthetic_mini_pipeline import _output_dir
from scripts.run_synthetic_mini_pipeline import _region_semantic_occupancy
from scripts.run_synthetic_mini_pipeline import _select_one_se_lowest_lambda
from scripts.run_synthetic_mini_pipeline import _student_distribution_summary
from scripts.run_synthetic_mini_pipeline import _training_metrics


def _runner_args(**overrides):
    values = {
        "enforce_paper_config": True,
        "domain": "code",
        "vocab_size": 64,
        "script_len": 47,
        "sample_count": 4_600,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 512,
        "dropout": 0.1,
        "kappa_continuation": "expectation",
        "max_kappa_states": 0,
        "analysis_count": 100,
        "train_count": 4_000,
        "val_count": 500,
        "epochs": 24,
        "batch_size": 256,
        "lr": 2e-4,
        "lr_schedule": "warmup_cosine",
        "lr_warmup_epochs": 6,
        "early_stopping_patience": 6,
        "analysis_path_policy": "sample",
        "train_path_policy": "sample",
        "ref_only": False,
        "fixed_hybrid_lambda": None,
        "hybrid_training_policy": "ambiguity_scaled",
        "hybrid_lambda_selection_policy": "min_validation_teacher_forced_kl",
        "hybrid_lambdas": "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        "hybrid_selection_count": 500,
        "rollout_repeats": 30,
        "paper_config": "paper",
        "hard_kd_lr": None,
        "hard_kd_early_stopping_patience": None,
        "hard_kd_early_stopping_metric": "val_teacher_forced_kl",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _exact_check_args(**overrides):
    values = {
        "enforce_paper_config": True,
        "domain": "code",
        "analysis_count": 100,
        "vocab_size": 64,
        "script_len": 47,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 512,
        "dropout": 0.1,
        "max_states": 0,
        "device": "cuda",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _oracle_check_args(**overrides):
    values = {
        "enforce_paper_config": True,
        "domain": "code",
        "analysis_count": 100,
        "vocab_size": 64,
        "script_len": 47,
        "max_states": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_runner_accepts_paper_configuration() -> None:
    enforce_runner(_runner_args())


def test_runner_accepts_negative_control_paper_configuration() -> None:
    enforce_runner(
        _runner_args(
            domain="negative_control",
            script_len=35,
            sample_count=2_600,
            train_count=2_000,
        )
    )


def test_runner_rejects_nonpaper_architecture_and_budget() -> None:
    with pytest.raises(SystemExit) as exc:
        enforce_runner(
            _runner_args(
                d_model=64,
                n_layers=1,
                epochs=12,
                lr_schedule="constant",
                fixed_hybrid_lambda=0.5,
            )
        )
    message = str(exc.value)
    assert "--d-model must be 128" in message
    assert "--n-layers must be 2" in message
    assert "--epochs must be 24" in message
    assert "--lr-schedule must be warmup_cosine for paper" in message
    assert "--fixed-hybrid-lambda must be unset" in message


def test_runner_rejects_wrong_negative_control_counts() -> None:
    with pytest.raises(SystemExit) as exc:
        enforce_runner(_runner_args(domain="negative_control", script_len=35))
    message = str(exc.value)
    assert "--train-count must be 2000" in message
    assert "--sample-count must be 2600" in message


def test_synthetic_flexible_distribution_is_low_top1_spread() -> None:
    assert FLEXIBLE_SPREAD_LOGIT_RANGE == pytest.approx(2.5)
    bundle = build_synthetic_domain("code", sample_count=1, script_len=47, vocab_size=64)
    state = bundle.oracle.initial_state(0)
    while bundle.oracle.semantic_tag(state) != "equivalent_implementation":
        teacher_action = max(bundle.oracle.next_dist(state), key=bundle.oracle.next_dist(state).get)
        state = bundle.oracle.step(state, int(teacher_action))
    probs = sorted(bundle.oracle.next_dist(state).values(), reverse=True)
    assert len(probs) == 8
    assert probs[0] == pytest.approx(0.319, abs=0.001)
    assert 0.65 < 1.0 - probs[0] < 0.70


def test_exact_check_rejects_nonpaper_shape() -> None:
    with pytest.raises(SystemExit) as exc:
        enforce_exact_check(_exact_check_args(vocab_size=128, script_len=64, dropout=0.2))
    message = str(exc.value)
    assert "--vocab-size must be 64" in message
    assert "--script-len must be 47 for code" in message
    assert "--dropout must be 0.1" in message


def test_oracle_check_rejects_state_subsampling_and_large_vocab() -> None:
    with pytest.raises(SystemExit) as exc:
        enforce_oracle_check(_oracle_check_args(vocab_size=128, max_states=64))
    message = str(exc.value)
    assert "--vocab-size must be 64" in message
    assert "--max-states must be 0" in message


def test_runner_seed_dir_layout_matches_manifest_builder_expectation(tmp_path) -> None:
    assert _output_dir(tmp_path, "code", None) == tmp_path / "code"
    assert _output_dir(tmp_path, "code", "seed_0") == tmp_path / "code" / "seed_0"


def test_training_metrics_flags_hard_underfit_and_edge_lambda() -> None:
    metrics = _training_metrics(
        {
            "hard_kd": {"best_val_teacher_forced_kl": [0.30], "val_teacher_forced_kl": [0.5, 0.3], "best_teacher_forced_kl_epoch": [1.0]},
            "soft_kd": {"best_val_teacher_forced_kl": [0.02]},
            "hybrid_kd": {"best_val_teacher_forced_kl": [0.03]},
        },
        {
            0.1: (0.40, None, None),
            0.2: (0.30, None, None),
            0.3: (0.20, None, None),
        },
        0.3,
        selection_policy="min_validation_teacher_forced_kl",
    )
    assert metrics["hard_to_soft_best_val_kl_ratio"] == pytest.approx(15.0)
    assert "hard_kd_best_val_kl_exceeds_soft_kd_by_more_than_5x" in metrics["warnings"]
    assert "hybrid_validation_kl_is_monotone_and_selected_lambda_is_on_grid_edge" in metrics["warnings"]
    assert "hard_kd_best_validation_epoch_is_at_training_tail" in metrics["warnings"]


def test_training_metrics_does_not_blame_teacher_forced_edge_for_rollout_selection() -> None:
    metrics = _training_metrics(
        {
            "hard_kd": {"best_val_teacher_forced_kl": [0.30], "val_teacher_forced_kl": [0.3], "best_teacher_forced_kl_epoch": [0.0]},
            "soft_kd": {"best_val_teacher_forced_kl": [0.02]},
            "hybrid_kd": {"best_val_teacher_forced_kl": [0.03]},
        },
        {
            0.1: (0.40, None, None),
            0.2: (0.30, None, None),
            0.3: (0.20, None, None),
        },
        0.3,
        selection_policy="min_validation_rollout_eb",
    )
    assert metrics["hybrid_lambda_selection_policy"] == "min_validation_rollout_eb"
    assert "hybrid_validation_kl_is_monotone_and_selected_lambda_is_on_grid_edge" not in metrics["warnings"]


def test_one_se_lambda_selector_prefers_lowest_near_best_rollout_lambda() -> None:
    selected = _select_one_se_lowest_lambda(
        {0.1: 1.2762, 0.2: 1.2783, 0.9: 1.2741},
        {0.1: 0.0156, 0.2: 0.0156, 0.9: 0.0156},
    )
    assert selected == 0.1


def test_one_se_lambda_selector_uses_best_when_lower_lambdas_are_outside_band() -> None:
    selected = _select_one_se_lowest_lambda(
        {0.1: 1.30, 0.2: 1.29, 0.9: 1.20},
        {0.9: 0.02},
    )
    assert selected == 0.9


def test_region_semantic_occupancy_counts_ref_and_conf_splits() -> None:
    occupancy = _region_semantic_occupancy(
        [
            {"semantic_tag": "syntax_layout", "bridge_partition": True, "garden_partition": False, "bridge_confidence": True, "garden_confidence": False},
            {"semantic_tag": "operator", "bridge_partition": True, "garden_partition": False, "bridge_confidence": False, "garden_confidence": False},
            {"semantic_tag": "equivalent", "bridge_partition": False, "garden_partition": True, "bridge_confidence": False, "garden_confidence": True},
        ]
    )
    assert occupancy["bridge_partition"] == {"syntax_layout": 1, "operator": 1}
    assert occupancy["garden_partition"] == {"equivalent": 1}
    assert occupancy["bridge_confidence"] == {"syntax_layout": 1}
    assert occupancy["garden_confidence"] == {"equivalent": 1}


def test_distribution_summary_uses_model_device_for_prefix_tensor() -> None:
    class Oracle:
        def next_dist(self, state):
            return {2: 0.7, 3: 0.3}

    class State:
        prefix = (1, 2)

    class DeviceCheckingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.marker = torch.nn.Parameter(torch.empty((), device="meta"))

        def forward(self, input_ids):
            assert input_ids.device == self.marker.device
            return torch.zeros((input_ids.shape[0], input_ids.shape[1], 4), dtype=torch.float32)

    summary = _student_distribution_summary(Oracle(), DeviceCheckingModel(), [State()])
    assert summary["states"] == 1


def test_clean_total_risk_summary_handles_empty_region_kl() -> None:
    clean_kl = {
        "overall_kl": 0.2,
        "bridge_kl": None,
        "garden_kl": 0.3,
    }
    expected_intervention = SimpleNamespace(
        overall=SimpleNamespace(expected_eb=0.1),
        bridge=SimpleNamespace(expected_eb=0.4),
        garden=SimpleNamespace(expected_eb=0.2),
    )

    summary = _clean_total_risk_summary(clean_kl, expected_intervention)

    assert summary["overall_total"] == pytest.approx(0.3)
    assert summary["bridge_total"] is None
    assert summary["garden_total"] == pytest.approx(0.5)
