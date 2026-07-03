from __future__ import annotations

import json

from bridge_garden_v2.synthetic_manifest import verify_synthetic_manifest
from scripts.build_synthetic_manifest import _load_domain_summary, _load_negative_control_summary, _negative_control_result


def _valid_manifest() -> dict:
    return {
        "setup": {
            "domains": ["code", "math", "dialogue"],
            "negative_control": "ngram_control",
            "domain_configs": {
                name: {
                    "vocab_size": 64,
                    "max_len": {"code": 47, "math": 43, "dialogue": 35}[name],
                    "analysis_size": 100,
                    "exact_continuation": True,
                    "full_vocab_kappa": True,
                    "uses_global_eval_token_ids": True,
                    "uses_state_local_eval_token_ids": False,
                    "oracle_teacher": True,
                    "kappa_continuation": "expectation",
                    "max_kappa_states": 0,
                    "eval_token_count": 62,
                    "eos_in_eval": True,
                    "pad_in_eval": False,
                    "bos_in_eval": False,
                    "kappa_state_count": 100 * {"code": 47, "math": 43, "dialogue": 35}[name],
                    "estimated_q_values": 100 * {"code": 47, "math": 43, "dialogue": 35}[name] * 62,
                    "fixed_hybrid_lambda": None,
                    "selected_hybrid_lambda": 0.5,
                    "hybrid_lambda_grid": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    "hybrid_lambda_selection_policy": "min_validation_teacher_forced_kl",
                    "hybrid_training_policy": "ambiguity_scaled",
                    "hybrid_selection_count": 1000,
                    "future_support": {"max": 4096, "p95": 4096, "p50": 2048},
                    "future_support_budget": 100_000,
                }
                for name in ["code", "math", "dialogue"]
            },
            "kappa_split": {
                "bridge_top_quantile": 0.2,
                "garden_bottom_quantile": 0.2,
            },
            "seeds": [0],
            "methods": ["ce_ref", "hard_kd", "soft_kd", "hybrid_kd"],
        },
        "results": {
            "domains": {
                name: {
                    "kappa_feasible": True,
                    "regional_crossover_metric": "ref_split_kappa_contribution.absolute_contribution",
                    "hard_soft_crossover_ref": True,
                    "position_rollout_bridge_crossover_passes": False,
                    "position_rollout_garden_crossover_passes": True,
                    "hybrid_beats_best_pure_overall": True,
                    "interpretability_verification_passes": True,
                    "kappa_spread_non_degenerate": True,
                    "kappa_spread": {
                        "top_bottom_ratio": 2.2,
                        "coefficient_of_variation": 0.6,
                    },
                    "training_converged": True,
                    "statistics": {
                        "seed_count": 3,
                        "bootstrap_protocol": "paired_seed_bootstrap",
                        "bridge_hard_over_soft_delta": {
                            "mean": 0.10,
                            "ci95_low": 0.04,
                            "ci95_high": 0.15,
                            "cohen_d": 2.0,
                        },
                        "garden_soft_over_hard_delta": {
                            "mean": 0.12,
                            "ci95_low": 0.05,
                            "ci95_high": 0.18,
                            "cohen_d": 2.2,
                        },
                        "hybrid_gain_over_best_pure": {
                            "mean": 0.03,
                            "ci95_low": 0.01,
                            "ci95_high": 0.05,
                            "cohen_d": 1.5,
                            "positive_seed_count": 3,
                        },
                    },
                    "interpretability": {
                        "heatmaps": [
                            f"figures/{name}_sample_{idx:03d}_kappa_heatmap.png"
                            for idx in range(5)
                        ],
                        "kappa_state_table": f"tables/{name}_kappa_state_table.json",
                        "top20_expected_high_risk_count": 17,
                        "bottom20_expected_flexible_count": 16,
                        "domain_interpretation": (
                            f"{name} high-kappa states align with semantic-risk choices, "
                            "while low-kappa states align with flexible equivalent realizations."
                        ),
                    },
                    "confidence_proxy": {
                        "spearman_rho": 0.82,
                        "bridge_overlap": 0.75,
                        "garden_overlap": 0.74,
                        "hard_soft_crossover_conf": True,
                        "regional_crossover_metric": "confidence_split_kappa_contribution.absolute_contribution",
                        "position_rollout_crossover_conf": False,
                    },
                }
                for name in ["code", "math", "dialogue"]
            },
            "negative_control": {"fails_same_pattern": True},
        },
        "provenance": {
            "git_commit": "abcdef123",
            "single_manifest_for_all_figures_tables": True,
        },
        "outputs": {
            "figures": ["figures/main.png"],
            "tables": ["tables/main.tsv"],
        },
    }


def test_synthetic_manifest_accepts_complete_contract() -> None:
    gates = verify_synthetic_manifest(_valid_manifest())
    failures = [gate for gate in gates if not gate.passed]
    assert failures == []


def test_synthetic_manifest_rejects_missing_hard_soft_crossover() -> None:
    manifest = _valid_manifest()
    manifest["results"]["domains"]["math"]["hard_soft_crossover_ref"] = False
    gates = verify_synthetic_manifest(manifest)
    failed = {gate.name for gate in gates if not gate.passed}
    assert "math_hard_soft_crossover_ref" in failed


def test_synthetic_manifest_requires_theorem_facing_crossover_metric() -> None:
    manifest = _valid_manifest()
    manifest["results"]["domains"]["code"]["regional_crossover_metric"] = "ref_split_position_rollout.eb"
    manifest["results"]["domains"]["code"]["confidence_proxy"]["regional_crossover_metric"] = "confidence_split_position_rollout.eb"
    del manifest["results"]["domains"]["code"]["position_rollout_bridge_crossover_passes"]

    gates = verify_synthetic_manifest(manifest)

    failed = {gate.name for gate in gates if not gate.passed}
    assert "code_regional_crossover_metric" in failed
    assert "code_confidence_crossover_metric" in failed
    assert "code_position_rollout_bridge_metric_present" in failed


def test_synthetic_manifest_rejects_weak_confidence_proxy() -> None:
    manifest = _valid_manifest()
    manifest["results"]["domains"]["dialogue"]["confidence_proxy"]["spearman_rho"] = None
    gates = verify_synthetic_manifest(manifest)
    failed = {gate.name for gate in gates if not gate.passed}
    assert "dialogue_confidence_rho_recorded" in failed


def test_synthetic_manifest_rejects_non_paper_kappa_setup() -> None:
    manifest = _valid_manifest()
    domain = manifest["setup"]["domain_configs"]["code"]
    domain["kappa_continuation"] = "mode"
    domain["max_kappa_states"] = 64
    domain["eval_token_count"] = 32
    domain["eos_in_eval"] = False
    domain["estimated_q_values"] = domain["kappa_state_count"] * 32
    gates = verify_synthetic_manifest(manifest)
    failed = {gate.name for gate in gates if not gate.passed}
    assert "code_kappa_expectation_continuation" in failed
    assert "code_all_kappa_states" in failed
    assert "code_global_eval_token_count" in failed
    assert "code_eos_in_eval" in failed
    assert "code_estimated_q_values" in failed


def test_synthetic_manifest_rejects_wrong_hybrid_contract() -> None:
    manifest = _valid_manifest()
    manifest["setup"]["domain_configs"]["dialogue"]["fixed_hybrid_lambda"] = 0.25
    manifest["setup"]["domain_configs"]["dialogue"]["selected_hybrid_lambda"] = 0.95
    manifest["setup"]["domain_configs"]["dialogue"]["hybrid_lambda_grid"] = [0.25, 0.5, 0.75]
    manifest["setup"]["domain_configs"]["dialogue"]["hybrid_lambda_selection_policy"] = "analysis_best"
    manifest["setup"]["domain_configs"]["dialogue"]["hybrid_training_policy"] = "role_aware_expected_role_gate"
    manifest["setup"]["domain_configs"]["dialogue"]["hybrid_selection_count"] = 50
    gates = verify_synthetic_manifest(manifest)
    failed = {gate.name for gate in gates if not gate.passed}
    assert "dialogue_no_fixed_hybrid_lambda" in failed
    assert "dialogue_hybrid_lambda_grid" in failed
    assert "dialogue_selected_hybrid_lambda" in failed
    assert "dialogue_hybrid_selection_policy" in failed
    assert "dialogue_hybrid_training_policy" in failed
    assert "dialogue_full_validation_hybrid_selection" in failed


def test_synthetic_manifest_rejects_state_local_eval_token_path() -> None:
    manifest = _valid_manifest()
    domain = manifest["setup"]["domain_configs"]["math"]
    domain["uses_global_eval_token_ids"] = False
    domain["uses_state_local_eval_token_ids"] = True
    gates = verify_synthetic_manifest(manifest)
    failed = {gate.name for gate in gates if not gate.passed}
    assert "math_uses_global_eval_token_ids" in failed
    assert "math_does_not_use_state_local_eval_token_ids" in failed


def test_synthetic_manifest_requires_interpretability_evidence() -> None:
    manifest = _valid_manifest()
    manifest["results"]["domains"]["code"]["interpretability"]["heatmaps"] = []
    manifest["results"]["domains"]["code"]["interpretability"]["top20_expected_high_risk_count"] = 0
    gates = verify_synthetic_manifest(manifest)
    failed = {gate.name for gate in gates if not gate.passed}
    assert "code_heatmaps_present" in failed
    assert "code_high_kappa_interpretability" in failed


def test_synthetic_manifest_treats_statistics_as_metric() -> None:
    manifest = _valid_manifest()
    stats = manifest["results"]["domains"]["code"]["statistics"]
    stats["seed_count"] = 2
    stats["bridge_hard_over_soft_delta"]["mean"] = -0.01
    stats["hybrid_gain_over_best_pure"]["cohen_d"] = None
    stats["hybrid_gain_over_best_pure"]["positive_seed_count"] = 1

    gates = verify_synthetic_manifest(manifest)

    failed = {gate.name for gate in gates if not gate.passed}
    assert not any(name.startswith("code_statistical") for name in failed)
    assert "code_bridge_delta_mean_positive" not in failed
    assert "code_effect_sizes_present" not in failed
    assert "code_hybrid_gain_directional_seeds" not in failed


def test_synthetic_manifest_accepts_any_nonempty_declared_seed_set() -> None:
    manifest = _valid_manifest()
    manifest["setup"]["seeds"] = [10]

    gates = verify_synthetic_manifest(manifest)

    failed = {gate.name for gate in gates if not gate.passed}
    assert "accepted_seed_declared" not in failed


def test_synthetic_manifest_rejects_empty_seed_set() -> None:
    manifest = _valid_manifest()
    manifest["setup"]["seeds"] = []

    gates = verify_synthetic_manifest(manifest)

    failed = {gate.name for gate in gates if not gate.passed}
    assert "accepted_seed_declared" in failed


def test_manifest_builder_aggregates_seed_summaries_conservatively(tmp_path) -> None:
    run_root = tmp_path / "run"
    domain_root = run_root / "code"
    for seed, hard_bridge, high_count in [
        (0, 0.10, 17),
        (1, 0.12, 16),
        (2, 0.11, 16),
    ]:
        seed_root = domain_root / f"seed_{seed}"
        seed_root.mkdir(parents=True)
        summary = {
            "domain": "code",
            "train_sample_seed": seed,
            "analysis_sample_seed": seed,
            "confidence_rho": 0.8 + 0.02 * seed,
            "bridge_overlap": 0.75,
            "garden_overlap": 0.76,
            "top20_semantic_counts": {"operator": high_count},
            "bottom20_semantic_counts": {"equivalent_implementation": 16 + seed},
            "methods": {
                "hard_kd": {
                    "train_loss": [0.4, 0.2],
                    "val_teacher_forced_kl": [0.3, 0.2],
                    "best_val_teacher_forced_kl": [0.2],
                    "ref_split_kappa_contribution": {
                        "bridge_absolute_contribution": hard_bridge,
                        "garden_absolute_contribution": 0.30 + seed * 0.01,
                    },
                    "ref_split_position_rollout": {"bridge_eb": hard_bridge, "garden_eb": 0.30 + seed * 0.01},
                    "overall_position_rollout": {"overall_eb": 0.22 + seed * 0.01},
                },
                "soft_kd": {
                    "train_loss": [0.35, 0.18],
                    "val_teacher_forced_kl": [0.25, 0.16],
                    "best_val_teacher_forced_kl": [0.16],
                    "ref_split_kappa_contribution": {
                        "bridge_absolute_contribution": 0.20,
                        "garden_absolute_contribution": 0.12,
                    },
                    "ref_split_position_rollout": {"bridge_eb": 0.20, "garden_eb": 0.12},
                    "overall_position_rollout": {"overall_eb": 0.20 + seed * 0.01},
                },
                "hybrid_kd": {
                    "train_loss": [0.32, 0.17],
                    "val_teacher_forced_kl": [0.24, 0.15],
                    "best_val_teacher_forced_kl": [0.15],
                    "overall_position_rollout": {"overall_eb": 0.09},
                },
            },
        }
        (seed_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    aggregated = _load_domain_summary(run_root, "code")

    assert aggregated["seeds"] == [0, 1, 2]
    assert aggregated["confidence_rho"] == 0.82
    assert aggregated["top20_expected_high_risk_count_min"] == 16
    assert aggregated["bottom20_expected_flexible_count_min"] == 16
    assert abs(aggregated["methods"]["hard_kd"]["ref_split_position_rollout"]["bridge_eb"] - 0.11) < 1e-12
    assert aggregated["statistical_report"]["seed_count"] == 3
    assert aggregated["statistical_report"]["bootstrap_protocol"] == "paired_seed_bootstrap"
    assert aggregated["statistical_report"]["bridge_hard_over_soft_delta"]["mean"] > 0
    assert aggregated["statistical_report"]["hybrid_gain_over_best_pure"]["cohen_d"] > 0


def test_negative_control_requires_all_pattern_failures() -> None:
    summary = {
        "confidence_rho": 0.2,
        "top20_semantic_counts": {"surface_ngram": 20},
        "bottom20_semantic_counts": {"surface_ngram": 20},
        "methods": {
            "hard_kd": {
                "ref_split_kappa_contribution": {
                    "bridge_absolute_contribution": 0.10,
                    "garden_absolute_contribution": 0.20,
                },
                "confidence_split_kappa_contribution": {
                    "bridge_absolute_contribution": 0.10,
                    "garden_absolute_contribution": 0.20,
                },
                "ref_split_position_rollout": {"bridge_eb": 0.10, "garden_eb": 0.20},
                "confidence_split_position_rollout": {"bridge_eb": 0.10, "garden_eb": 0.20},
                "overall_position_rollout": {"overall_eb": 0.20},
            },
            "soft_kd": {
                "ref_split_kappa_contribution": {
                    "bridge_absolute_contribution": 0.30,
                    "garden_absolute_contribution": 0.10,
                },
                "confidence_split_kappa_contribution": {
                    "bridge_absolute_contribution": 0.30,
                    "garden_absolute_contribution": 0.10,
                },
                "ref_split_position_rollout": {"bridge_eb": 0.30, "garden_eb": 0.10},
                "confidence_split_position_rollout": {"bridge_eb": 0.30, "garden_eb": 0.10},
                "overall_position_rollout": {"overall_eb": 0.19},
            },
            "hybrid_kd": {"overall_position_rollout": {"overall_eb": 0.18}},
        },
    }

    result = _negative_control_result(summary)

    assert result["no_semantic_separation"] is True
    assert result["no_ref_crossover"] is False
    assert result["no_confidence_crossover"] is False
    assert result["no_hybrid_advantage"] is False
    assert result["fails_same_pattern"] is False


def test_negative_control_passes_when_pattern_is_absent() -> None:
    summary = {
        "confidence_rho": 0.2,
        "top20_semantic_counts": {"surface_ngram": 20},
        "bottom20_semantic_counts": {"surface_ngram": 20},
        "methods": {
            "hard_kd": {
                "ref_split_kappa_contribution": {
                    "bridge_absolute_contribution": 0.20,
                    "garden_absolute_contribution": 0.20,
                },
                "confidence_split_kappa_contribution": {
                    "bridge_absolute_contribution": 0.20,
                    "garden_absolute_contribution": 0.20,
                },
                "ref_split_position_rollout": {"bridge_eb": 0.20, "garden_eb": 0.20},
                "confidence_split_position_rollout": {"bridge_eb": 0.20, "garden_eb": 0.20},
                "overall_position_rollout": {"overall_eb": 0.20},
            },
            "soft_kd": {
                "ref_split_kappa_contribution": {
                    "bridge_absolute_contribution": 0.20,
                    "garden_absolute_contribution": 0.20,
                },
                "confidence_split_kappa_contribution": {
                    "bridge_absolute_contribution": 0.20,
                    "garden_absolute_contribution": 0.20,
                },
                "ref_split_position_rollout": {"bridge_eb": 0.20, "garden_eb": 0.20},
                "confidence_split_position_rollout": {"bridge_eb": 0.20, "garden_eb": 0.20},
                "overall_position_rollout": {"overall_eb": 0.19},
            },
            "hybrid_kd": {"overall_position_rollout": {"overall_eb": 0.21}},
        },
    }

    result = _negative_control_result(summary)

    assert result["fails_same_pattern"] is True


def test_negative_control_accepts_tiny_or_uncertain_effects() -> None:
    summary = {
        "confidence_rho": 0.2,
        "top20_semantic_counts": {"surface_ngram": 20},
        "bottom20_semantic_counts": {"surface_ngram": 20},
        "statistical_report": {
            "bridge_hard_over_soft_delta": {"mean": 0.05, "ci95_low": -0.02, "ci95_high": 0.12},
            "garden_soft_over_hard_delta": {"mean": 0.04, "ci95_low": -0.03, "ci95_high": 0.11},
            "hybrid_gain_over_best_pure": {"mean": 0.03, "ci95_low": -0.01, "ci95_high": 0.08},
        },
        "methods": {
            "hard_kd": {
                "ref_split_kappa_contribution": {
                    "bridge_absolute_contribution": 0.10,
                    "garden_absolute_contribution": 0.30,
                },
                "confidence_split_kappa_contribution": {
                    "bridge_absolute_contribution": 0.20,
                    "garden_absolute_contribution": 0.20,
                },
                "ref_split_position_rollout": {"bridge_eb": 0.10, "garden_eb": 0.30},
                "confidence_split_position_rollout": {"bridge_eb": 0.20, "garden_eb": 0.20},
                "overall_position_rollout": {"overall_eb": 0.20},
            },
            "soft_kd": {
                "ref_split_kappa_contribution": {
                    "bridge_absolute_contribution": 0.15,
                    "garden_absolute_contribution": 0.26,
                },
                "confidence_split_kappa_contribution": {
                    "bridge_absolute_contribution": 0.20,
                    "garden_absolute_contribution": 0.20,
                },
                "ref_split_position_rollout": {"bridge_eb": 0.15, "garden_eb": 0.26},
                "confidence_split_position_rollout": {"bridge_eb": 0.20, "garden_eb": 0.20},
                "overall_position_rollout": {"overall_eb": 0.19},
            },
            "hybrid_kd": {"overall_position_rollout": {"overall_eb": 0.18}},
        },
    }

    result = _negative_control_result(summary)

    assert result["no_ref_crossover"] is True
    assert result["no_hybrid_advantage"] is True
    assert result["fails_same_pattern"] is True


def test_manifest_builder_loads_seeded_negative_control_summary(tmp_path) -> None:
    run_root = tmp_path / "run"
    for seed, rho in [(0, 0.2), (1, 0.4)]:
        seed_root = run_root / "negative_control" / f"seed_{seed}"
        seed_root.mkdir(parents=True)
        summary = {
            "domain": "negative_control",
            "train_sample_seed": seed,
            "analysis_sample_seed": seed,
            "model_seed": seed,
            "confidence_rho": rho,
            "bridge_overlap": 0.1,
            "garden_overlap": 0.1,
            "top20_semantic_counts": {"surface_ngram": 20},
            "bottom20_semantic_counts": {"surface_ngram": 20},
            "methods": {},
        }
        (seed_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    summary = _load_negative_control_summary(run_root)

    assert summary is not None
    assert summary["domain"] == "negative_control"
    assert summary["seeds"] == [0, 1]
    assert summary["confidence_rho"] == 0.30000000000000004
