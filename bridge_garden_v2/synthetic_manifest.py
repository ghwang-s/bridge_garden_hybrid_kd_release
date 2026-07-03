from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import json
import math


MAIN_DOMAINS = ("code", "math", "dialogue")
EXPECTED_SCRIPT_LENS = {
    "code": 47,
    "math": 43,
    "dialogue": 35,
}
EXPECTED_HYBRID_LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


def _get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _seed_set(value: Any) -> set[int] | None:
    if not isinstance(value, list):
        return None
    try:
        return {int(seed) for seed in value}
    except (TypeError, ValueError):
        return None


def _lambda_grid_matches(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != len(EXPECTED_HYBRID_LAMBDAS):
        return False
    try:
        observed = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return all(abs(left - right) <= 1e-12 for left, right in zip(observed, EXPECTED_HYBRID_LAMBDAS))


def _gate(name: str, passed: bool, detail: str) -> GateResult:
    return GateResult(name=name, passed=bool(passed), detail=detail)


def verify_synthetic_manifest(manifest: Mapping[str, Any]) -> list[GateResult]:
    """Verify the manifest-level contract for synthetic evidence.

    This verifier intentionally checks only facts that should be machine-readable.
    It does not replace code review or semantic verification; those must be recorded in
    the manifest and backed by their own artifacts.
    """
    gates: list[GateResult] = []

    domains = _get(manifest, "setup.domains")
    gates.append(_gate(
        "main_domains_present",
        isinstance(domains, list) and set(MAIN_DOMAINS).issubset(set(domains)),
        f"setup.domains={domains!r}",
    ))

    negative_control = _get(manifest, "setup.negative_control")
    gates.append(_gate(
        "negative_control_declared",
        isinstance(negative_control, str) and bool(negative_control.strip()),
        f"setup.negative_control={negative_control!r}",
    ))

    for domain in MAIN_DOMAINS:
        prefix = f"setup.domain_configs.{domain}"
        vocab_size = _get(manifest, f"{prefix}.vocab_size")
        max_len = _get(manifest, f"{prefix}.max_len")
        analysis_size = _get(manifest, f"{prefix}.analysis_size")
        exact_continuation = _get(manifest, f"{prefix}.exact_continuation")
        full_vocab_kappa = _get(manifest, f"{prefix}.full_vocab_kappa")
        uses_global_eval_token_ids = _get(manifest, f"{prefix}.uses_global_eval_token_ids")
        uses_state_local_eval_token_ids = _get(manifest, f"{prefix}.uses_state_local_eval_token_ids")
        oracle_teacher = _get(manifest, f"{prefix}.oracle_teacher")
        kappa_continuation = _get(manifest, f"{prefix}.kappa_continuation")
        max_kappa_states = _get(manifest, f"{prefix}.max_kappa_states")
        eval_token_count = _get(manifest, f"{prefix}.eval_token_count")
        eos_in_eval = _get(manifest, f"{prefix}.eos_in_eval")
        pad_in_eval = _get(manifest, f"{prefix}.pad_in_eval")
        bos_in_eval = _get(manifest, f"{prefix}.bos_in_eval")
        kappa_state_count = _get(manifest, f"{prefix}.kappa_state_count")
        estimated_q_values = _get(manifest, f"{prefix}.estimated_q_values")
        fixed_hybrid_lambda = _get(manifest, f"{prefix}.fixed_hybrid_lambda")
        selected_hybrid_lambda = _get(manifest, f"{prefix}.selected_hybrid_lambda")
        hybrid_lambda_grid = _get(manifest, f"{prefix}.hybrid_lambda_grid")
        hybrid_lambda_selection_policy = _get(manifest, f"{prefix}.hybrid_lambda_selection_policy")
        hybrid_training_policy = _get(manifest, f"{prefix}.hybrid_training_policy")
        hybrid_selection_count = _get(manifest, f"{prefix}.hybrid_selection_count")
        future_support = _get(manifest, f"{prefix}.future_support", {})
        future_support_max = _get(future_support, "max")
        future_support_budget = _get(manifest, f"{prefix}.future_support_budget", 100_000)
        expected_eval_token_count = (vocab_size - 2) if isinstance(vocab_size, int) else None
        expected_q_values = (
            kappa_state_count * expected_eval_token_count
            if isinstance(kappa_state_count, int) and isinstance(expected_eval_token_count, int)
            else None
        )
        expected_kappa_state_count = (
            analysis_size * max_len
            if isinstance(analysis_size, int) and isinstance(max_len, int)
            else None
        )
        gates.extend([
            _gate(f"{domain}_vocab_size", vocab_size == 64, f"vocab_size={vocab_size!r}"),
            _gate(f"{domain}_max_len", max_len == EXPECTED_SCRIPT_LENS[domain], f"max_len={max_len!r}"),
            _gate(f"{domain}_analysis_size", analysis_size == 100, f"analysis_size={analysis_size!r}"),
            _gate(f"{domain}_exact_continuation", exact_continuation is True, f"exact_continuation={exact_continuation!r}"),
            _gate(f"{domain}_full_vocab_kappa", full_vocab_kappa is True, f"full_vocab_kappa={full_vocab_kappa!r}"),
            _gate(f"{domain}_uses_global_eval_token_ids", uses_global_eval_token_ids is True, f"uses_global_eval_token_ids={uses_global_eval_token_ids!r}"),
            _gate(
                f"{domain}_does_not_use_state_local_eval_token_ids",
                uses_state_local_eval_token_ids is False,
                f"uses_state_local_eval_token_ids={uses_state_local_eval_token_ids!r}",
            ),
            _gate(f"{domain}_oracle_teacher", oracle_teacher is True, f"oracle_teacher={oracle_teacher!r}"),
            _gate(f"{domain}_kappa_expectation_continuation", kappa_continuation == "expectation", f"kappa_continuation={kappa_continuation!r}"),
            _gate(f"{domain}_all_kappa_states", max_kappa_states == 0, f"max_kappa_states={max_kappa_states!r}"),
            _gate(
                f"{domain}_global_eval_token_count",
                eval_token_count == expected_eval_token_count,
                f"eval_token_count={eval_token_count!r}, expected={expected_eval_token_count!r}",
            ),
            _gate(f"{domain}_eos_in_eval", eos_in_eval is True, f"eos_in_eval={eos_in_eval!r}"),
            _gate(f"{domain}_pad_excluded_from_eval", pad_in_eval is False, f"pad_in_eval={pad_in_eval!r}"),
            _gate(f"{domain}_bos_excluded_from_eval", bos_in_eval is False, f"bos_in_eval={bos_in_eval!r}"),
            _gate(
                f"{domain}_kappa_state_count",
                isinstance(kappa_state_count, int) and kappa_state_count == expected_kappa_state_count,
                f"kappa_state_count={kappa_state_count!r}, expected={expected_kappa_state_count!r}",
            ),
            _gate(
                f"{domain}_estimated_q_values",
                estimated_q_values == expected_q_values and isinstance(estimated_q_values, int) and estimated_q_values > 0,
                f"estimated_q_values={estimated_q_values!r}, expected={expected_q_values!r}",
            ),
            _gate(
                f"{domain}_no_fixed_hybrid_lambda",
                fixed_hybrid_lambda is None,
                f"fixed_hybrid_lambda={fixed_hybrid_lambda!r}",
            ),
            _gate(
                f"{domain}_hybrid_lambda_grid",
                _lambda_grid_matches(hybrid_lambda_grid),
                f"hybrid_lambda_grid={hybrid_lambda_grid!r}",
            ),
            _gate(
                f"{domain}_selected_hybrid_lambda",
                _is_number(selected_hybrid_lambda)
                and isinstance(hybrid_lambda_grid, list)
                and any(abs(float(selected_hybrid_lambda) - float(lam)) <= 1e-12 for lam in hybrid_lambda_grid),
                f"selected_hybrid_lambda={selected_hybrid_lambda!r}, grid={hybrid_lambda_grid!r}",
            ),
            _gate(
                f"{domain}_hybrid_selection_policy",
                hybrid_lambda_selection_policy == "min_validation_teacher_forced_kl",
                f"hybrid_lambda_selection_policy={hybrid_lambda_selection_policy!r}",
            ),
            _gate(
                f"{domain}_hybrid_training_policy",
                hybrid_training_policy == "ambiguity_scaled",
                f"hybrid_training_policy={hybrid_training_policy!r}",
            ),
            _gate(
                f"{domain}_full_validation_hybrid_selection",
                hybrid_selection_count == 1000,
                f"hybrid_selection_count={hybrid_selection_count!r}",
            ),
            _gate(
                f"{domain}_future_support_feasible",
                _is_number(future_support_max)
                and _is_number(future_support_budget)
                and float(future_support_max) <= float(future_support_budget),
                f"future_support.max={future_support_max!r}, budget={future_support_budget!r}",
            ),
        ])

    split = _get(manifest, "setup.kappa_split", {})
    gates.append(_gate(
        "kappa_split_top_bottom_20",
        _get(split, "bridge_top_quantile") == 0.2 and _get(split, "garden_bottom_quantile") == 0.2,
        f"kappa_split={split!r}",
    ))

    seeds = _get(manifest, "setup.seeds")
    seed_set = _seed_set(seeds)
    gates.append(_gate(
        "accepted_seed_declared",
        seed_set is not None and len(seed_set) >= 1,
        f"seeds={seeds!r}",
    ))

    methods = _get(manifest, "setup.methods")
    required_methods = {"ce_ref", "hard_kd", "soft_kd", "hybrid_kd"}
    gates.append(_gate(
        "required_methods",
        isinstance(methods, list) and required_methods.issubset(set(methods)),
        f"methods={methods!r}",
    ))

    for domain in MAIN_DOMAINS:
        prefix = f"results.domains.{domain}"
        gates.extend([
            _gate(
                f"{domain}_kappa_feasible",
                _get(manifest, f"{prefix}.kappa_feasible") is True,
                f"kappa_feasible={_get(manifest, f'{prefix}.kappa_feasible')!r}",
            ),
            _gate(
                f"{domain}_regional_crossover_metric",
                _get(manifest, f"{prefix}.regional_crossover_metric") == "ref_split_kappa_contribution.absolute_contribution",
                f"regional_crossover_metric={_get(manifest, f'{prefix}.regional_crossover_metric')!r}",
            ),
            _gate(
                f"{domain}_hard_soft_crossover_ref",
                _get(manifest, f"{prefix}.hard_soft_crossover_ref") is True,
                f"hard_soft_crossover_ref={_get(manifest, f'{prefix}.hard_soft_crossover_ref')!r}",
            ),
            _gate(
                f"{domain}_position_rollout_bridge_metric_present",
                isinstance(_get(manifest, f"{prefix}.position_rollout_bridge_crossover_passes"), bool),
                f"position_rollout_bridge_crossover_passes={_get(manifest, f'{prefix}.position_rollout_bridge_crossover_passes')!r}",
            ),
            _gate(
                f"{domain}_position_rollout_garden_metric_present",
                isinstance(_get(manifest, f"{prefix}.position_rollout_garden_crossover_passes"), bool),
                f"position_rollout_garden_crossover_passes={_get(manifest, f'{prefix}.position_rollout_garden_crossover_passes')!r}",
            ),
            _gate(
                f"{domain}_hybrid_overall",
                _get(manifest, f"{prefix}.hybrid_beats_best_pure_overall") is True,
                f"hybrid_beats_best_pure_overall={_get(manifest, f'{prefix}.hybrid_beats_best_pure_overall')!r}",
            ),
            _gate(
                f"{domain}_interpretability_verification",
                _get(manifest, f"{prefix}.interpretability_verification_passes") is True,
                f"interpretability_verification_passes={_get(manifest, f'{prefix}.interpretability_verification_passes')!r}",
            ),
            _gate(
                f"{domain}_kappa_spread",
                _get(manifest, f"{prefix}.kappa_spread_non_degenerate") is True,
                f"kappa_spread={_get(manifest, f'{prefix}.kappa_spread')!r}",
            ),
            _gate(
                f"{domain}_training_converged",
                _get(manifest, f"{prefix}.training_converged") is True,
                f"training_converged={_get(manifest, f'{prefix}.training_converged')!r}",
            ),
        ])

        interpretability = _get(manifest, f"{prefix}.interpretability", {})
        heatmaps = _get(interpretability, "heatmaps")
        kappa_state_table = _get(interpretability, "kappa_state_table")
        top_expected = _get(interpretability, "top20_expected_high_risk_count")
        bottom_expected = _get(interpretability, "bottom20_expected_flexible_count")
        explanation = _get(interpretability, "domain_interpretation")
        gates.extend([
            _gate(
                f"{domain}_heatmaps_present",
                isinstance(heatmaps, list) and len(heatmaps) >= 1,
                f"heatmaps={heatmaps!r}",
            ),
            _gate(
                f"{domain}_kappa_state_table",
                isinstance(kappa_state_table, str) and bool(kappa_state_table.strip()),
                f"kappa_state_table={kappa_state_table!r}",
            ),
            _gate(
                f"{domain}_high_kappa_interpretability",
                isinstance(top_expected, int) and top_expected > 0,
                f"top20_expected_high_risk_count={top_expected!r}",
            ),
            _gate(
                f"{domain}_low_kappa_interpretability",
                isinstance(bottom_expected, int) and bottom_expected > 0,
                f"bottom20_expected_flexible_count={bottom_expected!r}",
            ),
            _gate(
                f"{domain}_interpretation_written",
                isinstance(explanation, str) and len(explanation.strip()) >= 40,
                f"domain_interpretation={explanation!r}",
            ),
        ])

        rho = _get(manifest, f"{prefix}.confidence_proxy.spearman_rho")
        bridge_overlap = _get(manifest, f"{prefix}.confidence_proxy.bridge_overlap")
        garden_overlap = _get(manifest, f"{prefix}.confidence_proxy.garden_overlap")
        conf_crossover = _get(manifest, f"{prefix}.confidence_proxy.hard_soft_crossover_conf")
        conf_metric = _get(manifest, f"{prefix}.confidence_proxy.regional_crossover_metric")
        conf_position_diag = _get(manifest, f"{prefix}.confidence_proxy.position_rollout_crossover_conf")
        gates.extend([
            _gate(f"{domain}_confidence_rho_recorded", _is_number(rho), f"rho={rho!r}"),
            _gate(f"{domain}_confidence_bridge_overlap_recorded", _is_number(bridge_overlap), f"bridge_overlap={bridge_overlap!r}"),
            _gate(f"{domain}_confidence_garden_overlap_recorded", _is_number(garden_overlap), f"garden_overlap={garden_overlap!r}"),
            _gate(
                f"{domain}_confidence_crossover_metric",
                conf_metric == "confidence_split_kappa_contribution.absolute_contribution",
                f"regional_crossover_metric={conf_metric!r}",
            ),
            _gate(f"{domain}_confidence_crossover_recorded", isinstance(conf_crossover, bool), f"hard_soft_crossover_conf={conf_crossover!r}"),
            _gate(
                f"{domain}_confidence_position_rollout_metric_present",
                isinstance(conf_position_diag, bool),
                f"position_rollout_crossover_conf={conf_position_diag!r}",
            ),
        ])

    provenance = _get(manifest, "provenance", {})
    gates.extend([
        _gate("git_commit_recorded", isinstance(_get(provenance, "git_commit"), str) and len(_get(provenance, "git_commit")) >= 7, f"git_commit={_get(provenance, 'git_commit')!r}"),
        _gate("single_manifest_for_outputs", _get(provenance, "single_manifest_for_all_figures_tables") is True, f"single_manifest={_get(provenance, 'single_manifest_for_all_figures_tables')!r}"),
    ])

    output_artifacts = _get(manifest, "outputs", {})
    figures = _get(output_artifacts, "figures")
    tables = _get(output_artifacts, "tables")
    gates.extend([
        _gate("figures_declared", isinstance(figures, list) and len(figures) > 0, f"figures={figures!r}"),
        _gate("tables_declared", isinstance(tables, list) and len(tables) > 0, f"tables={tables!r}"),
    ])

    return gates


def load_manifest(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_manifest_file(path: str | Path) -> tuple[bool, list[GateResult]]:
    gates = verify_synthetic_manifest(load_manifest(path))
    return all(gate.passed for gate in gates), gates
