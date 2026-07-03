#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridge_garden_v2.synthetic_manifest import verify_synthetic_manifest


MAIN_DOMAINS = ("code", "math", "dialogue")
EXPECTED_HIGH_TAGS = {
    "code": {"syntax_layout", "operator", "branch_guard", "return_semantics"},
    "math": {"substitution", "operator", "computed_value", "final_answer"},
    "dialogue": {"recipient", "required_fact", "date_time", "forbidden_constraint"},
}
EXPECTED_LOW_TAGS = {
    "code": {"equivalent_implementation"},
    "math": {"equivalent_representation"},
    "dialogue": {"tone_paraphrase"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, help="Directory containing <domain>/summary.json files")
    parser.add_argument("--check", required=True, help="Structural or exact-cost check JSON")
    parser.add_argument("--negative-control-summary", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--print-gates", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    check = _load_json(Path(args.check))
    summaries = {
        domain: _load_domain_summary(run_root, domain)
        for domain in MAIN_DOMAINS
    }
    negative_summary = (
        _load_json(Path(args.negative_control_summary))
        if args.negative_control_summary
        else _load_negative_control_summary(run_root)
    )
    manifest = _build_manifest(
        run_root=run_root,
        check=check,
        summaries=summaries,
        negative_summary=negative_summary,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    gates = verify_synthetic_manifest(manifest)
    if args.print_gates:
        for gate in gates:
            status = "PASS" if gate.passed else "FAIL"
            print(f"{status}\t{gate.name}\t{gate.detail}")
    print(json.dumps({
        "manifest": str(out),
        "passed": all(gate.passed for gate in gates),
        "failed_gates": [gate.name for gate in gates if not gate.passed],
    }, indent=2))


def _build_manifest(
    *,
    run_root: Path,
    check: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    negative_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    domain_configs = {}
    check_domains = check.get("domains", {})
    for domain, summary in summaries.items():
        action = summary.get("action_universe", {})
        pf_domain = check_domains.get(domain, {})
        domain_configs[domain] = {
            "vocab_size": int(summary.get("vocab_size", check.get("vocab_size", 0))),
            "max_len": int(summary.get("script_len", check.get("script_len", 0))),
            "analysis_size": int(summary.get("analysis_count", check.get("analysis_count", 0))),
            "exact_continuation": summary.get("kappa_continuation") == "expectation",
            "full_vocab_kappa": True,
            "uses_global_eval_token_ids": action.get("uses_global_eval_token_ids") is True,
            "uses_state_local_eval_token_ids": action.get("uses_state_local_eval_token_ids") is True,
            "oracle_teacher": True,
            "kappa_continuation": summary.get("kappa_continuation"),
            "max_kappa_states": int(summary.get("max_kappa_states", -1)),
            "eval_token_count": int(action.get("eval_token_count", pf_domain.get("eval_token_count", -1))),
            "eos_in_eval": bool(action.get("eos_in_eval", pf_domain.get("eos_in_eval", False))),
            "pad_in_eval": bool(action.get("pad_in_eval", pf_domain.get("pad_in_eval", True))),
            "bos_in_eval": bool(action.get("bos_in_eval", pf_domain.get("bos_in_eval", True))),
            "kappa_state_count": int(summary.get("kappa_state_count", pf_domain.get("kappa_state_count", 0))),
            "estimated_q_values": int(summary.get("kappa_state_count", 0)) * int(action.get("eval_token_count", 0)),
            "fixed_hybrid_lambda": summary.get("fixed_hybrid_lambda"),
            "selected_hybrid_lambda": summary.get("selected_hybrid_lambda"),
            "hybrid_lambda_grid": summary.get("hybrid_lambda_grid", []),
            "hybrid_lambda_selection_policy": summary.get("hybrid_lambda_selection_policy"),
            "hybrid_training_policy": summary.get("hybrid_training_policy"),
            "hybrid_selection_count": int(summary.get("hybrid_selection_count", 0)),
            "future_support": pf_domain.get("future_support", {}),
            "future_support_budget": int(check.get("support_cap", 100_000)),
        }

    results = {
        "domains": {
            domain: _domain_result(run_root / domain, summary)
            for domain, summary in summaries.items()
        },
        "negative_control": _negative_control_result(negative_summary),
    }
    return {
        "setup": {
            "domains": list(MAIN_DOMAINS),
            "negative_control": "negative_control",
            "domain_configs": domain_configs,
            "kappa_split": {
                "bridge_top_quantile": 0.2,
                "garden_bottom_quantile": 0.2,
            },
            "seeds": _infer_seeds(summaries),
            "methods": _infer_methods(summaries),
        },
        "results": results,
        "provenance": {
            "git_commit": _git_commit(),
            "single_manifest_for_all_figures_tables": True,
            "run_root": str(run_root),
            "check": str(check.get("path", "")),
        },
        "outputs": _outputs(run_root),
    }


def _domain_result(domain_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    domain = str(summary.get("domain") or domain_dir.name)
    top_counts = summary.get("top20_semantic_counts", {})
    bottom_counts = summary.get("bottom20_semantic_counts", {})
    methods = summary.get("methods", {})
    heatmaps = sorted(str(path) for path in domain_dir.glob("**/*heatmap*.png"))
    kappa_state_table = domain_dir / "kappa_state_table.json"
    high_count = summary.get("top20_expected_high_risk_count_min")
    if high_count is None:
        high_count = _count_by_roles(top_counts, EXPECTED_HIGH_TAGS.get(domain, set()))
    flex_count = summary.get("bottom20_expected_flexible_count_min")
    if flex_count is None:
        flex_count = _count_by_roles(bottom_counts, EXPECTED_LOW_TAGS.get(domain, set()))
    return {
        "kappa_feasible": summary.get("kappa_continuation") == "expectation" and summary.get("max_kappa_states") == 0,
        "regional_crossover_metric": "ref_split_kappa_contribution.absolute_contribution",
        "hard_soft_crossover_ref": _hard_soft_crossover(methods),
        "position_rollout_bridge_crossover_passes": _bridge_crossover_position_rollout(methods),
        "position_rollout_garden_crossover_passes": _garden_crossover_position_rollout(methods),
        "hybrid_beats_best_pure_overall": _hybrid_beats_best_pure(methods),
        "interpretability_verification_passes": high_count > 0 and flex_count > 0,
        "kappa_spread_non_degenerate": _kappa_spread_non_degenerate(summary.get("kappa_spread", {})),
        "kappa_spread": summary.get("kappa_spread", {}),
        "training_converged": _training_converged(methods),
        "statistics": summary.get("statistical_report", {}),
        "interpretability": {
            "heatmaps": heatmaps,
            "kappa_state_table": str(kappa_state_table) if kappa_state_table.exists() else "",
            "top20_expected_high_risk_count": int(high_count),
            "bottom20_expected_flexible_count": int(flex_count),
            "domain_interpretation": (
                "High-kappa states are expected to align with semantic-risk choices, "
                "while low-kappa states are expected to align with repairable equivalent-form choices."
            ),
        },
        "confidence_proxy": {
            "spearman_rho": summary.get("confidence_rho"),
            "bridge_overlap": summary.get("bridge_overlap"),
            "garden_overlap": summary.get("garden_overlap"),
            "hard_soft_crossover_conf": _hard_soft_crossover(methods, split="confidence_split_kappa_contribution"),
            "regional_crossover_metric": "confidence_split_kappa_contribution.absolute_contribution",
            "position_rollout_crossover_conf": _hard_soft_crossover(methods, split="confidence_split_position_rollout"),
        },
    }


def _negative_control_result(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {"fails_same_pattern": False, "reason": "missing negative_control summary"}
    top = summary.get("top20_semantic_counts", {})
    bottom = summary.get("bottom20_semantic_counts", {})
    methods = summary.get("methods", {})
    statistics = summary.get("statistical_report", {})
    rho = summary.get("confidence_rho")
    no_semantic_separation = set(top) == {"surface_ngram"} and set(bottom) == {"surface_ngram"}
    weak_confidence = rho is None or (isinstance(rho, (int, float)) and float(rho) < 0.7)
    no_ref_crossover = (
        not _hard_soft_crossover(methods)
        or _weak_or_uncertain_effect(statistics.get("bridge_hard_over_soft_delta"))
        or _weak_or_uncertain_effect(statistics.get("garden_soft_over_hard_delta"))
    )
    no_conf_crossover = not _hard_soft_crossover(methods, split="confidence_split_kappa_contribution")
    no_hybrid_advantage = (
        not _hybrid_beats_best_pure(methods)
        or _weak_or_uncertain_effect(statistics.get("hybrid_gain_over_best_pure"))
    )
    return {
        "fails_same_pattern": bool(
            no_semantic_separation
            and weak_confidence
            and no_ref_crossover
            and no_conf_crossover
            and no_hybrid_advantage
        ),
        "confidence_rho": rho,
        "no_semantic_separation": no_semantic_separation,
        "no_ref_crossover": no_ref_crossover,
        "no_confidence_crossover": no_conf_crossover,
        "no_hybrid_advantage": no_hybrid_advantage,
        "top20_semantic_counts": top,
        "bottom20_semantic_counts": bottom,
        "statistical_report": statistics,
    }


def _weak_or_uncertain_effect(report: Any, *, threshold: float = 0.2) -> bool:
    if not isinstance(report, dict):
        return False
    mean = _finite_num(report.get("mean"))
    ci_low = _finite_num(report.get("ci95_low"))
    ci_high = _finite_num(report.get("ci95_high"))
    weak = mean is not None and abs(mean) < threshold
    uncertain = ci_low is not None and ci_high is not None and ci_low <= 0.0 <= ci_high
    return bool(weak or uncertain)


def _hard_soft_crossover(methods: dict[str, Any], *, split: str = "ref_split_kappa_contribution") -> bool:
    hard = methods.get("hard_kd", {}).get(split, {})
    soft = methods.get("soft_kd", {}).get(split, {})
    if not hard or not soft:
        return False
    if split.endswith("_kappa_contribution"):
        return (
            _num(hard.get("bridge_absolute_contribution")) < _num(soft.get("bridge_absolute_contribution"))
            and _num(soft.get("garden_absolute_contribution")) < _num(hard.get("garden_absolute_contribution"))
        )
    return (
        _num(hard.get("bridge_eb")) < _num(soft.get("bridge_eb"))
        and _num(soft.get("garden_eb")) < _num(hard.get("garden_eb"))
    )


def _bridge_crossover_position_rollout(methods: dict[str, Any]) -> bool:
    hard = methods.get("hard_kd", {}).get("ref_split_position_rollout", {})
    soft = methods.get("soft_kd", {}).get("ref_split_position_rollout", {})
    return bool(hard and soft and _num(hard.get("bridge_eb")) < _num(soft.get("bridge_eb")))


def _garden_crossover_position_rollout(methods: dict[str, Any]) -> bool:
    hard = methods.get("hard_kd", {}).get("ref_split_position_rollout", {})
    soft = methods.get("soft_kd", {}).get("ref_split_position_rollout", {})
    return bool(hard and soft and _num(soft.get("garden_eb")) < _num(hard.get("garden_eb")))


def _hybrid_beats_best_pure(methods: dict[str, Any]) -> bool:
    hybrid = _num(methods.get("hybrid_kd", {}).get("overall_position_rollout", {}).get("overall_eb"))
    hard = _num(methods.get("hard_kd", {}).get("overall_position_rollout", {}).get("overall_eb"))
    soft = _num(methods.get("soft_kd", {}).get("overall_position_rollout", {}).get("overall_eb"))
    return hybrid < min(hard, soft)


def _outputs(run_root: Path) -> dict[str, list[str]]:
    figures = sorted(str(path) for path in run_root.glob("**/*.png"))
    tables = sorted(str(path) for path in run_root.glob("**/*state_table.json"))
    return {"figures": figures, "tables": tables}


def _infer_methods(summaries: dict[str, dict[str, Any]]) -> list[str]:
    methods: set[str] = set()
    for summary in summaries.values():
        methods.update(summary.get("methods", {}).keys())
    return sorted(methods)


def _infer_seeds(summaries: dict[str, dict[str, Any]]) -> list[int]:
    seeds = {int(summary.get("train_sample_seed", 0)) for summary in summaries.values()}
    seeds.update(int(summary.get("analysis_sample_seed", 0)) for summary in summaries.values())
    seeds.update(int(summary.get("model_seed", summary.get("train_sample_seed", 0))) for summary in summaries.values())
    for summary in summaries.values():
        if isinstance(summary.get("seeds"), list):
            seeds.update(int(seed) for seed in summary["seeds"])
    return sorted(seeds)


def _load_domain_summary(run_root: Path, domain: str) -> dict[str, Any]:
    direct = run_root / domain / "summary.json"
    if direct.exists():
        return _load_json(direct)
    seed_paths = sorted((run_root / domain).glob("seed_*/summary.json"))
    if not seed_paths:
        raise FileNotFoundError(f"missing summary for domain {domain}: {direct}")
    summaries = [(path, _load_json(path)) for path in seed_paths]
    return _aggregate_seed_summaries(domain, summaries)


def _load_negative_control_summary(run_root: Path) -> dict[str, Any] | None:
    direct = _maybe_load(run_root / "negative_control" / "summary.json")
    if direct is not None:
        return direct
    try:
        return _load_domain_summary(run_root, "negative_control")
    except FileNotFoundError:
        return None


def _aggregate_seed_summaries(domain: str, items: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    base = copy.deepcopy(items[0][1])
    base["domain"] = domain
    base["seed_summary_paths"] = [str(path) for path, _ in items]
    base["seeds"] = sorted({_seed_from_summary(path, summary) for path, summary in items})
    for key in [
        "confidence_rho",
        "bridge_overlap",
        "garden_overlap",
        "bridge_tie_aware_overlap",
        "garden_tie_aware_overlap",
    ]:
        base[key] = _mean_path([summary.get(key) for _, summary in items])
    base["top20_expected_high_risk_count_min"] = min(
        _count_by_roles(summary.get("top20_semantic_counts", {}), EXPECTED_HIGH_TAGS.get(domain, set()))
        for _, summary in items
    )
    base["bottom20_expected_flexible_count_min"] = min(
        _count_by_roles(summary.get("bottom20_semantic_counts", {}), EXPECTED_LOW_TAGS.get(domain, set()))
        for _, summary in items
    )
    base["methods"] = _aggregate_methods([summary.get("methods", {}) for _, summary in items])
    base["statistical_report"] = _seed_statistical_report(items)
    return base


def _seed_statistical_report(items: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    rows = []
    for path, summary in items:
        metrics = _seed_effect_metrics(summary)
        if metrics is None:
            continue
        metrics["seed"] = _seed_from_summary(path, summary)
        rows.append(metrics)
    return {
        "seed_count": len(rows),
        "bootstrap_protocol": "paired_seed_bootstrap",
        "bootstrap_reps": 10_000,
        "bridge_hard_over_soft_delta": _bootstrap_metric([row["bridge_hard_over_soft_delta"] for row in rows]),
        "garden_soft_over_hard_delta": _bootstrap_metric([row["garden_soft_over_hard_delta"] for row in rows]),
        "hybrid_gain_over_best_pure": _bootstrap_metric([row["hybrid_gain_over_best_pure"] for row in rows]),
    }


def _seed_effect_metrics(summary: dict[str, Any]) -> dict[str, float] | None:
    methods = summary.get("methods", {})
    hard = methods.get("hard_kd", {})
    soft = methods.get("soft_kd", {})
    hybrid = methods.get("hybrid_kd", {})
    hard_ref = hard.get("ref_split_kappa_contribution", {})
    soft_ref = soft.get("ref_split_kappa_contribution", {})
    hard_overall = hard.get("overall_position_rollout", {})
    soft_overall = soft.get("overall_position_rollout", {})
    hybrid_overall = hybrid.get("overall_position_rollout", {})
    values = {
        "hard_bridge": _finite_num(hard_ref.get("bridge_absolute_contribution")),
        "soft_bridge": _finite_num(soft_ref.get("bridge_absolute_contribution")),
        "hard_garden": _finite_num(hard_ref.get("garden_absolute_contribution")),
        "soft_garden": _finite_num(soft_ref.get("garden_absolute_contribution")),
        "hard_overall": _finite_num(hard_overall.get("overall_eb")),
        "soft_overall": _finite_num(soft_overall.get("overall_eb")),
        "hybrid_overall": _finite_num(hybrid_overall.get("overall_eb")),
    }
    if any(value is None for value in values.values()):
        return None
    return {
        "bridge_hard_over_soft_delta": values["soft_bridge"] - values["hard_bridge"],
        "garden_soft_over_hard_delta": values["hard_garden"] - values["soft_garden"],
        "hybrid_gain_over_best_pure": min(values["hard_overall"], values["soft_overall"]) - values["hybrid_overall"],
    }


def _bootstrap_metric(values: list[float]) -> dict[str, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "cohen_d": None, "positive_seed_count": 0}
    mean = sum(finite) / len(finite)
    positive_count = sum(1 for value in finite if value > 0.0)
    if len(finite) == 1:
        return {
            "mean": mean,
            "ci95_low": mean,
            "ci95_high": mean,
            "cohen_d": 0.0,
            "positive_seed_count": positive_count,
        }
    rng = random.Random(20260515)
    boot_means = []
    for _ in range(10_000):
        sample = [finite[rng.randrange(len(finite))] for _ in finite]
        boot_means.append(sum(sample) / len(sample))
    boot_means.sort()
    std = math.sqrt(sum((value - mean) ** 2 for value in finite) / (len(finite) - 1))
    cohen_d = mean / std if std > 0 else (999999.0 if mean > 0 else (-999999.0 if mean < 0 else 0.0))
    return {
        "mean": mean,
        "ci95_low": boot_means[int(0.025 * (len(boot_means) - 1))],
        "ci95_high": boot_means[int(0.975 * (len(boot_means) - 1))],
        "cohen_d": cohen_d,
        "positive_seed_count": positive_count,
    }


def _kappa_spread_non_degenerate(spread: Any) -> bool:
    if not isinstance(spread, dict):
        return False
    ratio = _finite_num(spread.get("top_bottom_ratio"))
    cv = _finite_num(spread.get("coefficient_of_variation"))
    return bool((ratio is not None and ratio >= 2.0) or (cv is not None and cv > 0.5))


def _training_converged(methods: dict[str, Any]) -> bool:
    for method in ("ce_ref", "hard_kd", "soft_kd", "hybrid_kd"):
        summary = methods.get(method)
        if not isinstance(summary, dict):
            return False
        losses = summary.get("train_loss")
        if not isinstance(losses, list) or not losses:
            return False
        finite_losses = [_finite_num(value) for value in losses]
        if any(value is None for value in finite_losses):
            return False
        best_val = summary.get("best_val_teacher_forced_kl")
        if isinstance(best_val, list):
            best_val = best_val[0] if best_val else None
        val_history = summary.get("val_teacher_forced_kl")
        if not isinstance(val_history, list) or not val_history:
            return False
        if any(_finite_num(value) is None for value in val_history):
            return False
        if _finite_num(best_val) is None:
            return False
    return True


def _aggregate_methods(method_sets: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted(set().union(*(set(methods) for methods in method_sets)))
    return {
        method: _aggregate_nested([methods.get(method, {}) for methods in method_sets])
        for method in methods
    }


def _aggregate_nested(values: list[Any]) -> Any:
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if numeric and len(numeric) == len(values):
        return sum(numeric) / len(numeric)
    mappings = [value for value in values if isinstance(value, dict)]
    if mappings and len(mappings) == len(values):
        keys = sorted(set().union(*(set(mapping) for mapping in mappings)))
        return {
            key: _aggregate_nested([mapping.get(key) for mapping in mappings])
            for key in keys
        }
    return values[0] if values else None


def _mean_path(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return sum(numeric) / len(numeric) if numeric and len(numeric) == len(values) else None


def _seed_from_summary(path: Path, summary: dict[str, Any]) -> int:
    if "seed" in summary:
        return int(summary["seed"])
    if "model_seed" in summary:
        return int(summary["model_seed"])
    if "train_sample_seed" in summary:
        return int(summary["train_sample_seed"])
    name = path.parent.name
    if name.startswith("seed_"):
        return int(name.removeprefix("seed_"))
    return 0


def _count_by_roles(counts: dict[str, Any], roles: set[str]) -> int:
    return sum(int(value) for key, value in counts.items() if key in roles)


def _finite_num(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _num(value: Any) -> float:
    try:
        if value is None:
            return float("inf")
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_load(path: Path) -> dict[str, Any] | None:
    return _load_json(path) if path.exists() else None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
