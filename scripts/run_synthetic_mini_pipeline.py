#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridge_garden_v2.synthetic_domains import DOMAIN_SCRIPT_LENS, build_synthetic_domain
from bridge_garden_v2.synthetic_evaluator import (
    synthetic_evaluation_from_results,
    collect_clean_path_states,
    collect_sampled_path_states,
    evaluate_oracle_states,
    positions_from_state_ids,
)
from bridge_garden_v2.synthetic_visuals import render_kappa_confidence_scatter, render_token_kappa_heatmap
from bridge_garden_v2.modeling import CausalTransformerLM
from bridge_garden_v2.oracle_dataset import materialize_mode_paths, materialize_sampled_paths
from bridge_garden_v2.oracle_eval import (
    summarize_oracle_eb,
    summarize_oracle_expected_intervention,
    summarize_oracle_kappa_contribution,
    summarize_oracle_eb_by_positions,
    summarize_oracle_local_intervention_eb,
    summarize_oracle_sampled_rollout_statuses,
    trace_oracle_rollouts,
)
from bridge_garden_v2.oracle_student import NeuralStudentLoss
from bridge_garden_v2.oracle_training import train_oracle_student
from bridge_garden_v2.schema import ModelConfig


SYNTHETIC_DOMAINS = ["code", "math", "dialogue", "negative_control"]
SYNTHETIC_HYBRID_LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SYNTHETIC_PAPER_CONFIGS = {
    "paper": {
        "main_train_count": 4_000,
        "negative_train_count": 2_000,
        "val_count": 500,
        "analysis_count": 100,
        "rollout_repeats_min": 30,
        "lr": 2e-4,
        "lr_schedule": "warmup_cosine",
        "lr_warmup_epochs": 6,
        "early_stopping_patience": 6,
        "hard_kd_lr": None,
        "hard_kd_early_stopping_patience": None,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=SYNTHETIC_DOMAINS, default="code")
    parser.add_argument("--out-dir", default="bridge_garden_v2/synthetic_mini")
    parser.add_argument("--seed-dir-name", default=None, help="Optional seed subdirectory under <out-dir>/<domain>, e.g. seed_0")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--analysis-count", type=int, default=2)
    parser.add_argument("--script-len", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=["constant", "warmup_cosine"], default="constant")
    parser.add_argument("--lr-warmup-epochs", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--hard-kd-early-stopping-patience", type=int, default=None)
    parser.add_argument(
        "--hard-kd-early-stopping-metric",
        choices=["val_teacher_forced_kl", "val_method_loss"],
        default="val_teacher_forced_kl",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-kappa-states", type=int, default=0, help="0 means evaluate all non-terminal analysis states")
    parser.add_argument("--hybrid-lambdas", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--fixed-hybrid-lambda", type=float, default=None)
    parser.add_argument(
        "--hybrid-training-policy",
        choices=[
            "ambiguity_scaled",
            "role_aware_expected_role_gate",
            "bridge_hard_ambiguity_scaled",
        ],
        default="ambiguity_scaled",
    )
    parser.add_argument("--hybrid-selection-count", type=int, default=0, help="Validation sequences used only for non-main hybrid lambda selection")
    parser.add_argument(
        "--hybrid-lambda-selection-policy",
        choices=[
            "min_validation_teacher_forced_kl",
            "min_validation_rollout_eb",
            "min_validation_rollout_eb_one_se_lowest_lambda",
        ],
        default="min_validation_teacher_forced_kl",
    )
    parser.add_argument("--ref-only", action="store_true", help="Only train/evaluate ce_ref; setup query, not a full result")
    parser.add_argument("--train-path-policy", choices=["sample", "mode"], default="sample")
    parser.add_argument("--train-sample-seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--analysis-path-policy", choices=["sample", "mode"], default="sample")
    parser.add_argument("--analysis-sample-seed", type=int, default=0)
    parser.add_argument("--kappa-continuation", choices=["expectation", "mode"], default="expectation")
    parser.add_argument("--enforce-paper-config", action="store_true", help="Reject metric settings that deviate from the paper-defined κ setup")
    parser.add_argument(
        "--paper-config",
        choices=sorted(SYNTHETIC_PAPER_CONFIGS),
        default="paper",
        help="Named synthetic configuration used in the paper.",
    )
    parser.add_argument("--rollout-repeats", type=int, default=50)
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument("--violation-penalty", type=float, default=2.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--method-eval-devices",
        default="",
        help=(
            "Optional comma-separated devices for parallel method evaluation, "
            "for example cuda:0,cuda:1,cuda:2,cuda:3. Empty means serial."
        ),
    )
    parser.add_argument("--num-heatmaps", type=int, default=5)
    parser.add_argument("--progress-json", default=None, help="Optional JSON progress file for long runs")
    parser.add_argument("--ref-kappa-worker-payload", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ref-kappa-worker-output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--method-eval-worker-payload", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--method-eval-worker-output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hard-kd-lr", type=float, default=None, help="Optional hard-KD-specific peak learning rate.")
    args = parser.parse_args()
    if args.ref_kappa_worker_payload:
        if not args.ref_kappa_worker_output:
            raise SystemExit("--ref-kappa-worker-output is required with --ref-kappa-worker-payload")
        _run_ref_kappa_worker(Path(args.ref_kappa_worker_payload), Path(args.ref_kappa_worker_output))
        return
    if args.method_eval_worker_payload:
        if not args.method_eval_worker_output:
            raise SystemExit("--method-eval-worker-output is required with --method-eval-worker-payload")
        _run_method_eval_worker(Path(args.method_eval_worker_payload), Path(args.method_eval_worker_output))
        return
    if args.script_len is None:
        args.script_len = DOMAIN_SCRIPT_LENS[args.domain]
    _enforce_paper_config(args)
    device = _resolve_device(args.device)
    model_seed = int(args.train_sample_seed if args.model_seed is None else args.model_seed)

    out_dir = _output_dir(Path(args.out_dir), args.domain, args.seed_dir_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_progress(args, out_dir, "starting", {"device": str(device)})
    required_samples = args.train_count + args.val_count + args.analysis_count
    if args.sample_count < required_samples:
        args.sample_count = required_samples
    bundle = build_synthetic_domain(
        args.domain,
        sample_count=args.sample_count,
        script_len=args.script_len,
        vocab_size=args.vocab_size,
    )
    train_ids = list(range(args.train_count))
    val_ids = list(range(args.train_count, args.train_count + args.val_count))
    analysis_ids = list(range(args.train_count + args.val_count, required_samples))
    if args.train_path_policy == "sample":
        batch = materialize_sampled_paths(
            bundle.oracle,
            sample_ids=train_ids,
            vocab_size=len(bundle.vocab),
            seed=args.train_sample_seed,
        )
        val_batch = materialize_sampled_paths(
            bundle.oracle,
            sample_ids=val_ids,
            vocab_size=len(bundle.vocab),
            seed=args.train_sample_seed,
        )
    else:
        batch = materialize_mode_paths(bundle.oracle, sample_ids=train_ids, vocab_size=len(bundle.vocab))
        val_batch = materialize_mode_paths(bundle.oracle, sample_ids=val_ids, vocab_size=len(bundle.vocab))
    _write_progress(
        args,
        out_dir,
        "data_ready",
        {
            "train_count": args.train_count,
            "val_count": args.val_count,
            "analysis_count": args.analysis_count,
            "sample_count": args.sample_count,
        },
    )
    model_cfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    methods = ["ce_ref"] if args.ref_only else ["ce_ref", "hard_kd", "soft_kd"]
    models = {}
    histories = {}
    for method in methods:
        method_lr = _method_learning_rate(args, method)
        method_patience = _method_early_stopping_patience(args, method)
        model, history = train_oracle_student(
            method=method,
            input_ids=batch.input_ids,
            target_ids=batch.target_ids,
            teacher_probs=batch.teacher_probs,
            vocab_size=len(bundle.vocab),
            model_config=model_cfg,
            max_len=batch.input_ids.shape[1],
            pad_id=bundle.oracle.pad_id,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=method_lr,
            lr_schedule=args.lr_schedule,
            lr_warmup_epochs=args.lr_warmup_epochs,
            lambda_soft=0.5,
            seed=model_seed,
            device=device,
            val_input_ids=val_batch.input_ids,
            val_target_ids=val_batch.target_ids,
            val_teacher_probs=val_batch.teacher_probs,
            early_stopping_patience=method_patience,
            early_stopping_metric=_method_early_stopping_metric(args, method),
        )
        models[method] = model
        histories[method] = history
        _write_progress(args, out_dir, "method_trained", {"method": method})

    hybrid_candidates = {}
    best_lambda = None
    hybrid_validation_rollout_eb = {}
    if not args.ref_only:
        lambda_grid = (
            [float(args.fixed_hybrid_lambda)]
            if args.fixed_hybrid_lambda is not None
            else [float(x) for x in args.hybrid_lambdas.split(",") if x.strip()]
        )
        for lam in lambda_grid:
            _write_progress(args, out_dir, "hybrid_candidate_started", {"lambda": lam})
            hybrid_role_ids = _hybrid_role_ids(batch.role_ids, args.hybrid_training_policy)
            hybrid_val_role_ids = _hybrid_role_ids(val_batch.role_ids, args.hybrid_training_policy)
            model, history = train_oracle_student(
                method="hybrid_kd",
                input_ids=batch.input_ids,
                target_ids=batch.target_ids,
                teacher_probs=batch.teacher_probs,
                vocab_size=len(bundle.vocab),
                model_config=model_cfg,
                max_len=batch.input_ids.shape[1],
                pad_id=bundle.oracle.pad_id,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=_method_learning_rate(args, "hybrid_kd"),
                lr_schedule=args.lr_schedule,
                lr_warmup_epochs=args.lr_warmup_epochs,
                lambda_soft=lam,
                seed=model_seed,
                device=device,
                val_input_ids=val_batch.input_ids,
                val_target_ids=val_batch.target_ids,
                val_teacher_probs=val_batch.teacher_probs,
                role_ids=hybrid_role_ids,
                val_role_ids=hybrid_val_role_ids,
                early_stopping_patience=_method_early_stopping_patience(args, "hybrid_kd"),
                early_stopping_metric=_method_early_stopping_metric(args, "hybrid_kd"),
            )
            if args.fixed_hybrid_lambda is not None:
                val_risk = float("nan")
            else:
                val_risk = _best_val_teacher_forced_kl(history)
            hybrid_candidates[lam] = (val_risk, model, history)
            _write_progress(
                args,
                out_dir,
                "hybrid_candidate_trained",
                {"lambda": lam, "validation_teacher_forced_kl": _finite(val_risk)},
            )
        if args.fixed_hybrid_lambda is not None:
            best_lambda = float(args.fixed_hybrid_lambda)
        elif args.hybrid_lambda_selection_policy in {
            "min_validation_rollout_eb",
            "min_validation_rollout_eb_one_se_lowest_lambda",
        }:
            selection_ids = val_ids[: int(args.hybrid_selection_count)]
            if not selection_ids:
                raise SystemExit("--hybrid-selection-count must be positive for validation rollout lambda selection")
            rollout_scores = {}
            rollout_score_ses = {}
            for lam, (_val_risk, model, _history) in hybrid_candidates.items():
                _write_progress(args, out_dir, "hybrid_candidate_rollout_selection_started", {"lambda": lam})
                score = summarize_oracle_eb(
                    oracle=bundle.oracle,
                    model=model,
                    vocab_size=len(bundle.vocab),
                    sample_ids=selection_ids,
                    rollout_repeats=args.rollout_repeats,
                    rollout_seed=args.rollout_seed,
                    device=device,
                )
                rollout_scores[lam] = float(score.exposure_bias)
                rollout_score_ses[lam] = float(score.exposure_bias_se)
                hybrid_validation_rollout_eb[lam] = {
                    "overall_eb": _finite(score.exposure_bias),
                    "overall_eb_se": _finite(score.exposure_bias_se),
                    "teacher_forced_kl": _finite(score.teacher_forced_kl),
                    "rollout_kl": _finite(score.rollout_kl),
                }
                _write_progress(
                    args,
                    out_dir,
                    "hybrid_candidate_rollout_selection_completed",
                    {
                        "lambda": lam,
                        "validation_rollout_eb": _finite(score.exposure_bias),
                        "validation_rollout_eb_se": _finite(score.exposure_bias_se),
                    },
                )
            if args.hybrid_lambda_selection_policy == "min_validation_rollout_eb_one_se_lowest_lambda":
                best_lambda = _select_one_se_lowest_lambda(rollout_scores, rollout_score_ses)
            else:
                best_lambda = min(rollout_scores, key=lambda lam: rollout_scores[lam])
        else:
            best_lambda = min(hybrid_candidates, key=lambda lam: hybrid_candidates[lam][0])
        models["hybrid_kd"] = hybrid_candidates[best_lambda][1]
        histories["hybrid_kd"] = hybrid_candidates[best_lambda][2]
        _write_progress(args, out_dir, "hybrid_selected", {"lambda": best_lambda})

    if args.analysis_path_policy == "sample":
        all_analysis_states = collect_sampled_path_states(bundle.oracle, analysis_ids, seed=args.analysis_sample_seed)
    else:
        all_analysis_states = []
        for sample_id in analysis_ids:
            all_analysis_states.extend(collect_clean_path_states(bundle.oracle, sample_count=sample_id + 1)[-args.script_len:])
    states = list(all_analysis_states)
    if args.max_kappa_states > 0:
        states = states[: args.max_kappa_states]
    _write_progress(args, out_dir, "analysis_states_ready", {"kappa_states": len(states)})

    _write_progress(args, out_dir, "ref_kappa_started", {"kappa_states": len(states)})
    kappa_eval = _evaluate_ref_kappa(
        args=args,
        out_dir=out_dir,
        bundle=bundle,
        model_cfg=model_cfg,
        model_max_len=int(batch.input_ids.shape[1]),
        ref_model=models["ce_ref"],
        states=states,
        fallback_device=device,
    )
    _write_progress(args, out_dir, "ref_kappa_completed", {"kappa_state_count": len(kappa_eval.scores)})
    bridge_positions = positions_from_state_ids(kappa_eval.kappa_split.bridge)
    garden_positions = positions_from_state_ids(kappa_eval.kappa_split.garden)
    conf_bridge_positions = positions_from_state_ids(kappa_eval.confidence_split.bridge)
    conf_garden_positions = positions_from_state_ids(kappa_eval.confidence_split.garden)

    method_summaries, rollout_traces = _evaluate_methods(
        args=args,
        out_dir=out_dir,
        bundle=bundle,
        model_cfg=model_cfg,
        model_max_len=int(batch.input_ids.shape[1]),
        models=models,
        histories=histories,
        kappa_eval=kappa_eval,
        analysis_ids=analysis_ids,
        fallback_device=device,
    )

    heatmap_paths = []
    _write_progress(args, out_dir, "visualization_started", {"num_heatmaps": args.num_heatmaps})
    heatmap_sample_ids = _heatmap_sample_ids(all_analysis_states, kappa_eval, args.num_heatmaps)
    for heatmap_sample_id in heatmap_sample_ids:
        sample_states = [
            state for state in all_analysis_states
            if int(getattr(state, "sample_id", -1)) == int(heatmap_sample_id)
        ]
        tokens, tags, kappas, heatmap_regions = _heatmap_rows(bundle, kappa_eval, sample_states)
        heatmap_paths.append(render_token_kappa_heatmap(
            tokens=tokens,
            kappa_values=kappas,
            semantic_tags=tags,
            region_labels=heatmap_regions,
            output_path=out_dir / f"kappa_heatmap_sample_{heatmap_sample_id:03d}.png",
            title=f"{args.domain} sample {heatmap_sample_id} κ_ref",
        ))
    state_table = _state_table(bundle, kappa_eval, all_analysis_states)
    (out_dir / "kappa_state_table.json").write_text(json.dumps(state_table, indent=2), encoding="utf-8")
    (out_dir / "rollout_traces.json").write_text(json.dumps(rollout_traces, indent=2), encoding="utf-8")
    scatter_rows = _scatter_rows(kappa_eval)
    scatter_path = render_kappa_confidence_scatter(
        kappa_values=[row["kappa"] for row in scatter_rows],
        confidence_values=[row["confidence"] for row in scatter_rows],
        semantic_tags=[row["semantic_tag"] for row in scatter_rows],
        region_labels=[row["region_label"] for row in scatter_rows],
        output_path=out_dir / "mini_kappa_confidence_scatter.png",
        title=f"{args.domain} κ_ref vs teacher confidence",
    )
    summary = {
        "domain": args.domain,
        "seed_dir_name": args.seed_dir_name,
        "sample_count": args.sample_count,
        "train_count": args.train_count,
        "val_count": args.val_count,
        "analysis_count": args.analysis_count,
        "script_len": args.script_len,
        "vocab_size": args.vocab_size,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "analysis_ids": analysis_ids,
        "selected_hybrid_lambda": best_lambda,
        "fixed_hybrid_lambda": args.fixed_hybrid_lambda,
        "hybrid_lambda_grid": [float(x) for x in args.hybrid_lambdas.split(",") if x.strip()],
        "hybrid_lambda_selection_policy": args.hybrid_lambda_selection_policy,
        "hybrid_validation_rollout_eb": {str(key): value for key, value in hybrid_validation_rollout_eb.items()},
        "hybrid_training_policy": args.hybrid_training_policy,
        "hybrid_selection_count": args.hybrid_selection_count,
        "ref_only": args.ref_only,
        "train_path_policy": args.train_path_policy,
        "train_sample_seed": args.train_sample_seed,
        "model_seed": model_seed,
        "analysis_path_policy": args.analysis_path_policy,
        "analysis_sample_seed": args.analysis_sample_seed,
        "max_kappa_states": args.max_kappa_states,
        "kappa_continuation": args.kappa_continuation,
        "enforce_paper_config": args.enforce_paper_config,
        "paper_config": args.paper_config,
        "rollout_repeats": args.rollout_repeats,
        "rollout_seed": args.rollout_seed,
        "violation_penalty": args.violation_penalty,
        "lr": args.lr,
        "method_lr_overrides": {
            "hard_kd": args.hard_kd_lr,
        },
        "method_early_stopping_patience_overrides": {
            "hard_kd": args.hard_kd_early_stopping_patience,
        },
        "method_early_stopping_metric_overrides": {
            "hard_kd": args.hard_kd_early_stopping_metric,
        },
        "lr_schedule": args.lr_schedule,
        "lr_warmup_epochs": args.lr_warmup_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "device": str(device),
        "requested_device": args.device,
        "method_eval_devices": _method_eval_devices(args.method_eval_devices, device),
        "num_heatmaps": args.num_heatmaps,
        "model_config": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
        },
        "action_universe": {
            "eval_token_count": len(bundle.oracle.eval_token_ids),
            "eos_in_eval": bundle.oracle.eos_id in bundle.oracle.eval_token_ids,
            "pad_in_eval": bundle.oracle.pad_id in bundle.oracle.eval_token_ids,
            "bos_in_eval": bundle.oracle.bos_id in bundle.oracle.eval_token_ids,
            "expected_eval_token_count": len(bundle.vocab) - 2,
            "uses_global_eval_token_ids": True,
            "uses_state_local_eval_token_ids": False,
        },
        "hybrid_validation_teacher_forced_kl": {str(lam): _finite(item[0]) for lam, item in hybrid_candidates.items()},
        "training_metrics": _training_metrics(
            method_summaries,
            hybrid_candidates,
            best_lambda,
            selection_policy=args.hybrid_lambda_selection_policy,
        ),
        "kappa_state_count": len(kappa_eval.scores),
        "kappa_spread": _kappa_spread_summary(kappa_eval),
        "kappa_region_semantic_occupancy": _region_semantic_occupancy(state_table),
        "confidence_rho": _finite(kappa_eval.confidence_rho),
        "bridge_overlap": _finite(kappa_eval.bridge_overlap),
        "garden_overlap": _finite(kappa_eval.garden_overlap),
        "bridge_tie_aware_overlap": _finite(kappa_eval.bridge_tie_aware_overlap),
        "garden_tie_aware_overlap": _finite(kappa_eval.garden_tie_aware_overlap),
        "top20_semantic_counts": kappa_eval.top20_semantic_counts,
        "bottom20_semantic_counts": kappa_eval.bottom20_semantic_counts,
        "bridge_positions": sorted(bridge_positions),
        "garden_positions": sorted(garden_positions),
        "confidence_bridge_positions": sorted(conf_bridge_positions),
        "confidence_garden_positions": sorted(conf_garden_positions),
        "confidence_bridge_tie_band_positions": sorted(positions_from_state_ids(kappa_eval.confidence_split.bridge_tie_band)),
        "confidence_garden_tie_band_positions": sorted(positions_from_state_ids(kappa_eval.confidence_split.garden_tie_band)),
        "position_split_metrics": {
            "ref": _position_split_metrics(kappa_eval.kappa_split.bridge, kappa_eval.kappa_split.garden),
            "confidence": _position_split_metrics(kappa_eval.confidence_split.bridge, kappa_eval.confidence_split.garden),
        },
        "distribution_metrics": _distribution_metrics(bundle, kappa_eval, models),
        "methods": method_summaries,
        "heatmap": str(heatmap_paths[0]) if heatmap_paths else None,
        "heatmaps": [str(path) for path in heatmap_paths],
        "heatmap_sample_ids": heatmap_sample_ids,
        "confidence_scatter": str(scatter_path),
        "kappa_state_table": str(out_dir / "kappa_state_table.json"),
        "rollout_traces": str(out_dir / "rollout_traces.json"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_progress(args, out_dir, "completed", {"summary": str(out_dir / "summary.json")})
    print(json.dumps(summary, indent=2))


def _write_progress(args: argparse.Namespace, out_dir: Path, stage: str, extra: dict | None = None) -> None:
    progress_path = Path(args.progress_json) if args.progress_json else out_dir / "pipeline_progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "running" if stage != "completed" else "completed",
        "stage": stage,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "domain": args.domain,
        "seed_dir_name": args.seed_dir_name,
        "train_sample_seed": args.train_sample_seed,
        "analysis_sample_seed": args.analysis_sample_seed,
        "model_seed": args.model_seed,
        "extra": extra or {},
    }
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _method_eval_devices(raw: str, fallback_device: torch.device) -> list[str]:
    devices = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if devices:
        return [str(fallback_device) if item == "auto" else item for item in devices]
    return [str(fallback_device)]


def _method_learning_rate(args: argparse.Namespace, method: str) -> float:
    if method == "hard_kd" and args.hard_kd_lr is not None:
        return float(args.hard_kd_lr)
    return float(args.lr)


def _method_early_stopping_patience(args: argparse.Namespace, method: str) -> int:
    if method == "hard_kd" and args.hard_kd_early_stopping_patience is not None:
        return int(args.hard_kd_early_stopping_patience)
    return int(args.early_stopping_patience)


def _method_early_stopping_metric(args: argparse.Namespace, method: str) -> str:
    if method == "hard_kd":
        return str(args.hard_kd_early_stopping_metric)
    return "val_teacher_forced_kl"


def _hybrid_role_ids(role_ids: torch.Tensor, policy: str) -> torch.Tensor | None:
    if policy == "ambiguity_scaled":
        return None
    if policy == "role_aware_expected_role_gate":
        return role_ids
    if policy == "bridge_hard_ambiguity_scaled":
        return torch.where(role_ids == 1, torch.ones_like(role_ids), torch.zeros_like(role_ids))
    raise ValueError(f"unknown hybrid training policy: {policy}")


def _evaluate_ref_kappa(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    bundle,
    model_cfg: ModelConfig,
    model_max_len: int,
    ref_model: torch.nn.Module,
    states: list,
    fallback_device: torch.device,
):
    devices = _method_eval_devices(args.method_eval_devices, fallback_device)
    if len(devices) <= 1 or len(states) <= 1:
        ref_loss = NeuralStudentLoss(
            ref_model,
            vocab_size=len(bundle.vocab),
            violation_penalty=args.violation_penalty,
            device=fallback_device,
        )
        return evaluate_oracle_states(
            bundle.oracle,
            ref_loss,
            states,
            continuation=args.kappa_continuation,
        )

    chunks = _contiguous_chunks(states, max_chunks=min(len(devices), len(states)))
    _write_progress(
        args,
        out_dir,
        "parallel_ref_kappa_started",
        {"kappa_states": len(states), "devices": devices[: len(chunks)], "chunks": [len(chunk) for chunk in chunks]},
    )
    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in ref_model.state_dict().items()
    }
    with tempfile.TemporaryDirectory(prefix="synthetic_ref_kappa_") as tmp_name:
        tmp_dir = Path(tmp_name)
        processes = {}
        log_handles = {}
        for index, chunk in enumerate(chunks):
            device = devices[index % len(devices)]
            payload = {
                "domain": args.domain,
                "sample_count": args.sample_count,
                "script_len": args.script_len,
                "vocab_size": args.vocab_size,
                "model_config": model_cfg,
                "model_max_len": int(model_max_len),
                "state_dict": state_dict,
                "states": list(chunk),
                "device": device,
                "kappa_continuation": args.kappa_continuation,
                "violation_penalty": args.violation_penalty,
            }
            payload_path = tmp_dir / f"ref_kappa_{index}.payload.pkl"
            output_path = tmp_dir / f"ref_kappa_{index}.result.pkl"
            stdout_path = tmp_dir / f"ref_kappa_{index}.stdout"
            stderr_path = tmp_dir / f"ref_kappa_{index}.stderr"
            with payload_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            stdout_handle = stdout_path.open("wb")
            stderr_handle = stderr_path.open("wb")
            log_handles[index] = (stdout_handle, stderr_handle)
            _write_progress(
                args,
                out_dir,
                "ref_kappa_worker_started",
                {"worker": index, "device": device, "kappa_states": len(chunk)},
            )
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--ref-kappa-worker-payload",
                    str(payload_path),
                    "--ref-kappa-worker-output",
                    str(output_path),
                ],
                cwd=str(ROOT),
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            processes[index] = (proc, output_path, stdout_path, stderr_path, device)
        merged_results = []
        for index in range(len(chunks)):
            proc, output_path, stdout_path, stderr_path, device = processes[index]
            returncode = proc.wait()
            for handle in log_handles[index]:
                handle.close()
            if returncode != 0:
                raise RuntimeError(
                    f"ref kappa worker failed for chunk={index} on {device} with returncode={returncode}\n"
                    f"stdout tail:\n{_tail_text(stdout_path)}\n"
                    f"stderr tail:\n{_tail_text(stderr_path)}"
                )
            with output_path.open("rb") as handle:
                chunk_results = pickle.load(handle)
            merged_results.extend(chunk_results)
            _write_progress(
                args,
                out_dir,
                "ref_kappa_worker_completed",
                {"worker": index, "device": device, "kappa_states": len(chunk_results)},
            )
    if len(merged_results) != len(states):
        raise RuntimeError(f"ref kappa worker merge produced {len(merged_results)} results for {len(states)} states")
    return synthetic_evaluation_from_results(bundle.oracle, states, tuple(merged_results))


def _contiguous_chunks(items: list, *, max_chunks: int) -> list[list]:
    if max_chunks <= 1:
        return [list(items)]
    size = math.ceil(len(items) / max_chunks)
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


def _run_ref_kappa_worker(payload_path: Path, output_path: Path) -> None:
    torch.set_num_threads(1)
    with payload_path.open("rb") as handle:
        payload = pickle.load(handle)
    device = _worker_device(str(payload["device"]))
    bundle = build_synthetic_domain(
        str(payload["domain"]),
        sample_count=int(payload["sample_count"]),
        script_len=int(payload["script_len"]),
        vocab_size=int(payload["vocab_size"]),
    )
    model = CausalTransformerLM(
        vocab_size=len(bundle.vocab),
        config=payload["model_config"],
        max_len=int(payload["model_max_len"]),
        pad_id=bundle.oracle.pad_id,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    loss = NeuralStudentLoss(
        model,
        vocab_size=len(bundle.vocab),
        violation_penalty=float(payload["violation_penalty"]),
        device=device,
    )
    result = evaluate_oracle_states(
        bundle.oracle,
        loss,
        list(payload["states"]),
        continuation=str(payload["kappa_continuation"]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(tuple(result.kappa_results), handle, protocol=pickle.HIGHEST_PROTOCOL)


def _evaluate_methods(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    bundle,
    model_cfg: ModelConfig,
    model_max_len: int,
    models: dict[str, torch.nn.Module],
    histories: dict[str, dict],
    kappa_eval,
    analysis_ids: list[int],
    fallback_device: torch.device,
) -> tuple[dict, dict]:
    method_order = list(models.keys())
    devices = _method_eval_devices(args.method_eval_devices, fallback_device)
    payloads = {
        method: {
            "method": method,
            "device": devices[index % len(devices)],
            "domain": args.domain,
            "sample_count": args.sample_count,
            "script_len": args.script_len,
            "vocab_size": args.vocab_size,
            "model_config": model_cfg,
            "model_max_len": int(model_max_len),
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in models[method].state_dict().items()
            },
            "history": histories[method],
            "kappa_eval": kappa_eval,
            "analysis_ids": list(analysis_ids),
            "kappa_continuation": args.kappa_continuation,
            "rollout_repeats": args.rollout_repeats,
            "rollout_seed": args.rollout_seed,
            "violation_penalty": args.violation_penalty,
        }
        for index, method in enumerate(method_order)
    }

    if len(devices) <= 1 or len(method_order) <= 1:
        summaries = {}
        traces = {}
        for method in method_order:
            _write_progress(args, out_dir, "method_eval_started", {"method": method, "device": payloads[method]["device"]})
            result_method, summary, trace = _evaluate_method_worker(payloads[method])
            summaries[result_method] = summary
            traces[result_method] = trace
            _write_progress(args, out_dir, "method_eval_completed", {"method": method, "device": payloads[method]["device"]})
        return (
            {method: summaries[method] for method in method_order},
            {method: traces[method] for method in method_order},
        )

    _write_progress(
        args,
        out_dir,
        "parallel_method_eval_started",
        {"methods": method_order, "devices": devices},
    )
    summaries = {}
    traces = {}
    with tempfile.TemporaryDirectory(prefix="synthetic_method_eval_") as tmp_name:
        tmp_dir = Path(tmp_name)
        max_parallel = max(1, min(len(devices), len(method_order)))
        for start in range(0, len(method_order), max_parallel):
            wave = method_order[start : start + max_parallel]
            processes = {}
            log_handles = {}
            for method in wave:
                payload = payloads[method]
                _write_progress(args, out_dir, "method_eval_started", {"method": method, "device": payload["device"]})
                payload_path = tmp_dir / f"{method}.payload.pkl"
                output_path = tmp_dir / f"{method}.result.pkl"
                stdout_path = tmp_dir / f"{method}.stdout"
                stderr_path = tmp_dir / f"{method}.stderr"
                with payload_path.open("wb") as handle:
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                stdout_handle = stdout_path.open("wb")
                stderr_handle = stderr_path.open("wb")
                log_handles[method] = (stdout_handle, stderr_handle)
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--method-eval-worker-payload",
                        str(payload_path),
                        "--method-eval-worker-output",
                        str(output_path),
                    ],
                    cwd=str(ROOT),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )
                processes[method] = (proc, output_path, stdout_path, stderr_path, payload["device"])
            for method in wave:
                proc, output_path, stdout_path, stderr_path, device = processes[method]
                returncode = proc.wait()
                for handle in log_handles[method]:
                    handle.close()
                if returncode != 0:
                    raise RuntimeError(
                        f"method eval worker failed for {method} on {device} with returncode={returncode}\n"
                        f"stdout tail:\n{_tail_text(stdout_path)}\n"
                        f"stderr tail:\n{_tail_text(stderr_path)}"
                    )
                with output_path.open("rb") as handle:
                    result_method, summary, trace = pickle.load(handle)
                summaries[result_method] = summary
                traces[result_method] = trace
                _write_progress(args, out_dir, "method_eval_completed", {"method": method, "device": device})
    return (
        {method: summaries[method] for method in method_order},
        {method: traces[method] for method in method_order},
    )


def _run_method_eval_worker(payload_path: Path, output_path: Path) -> None:
    with payload_path.open("rb") as handle:
        payload = pickle.load(handle)
    result = _evaluate_method_worker(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _tail_text(path: Path, *, limit: int = 4000) -> str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return f"<unable to read {path}: {exc}>"
    return data[-limit:].decode("utf-8", errors="replace")


def _evaluate_method_worker(payload: dict) -> tuple[str, dict, list[dict]]:
    torch.set_num_threads(1)
    method = str(payload["method"])
    device = _worker_device(str(payload["device"]))
    bundle = build_synthetic_domain(
        str(payload["domain"]),
        sample_count=int(payload["sample_count"]),
        script_len=int(payload["script_len"]),
        vocab_size=int(payload["vocab_size"]),
    )
    model = CausalTransformerLM(
        vocab_size=len(bundle.vocab),
        config=payload["model_config"],
        max_len=int(payload["model_max_len"]),
        pad_id=bundle.oracle.pad_id,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if model.training:
        raise RuntimeError(f"{method} method eval must not run with model.training=True")

    kappa_eval = payload["kappa_eval"]
    analysis_ids = list(payload["analysis_ids"])
    states = [score.state_id for score in kappa_eval.scores]
    bridge_positions = positions_from_state_ids(kappa_eval.kappa_split.bridge)
    garden_positions = positions_from_state_ids(kappa_eval.kappa_split.garden)
    conf_bridge_positions = positions_from_state_ids(kappa_eval.confidence_split.bridge)
    conf_garden_positions = positions_from_state_ids(kappa_eval.confidence_split.garden)
    method_loss = NeuralStudentLoss(
        model,
        vocab_size=len(bundle.vocab),
        violation_penalty=float(payload["violation_penalty"]),
        device=device,
    )
    if method == "ce_ref":
        method_kappa_eval = kappa_eval
    else:
        method_kappa_eval = evaluate_oracle_states(
            bundle.oracle,
            method_loss,
            states,
            continuation=str(payload["kappa_continuation"]),
        )
    ref_regions = summarize_oracle_eb_by_positions(
        oracle=bundle.oracle,
        model=model,
        vocab_size=len(bundle.vocab),
        sample_ids=analysis_ids,
        bridge_positions=bridge_positions,
        garden_positions=garden_positions,
        rollout_repeats=int(payload["rollout_repeats"]),
        rollout_seed=int(payload["rollout_seed"]),
        device=device,
    )
    conf_regions = summarize_oracle_eb_by_positions(
        oracle=bundle.oracle,
        model=model,
        vocab_size=len(bundle.vocab),
        sample_ids=analysis_ids,
        bridge_positions=conf_bridge_positions,
        garden_positions=conf_garden_positions,
        rollout_repeats=int(payload["rollout_repeats"]),
        rollout_seed=int(payload["rollout_seed"]),
        device=device,
    )
    local_regions = summarize_oracle_local_intervention_eb(
        oracle=bundle.oracle,
        model=model,
        vocab_size=len(bundle.vocab),
        states=states,
        bridge_states=kappa_eval.kappa_split.bridge,
        garden_states=kappa_eval.kappa_split.garden,
        device=device,
    )
    conf_local_regions = summarize_oracle_local_intervention_eb(
        oracle=bundle.oracle,
        model=model,
        vocab_size=len(bundle.vocab),
        states=states,
        bridge_states=kappa_eval.confidence_split.bridge,
        garden_states=kappa_eval.confidence_split.garden,
        device=device,
    )
    kappa_contrib = summarize_oracle_kappa_contribution(
        oracle=bundle.oracle,
        model=model,
        vocab_size=len(bundle.vocab),
        kappa_results=kappa_eval.kappa_results,
        bridge_states=kappa_eval.kappa_split.bridge,
        garden_states=kappa_eval.kappa_split.garden,
        device=device,
    )
    conf_kappa_contrib = summarize_oracle_kappa_contribution(
        oracle=bundle.oracle,
        model=model,
        vocab_size=len(bundle.vocab),
        kappa_results=kappa_eval.kappa_results,
        bridge_states=kappa_eval.confidence_split.bridge,
        garden_states=kappa_eval.confidence_split.garden,
        device=device,
    )
    expected_intervention = summarize_oracle_expected_intervention(
        model=model,
        kappa_results=method_kappa_eval.kappa_results,
        bridge_states=kappa_eval.kappa_split.bridge,
        garden_states=kappa_eval.kappa_split.garden,
        device=device,
    )
    conf_expected_intervention = summarize_oracle_expected_intervention(
        model=model,
        kappa_results=method_kappa_eval.kappa_results,
        bridge_states=kappa_eval.confidence_split.bridge,
        garden_states=kappa_eval.confidence_split.garden,
        device=device,
    )
    clean_state_kl = _clean_state_kl_summary(
        bundle.oracle,
        method_loss,
        states,
        kappa_eval.kappa_split.bridge,
        kappa_eval.kappa_split.garden,
    )
    conf_clean_state_kl = _clean_state_kl_summary(
        bundle.oracle,
        method_loss,
        states,
        kappa_eval.confidence_split.bridge,
        kappa_eval.confidence_split.garden,
    )
    history = payload["history"]
    summary = {
        "train_loss": history["train_loss"],
        "learning_rate": history.get("learning_rate"),
        "lr_schedule": history.get("lr_schedule"),
        "lr_peak": history.get("lr_peak"),
        "lr_warmup_epochs": history.get("lr_warmup_epochs"),
        "val_teacher_forced_kl": history.get("val_teacher_forced_kl"),
        "val_method_loss": history.get("val_method_loss"),
        "val_early_stopping_objective": history.get("val_early_stopping_objective"),
        "early_stopping_metric": history.get("early_stopping_metric"),
        "early_stopping_relative_min_delta": history.get("early_stopping_relative_min_delta"),
        "early_stopping_patience": history.get("early_stopping_patience"),
        "best_epoch": history.get("best_epoch"),
        "best_val_early_stopping_objective": history.get("best_val_early_stopping_objective"),
        "best_teacher_forced_kl_epoch": history.get("best_teacher_forced_kl_epoch"),
        "best_val_teacher_forced_kl": history.get("best_val_teacher_forced_kl"),
        "method_confidence_rho": _finite(method_kappa_eval.confidence_rho),
        "ref_split_clean_state_kl": clean_state_kl,
        "confidence_split_clean_state_kl": conf_clean_state_kl,
        "ref_split_expected_intervention": _expected_intervention_summary(expected_intervention),
        "confidence_split_expected_intervention": _expected_intervention_summary(conf_expected_intervention),
        "ref_split_clean_state_total_risk": _clean_total_risk_summary(clean_state_kl, expected_intervention),
        "confidence_split_clean_state_total_risk": _clean_total_risk_summary(conf_clean_state_kl, conf_expected_intervention),
        "ref_split_kappa_contribution": _kappa_contribution_summary(kappa_contrib),
        "confidence_split_kappa_contribution": _kappa_contribution_summary(conf_kappa_contrib),
        "overall_position_rollout": _region_summary(ref_regions),
        "ref_split_position_rollout": _region_summary(ref_regions),
        "confidence_split_position_rollout": _region_summary(conf_regions),
        "sampled_rollout_status_metric": summarize_oracle_sampled_rollout_statuses(
            oracle=bundle.oracle,
            model=model,
            sample_ids=analysis_ids,
            rollout_repeats=int(payload["rollout_repeats"]),
            rollout_seed=int(payload["rollout_seed"]),
            device=device,
        ),
        "ref_split_local_intervention_metric": _local_region_summary(local_regions),
        "confidence_split_local_intervention_metric": _local_region_summary(conf_local_regions),
    }
    trace = _tokenize_trace(
        trace_oracle_rollouts(
            oracle=bundle.oracle,
            model=model,
            sample_ids=analysis_ids,
            device=device,
        ),
        bundle.vocab,
    )
    return method, summary, trace


def _worker_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"{name} requested but CUDA is unavailable")
        index = 0 if device.index is None else int(device.index)
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"{name} requested but only {torch.cuda.device_count()} CUDA devices are visible")
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    return device


def _kappa_spread_summary(kappa_eval) -> dict:
    values = [float(score.kappa) for score in kappa_eval.scores if math.isfinite(float(score.kappa))]
    bridge_set = set(kappa_eval.kappa_split.bridge)
    garden_set = set(kappa_eval.kappa_split.garden)
    bridge_values = [
        float(score.kappa) for score in kappa_eval.scores
        if score.state_id in bridge_set and math.isfinite(float(score.kappa))
    ]
    garden_values = [
        float(score.kappa) for score in kappa_eval.scores
        if score.state_id in garden_set and math.isfinite(float(score.kappa))
    ]

    def _mean(items: list[float]) -> float | None:
        return _finite(sum(items) / len(items)) if items else None

    mean = _mean(values)
    if values and mean is not None:
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        cv = _finite(std / abs(mean)) if abs(mean) > 1e-12 else None
    else:
        cv = None
    bridge_mean = _mean(bridge_values)
    garden_mean = _mean(garden_values)
    if bridge_mean is not None and garden_mean is not None and abs(garden_mean) > 1e-12:
        top_bottom_ratio = _finite(bridge_mean / garden_mean)
    else:
        top_bottom_ratio = None
    return {
        "state_count": len(values),
        "bridge_mean": bridge_mean,
        "garden_mean": garden_mean,
        "top_bottom_ratio": top_bottom_ratio,
        "coefficient_of_variation": cv,
    }


def _training_metrics(
    method_summaries: dict,
    hybrid_candidates: dict,
    selected_lambda: float | None,
    *,
    selection_policy: str,
) -> dict:
    hard_best = _history_best_val(method_summaries.get("hard_kd", {}))
    soft_best = _history_best_val(method_summaries.get("soft_kd", {}))
    hard_soft_ratio = None
    if hard_best is not None and soft_best is not None and abs(soft_best) > 1e-12:
        hard_soft_ratio = _finite(hard_best / soft_best)
    hybrid_values = {
        float(lam): float(item[0])
        for lam, item in hybrid_candidates.items()
        if item and item[0] is not None and math.isfinite(float(item[0]))
    }
    sorted_hybrid = sorted(hybrid_values.items())
    hybrid_monotone_decreasing = (
        len(sorted_hybrid) >= 2
        and all(right <= left for (_, left), (_, right) in zip(sorted_hybrid, sorted_hybrid[1:]))
    )
    selected_at_grid_edge = False
    if selected_lambda is not None and sorted_hybrid:
        lambdas = [lam for lam, _ in sorted_hybrid]
        selected_at_grid_edge = abs(float(selected_lambda) - min(lambdas)) <= 1e-12 or abs(float(selected_lambda) - max(lambdas)) <= 1e-12
    warnings = []
    if hard_soft_ratio is not None and hard_soft_ratio > 5.0:
        warnings.append("hard_kd_best_val_kl_exceeds_soft_kd_by_more_than_5x")
    if (
        selection_policy == "min_validation_teacher_forced_kl"
        and hybrid_monotone_decreasing
        and selected_at_grid_edge
    ):
        warnings.append("hybrid_validation_kl_is_monotone_and_selected_lambda_is_on_grid_edge")
    tail = _history_tail_metric(method_summaries.get("hard_kd", {}))
    if tail.get("best_epoch_at_tail"):
        warnings.append("hard_kd_best_validation_epoch_is_at_training_tail")
    return {
        "best_val_teacher_forced_kl": {
            "hard_kd": hard_best,
            "soft_kd": soft_best,
            "hybrid_kd": _history_best_val(method_summaries.get("hybrid_kd", {})),
        },
        "hard_to_soft_best_val_kl_ratio": hard_soft_ratio,
        "hybrid_validation_monotone_decreasing": hybrid_monotone_decreasing,
        "hybrid_selected_lambda_at_grid_edge": selected_at_grid_edge,
        "hybrid_lambda_selection_policy": selection_policy,
        "hard_kd_tail": tail,
        "warnings": warnings,
    }


def _select_one_se_lowest_lambda(
    rollout_scores: dict[float, float],
    rollout_score_ses: dict[float, float],
) -> float:
    if not rollout_scores:
        raise ValueError("no rollout scores available")
    best_lambda = min(rollout_scores, key=lambda lam: rollout_scores[lam])
    threshold = float(rollout_scores[best_lambda]) + max(0.0, float(rollout_score_ses.get(best_lambda, 0.0)))
    eligible = [
        float(lam)
        for lam, score in rollout_scores.items()
        if float(score) <= threshold
    ]
    if not eligible:
        return float(best_lambda)
    return min(eligible)


def _history_best_val(summary: dict) -> float | None:
    values = summary.get("best_val_teacher_forced_kl")
    if isinstance(values, list) and values:
        value = float(values[0])
        return _finite(value) if math.isfinite(value) else None
    values = summary.get("val_teacher_forced_kl")
    if isinstance(values, list) and values:
        finite = [float(value) for value in values if math.isfinite(float(value))]
        return _finite(min(finite)) if finite else None
    return None


def _history_tail_metric(summary: dict) -> dict:
    values = summary.get("val_teacher_forced_kl")
    best_epochs = summary.get("best_teacher_forced_kl_epoch")
    if not isinstance(values, list) or not values:
        return {
            "num_val_epochs": 0,
            "best_epoch": None,
            "last_val_teacher_forced_kl": None,
            "best_val_teacher_forced_kl": None,
            "last_to_best_ratio": None,
            "best_epoch_at_tail": False,
        }
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    best_value = min(finite_values) if finite_values else None
    last_value = float(values[-1]) if math.isfinite(float(values[-1])) else None
    best_epoch = int(float(best_epochs[0])) if isinstance(best_epochs, list) and best_epochs else None
    last_to_best_ratio = None
    if best_value is not None and last_value is not None and abs(best_value) > 1e-12:
        last_to_best_ratio = _finite(last_value / best_value)
    return {
        "num_val_epochs": len(values),
        "best_epoch": best_epoch,
        "last_val_teacher_forced_kl": _finite(last_value) if last_value is not None else None,
        "best_val_teacher_forced_kl": _finite(best_value) if best_value is not None else None,
        "last_to_best_ratio": last_to_best_ratio,
        "best_epoch_at_tail": best_epoch is not None and best_epoch >= max(0, len(values) - 2),
    }


def _region_semantic_occupancy(table: list[dict]) -> dict:
    out: dict[str, dict[str, int]] = {"bridge_partition": {}, "garden_partition": {}, "bridge_confidence": {}, "garden_confidence": {}}
    for row in table:
        tag = str(row.get("semantic_tag"))
        for key in out:
            if bool(row.get(key)):
                out[key][tag] = out[key].get(tag, 0) + 1
    return out


def _enforce_paper_config(args: argparse.Namespace) -> None:
    if not args.enforce_paper_config:
        return
    errors = []
    setup_name = getattr(args, "paper_config", "paper")
    setup = SYNTHETIC_PAPER_CONFIGS[setup_name]
    expected_train_count = int(setup["negative_train_count"] if args.domain == "negative_control" else setup["main_train_count"])
    expected_val_count = int(setup["val_count"])
    expected_analysis_count = int(setup["analysis_count"])
    expected_sample_count = expected_train_count + expected_val_count + expected_analysis_count
    expected_script_len = DOMAIN_SCRIPT_LENS[args.domain]
    if args.vocab_size != 64:
        errors.append("--vocab-size must be 64")
    if args.script_len != expected_script_len:
        errors.append(f"--script-len must be {expected_script_len} for {args.domain}")
    if args.d_model != 128:
        errors.append("--d-model must be 128")
    if args.n_heads != 4:
        errors.append("--n-heads must be 4")
    if args.n_layers != 2:
        errors.append("--n-layers must be 2")
    if args.d_ff != 512:
        errors.append("--d-ff must be 512")
    if abs(float(args.dropout) - 0.1) > 1e-12:
        errors.append("--dropout must be 0.1")
    if args.kappa_continuation != "expectation":
        errors.append("--kappa-continuation must be expectation")
    if args.max_kappa_states != 0:
        errors.append("--max-kappa-states must be 0 so all non-terminal states are evaluated")
    if args.analysis_count != expected_analysis_count:
        errors.append(f"--analysis-count must be {expected_analysis_count} for {setup_name}")
    if args.train_count != expected_train_count:
        errors.append(f"--train-count must be {expected_train_count} for the paper configuration")
    if args.val_count != expected_val_count:
        errors.append(f"--val-count must be {expected_val_count} for the paper configuration")
    if args.sample_count != expected_sample_count:
        errors.append(f"--sample-count must be {expected_sample_count} for the paper configuration")
    if args.epochs != 24:
        errors.append("--epochs must be 24 as the max-epoch budget")
    if args.batch_size != 256:
        errors.append("--batch-size must be 256")
    if abs(float(args.lr) - float(setup["lr"])) > 1e-12:
        errors.append(f"--lr must be {setup['lr']} for {setup_name}")
    if args.lr_schedule != setup["lr_schedule"]:
        errors.append(f"--lr-schedule must be {setup['lr_schedule']} for {setup_name}")
    if args.lr_warmup_epochs != int(setup["lr_warmup_epochs"]):
        errors.append(f"--lr-warmup-epochs must be {setup['lr_warmup_epochs']} for {setup_name}")
    if args.early_stopping_patience != int(setup["early_stopping_patience"]):
        errors.append(f"--early-stopping-patience must be {setup['early_stopping_patience']} for {setup_name}")
    expected_hard_kd_lr = setup["hard_kd_lr"]
    actual_hard_kd_lr = getattr(args, "hard_kd_lr", None)
    if expected_hard_kd_lr is None:
        if actual_hard_kd_lr is not None:
            errors.append(f"--hard-kd-lr must be unset for {setup_name}")
    elif actual_hard_kd_lr is None or abs(float(actual_hard_kd_lr) - float(expected_hard_kd_lr)) > 1e-12:
        errors.append(f"--hard-kd-lr must be {expected_hard_kd_lr} for {setup_name}")
    expected_hard_kd_patience = setup.get("hard_kd_early_stopping_patience")
    actual_hard_kd_patience = getattr(args, "hard_kd_early_stopping_patience", None)
    if expected_hard_kd_patience is None:
        if actual_hard_kd_patience is not None:
            errors.append(f"--hard-kd-early-stopping-patience must be unset for {setup_name}")
    elif actual_hard_kd_patience != int(expected_hard_kd_patience):
        errors.append(
            f"--hard-kd-early-stopping-patience must be {expected_hard_kd_patience} for {setup_name}"
        )
    expected_hard_kd_metric = setup.get("hard_kd_early_stopping_metric", "val_teacher_forced_kl")
    actual_hard_kd_metric = getattr(args, "hard_kd_early_stopping_metric", "val_teacher_forced_kl")
    if actual_hard_kd_metric != expected_hard_kd_metric:
        errors.append(f"--hard-kd-early-stopping-metric must be {expected_hard_kd_metric} for {setup_name}")
    if args.analysis_path_policy != "sample":
        errors.append("--analysis-path-policy must be sample")
    if args.train_path_policy != "sample":
        errors.append("--train-path-policy must be sample")
    if args.ref_only:
        errors.append("--ref-only is a metric check and cannot be used for paper configuration runs")
    if args.fixed_hybrid_lambda is not None:
        errors.append("--fixed-hybrid-lambda must be unset; the paper configuration selects lambda on validation data")
    expected_selection_policy = setup.get("hybrid_lambda_selection_policy", "min_validation_teacher_forced_kl")
    actual_selection_policy = getattr(args, "hybrid_lambda_selection_policy", "min_validation_teacher_forced_kl")
    if actual_selection_policy != expected_selection_policy:
        errors.append(f"--hybrid-lambda-selection-policy must be {expected_selection_policy} for {setup_name}")
    expected_hybrid_training_policy = setup.get("hybrid_training_policy", "ambiguity_scaled")
    if args.hybrid_training_policy != expected_hybrid_training_policy:
        errors.append(
            f"--hybrid-training-policy must be {expected_hybrid_training_policy} for {setup_name}"
        )
    lambda_grid = [float(x) for x in args.hybrid_lambdas.split(",") if x.strip()]
    if len(lambda_grid) != len(SYNTHETIC_HYBRID_LAMBDAS) or any(
        abs(left - right) > 1e-12 for left, right in zip(lambda_grid, SYNTHETIC_HYBRID_LAMBDAS)
    ):
        errors.append("--hybrid-lambdas must be 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    if args.hybrid_selection_count != expected_val_count:
        errors.append(f"--hybrid-selection-count must be {expected_val_count} to select lambda on the full validation split")
    if args.rollout_repeats < int(setup["rollout_repeats_min"]):
        errors.append(f"--rollout-repeats must be at least {setup['rollout_repeats_min']} for {setup_name}")
    if errors:
        joined = "\n  - ".join(errors)
        raise SystemExit(f"--enforce-paper-config rejected this metric setup:\n  - {joined}")


def _output_dir(root: Path, domain: str, seed_dir_name: str | None) -> Path:
    if seed_dir_name:
        return root / domain / seed_dir_name
    return root / domain


def _region_summary(region) -> dict:
    return {
        "overall_eb": _finite(region.overall.exposure_bias),
        "overall_eb_se": _finite(region.overall.exposure_bias_se),
        "bridge_eb": _finite(region.bridge_eb),
        "bridge_eb_se": _finite(region.bridge_eb_se),
        "garden_eb": _finite(region.garden_eb),
        "garden_eb_se": _finite(region.garden_eb_se),
        "overall_tf_kl": _finite(region.overall.teacher_forced_kl),
        "overall_rollout_kl": _finite(region.overall.rollout_kl),
        "overall_rollout_kl_se": _finite(region.overall.rollout_kl_se),
        "rollout_repeats": region.overall.rollout_repeats,
    }


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is false")
    return torch.device(name)


def _best_val_teacher_forced_kl(history: dict) -> float:
    values = history.get("best_val_teacher_forced_kl") or history.get("val_teacher_forced_kl") or []
    if not values:
        raise RuntimeError("hybrid lambda selection requires validation teacher-forced KL history")
    return float(min(float(value) for value in values))


def _clean_state_kl_summary(oracle, loss: NeuralStudentLoss, states: list, bridge_states, garden_states) -> dict:
    bridge_set = set(bridge_states)
    garden_set = set(garden_states)
    rows = []
    for state in states:
        if oracle.is_terminal(state):
            continue
        rows.append((state, float(loss.loss_at_state(state, oracle))))

    def _mean(selected) -> float:
        values = [value for state, value in rows if state in selected]
        return _finite(sum(values) / len(values)) if values else None

    overall_values = [value for _, value in rows]
    return {
        "overall_kl": _finite(sum(overall_values) / len(overall_values)) if overall_values else None,
        "bridge_kl": _mean(bridge_set),
        "garden_kl": _mean(garden_set),
        "overall_states": len(rows),
        "bridge_states": len([state for state, _ in rows if state in bridge_set]),
        "garden_states": len([state for state, _ in rows if state in garden_set]),
    }


def _clean_total_risk_summary(clean_kl: dict, expected_intervention) -> dict:
    def _total(clean_value, intervention_value):
        if clean_value is None:
            return None
        return _finite(clean_value + intervention_value)

    return {
        "overall_total": _total(clean_kl["overall_kl"], expected_intervention.overall.expected_eb),
        "bridge_total": _total(clean_kl["bridge_kl"], expected_intervention.bridge.expected_eb),
        "garden_total": _total(clean_kl["garden_kl"], expected_intervention.garden.expected_eb),
        "overall_clean_kl": clean_kl["overall_kl"],
        "bridge_clean_kl": clean_kl["bridge_kl"],
        "garden_clean_kl": clean_kl["garden_kl"],
        "overall_expected_intervention": _finite(expected_intervention.overall.expected_eb),
        "bridge_expected_intervention": _finite(expected_intervention.bridge.expected_eb),
        "garden_expected_intervention": _finite(expected_intervention.garden.expected_eb),
    }


def _position_split_metrics(bridge_states, garden_states) -> dict:
    bridge_set = set(bridge_states)
    garden_set = set(garden_states)
    bridge_positions = positions_from_state_ids(bridge_set)
    garden_positions = positions_from_state_ids(garden_set)
    overlap = bridge_positions & garden_positions
    return {
        "bridge_state_count": len(bridge_set),
        "garden_state_count": len(garden_set),
        "state_overlap_count": len(bridge_set & garden_set),
        "bridge_position_count": len(bridge_positions),
        "garden_position_count": len(garden_positions),
        "position_overlap_count": len(overlap),
        "position_overlap_fraction_of_bridge": _finite(len(overlap) / len(bridge_positions)) if bridge_positions else None,
        "position_overlap_fraction_of_garden": _finite(len(overlap) / len(garden_positions)) if garden_positions else None,
        "position_rollout_regions_are_disjoint": len(overlap) == 0,
    }


def _distribution_metrics(bundle, kappa_eval, models: dict[str, torch.nn.Module]) -> dict:
    regions = {
        "overall": [score.state_id for score in kappa_eval.scores],
        "bridge": list(kappa_eval.kappa_split.bridge),
        "garden": list(kappa_eval.kappa_split.garden),
    }
    out = {"teacher": {}, "students": {}}
    for region_name, states in regions.items():
        out["teacher"][region_name] = _teacher_distribution_summary(bundle.oracle, states)
    for method, model in models.items():
        out["students"][method] = {
            region_name: _student_distribution_summary(bundle.oracle, model, states)
            for region_name, states in regions.items()
        }
    return out


def _teacher_distribution_summary(oracle, states: list) -> dict:
    rows = []
    for state in states:
        dist = oracle.next_dist(state)
        if not dist:
            continue
        probs = [float(p) for p in dist.values()]
        top1 = max(probs)
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        rows.append({
            "top1": top1,
            "entropy": entropy,
            "support_size": len(probs),
            "nonmode_support_mass": 1.0 - top1,
        })
    return _mean_row(rows)


def _student_distribution_summary(oracle, model: torch.nn.Module, states: list) -> dict:
    rows = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for state in states:
            dist = oracle.next_dist(state)
            if not dist:
                continue
            prefix = tuple(int(x) for x in getattr(state, "prefix"))
            input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
            logits = model(input_ids)[:, -1, :]
            probs = F.softmax(logits, dim=-1)[0].detach().cpu()
            support = [int(token_id) for token_id in dist]
            mode = max(dist, key=lambda token_id: (dist[token_id], -int(token_id)))
            support_mass = float(probs[support].sum())
            mode_prob = float(probs[int(mode)])
            rows.append({
                "teacher_support_mass": support_mass,
                "teacher_mode_prob": mode_prob,
                "teacher_nonmode_support_mass": support_mass - mode_prob,
                "off_teacher_support_mass": 1.0 - support_mass,
                "student_top1": float(probs.max()),
            })
    return _mean_row(rows)


def _mean_row(rows: list[dict]) -> dict:
    if not rows:
        return {"states": 0}
    keys = rows[0].keys()
    return {
        "states": len(rows),
        **{key: _finite(sum(float(row[key]) for row in rows) / len(rows)) for key in keys},
    }


def _local_region_summary(region) -> dict:
    return {
        "metric": True,
        "loss_to_go_continuation": "mode",
        "overall_eb": _finite(region.overall.exposure_bias),
        "bridge_eb": _finite(region.bridge.exposure_bias),
        "garden_eb": _finite(region.garden.exposure_bias),
        "overall_teacher_mode_loss": _finite(region.overall.teacher_mode_loss),
        "overall_student_action_loss": _finite(region.overall.student_action_loss),
        "bridge_states": region.bridge.states,
        "garden_states": region.garden.states,
        "bridge_student_in_teacher_support_rate": _finite(region.bridge.student_in_teacher_support_rate),
        "garden_student_in_teacher_support_rate": _finite(region.garden.student_in_teacher_support_rate),
        "bridge_student_violation_rate": _finite(region.bridge.student_violation_rate),
        "garden_student_violation_rate": _finite(region.garden.student_violation_rate),
    }


def _kappa_contribution_summary(region) -> dict:
    return {
        "overall_contribution": _finite(region.overall.contribution),
        "bridge_contribution": _finite(region.bridge.contribution),
        "garden_contribution": _finite(region.garden.contribution),
        "overall_absolute_contribution": _finite(region.overall.absolute_contribution),
        "bridge_absolute_contribution": _finite(region.bridge.absolute_contribution),
        "garden_absolute_contribution": _finite(region.garden.absolute_contribution),
        "overall_l1_delta": _finite(region.overall.mean_l1_delta),
        "bridge_l1_delta": _finite(region.bridge.mean_l1_delta),
        "garden_l1_delta": _finite(region.garden.mean_l1_delta),
        "overall_teacher_kl": _finite(region.overall.mean_teacher_kl),
        "bridge_teacher_kl": _finite(region.bridge.mean_teacher_kl),
        "garden_teacher_kl": _finite(region.garden.mean_teacher_kl),
        "bridge_states": region.bridge.states,
        "garden_states": region.garden.states,
    }


def _expected_intervention_summary(region) -> dict:
    return {
        "overall_eb": _finite(region.overall.expected_eb),
        "bridge_eb": _finite(region.bridge.expected_eb),
        "garden_eb": _finite(region.garden.expected_eb),
        "overall_teacher_expected_loss": _finite(region.overall.teacher_expected_loss),
        "overall_student_expected_loss": _finite(region.overall.student_expected_loss),
        "bridge_states": region.bridge.states,
        "garden_states": region.garden.states,
        "overall_student_eval_mass": _finite(region.overall.mean_student_eval_mass),
        "bridge_student_eval_mass": _finite(region.bridge.mean_student_eval_mass),
        "garden_student_eval_mass": _finite(region.garden.mean_student_eval_mass),
    }


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _heatmap_sample_ids(all_analysis_states: list, kappa_eval, num_heatmaps: int) -> list[int]:
    if int(num_heatmaps) <= 0:
        return []
    ids = []
    kappa_samples = {
        int(getattr(score.state_id, "sample_id", -1))
        for score in kappa_eval.scores
    }
    for state in all_analysis_states:
        sample_id = int(getattr(state, "sample_id", -1))
        if sample_id in kappa_samples and sample_id not in ids:
            ids.append(sample_id)
        if len(ids) >= max(0, int(num_heatmaps)):
            break
    return ids


def _heatmap_rows(bundle, kappa_eval, path_states: list):
    state_to_kappa = {score.state_id: score.kappa for score in kappa_eval.scores}
    state_to_region = _region_labels(kappa_eval)
    realized_tokens = _realized_token_map(bundle, path_states)
    tokens = []
    tags = []
    kappas = []
    region_labels = []
    for state in path_states:
        if bundle.oracle.is_terminal(state):
            continue
        dist = bundle.oracle.next_dist(state)
        if not dist:
            continue
        if state not in state_to_kappa:
            raise RuntimeError(f"missing kappa row for heatmap state sample={getattr(state, 'sample_id', None)} position={getattr(state, 'position', None)}")
        action, _source = realized_tokens[state]
        tokens.append(bundle.vocab[action])
        tags.append(bundle.oracle.semantic_tag(state))
        kappas.append(state_to_kappa[state])
        region_labels.append(state_to_region.get(state, ""))
    return tokens, tags, kappas, region_labels


def _state_table(bundle, kappa_eval, all_analysis_states: list) -> list[dict]:
    bridge = set(kappa_eval.kappa_split.bridge)
    garden = set(kappa_eval.kappa_split.garden)
    conf_bridge = set(kappa_eval.confidence_split.bridge)
    conf_garden = set(kappa_eval.confidence_split.garden)
    conf_bridge_tie_band = set(kappa_eval.confidence_split.bridge_tie_band)
    conf_garden_tie_band = set(kappa_eval.confidence_split.garden_tie_band)
    realized_tokens = _realized_token_map(bundle, all_analysis_states)
    rows = []
    for score in sorted(kappa_eval.scores, key=lambda item: (getattr(item.state_id, "sample_id", 0), getattr(item.state_id, "position", 0))):
        state = score.state_id
        prefix_ids = tuple(getattr(state, "prefix", ()))
        dist = bundle.oracle.next_dist(state)
        top_next = sorted(dist.items(), key=lambda item: (-float(item[1]), int(item[0])))[:5]
        visible_token_id, visible_token_source = realized_tokens[state]
        rows.append({
            "sample_id": getattr(state, "sample_id", None),
            "position": getattr(state, "position", None),
            "status": str(getattr(state, "status", "")),
            "style": int(getattr(state, "style", 0)),
            "prefix_tail": [bundle.vocab[token_id] for token_id in prefix_ids[-10:]],
            "visible_token": bundle.vocab[int(visible_token_id)],
            "visible_token_id": int(visible_token_id),
            "visible_token_source": visible_token_source,
            "next_dist_top": [
                {"token": bundle.vocab[int(token_id)], "prob": float(prob)}
                for token_id, prob in top_next
            ],
            "expected_role": bundle.oracle.expected_role(state),
            "semantic_tag": score.semantic_tag,
            "kappa": score.kappa,
            "confidence": score.confidence,
            "bridge_partition": state in bridge,
            "garden_partition": state in garden,
            "bridge_confidence": state in conf_bridge,
            "garden_confidence": state in conf_garden,
            "bridge_confidence_tie_band": state in conf_bridge_tie_band,
            "garden_confidence_tie_band": state in conf_garden_tie_band,
        })
    return rows


def _realized_token_map(bundle, path_states: list) -> dict:
    realized = {}
    for idx, state in enumerate(path_states):
        if bundle.oracle.is_terminal(state):
            continue
        dist = bundle.oracle.next_dist(state)
        if not dist:
            continue
        next_state = path_states[idx + 1] if idx + 1 < len(path_states) else None
        if (
            next_state is not None
            and int(getattr(next_state, "sample_id", -2)) == int(getattr(state, "sample_id", -1))
            and len(tuple(getattr(next_state, "prefix", ()))) > len(tuple(getattr(state, "prefix", ())))
        ):
            realized[state] = (int(tuple(getattr(next_state, "prefix"))[-1]), "sampled_path")
        else:
            realized[state] = (int(max(dist, key=lambda token_id: (dist[token_id], -int(token_id)))), "teacher_top")
    return realized


def _scatter_rows(kappa_eval) -> list[dict]:
    state_to_region = _region_labels(kappa_eval)
    rows = []
    for score in kappa_eval.scores:
        rows.append({
            "kappa": score.kappa,
            "confidence": score.confidence,
            "semantic_tag": score.semantic_tag,
            "region_label": state_to_region.get(score.state_id, ""),
        })
    return rows


def _region_labels(kappa_eval) -> dict:
    labels = {}
    for state in kappa_eval.kappa_split.bridge:
        labels[state] = "Bκ"
    for state in kappa_eval.kappa_split.garden:
        labels[state] = "Gκ"
    for state in kappa_eval.confidence_split.bridge:
        labels[state] = f"{labels.get(state, '')}/Bc".strip("/")
    for state in kappa_eval.confidence_split.garden:
        labels[state] = f"{labels.get(state, '')}/Gc".strip("/")
    for state in kappa_eval.confidence_split.bridge_tie_band:
        if state not in kappa_eval.confidence_split.bridge:
            labels[state] = f"{labels.get(state, '')}/BcTie".strip("/")
    for state in kappa_eval.confidence_split.garden_tie_band:
        if state not in kappa_eval.confidence_split.garden:
            labels[state] = f"{labels.get(state, '')}/GcTie".strip("/")
    return labels


def _tokenize_trace(rows: list[dict], vocab: tuple[str, ...]) -> list[dict]:
    converted = []
    for row in rows:
        item = dict(row)
        item["teacher_token"] = vocab[int(row["teacher_action"])]
        item["student_token"] = vocab[int(row["student_action"])]
        item["realized_student_token"] = vocab[int(row["realized_student_action"])]
        converted.append(item)
    return converted


if __name__ == "__main__":
    main()
