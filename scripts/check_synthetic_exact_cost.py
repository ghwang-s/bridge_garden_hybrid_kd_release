#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridge_garden_v2.synthetic_domains import DOMAIN_SCRIPT_LENS, build_synthetic_domain
from bridge_garden_v2.synthetic_evaluator import collect_sampled_path_states, estimate_teacher_support_size
from bridge_garden_v2.exact_oracle import exact_kappa_for_state
from bridge_garden_v2.modeling import CausalTransformerLM
from bridge_garden_v2.oracle_student import NeuralStudentLoss
from bridge_garden_v2.schema import ModelConfig


SYNTHETIC_DOMAINS = ["code", "math", "dialogue", "negative_control"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=SYNTHETIC_DOMAINS, default="code")
    parser.add_argument("--analysis-count", type=int, default=100)
    parser.add_argument("--analysis-sample-seed", type=int, default=0)
    parser.add_argument("--script-len", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--max-states", type=int, default=0, help="0 means all non-terminal prefix states")
    parser.add_argument("--support-cap", type=int, default=100_000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--violation-penalty", type=float, default=2.0)
    parser.add_argument("--max-batch-rows", type=int, default=8192)
    parser.add_argument("--enforce-paper-config", action="store_true")
    parser.add_argument("--out", default="bridge_garden_v2/synthetic_exact_cost_check.json")
    args = parser.parse_args()
    if args.script_len is None:
        args.script_len = DOMAIN_SCRIPT_LENS[args.domain]
    _enforce_paper_config(args)

    device = _resolve_device(args.device)
    bundle = build_synthetic_domain(
        args.domain,
        sample_count=args.analysis_count,
        script_len=args.script_len,
        vocab_size=args.vocab_size,
    )
    analysis_ids = list(range(args.analysis_count))
    states = collect_sampled_path_states(bundle.oracle, analysis_ids, seed=args.analysis_sample_seed)
    all_kappa_state_count = len(states)
    if args.max_states > 0:
        states = states[: args.max_states]

    model = CausalTransformerLM(
        vocab_size=len(bundle.vocab),
        config=ModelConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
        ),
        max_len=args.script_len + 1,
        pad_id=bundle.oracle.pad_id,
    )
    loss = NeuralStudentLoss(
        model,
        vocab_size=len(bundle.vocab),
        violation_penalty=args.violation_penalty,
        device=device,
        max_batch_rows=args.max_batch_rows,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    rows = []
    started = time.perf_counter()
    base_report = {
        "domain": args.domain,
        "analysis_count": args.analysis_count,
        "analysis_sample_seed": args.analysis_sample_seed,
        "script_len": args.script_len,
        "vocab_size": args.vocab_size,
        "device": str(device),
        "model_config": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
            "max_batch_rows": args.max_batch_rows,
        },
        "kappa_definition": {
            "kappa_continuation": "expectation",
            "max_states": args.max_states,
            "all_kappa_state_count": all_kappa_state_count,
            "evaluated_state_count": len(states),
            "eval_token_count": len(bundle.oracle.eval_token_ids),
            "eos_in_eval": bundle.oracle.eos_id in bundle.oracle.eval_token_ids,
            "pad_in_eval": bundle.oracle.pad_id in bundle.oracle.eval_token_ids,
            "bos_in_eval": bundle.oracle.bos_id in bundle.oracle.eval_token_ids,
            "uses_global_eval_token_ids": True,
            "uses_state_local_eval_token_ids": False,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for idx, state in enumerate(states):
        before = time.perf_counter()
        cache_before = len(loss._batch_loss_cache)
        calls_before = loss.batch_forward_calls
        rows_before = loss.batch_forward_rows
        support_size = estimate_teacher_support_size(bundle.oracle, state, cap=args.support_cap)
        result = exact_kappa_for_state(
            bundle.oracle,
            loss,
            state,
            continuation="expectation",
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - before
        rows.append({
            "index": idx,
            "sample_id": int(getattr(state, "sample_id")),
            "position": int(getattr(state, "position")),
            "semantic_tag": bundle.oracle.semantic_tag(state),
            "expected_role": bundle.oracle.expected_role(state),
            "teacher_support_paths": support_size,
            "eval_token_count": len(result.action_ids),
            "seconds": elapsed,
            "new_loss_cache_entries": len(loss._batch_loss_cache) - cache_before,
            "new_batch_forward_calls": loss.batch_forward_calls - calls_before,
            "new_batch_forward_rows": loss.batch_forward_rows - rows_before,
            "kappa": float(result.state_kappa),
            "teacher_confidence": float(result.teacher_confidence),
            "q_min": min(result.q_values),
            "q_max": max(result.q_values),
            "qbar": float(result.qbar),
            "frontier_stats": dict(result.computation_stats or {}),
        })
        _write_report(
            out_path,
            base_report=base_report,
            status="running",
            started=started,
            rows=rows,
            loss=loss,
            device=device,
        )
        print(json.dumps(rows[-1]), flush=True)

    report = _write_report(
        out_path,
        base_report=base_report,
        status="completed",
        started=started,
        rows=rows,
        loss=loss,
        device=device,
    )
    print(json.dumps(report, indent=2), flush=True)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is false")
    return torch.device(name)


def _enforce_paper_config(args: argparse.Namespace) -> None:
    if not args.enforce_paper_config:
        return
    errors = []
    if args.analysis_count != 100:
        errors.append("--analysis-count must be 100")
    if args.vocab_size != 64:
        errors.append("--vocab-size must be 64")
    if args.script_len != DOMAIN_SCRIPT_LENS[args.domain]:
        errors.append(f"--script-len must be {DOMAIN_SCRIPT_LENS[args.domain]} for {args.domain}")
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
    if args.max_states != 0:
        errors.append("--max-states must be 0 so all non-terminal prefix states are evaluated")
    if args.device != "cuda":
        errors.append("--device must be cuda for GPU cost check")
    if errors:
        joined = "\n  - ".join(errors)
        raise SystemExit(f"--enforce-paper-config rejected this metric exact-cost check:\n  - {joined}")


def _cuda_stats(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"available": False}
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(device),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _write_report(
    path: Path,
    *,
    base_report: dict,
    status: str,
    started: float,
    rows: list[dict],
    loss: NeuralStudentLoss,
    device: torch.device,
) -> dict:
    total_seconds = time.perf_counter() - started
    all_kappa = int(base_report["kappa_definition"]["all_kappa_state_count"])
    report = {
        **base_report,
        "status": status,
        "timing": {
            "total_seconds": total_seconds,
            "seconds_per_state": total_seconds / len(rows) if rows else None,
            "projected_seconds_all_kappa_states": (
                total_seconds * all_kappa / len(rows)
                if rows else None
            ),
        },
        "student_loss_stats": {
            "cache_entries": len(loss._batch_loss_cache),
            "batch_forward_calls": loss.batch_forward_calls,
            "batch_forward_rows": loss.batch_forward_rows,
            "cache_miss_entries": loss.cache_miss_entries,
        },
        "cuda": _cuda_stats(device),
        "rows": rows,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
