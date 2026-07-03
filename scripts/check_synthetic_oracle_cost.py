#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Hashable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridge_garden_v2.synthetic_domains import DOMAIN_SCRIPT_LENS, build_synthetic_domain
from bridge_garden_v2.synthetic_evaluator import collect_sampled_path_states, estimate_teacher_support_size


SYNTHETIC_DOMAINS = ["code", "math", "dialogue", "negative_control"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=SYNTHETIC_DOMAINS, default="code")
    parser.add_argument("--analysis-count", type=int, default=100)
    parser.add_argument("--analysis-sample-seed", type=int, default=0)
    parser.add_argument("--script-len", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--max-states", type=int, default=0, help="0 means all non-terminal analysis states")
    parser.add_argument("--support-cap", type=int, default=100_000)
    parser.add_argument("--enforce-paper-config", action="store_true")
    parser.add_argument("--out", default="bridge_garden_v2/synthetic_oracle_cost_check.json")
    args = parser.parse_args()
    if args.script_len is None:
        args.script_len = DOMAIN_SCRIPT_LENS[args.domain]
    _enforce_paper_config(args)

    bundle = build_synthetic_domain(
        args.domain,
        sample_count=args.analysis_count,
        script_len=args.script_len,
        vocab_size=args.vocab_size,
    )
    states = collect_sampled_path_states(
        bundle.oracle,
        list(range(args.analysis_count)),
        seed=args.analysis_sample_seed,
    )
    all_kappa_state_count = len(states)
    if args.max_states > 0:
        states = states[: args.max_states]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    base = {
        "domain": args.domain,
        "analysis_count": args.analysis_count,
        "analysis_sample_seed": args.analysis_sample_seed,
        "script_len": args.script_len,
        "vocab_size": args.vocab_size,
        "kappa_definition": {
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
    for idx, state in enumerate(states):
        before = time.perf_counter()
        row = _oracle_frontier_cost(
            bundle.oracle,
            state,
            eval_token_ids=bundle.oracle.eval_token_ids,
            support_cap=args.support_cap,
        )
        row.update({
            "index": idx,
            "sample_id": int(getattr(state, "sample_id")),
            "position": int(getattr(state, "position")),
            "semantic_tag": bundle.oracle.semantic_tag(state),
            "expected_role": bundle.oracle.expected_role(state),
            "seconds": time.perf_counter() - before,
        })
        rows.append(row)
        _write_report(out_path, base, "running", started, rows)
        print(json.dumps(row), flush=True)
    report = _write_report(out_path, base, "completed", started, rows)
    print(json.dumps(report, indent=2), flush=True)


def _oracle_frontier_cost(oracle, state: Hashable, *, eval_token_ids: tuple[int, ...], support_cap: int) -> dict:
    support_size = estimate_teacher_support_size(oracle, state, cap=support_cap)
    initial_states = [oracle.step(state, int(action_id)) for action_id in eval_token_ids]
    initial_unique = set(initial_states)
    initial_status = Counter(str(getattr(next_state, "status", "")) for next_state in initial_states)
    frontier = set(initial_unique)
    expanded_total = 0
    max_frontier = len(frontier)
    max_active = 0
    by_depth = []
    depth = 0
    while frontier:
        active = [
            current for current in frontier
            if not oracle.is_terminal(current) and oracle.remaining_horizon(current) > 0
        ]
        if not active:
            break
        status_counts = Counter(str(getattr(current, "status", "")) for current in active)
        tag_counts = Counter(str(oracle.semantic_tag(current)) for current in active)
        by_depth.append({
            "depth": depth,
            "active_states": len(active),
            "status_counts": dict(status_counts),
            "semantic_tag_counts": dict(tag_counts),
        })
        expanded_total += len(active)
        max_active = max(max_active, len(active))
        next_frontier = set()
        for current in active:
            for action_id, prob in oracle.next_dist(current).items():
                if float(prob) > 0.0:
                    next_frontier.add(oracle.step(current, int(action_id)))
        frontier = next_frontier
        max_frontier = max(max_frontier, len(frontier))
        depth += 1
    return {
        "teacher_support_paths": support_size,
        "initial_unique_states": len(initial_unique),
        "initial_status_counts": dict(initial_status),
        "expanded_state_total": expanded_total,
        "max_frontier_states": max_frontier,
        "max_active_states": max_active,
        "depth_count": len(by_depth),
        "by_depth": by_depth,
    }


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
    if args.max_states != 0:
        errors.append("--max-states must be 0 so all non-terminal states are evaluated")
    if errors:
        joined = "\n  - ".join(errors)
        raise SystemExit(f"--enforce-paper-config rejected this metric oracle-cost check:\n  - {joined}")


def _write_report(path: Path, base: dict, status: str, started: float, rows: list[dict]) -> dict:
    elapsed = time.perf_counter() - started
    by_tag: dict[str, list[int]] = {}
    by_role: dict[str, list[int]] = {}
    for row in rows:
        by_tag.setdefault(str(row["semantic_tag"]), []).append(int(row["expanded_state_total"]))
        by_role.setdefault(str(row["expected_role"]), []).append(int(row["expanded_state_total"]))
    report = {
        **base,
        "status": status,
        "timing": {
            "total_seconds": elapsed,
            "seconds_per_state": elapsed / len(rows) if rows else None,
        },
        "summary": {
            "evaluated_rows": len(rows),
            "expanded_state_total": sum(int(row["expanded_state_total"]) for row in rows),
            "max_active_states": max((int(row["max_active_states"]) for row in rows), default=0),
            "max_frontier_states": max((int(row["max_frontier_states"]) for row in rows), default=0),
            "expanded_by_tag": _summarize_groups(by_tag),
            "expanded_by_role": _summarize_groups(by_role),
        },
        "rows": rows,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _summarize_groups(groups: dict[str, list[int]]) -> dict[str, dict]:
    out = {}
    for key, values in groups.items():
        ordered = sorted(values)
        out[key] = {
            "count": len(values),
            "min": ordered[0],
            "p50": ordered[len(ordered) // 2],
            "max": ordered[-1],
            "mean": sum(values) / len(values),
        }
    return out


if __name__ == "__main__":
    main()
