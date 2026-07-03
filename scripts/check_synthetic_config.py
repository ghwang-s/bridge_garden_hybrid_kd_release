#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridge_garden_v2.synthetic_domains import build_synthetic_domain
from bridge_garden_v2.synthetic_evaluator import collect_sampled_path_states, estimate_teacher_support_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="code,math,dialogue")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--analysis-count", type=int, default=4)
    parser.add_argument("--script-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--analysis-sample-seed", type=int, default=0)
    parser.add_argument("--support-cap", type=int, default=100_000)
    parser.add_argument("--out", default="bridge_garden_v2/synthetic_config_check.json")
    args = parser.parse_args()

    report = {
        "sample_count": args.sample_count,
        "analysis_count": args.analysis_count,
        "script_len": args.script_len,
        "vocab_size": args.vocab_size,
        "analysis_sample_seed": args.analysis_sample_seed,
        "support_cap": args.support_cap,
        "domains": {},
    }
    for domain in [item.strip() for item in args.domains.split(",") if item.strip()]:
        bundle = build_synthetic_domain(
            domain,
            sample_count=args.sample_count,
            script_len=args.script_len,
            vocab_size=args.vocab_size,
        )
        analysis_ids = list(range(args.sample_count - args.analysis_count, args.sample_count))
        states = collect_sampled_path_states(bundle.oracle, analysis_ids, seed=args.analysis_sample_seed)
        kappa_states = list(states)
        support_rows = [
            estimate_teacher_support_size(bundle.oracle, state, cap=args.support_cap)
            for state in kappa_states
        ]
        tag_counts: dict[str, int] = {}
        for state in kappa_states:
            tag = bundle.oracle.semantic_tag(state)
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        report["domains"][domain] = {
            "eval_token_count": len(bundle.oracle.eval_token_ids),
            "eos_in_eval": bundle.oracle.eos_id in bundle.oracle.eval_token_ids,
            "pad_in_eval": bundle.oracle.pad_id in bundle.oracle.eval_token_ids,
            "bos_in_eval": bundle.oracle.bos_id in bundle.oracle.eval_token_ids,
            "analysis_ids": analysis_ids,
            "kappa_state_count": len(kappa_states),
            "semantic_tag_counts": tag_counts,
            "future_support": _summary(support_rows),
            "estimated_q_values": len(kappa_states) * len(bundle.oracle.eval_token_ids),
        }

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _summary(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    vals = sorted(int(v) for v in values)
    return {
        "count": len(vals),
        "min": vals[0],
        "p50": vals[len(vals) // 2],
        "p95": vals[min(len(vals) - 1, int(0.95 * (len(vals) - 1)))],
        "max": vals[-1],
    }


if __name__ == "__main__":
    main()
