#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MAIN_DOMAINS = ("code", "math", "dialogue")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine per-domain exact-cost check JSON files.")
    parser.add_argument("--code", required=True)
    parser.add_argument("--math", required=True)
    parser.add_argument("--dialogue", required=True)
    parser.add_argument("--support-cap", type=int, default=100_000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    inputs = {
        "code": Path(args.code),
        "math": Path(args.math),
        "dialogue": Path(args.dialogue),
    }
    summary = build_check_summary(inputs, support_cap=args.support_cap)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary["path"] = str(out)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "domains": list(summary["domains"])}, indent=2))


def build_check_summary(paths: dict[str, Path], *, support_cap: int = 100_000) -> dict[str, Any]:
    domains = {}
    for domain in MAIN_DOMAINS:
        if domain not in paths:
            raise ValueError(f"missing check path for {domain}")
        report = _load_json(paths[domain])
        domains[domain] = _domain_summary(domain, report, source_path=paths[domain])
    return {
        "support_cap": int(support_cap),
        "domains": domains,
    }


def _domain_summary(domain: str, report: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    config = report.get("kappa_definition", {})
    rows = report.get("rows", [])
    if report.get("domain") != domain:
        raise ValueError(f"{source_path} has domain={report.get('domain')!r}, expected {domain!r}")
    if report.get("status") != "completed":
        raise ValueError(f"{source_path} status={report.get('status')!r}, expected completed")
    all_kappa = int(config.get("all_kappa_state_count", config.get("all_eligible_state_count", 0)))
    evaluated = int(config.get("evaluated_state_count", len(rows)))
    if evaluated != all_kappa or len(rows) != all_kappa:
        raise ValueError(
            f"{source_path} is not all-state complete: rows={len(rows)}, "
            f"evaluated={evaluated}, all_kappa={all_kappa}"
        )
    return {
        "source_path": str(source_path),
        "status": report.get("status"),
        "analysis_count": report.get("analysis_count"),
        "analysis_sample_seed": report.get("analysis_sample_seed"),
        "script_len": report.get("script_len"),
        "vocab_size": report.get("vocab_size"),
        "device": report.get("device"),
        "kappa_state_count": all_kappa,
        "evaluated_state_count": evaluated,
        "eval_token_count": config.get("eval_token_count"),
        "eos_in_eval": config.get("eos_in_eval"),
        "pad_in_eval": config.get("pad_in_eval"),
        "bos_in_eval": config.get("bos_in_eval"),
        "uses_global_eval_token_ids": config.get("uses_global_eval_token_ids"),
        "uses_state_local_eval_token_ids": config.get("uses_state_local_eval_token_ids"),
        "future_support": _support_summary(rows),
        "timing": report.get("timing", {}),
        "student_loss_stats": report.get("student_loss_stats", {}),
        "cuda": report.get("cuda", {}),
    }


def _support_summary(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    values = sorted(
        int(row["teacher_support_paths"])
        for row in rows
        if row.get("teacher_support_paths") is not None
    )
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "min": values[0],
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": values[-1],
    }


def _percentile(values: list[int], q: float) -> float:
    if len(values) == 1:
        return float(values[0])
    pos = q * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data, _ = json.JSONDecoder().raw_decode(raw)
    return data


if __name__ == "__main__":
    main()
