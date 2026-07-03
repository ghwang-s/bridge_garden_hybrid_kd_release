#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values: np.ndarray, positions: List[int]) -> float:
    if not positions:
        return float("nan")
    return float(values[positions].mean())


def _weighted_mean(values: np.ndarray, counts: np.ndarray, positions: List[int]) -> float:
    if not positions:
        return float("nan")
    pos_counts = counts[positions]
    denom = float(pos_counts.sum())
    if denom <= 0:
        return float("nan")
    return float(np.dot(values[positions], pos_counts) / denom)


def _valid_positions(counts: np.ndarray) -> List[int]:
    return np.flatnonzero(counts > 0).tolist()


def _compute_row(summary: Dict[str, object], positions: List[int], label: str) -> Dict[str, object]:
    eb = np.array(summary["per_step_eb"], dtype=np.float64)
    counts = np.array(summary["per_step_valid_counts"], dtype=np.float64)
    valid_positions = [pos for pos in positions if 0 <= pos < len(eb) and counts[pos] > 0]
    return {
        "region": label,
        "count": len(valid_positions),
        "eb": _mean(eb, valid_positions),
        "eb_weighted": _weighted_mean(eb, counts, valid_positions),
    }


def _best_hybrid(methods: Dict[str, Dict[str, object]], prefer_weighted: bool) -> str | None:
    hybrid_names = [name for name in methods if name.startswith("hybrid_")]
    if not hybrid_names:
        return None
    key = "overall_eb_weighted" if prefer_weighted else "overall_eb"
    return min(hybrid_names, key=lambda name: float(methods[name].get(key, float("inf"))))


def evaluate_artifact(artifact_root: Path, prefer_weighted: bool) -> Dict[str, object]:
    summary_path = artifact_root / "reports" / "train_summary.json"
    payload = _load_json(summary_path)
    methods = payload["methods"]

    metrics_dir = artifact_root / "metrics"
    oracle_payload = payload.get("oracle_positions")
    if oracle_payload is None and (metrics_dir / "oracle_positions.json").exists():
        oracle_payload = _load_json(metrics_dir / "oracle_positions.json")

    region_masks = payload.get("region_masks")
    if region_masks is None and (metrics_dir / "region_masks.json").exists():
        region_masks = _load_json(metrics_dir / "region_masks.json")
    if region_masks is None:
        raise KeyError("region_masks not found in train_summary.json or metrics/region_masks.json")

    region_overlap = payload.get("region_overlap")
    if region_overlap is None and (metrics_dir / "region_overlap.json").exists():
        region_overlap = _load_json(metrics_dir / "region_overlap.json")

    oracle_positions = oracle_payload["role_positions"] if oracle_payload is not None else None
    best_hybrid = _best_hybrid(methods, prefer_weighted)
    selected_methods = ["hard", "soft"] + ([best_hybrid] if best_hybrid else [])

    region_sets = {
        "overall": None,
        "kappa_bridge": region_masks["bridge_positions"],
        "kappa_garden": region_masks["garden_positions"],
    }
    if oracle_positions is not None:
        region_sets["oracle_bridge"] = oracle_positions.get("bridge", [])
        region_sets["oracle_garden"] = oracle_positions.get("garden", [])

    rows = []
    for method_name in selected_methods:
        if method_name is None:
            continue
        method_summary = methods[method_name]
        counts = np.array(method_summary["per_step_valid_counts"], dtype=np.float64)
        overall_positions = _valid_positions(counts)
        for region_name, positions in region_sets.items():
            pos = overall_positions if positions is None else positions
            row = _compute_row(method_summary, pos, region_name)
            row["method"] = method_name
            rows.append(row)

    return {
        "artifact_root": str(artifact_root),
        "kappa_reference_method": payload.get("kappa_reference_method"),
        "best_hybrid": best_hybrid,
        "has_oracle_positions": oracle_positions is not None,
        "region_overlap": region_overlap,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate synthetic EB artifact with oracle/kappa and weighted/unweighted views")
    parser.add_argument("artifact_root", type=Path, help="Artifact root containing reports/train_summary.json")
    parser.add_argument("--prefer-weighted", action="store_true", help="Select the representative hybrid by weighted overall EB")
    args = parser.parse_args()

    report = evaluate_artifact(args.artifact_root.expanduser().resolve(), prefer_weighted=args.prefer_weighted)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
