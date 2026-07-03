from __future__ import annotations

import json

import pytest

from scripts.build_synthetic_cost_summary import build_check_summary


SCRIPT_LENS = {"code": 47, "math": 43, "dialogue": 35}


def _report(domain: str, *, status: str = "completed", rows: int = 2) -> dict:
    row_items = [
        {"teacher_support_paths": value}
        for value in range(1, rows + 1)
    ]
    return {
        "domain": domain,
        "status": status,
        "analysis_count": 100,
        "analysis_sample_seed": 0,
        "script_len": SCRIPT_LENS[domain],
        "vocab_size": 64,
        "device": "cuda",
        "kappa_definition": {
            "all_kappa_state_count": rows,
            "evaluated_state_count": rows,
            "eval_token_count": 62,
            "eos_in_eval": True,
            "pad_in_eval": False,
            "bos_in_eval": False,
            "uses_global_eval_token_ids": True,
            "uses_state_local_eval_token_ids": False,
        },
        "timing": {"total_seconds": 1.0},
        "student_loss_stats": {"batch_forward_rows": 10},
        "cuda": {"available": True},
        "rows": row_items,
    }


def test_build_check_summary_combines_completed_domains(tmp_path) -> None:
    paths = {}
    for domain in ["code", "math", "dialogue"]:
        path = tmp_path / f"{domain}.json"
        path.write_text(json.dumps(_report(domain, rows=3)), encoding="utf-8")
        paths[domain] = path

    summary = build_check_summary(paths)

    assert set(summary["domains"]) == {"code", "math", "dialogue"}
    assert summary["domains"]["code"]["kappa_state_count"] == 3
    assert summary["domains"]["code"]["eval_token_count"] == 62
    assert summary["domains"]["code"]["future_support"]["max"] == 3
    assert summary["domains"]["code"]["future_support"]["p50"] == 2.0


def test_build_check_summary_rejects_incomplete_domain(tmp_path) -> None:
    paths = {}
    for domain in ["code", "math", "dialogue"]:
        path = tmp_path / f"{domain}.json"
        path.write_text(
            json.dumps(_report(domain, status="running" if domain == "dialogue" else "completed")),
            encoding="utf-8",
        )
        paths[domain] = path

    with pytest.raises(ValueError, match="expected completed"):
        build_check_summary(paths)
