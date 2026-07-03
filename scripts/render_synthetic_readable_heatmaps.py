#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridge_garden_v2.synthetic_domains import DOMAIN_SCRIPT_LENS, build_synthetic_domain


DOMAIN_TITLES = {
    "code": "Code Control Flow",
    "math": "Math Derivation",
    "dialogue": "Dialogue Constraint Following",
}

TAG_LABELS = {
    "operator": "operator",
    "branch_guard": "branch guard",
    "return_semantics": "return",
    "equivalent_implementation": "equiv impl",
    "substitution": "substitute",
    "computed_value": "computed",
    "final_answer": "final",
    "equivalent_representation": "equiv form",
    "recipient": "recipient",
    "required_fact": "required fact",
    "date_time": "date/time",
    "forbidden_constraint": "forbidden",
    "tone_paraphrase": "tone",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Render readable token-block κ heatmaps from synthetic kappa_state_table.json.")
    parser.add_argument("--domain", required=True, choices=["code", "math", "dialogue"])
    parser.add_argument("--kappa-state-table", required=True, type=Path)
    parser.add_argument("--summary", type=Path, help="Optional summary.json used to replay the teacher-sampled analysis path.")
    parser.add_argument("--sample-id", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = _load_json(args.kappa_state_table)
    by_position = {
        int(row["position"]): row
        for row in rows
        if int(row.get("sample_id", -1)) == int(args.sample_id)
    }
    if not by_position:
        raise SystemExit(f"sample_id {args.sample_id} not found in {args.kappa_state_table}")
    summary = _load_json(args.summary) if args.summary else None
    cells, token_source = _script_cells(args.domain, args.sample_id, by_position, summary)
    render(cells, args.domain, args.sample_id, args.output, token_source=token_source)
    return 0


def _load_json(path: Path):
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text.lstrip())
    return obj


def _script_cells(domain: str, sample_id: int, by_position: dict[int, dict], summary: dict | None) -> tuple[list[dict], str]:
    token_source = "teacher-mode fallback"
    sample_count = sample_id + 1
    if summary:
        analysis_ids = [int(item) for item in summary.get("analysis_ids", [])]
        if analysis_ids:
            sample_count = max(sample_count, max(analysis_ids) + 1)
    bundle = build_synthetic_domain(
        domain,
        sample_count=sample_count,
        script_len=DOMAIN_SCRIPT_LENS[domain],
        vocab_size=64,
    )
    script = bundle.oracle.scripts[sample_id]
    sampled_tokens = None
    if summary and summary.get("analysis_path_policy") == "sample":
        sampled_tokens = _teacher_sampled_tokens(bundle, sample_id, summary)
        token_source = "teacher-sampled analysis path"
    cells: list[dict] = []
    for pos, step in enumerate(script):
        if sampled_tokens is not None and pos < len(sampled_tokens):
            token = sampled_tokens[pos]
        elif step.semantic_tag == "terminal":
            token = "<eos>"
        else:
            token_id = int(max(step.dist, key=lambda item: (float(item[1]), -int(item[0])))[0])
            token = bundle.vocab[token_id]
        row = by_position.get(pos)
        cells.append(
            {
                "position": pos,
                "token": token,
                "semantic_tag": row["semantic_tag"] if row else step.semantic_tag,
                "expected_role": row["expected_role"] if row else step.expected_role,
                "kappa": float(row["kappa"]) if row else 0.0,
                "is_decision": row is not None,
                "region": _region(row) if row else "",
            }
        )
    return cells, token_source


def _teacher_sampled_tokens(bundle, sample_id: int, summary: dict) -> list[str]:
    analysis_ids = [int(item) for item in summary.get("analysis_ids", [])]
    if sample_id not in analysis_ids:
        raise ValueError(f"sample_id {sample_id} is not listed in summary analysis_ids")
    rng = random.Random(int(summary.get("analysis_sample_seed", 0)))
    for current_sample_id in analysis_ids:
        state = bundle.oracle.initial_state(current_sample_id)
        tokens: list[str] = []
        while not bundle.oracle.is_terminal(state):
            dist = bundle.oracle.next_dist(state)
            if not dist:
                break
            action_id = _sample_from_dist(dist, rng)
            tokens.append(bundle.vocab[action_id])
            state = bundle.oracle.step(state, action_id)
        if current_sample_id == sample_id:
            return tokens
    raise ValueError(f"sample_id {sample_id} was not replayed")


def _sample_from_dist(dist: dict[int, float], rng: random.Random) -> int:
    total = float(sum(dist.values()))
    if total <= 0.0:
        raise ValueError("cannot sample from empty distribution")
    threshold = rng.random() * total
    cdf = 0.0
    last = None
    for token_id, prob in sorted(dist.items()):
        last = int(token_id)
        cdf += float(prob)
        if threshold <= cdf:
            return int(token_id)
    if last is None:
        raise ValueError("cannot sample from empty distribution")
    return last


def _region(row: dict | None) -> str:
    if not row:
        return ""
    bridge = bool(row.get("bridge_partition"))
    garden = bool(row.get("garden_partition"))
    if bridge and garden:
        return "Bridge/Garden"
    if bridge:
        return "Bridge"
    if garden:
        return "Garden"
    return ""


def render(cells: list[dict], domain: str, sample_id: int, output: Path, *, token_source: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.patches import FancyBboxPatch

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = _layout_lines(cells, domain)
    values = np.array([abs(cell["kappa"]) for cell in cells if cell["is_decision"]], dtype=np.float64)
    max_log = float(np.max(np.log1p(values))) if values.size else 1.0
    if max_log <= 0:
        max_log = 1.0

    max_cols = max(sum(_cell_width(cell) + 0.08 for cell in line) for line in lines)
    fig_w = max(9.5, min(17.5, 0.72 * max_cols + 1.2))
    row_step = 1.18
    fig_h = 2.15 + row_step * len(lines) + 1.25
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    cmap = plt.get_cmap("RdBu_r")
    norm = colors.Normalize(vmin=-max_log, vmax=max_log)
    y = fig_h - 1.75
    ax.text(
        0.15,
        fig_h - 0.45,
        f"{DOMAIN_TITLES[domain]} sample {sample_id}: token-level κ heatmap",
        fontsize=13.5,
        fontweight="bold",
        va="bottom",
        color="#1f2933",
    )
    ax.text(
        0.15,
        fig_h - 0.83,
        "Every rendered token box is a non-terminal prefix state with exact κ; any gray box indicates a missing row and must fail verification. "
        f"Tokens show the {token_source}; Bridge/Garden labels use the κ split. Numbers are raw κ.",
        fontsize=8.8,
        va="bottom",
        color="#475467",
    )

    for line in lines:
        x = 0.15
        for cell in line:
            width = _cell_width(cell)
            raw = float(cell["kappa"])
            shown = math.copysign(math.log1p(abs(raw)), raw) if cell["is_decision"] else 0.0
            face = cmap(norm(shown)) if cell["is_decision"] else "#f0f2f4"
            edge = "#ffffff" if cell["is_decision"] else "#d8dde3"
            patch = FancyBboxPatch(
                (x, y),
                width,
                0.52,
                boxstyle="round,pad=0.035,rounding_size=0.055",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.0,
            )
            ax.add_patch(patch)
            token_color = "#101820" if abs(shown) < 0.72 * max_log else "#ffffff"
            if _display_token(cell["token"]).strip():
                ax.text(x + width / 2, y + 0.31, _display_token(cell["token"]), ha="center", va="center", fontsize=10.5, color=token_color)
            if cell["is_decision"]:
                label = TAG_LABELS.get(cell["semantic_tag"], cell["semantic_tag"])
                ax.text(x + width / 2, y - 0.07, label, ha="center", va="top", fontsize=8.2, color="#425466")
                ax.text(
                    x + width / 2,
                    y - 0.32,
                    _kappa_label(raw),
                    ha="center",
                    va="top",
                    fontsize=7.6,
                    color="#667085",
                )
                if cell["region"]:
                    ax.text(x + width / 2, y + 0.56, cell["region"], ha="center", va="bottom", fontsize=7.8, color="#344054")
            x += width + 0.08
        y -= row_step

    bar_width = min(8.0, max(4.6, 0.58 * max_cols))
    bar_x0 = 0.15 + max(0.0, (max_cols - bar_width) / 2.0)
    bar_x1 = bar_x0 + bar_width
    bar_y = max(0.72, y + 0.48)
    gradient = np.linspace(-max_log, max_log, 256, dtype=np.float64)[None, :]
    ax.imshow(
        gradient,
        cmap=cmap,
        norm=norm,
        extent=(bar_x0, bar_x1, bar_y, bar_y + 0.14),
        aspect="auto",
        zorder=0,
    )
    ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color="#111827", linewidth=0.8)
    ax.plot([bar_x0, bar_x1], [bar_y + 0.14, bar_y + 0.14], color="#111827", linewidth=0.8)
    tick_values = np.linspace(-max_log, max_log, 7)
    for value in tick_values:
        frac = (value + max_log) / (2 * max_log)
        tx = bar_x0 + frac * bar_width
        ax.plot([tx, tx], [bar_y - 0.04, bar_y], color="#111827", linewidth=0.8)
        ax.text(tx, bar_y - 0.11, f"{value:.0f}", ha="center", va="top", fontsize=8.0, color="#111827")
    ax.text(
        (bar_x0 + bar_x1) / 2,
        bar_y - 0.38,
        "color uses sign(κ) log(1 + |κ|); printed κ values are raw",
        ha="center",
        va="top",
        fontsize=8.8,
        color="#111827",
    )
    ax.set_xlim(0, max_cols + 0.6)
    ax.set_ylim(max(0.0, bar_y - 0.65), fig_h)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _layout_lines(cells: list[dict], domain: str) -> list[list[dict]]:
    if domain == "code":
        lines: list[list[dict]] = []
        current: list[dict] = []
        for cell in cells:
            token = cell["token"]
            current.append(cell)
            if token == "NEWLINE":
                if current:
                    lines.append(current)
                    current = []
        if current:
            lines.append(current)
        return lines

    limit = 14
    lines = []
    current = []
    for cell in cells:
        tok = cell["token"]
        if tok == "<eos>":
            continue
        current.append(cell)
        if tok == "." or len(current) >= limit:
            lines.append(current)
            current = []
    if current:
        lines.append(current)
    return lines


def _cell_width(cell: dict) -> float:
    token = _display_token(cell["token"])
    if token == "    ":
        return 0.42
    return max(0.68, min(2.7, 0.38 + 0.205 * len(token)))


def _display_token(token: str) -> str:
    if token == "    ":
        return "    "
    if token == "NEWLINE":
        return "newline"
    if token == "INDENT":
        return "indent"
    if token == "DEDENT":
        return "dedent"
    return token.replace("_", " ")


def _kappa_label(value: float) -> str:
    if abs(value) >= 100:
        return f"κ={value:.0f}"
    if abs(value) >= 10:
        return f"κ={value:.1f}"
    if abs(value) >= 0.05:
        return f"κ={value:.2f}"
    return "κ≈0"


if __name__ == "__main__":
    raise SystemExit(main())
