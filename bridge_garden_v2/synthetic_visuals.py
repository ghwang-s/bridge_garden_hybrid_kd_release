from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def render_token_kappa_heatmap(
    *,
    tokens: Sequence[str],
    kappa_values: Sequence[float],
    semantic_tags: Sequence[str],
    region_labels: Sequence[str] | None = None,
    output_path: str | Path,
    title: str,
) -> Path:
    """Render a compact token-level κ heatmap with semantic tag labels."""
    if not (len(tokens) == len(kappa_values) == len(semantic_tags)):
        raise ValueError("tokens, kappa_values, and semantic_tags must have equal length")
    if region_labels is not None and len(region_labels) != len(tokens):
        raise ValueError("region_labels must match tokens when provided")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = np.array(kappa_values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot render empty heatmap")
    finite = values[np.isfinite(values)]
    scale = float(np.max(np.abs(finite))) if finite.size else 1.0
    if scale <= 0.0:
        scale = 1.0

    fig_width = max(8.0, min(18.0, 0.28 * len(tokens)))
    fig, ax = plt.subplots(figsize=(fig_width, 2.8))
    ax.imshow(values.reshape(1, -1), aspect="auto", cmap="coolwarm", vmin=-scale, vmax=scale)
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(tokens)))
    if region_labels is None:
        labels = [f"{tok}\n{_short_tag(tag)}" for tok, tag in zip(tokens, semantic_tags)]
    else:
        labels = [
            f"{tok}\n{_short_tag(tag)}\n{region}"
            for tok, tag, region in zip(tokens, semantic_tags, region_labels)
        ]
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    for idx, value in enumerate(values):
        ax.text(idx, 0, f"{value:.1f}", ha="center", va="center", fontsize=6, color="black")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def render_kappa_confidence_scatter(
    *,
    kappa_values: Sequence[float],
    confidence_values: Sequence[float],
    semantic_tags: Sequence[str],
    region_labels: Sequence[str],
    output_path: str | Path,
    title: str,
) -> Path:
    """Render the observable confidence proxy against exact κ."""
    if not (len(kappa_values) == len(confidence_values) == len(semantic_tags) == len(region_labels)):
        raise ValueError("all scatter inputs must have equal length")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    x = np.array(kappa_values, dtype=np.float64)
    y = np.array(confidence_values, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    tags = sorted(set(semantic_tags))
    cmap = plt.get_cmap("tab10")
    for idx, tag in enumerate(tags):
        mask = np.array([item == tag for item in semantic_tags], dtype=bool)
        ax.scatter(x[mask], y[mask], s=46, alpha=0.85, color=cmap(idx % 10), label=_short_tag(tag))
    for idx, label in enumerate(region_labels):
        if label:
            ax.annotate(label, (x[idx], y[idx]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("exact state κ_ref")
    ax.set_ylabel("teacher confidence c_T")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25)
    if len(tags) <= 8:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _short_tag(tag: str) -> str:
    parts = tag.split("_")
    if len(parts) == 1:
        return tag[:8]
    return "".join(part[:3] for part in parts)[:10]
