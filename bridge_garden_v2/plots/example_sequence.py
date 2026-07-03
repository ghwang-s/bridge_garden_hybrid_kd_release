from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROLE_COLORS = {
    "commit": "#f4d35e",
    "bridge": "#ee6c4d",
    "garden": "#3d5a80",
    "middle": "#98c1d9",
    "special": "#c9ced6",
}


def plot_example_sequence(payload: Dict[str, object], output_path: str | Path, title: str | None = None) -> None:
    """
    Draw a readable token-level explanation figure.

    Expected payload keys:
      - tokens: list[str]
      - roles: list[str]
      - trace_rows: optional list[{"label": str, "text": str}]
      - line_breaks: optional list[int] token break indices
      - caption: optional str
    """
    tokens: List[str] = list(payload["tokens"])
    roles: List[str] = list(payload["roles"])
    trace_rows = list(payload.get("trace_rows", []))
    line_breaks = sorted(int(x) for x in payload.get("line_breaks", []))
    caption = str(payload.get("caption", ""))
    title = title or str(payload.get("title", "Example Sequence"))

    lines: List[List[tuple[str, str]]] = []
    current: List[tuple[str, str]] = []
    breaks = set(line_breaks)
    for idx, (token, role) in enumerate(zip(tokens, roles)):
        current.append((token, role))
        if idx in breaks:
            lines.append(current)
            current = []
    if current:
        lines.append(current)

    fig_height = 2.2 + 0.8 * len(lines) + 0.55 * len(trace_rows)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis("off")

    y = fig_height - 1.0
    ax.text(0.2, y + 0.5, title, fontsize=16, fontweight="bold", va="bottom")

    box_h = 0.42
    for line in lines:
        x = 0.2
        for token, role in line:
            width = max(0.6, 0.16 * len(token) + 0.28)
            color = ROLE_COLORS.get(role, ROLE_COLORS["middle"])
            patch = FancyBboxPatch(
                (x, y),
                width,
                box_h,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor=color,
                edgecolor="white",
                linewidth=1.0,
                alpha=0.95,
            )
            ax.add_patch(patch)
            ax.text(x + width / 2, y + box_h / 2, token, ha="center", va="center", fontsize=10, color="white")
            x += width + 0.08
        y -= 0.62

    for row in trace_rows:
        label = row.get("label", "")
        text = row.get("text", "")
        ax.text(0.2, y, f"{label}: {text}", fontsize=10.5, va="top", color="#253140")
        y -= 0.38

    legend_y = y - 0.2
    legend_items = [("commit", "Commit"), ("bridge", "Bridge"), ("garden", "Garden"), ("middle", "Middle")]
    x = 0.2
    for role, label in legend_items:
        patch = FancyBboxPatch(
            (x, legend_y),
            0.26,
            0.22,
            boxstyle="round,pad=0.04,rounding_size=0.05",
            facecolor=ROLE_COLORS[role],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
        )
        ax.add_patch(patch)
        ax.text(x + 0.33, legend_y + 0.11, label, va="center", fontsize=10)
        x += 1.55

    if caption:
        ax.text(0.2, legend_y - 0.38, caption, fontsize=10, color="#44505c", va="top")

    ax.set_xlim(0, 13)
    ax.set_ylim(0, fig_height)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
