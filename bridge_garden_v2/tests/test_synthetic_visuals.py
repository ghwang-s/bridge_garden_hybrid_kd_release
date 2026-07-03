from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bridge_garden_v2.synthetic_visuals import render_token_kappa_heatmap
from scripts.run_synthetic_mini_pipeline import _heatmap_rows, _state_table


def test_render_token_kappa_heatmap_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    output = render_token_kappa_heatmap(
        tokens=["x", "+", "y"],
        kappa_values=[3.0, 1.0, -0.5],
        semantic_tags=["variable_binding", "operator", "equivalent_form"],
        output_path=tmp_path / "heatmap.png",
        title="smoke",
    )
    assert output.exists()
    assert output.stat().st_size > 0


def test_render_token_kappa_heatmap_rejects_mismatched_lengths(tmp_path: Path) -> None:
    try:
        render_token_kappa_heatmap(
            tokens=["x"],
            kappa_values=[1.0, 2.0],
            semantic_tags=["operator"],
            output_path=tmp_path / "bad.png",
            title="bad",
        )
    except ValueError as exc:
        assert "equal length" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_heatmap_rows_require_kappa_for_every_visible_token() -> None:
    state = _State(sample_id=0, position=0, prefix=(1,))
    next_state = _State(sample_id=0, position=1, prefix=(1, 2))
    bundle = SimpleNamespace(oracle=_Oracle(), vocab={1: "<bos>", 2: "x", 3: "y"})
    kappa_eval = _KappaEval(scores=[])

    with pytest.raises(RuntimeError, match="missing kappa row"):
        _heatmap_rows(bundle, kappa_eval, [state, next_state])


def test_state_table_records_visible_token_trace() -> None:
    state = _State(sample_id=0, position=0, prefix=(1,))
    next_state = _State(sample_id=0, position=1, prefix=(1, 2))
    bundle = SimpleNamespace(oracle=_Oracle(), vocab={1: "<bos>", 2: "x", 3: "y"})
    kappa_eval = _KappaEval(scores=[SimpleNamespace(state_id=state, kappa=1.5, confidence=0.8, semantic_tag="operator")])

    rows = _state_table(bundle, kappa_eval, [state, next_state])

    assert rows[0]["visible_token"] == "x"
    assert rows[0]["visible_token_id"] == 2
    assert rows[0]["visible_token_source"] == "sampled_path"


class _State:
    def __init__(self, *, sample_id: int, position: int, prefix: tuple[int, ...]) -> None:
        self.sample_id = sample_id
        self.position = position
        self.prefix = prefix

    def __hash__(self) -> int:
        return hash((self.sample_id, self.position, self.prefix))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _State)
            and self.sample_id == other.sample_id
            and self.position == other.position
            and self.prefix == other.prefix
        )


class _Oracle:
    def is_terminal(self, state: _State) -> bool:
        return state.position >= 1

    def next_dist(self, state: _State) -> dict[int, float]:
        return {2: 0.6, 3: 0.4}

    def semantic_tag(self, state: _State) -> str:
        return "operator"

    def expected_role(self, state: _State) -> str:
        return "high_risk"


class _KappaEval:
    def __init__(self, *, scores: list) -> None:
        self.scores = scores
        empty = SimpleNamespace(bridge=[], garden=[], bridge_tie_band=[], garden_tie_band=[])
        self.kappa_split = SimpleNamespace(bridge=[], garden=[])
        self.confidence_split = empty
