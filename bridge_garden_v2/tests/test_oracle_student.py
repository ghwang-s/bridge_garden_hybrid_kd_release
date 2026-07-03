from __future__ import annotations

import torch
import torch.nn as nn

from bridge_garden_v2.oracle_student import NeuralStudentLoss
from bridge_garden_v2.scripted_oracle import ScriptedOracle, ScriptStep


class PrefixAgnosticModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits_param = nn.Parameter(logits.clone())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        return self.logits_param.view(1, 1, -1).expand(batch, seq_len, -1)


def test_neural_student_loss_near_zero_when_distribution_matches() -> None:
    oracle = ScriptedOracle(
        scripts=[[ScriptStep(dist=((2, 0.7), (3, 0.3)), semantic_tag="x", expected_role="x")]],
        eval_token_ids=(2, 3),
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )
    model = PrefixAgnosticModel(torch.log(torch.tensor([0.0, 0.0, 0.7, 0.3]).clamp(min=1e-8)))
    loss = NeuralStudentLoss(model, vocab_size=4, device=torch.device("cpu"))
    assert loss.loss_at_state(oracle.initial_state(0), oracle) < 1e-6


def test_neural_student_loss_requires_prefix_state() -> None:
    oracle = ScriptedOracle(
        scripts=[[ScriptStep(dist=((2, 1.0),), semantic_tag="x", expected_role="x")]],
        eval_token_ids=(2,),
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )
    model = PrefixAgnosticModel(torch.zeros(4))
    loss = NeuralStudentLoss(model, vocab_size=4, device=torch.device("cpu"))
    try:
        loss.loss_at_state(("bad",), oracle)
    except TypeError as exc:
        assert "prefix" in str(exc)
    else:
        raise AssertionError("expected TypeError")
