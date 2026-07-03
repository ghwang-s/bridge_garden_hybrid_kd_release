from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from bridge_garden_v2.metrics import build_valid_target_mask, rollout_kl_full
from bridge_garden_v2.pipeline import _masked_ce_loss_with_valid_targets, summarize_eb, compute_region_overlap
from bridge_garden_v2.schema import ModelConfig, RegionMaskBundle


class ToyLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size, device=x.device)
        logits[..., 1] = 4.0
        logits[..., 2] = 1.0
        return logits


def test_valid_target_mask_excludes_pad_targets() -> None:
    batch = torch.tensor([[1, 4, 5, 0, 0], [1, 3, 2, 6, 0]])
    mask = build_valid_target_mask(batch, pad_id=0)
    assert mask.tolist() == [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]]


def test_rollout_kl_full_stops_counting_after_eos() -> None:
    teacher = ToyLM(vocab_size=8)
    student = ToyLM(vocab_size=8)
    gold = torch.tensor([
        [1, 4, 2, 7, 0],  # EOS at position 2
        [1, 5, 6, 2, 0],  # EOS at position 3
    ])
    metric = rollout_kl_full(
        teacher=teacher,
        student=student,
        gold_batch=gold,
        bos_id=1,
        eos_id=2,
        pad_id=0,
    )
    assert metric.counts.tolist()[0] == 2.0
    assert metric.counts.tolist()[1] <= 2.0
    assert np.allclose(metric.values, 0.0)


def test_masked_ce_loss_uses_valid_targets_not_teacher_labels() -> None:
    logits = torch.tensor([[[3.0, 1.0], [1.0, 3.0]]])
    teacher_labels = torch.tensor([[1, 1]])
    valid_targets = torch.tensor([[5, 0]])
    loss = _masked_ce_loss_with_valid_targets(logits, teacher_labels, valid_targets, pad_id=0)
    expected = torch.nn.functional.cross_entropy(logits[:, :1, :].reshape(-1, 2), teacher_labels[:, :1].reshape(-1))
    assert torch.allclose(loss, expected)


def test_summarize_eb_reports_weighted_and_overlap() -> None:
    tf_metric = type('Metric', (), {'values': np.array([2.0, 20.0]), 'counts': np.array([2.0, 10.0])})
    rollout_metric = type('Metric', (), {'values': np.array([4.0, 30.0]), 'counts': np.array([2.0, 10.0])})
    region_masks = RegionMaskBundle(
        coverage_counts=[2, 10],
        valid_positions=[0, 1],
        bridge_positions=[0],
        garden_positions=[1],
        bridge_threshold=1.0,
        garden_threshold=0.0,
        position_mean_kappa=[1.0, 0.0],
    )
    oracle_positions = {'role_positions': {'commit': [], 'bridge': [0], 'garden': [1], 'middle': []}}
    overlap = compute_region_overlap(region_masks, oracle_positions)
    summary = summarize_eb(tf_metric, rollout_metric, region_masks, oracle_positions=oracle_positions, region_overlap=overlap)
    assert summary['overall_eb'] == 1.0
    assert np.isclose(summary['overall_eb_weighted'], 1.0)
    assert summary['oracle_role_summary']['bridge']['eb_weighted'] == 1.0
    assert summary['region_overlap']['bridge']['overlap_count'] == 1
