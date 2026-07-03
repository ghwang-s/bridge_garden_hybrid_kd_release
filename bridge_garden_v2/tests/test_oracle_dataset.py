from __future__ import annotations

import torch

from bridge_garden_v2.synthetic_domains import build_synthetic_domain
from bridge_garden_v2.oracle_dataset import materialize_mode_paths, materialize_sampled_paths


def test_materialize_mode_paths_shapes_and_probs() -> None:
    bundle = build_synthetic_domain("code", sample_count=2)
    batch = materialize_mode_paths(bundle.oracle, sample_ids=[0, 1], vocab_size=len(bundle.vocab))
    assert batch.input_ids.shape == (2, 47)
    assert batch.target_ids.shape == (2, 47)
    assert batch.teacher_probs.shape == (2, 47, 64)
    assert torch.allclose(batch.teacher_probs.sum(dim=-1), torch.ones(2, 47))
    assert len(batch.semantic_tags) == 2
    assert len(batch.semantic_tags[0]) == 47
    assert len(batch.expected_roles) == 2
    assert batch.role_ids.shape == (2, 47)
    assert 1 in batch.role_ids[0].tolist()
    assert 2 in batch.role_ids[0].tolist()


def test_materialize_mode_paths_uses_mode_targets() -> None:
    bundle = build_synthetic_domain("dialogue", sample_count=1)
    batch = materialize_mode_paths(bundle.oracle, sample_ids=[0], vocab_size=len(bundle.vocab))
    for pos in range(batch.target_ids.shape[1]):
        probs = batch.teacher_probs[0, pos]
        assert int(batch.target_ids[0, pos]) == int(torch.argmax(probs))


def test_materialize_sampled_paths_is_seeded_and_can_take_non_mode_targets() -> None:
    bundle = build_synthetic_domain("dialogue", sample_count=4)
    left = materialize_sampled_paths(bundle.oracle, sample_ids=[0, 1, 2, 3], vocab_size=len(bundle.vocab), seed=7)
    right = materialize_sampled_paths(bundle.oracle, sample_ids=[0, 1, 2, 3], vocab_size=len(bundle.vocab), seed=7)
    assert torch.equal(left.target_ids, right.target_ids)
    non_mode = 0
    for row in range(left.target_ids.shape[0]):
        for pos in range(left.target_ids.shape[1]):
            probs = left.teacher_probs[row, pos]
            if int(left.target_ids[row, pos]) != int(torch.argmax(probs)):
                non_mode += 1
    assert non_mode > 0
