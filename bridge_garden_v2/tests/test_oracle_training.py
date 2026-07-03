from __future__ import annotations

import torch
import pytest

from bridge_garden_v2.synthetic_domains import build_synthetic_domain
from bridge_garden_v2.oracle_dataset import materialize_mode_paths
from bridge_garden_v2.oracle_training import ce_loss, hard_kd_loss, hybrid_kd_loss, soft_kd_loss
from bridge_garden_v2.oracle_training import _epoch_learning_rate
from bridge_garden_v2.oracle_training import _is_significant_improvement
from bridge_garden_v2.oracle_training import train_oracle_student
from bridge_garden_v2.schema import ModelConfig


def test_hard_kd_loss_masks_pad_targets() -> None:
    logits = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
    teacher_probs = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    target_ids = torch.tensor([[0, 0]])
    loss = hard_kd_loss(logits, teacher_probs, target_ids, pad_id=0)
    assert torch.isfinite(loss)


def test_hard_kd_loss_uses_teacher_argmax_not_sampled_target() -> None:
    logits = torch.tensor([[[4.0, 0.0]]])
    teacher_probs = torch.tensor([[[0.9, 0.1]]])
    target_ids = torch.tensor([[1]])
    loss = hard_kd_loss(logits, teacher_probs, target_ids, pad_id=99)
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), torch.tensor([0]))
    assert torch.allclose(loss, expected)


def test_ce_loss_uses_sampled_target_not_teacher_argmax() -> None:
    logits = torch.tensor([[[4.0, 0.0]]])
    target_ids = torch.tensor([[1]])
    loss = ce_loss(logits, target_ids, pad_id=99)
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), torch.tensor([1]))
    assert torch.allclose(loss, expected)


def test_soft_kd_loss_zero_when_student_matches_teacher() -> None:
    teacher_probs = torch.tensor([[[0.8, 0.2], [0.4, 0.6]]])
    logits = teacher_probs.log()
    target_ids = torch.tensor([[3, 4]])
    loss = soft_kd_loss(logits, teacher_probs, target_ids, pad_id=0)
    assert loss.item() < 1e-6


def test_hybrid_kd_loss_interpolates() -> None:
    logits = torch.tensor([[[2.0, 0.0]]])
    teacher_probs = torch.tensor([[[0.7, 0.3]]])
    target_ids = torch.tensor([[1]])
    hard = hard_kd_loss(logits, teacher_probs, target_ids, pad_id=0)
    soft = soft_kd_loss(logits, teacher_probs, target_ids, pad_id=0)
    hybrid = hybrid_kd_loss(logits, teacher_probs, target_ids, pad_id=0, lambda_soft=0.25)
    local_lambda = 0.25 * ((1.0 - 0.7) / 0.5)
    assert torch.allclose(hybrid, local_lambda * soft + (1.0 - local_lambda) * hard)


def test_hybrid_kd_loss_uses_role_aware_gate_when_available() -> None:
    logits = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]])
    teacher_probs = torch.tensor([[[0.7, 0.3], [0.7, 0.3]]])
    target_ids = torch.tensor([[1, 1]])
    role_ids = torch.tensor([[1, 2]])
    hard = hard_kd_loss(logits[:, :1], teacher_probs[:, :1], target_ids[:, :1], pad_id=99)
    soft = soft_kd_loss(logits[:, 1:], teacher_probs[:, 1:], target_ids[:, 1:], pad_id=99)
    hybrid = hybrid_kd_loss(
        logits,
        teacher_probs,
        target_ids,
        pad_id=99,
        lambda_soft=0.25,
        role_ids=role_ids,
    )
    assert torch.allclose(hybrid, (hard + soft) / 2.0)


def test_train_oracle_student_runs_one_epoch() -> None:
    bundle = build_synthetic_domain("code", sample_count=2)
    batch = materialize_mode_paths(bundle.oracle, sample_ids=[0, 1], vocab_size=len(bundle.vocab))
    model, history = train_oracle_student(
        method="soft_kd",
        input_ids=batch.input_ids,
        target_ids=batch.target_ids,
        teacher_probs=batch.teacher_probs,
        vocab_size=len(bundle.vocab),
        model_config=ModelConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0),
        max_len=batch.input_ids.shape[1],
        pad_id=bundle.oracle.pad_id,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        device=torch.device("cpu"),
        role_ids=batch.role_ids,
    )
    assert history["train_loss"][0] > 0.0
    logits = model(batch.input_ids)
    assert logits.shape == (2, 47, 64)


def test_train_oracle_student_records_validation_kl() -> None:
    bundle = build_synthetic_domain("code", sample_count=3)
    batch = materialize_mode_paths(bundle.oracle, sample_ids=[0, 1], vocab_size=len(bundle.vocab))
    val_batch = materialize_mode_paths(bundle.oracle, sample_ids=[2], vocab_size=len(bundle.vocab))
    _, history = train_oracle_student(
        method="ce_ref",
        input_ids=batch.input_ids,
        target_ids=batch.target_ids,
        teacher_probs=batch.teacher_probs,
        vocab_size=len(bundle.vocab),
        model_config=ModelConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0),
        max_len=batch.input_ids.shape[1],
        pad_id=bundle.oracle.pad_id,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        device=torch.device("cpu"),
        val_input_ids=val_batch.input_ids,
        val_target_ids=val_batch.target_ids,
        val_teacher_probs=val_batch.teacher_probs,
        early_stopping_patience=3,
    )
    assert len(history["val_teacher_forced_kl"]) == len(history["train_loss"])
    assert len(history["val_early_stopping_objective"]) == len(history["train_loss"])
    assert history["early_stopping_metric"] == ["val_teacher_forced_kl"]
    assert history["best_epoch"][0] >= 0
    assert history["best_val_early_stopping_objective"][0] == min(history["val_early_stopping_objective"])
    assert history["best_val_teacher_forced_kl"][0] == min(history["val_teacher_forced_kl"])


def test_train_oracle_student_records_warmup_cosine_learning_rates() -> None:
    bundle = build_synthetic_domain("code", sample_count=3)
    batch = materialize_mode_paths(bundle.oracle, sample_ids=[0, 1], vocab_size=len(bundle.vocab))
    val_batch = materialize_mode_paths(bundle.oracle, sample_ids=[2], vocab_size=len(bundle.vocab))
    _, history = train_oracle_student(
        method="soft_kd",
        input_ids=batch.input_ids,
        target_ids=batch.target_ids,
        teacher_probs=batch.teacher_probs,
        vocab_size=len(bundle.vocab),
        model_config=ModelConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0),
        max_len=batch.input_ids.shape[1],
        pad_id=bundle.oracle.pad_id,
        epochs=5,
        batch_size=2,
        lr=3e-4,
        lr_schedule="warmup_cosine",
        lr_warmup_epochs=3,
        device=torch.device("cpu"),
        val_input_ids=val_batch.input_ids,
        val_target_ids=val_batch.target_ids,
        val_teacher_probs=val_batch.teacher_probs,
        early_stopping_patience=10,
    )
    assert history["lr_schedule"] == ["warmup_cosine"]
    assert history["lr_peak"] == [3e-4]
    assert history["lr_warmup_epochs"] == [3.0]
    assert history["learning_rate"] == [
        _epoch_learning_rate(peak_lr=3e-4, epoch=epoch, epochs=5, schedule="warmup_cosine", warmup_epochs=3)
        for epoch in range(len(history["learning_rate"]))
    ]
    assert history["learning_rate"][:3] == pytest.approx([1e-4, 2e-4, 3e-4])


def test_hard_kd_checkpoint_selection_uses_teacher_forced_kl() -> None:
    bundle = build_synthetic_domain("code", sample_count=3)
    batch = materialize_mode_paths(bundle.oracle, sample_ids=[0, 1], vocab_size=len(bundle.vocab))
    val_batch = materialize_mode_paths(bundle.oracle, sample_ids=[2], vocab_size=len(bundle.vocab))
    _, history = train_oracle_student(
        method="hard_kd",
        input_ids=batch.input_ids,
        target_ids=batch.target_ids,
        teacher_probs=batch.teacher_probs,
        vocab_size=len(bundle.vocab),
        model_config=ModelConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0),
        max_len=batch.input_ids.shape[1],
        pad_id=bundle.oracle.pad_id,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        device=torch.device("cpu"),
        val_input_ids=val_batch.input_ids,
        val_target_ids=val_batch.target_ids,
        val_teacher_probs=val_batch.teacher_probs,
        early_stopping_patience=3,
    )
    assert history["early_stopping_metric"] == ["val_teacher_forced_kl"]
    assert history["early_stopping_relative_min_delta"] == [0.0]
    assert len(history["val_early_stopping_objective"]) == len(history["train_loss"])
    assert history["best_val_early_stopping_objective"][0] == min(history["val_early_stopping_objective"])
    assert history["best_val_teacher_forced_kl"][0] == min(history["val_teacher_forced_kl"])
    assert history["best_epoch"] == history["best_teacher_forced_kl_epoch"]


def test_hard_kd_checkpoint_selection_can_use_method_loss() -> None:
    bundle = build_synthetic_domain("code", sample_count=3)
    batch = materialize_mode_paths(bundle.oracle, sample_ids=[0, 1], vocab_size=len(bundle.vocab))
    val_batch = materialize_mode_paths(bundle.oracle, sample_ids=[2], vocab_size=len(bundle.vocab))
    _, history = train_oracle_student(
        method="hard_kd",
        input_ids=batch.input_ids,
        target_ids=batch.target_ids,
        teacher_probs=batch.teacher_probs,
        vocab_size=len(bundle.vocab),
        model_config=ModelConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0),
        max_len=batch.input_ids.shape[1],
        pad_id=bundle.oracle.pad_id,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        device=torch.device("cpu"),
        val_input_ids=val_batch.input_ids,
        val_target_ids=val_batch.target_ids,
        val_teacher_probs=val_batch.teacher_probs,
        early_stopping_patience=3,
        early_stopping_metric="val_method_loss",
    )
    assert history["early_stopping_metric"] == ["val_method_loss"]
    assert len(history["val_method_loss"]) == len(history["train_loss"])
    assert history["val_early_stopping_objective"] == history["val_method_loss"]
    assert history["best_val_early_stopping_objective"][0] == min(history["val_method_loss"])
    assert history["best_val_teacher_forced_kl"][0] == min(history["val_teacher_forced_kl"])


def test_hard_kd_relative_min_delta_requires_material_improvement() -> None:
    assert _is_significant_improvement(0.99, 1.0, relative_min_delta=0.01) is False
    assert _is_significant_improvement(0.989, 1.0, relative_min_delta=0.01) is True
    assert _is_significant_improvement(0.999, 1.0, relative_min_delta=0.0) is True
