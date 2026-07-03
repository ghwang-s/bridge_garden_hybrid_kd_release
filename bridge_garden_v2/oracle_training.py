from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .modeling import CausalTransformerLM
from .schema import ModelConfig


def valid_target_mask(target_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    return (target_ids != pad_id).float()


def hard_kd_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor, target_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    mask = valid_target_mask(target_ids, pad_id)
    per_token = _hard_kd_per_token(student_logits, teacher_probs)
    return _masked_mean(per_token, mask)


def ce_loss(student_logits: torch.Tensor, target_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    per_token = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        target_ids.reshape(-1),
        ignore_index=pad_id,
        reduction="none",
    ).reshape(target_ids.shape)
    mask = valid_target_mask(target_ids, pad_id)
    return _masked_mean(per_token, mask)


def _hard_kd_per_token(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    hard_targets = torch.argmax(teacher_probs, dim=-1)
    return F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        hard_targets.reshape(-1),
        reduction="none",
    ).reshape(hard_targets.shape)


def soft_kd_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor, target_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    mask = valid_target_mask(target_ids, pad_id)
    per_token = _soft_kd_per_token(student_logits, teacher_probs)
    return _masked_mean(per_token, mask)


def _soft_kd_per_token(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(log_probs, teacher_probs, reduction="none").sum(dim=-1)


def hybrid_kd_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
    lambda_soft: float,
    role_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if not 0.0 <= lambda_soft <= 1.0:
        raise ValueError("lambda_soft must be in [0, 1]")
    mask = valid_target_mask(target_ids, pad_id)
    soft = _soft_kd_per_token(student_logits, teacher_probs)
    hard = _hard_kd_per_token(student_logits, teacher_probs)
    local_lambda = _hybrid_soft_weight(teacher_probs, lambda_soft=lambda_soft, role_ids=role_ids)
    per_token = local_lambda * soft + (1.0 - local_lambda) * hard
    return _masked_mean(per_token, mask)


def teacher_forced_kl_from_probs(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    return soft_kd_loss(student_logits, teacher_probs, target_ids, pad_id)


def train_oracle_student(
    *,
    method: str,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    teacher_probs: torch.Tensor,
    vocab_size: int,
    model_config: ModelConfig,
    max_len: int,
    pad_id: int,
    epochs: int,
    batch_size: int,
    lr: float,
    lr_schedule: str = "constant",
    lr_warmup_epochs: int = 0,
    lambda_soft: float = 0.5,
    seed: int = 0,
    device: torch.device | None = None,
    val_input_ids: torch.Tensor | None = None,
    val_target_ids: torch.Tensor | None = None,
    val_teacher_probs: torch.Tensor | None = None,
    role_ids: torch.Tensor | None = None,
    val_role_ids: torch.Tensor | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_metric: str = "val_teacher_forced_kl",
) -> tuple[CausalTransformerLM, dict[str, list[float]]]:
    """Train one student directly against oracle teacher probabilities."""
    if method not in {"ce_ref", "hard_kd", "soft_kd", "hybrid_kd"}:
        raise ValueError(f"unknown oracle student method: {method}")
    if lr_schedule not in {"constant", "warmup_cosine"}:
        raise ValueError(f"unknown learning-rate schedule: {lr_schedule}")
    if lr_warmup_epochs < 0:
        raise ValueError("lr_warmup_epochs must be non-negative")
    if lr_schedule == "constant" and lr_warmup_epochs != 0:
        raise ValueError("constant learning-rate schedule requires lr_warmup_epochs=0")
    if lr_schedule == "warmup_cosine" and lr_warmup_epochs <= 0:
        raise ValueError("warmup_cosine learning-rate schedule requires positive lr_warmup_epochs")
    if early_stopping_metric not in {"val_teacher_forced_kl", "val_method_loss"}:
        raise ValueError("early_stopping_metric must be val_teacher_forced_kl or val_method_loss")
    torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTransformerLM(vocab_size, model_config, max_len, pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    has_role_ids = role_ids is not None
    if role_ids is None:
        role_ids = torch.zeros_like(target_ids, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(input_ids, target_ids, teacher_probs, role_ids),
        batch_size=batch_size,
        shuffle=True,
    )
    relative_min_delta = 0.0
    history = {
        "train_loss": [],
        "learning_rate": [],
        "val_teacher_forced_kl": [],
        "val_method_loss": [],
        "val_early_stopping_objective": [],
        "lr_schedule": [lr_schedule],
        "lr_peak": [float(lr)],
        "lr_warmup_epochs": [float(lr_warmup_epochs)],
        "early_stopping_metric": [early_stopping_metric],
        "early_stopping_relative_min_delta": [relative_min_delta],
        "early_stopping_patience": [float(early_stopping_patience)]
        if early_stopping_patience is not None
        else [None],
    }
    best_state = None
    best_val = float("inf")
    best_epoch = -1
    best_teacher_forced_kl = float("inf")
    best_teacher_forced_kl_epoch = -1
    stale_epochs = 0
    for epoch in range(epochs):
        epoch_lr = _epoch_learning_rate(
            peak_lr=lr,
            epoch=epoch,
            epochs=epochs,
            schedule=lr_schedule,
            warmup_epochs=lr_warmup_epochs,
        )
        for group in optimizer.param_groups:
            group["lr"] = epoch_lr
        model.train()
        losses = []
        for batch_inp, batch_tgt, batch_probs, batch_roles in loader:
            batch_inp = batch_inp.to(device)
            batch_tgt = batch_tgt.to(device)
            batch_probs = batch_probs.to(device)
            batch_roles = batch_roles.to(device)
            logits = model(batch_inp)
            if method == "ce_ref":
                loss = ce_loss(logits, batch_tgt, pad_id)
            elif method == "hard_kd":
                loss = hard_kd_loss(logits, batch_probs, batch_tgt, pad_id)
            elif method == "soft_kd":
                loss = soft_kd_loss(logits, batch_probs, batch_tgt, pad_id)
            else:
                loss = hybrid_kd_loss(
                    logits,
                    batch_probs,
                    batch_tgt,
                    pad_id,
                    lambda_soft=lambda_soft,
                    role_ids=batch_roles if has_role_ids else None,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history["learning_rate"].append(float(epoch_lr))
        history["train_loss"].append(float(sum(losses) / max(1, len(losses))))
        if val_input_ids is not None and val_target_ids is not None and val_teacher_probs is not None:
            val_kl = _eval_teacher_forced_kl(
                model=model,
                input_ids=val_input_ids,
                target_ids=val_target_ids,
                teacher_probs=val_teacher_probs,
                role_ids=val_role_ids,
                pad_id=pad_id,
                batch_size=batch_size,
                device=device,
            )
            val_method_loss = _eval_method_validation_loss(
                method=method,
                model=model,
                input_ids=val_input_ids,
                target_ids=val_target_ids,
                teacher_probs=val_teacher_probs,
                role_ids=val_role_ids,
                pad_id=pad_id,
                batch_size=batch_size,
                device=device,
                lambda_soft=lambda_soft,
            )
            val_objective = val_kl if early_stopping_metric == "val_teacher_forced_kl" else val_method_loss
            history["val_teacher_forced_kl"].append(val_kl)
            history["val_method_loss"].append(val_method_loss)
            history["val_early_stopping_objective"].append(val_objective)
            if val_kl < best_teacher_forced_kl:
                best_teacher_forced_kl = val_kl
                best_teacher_forced_kl_epoch = epoch
            if _is_significant_improvement(val_objective, best_val, relative_min_delta=relative_min_delta):
                best_val = val_objective
                best_epoch = epoch
                stale_epochs = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale_epochs += 1
            if early_stopping_patience is not None and stale_epochs >= early_stopping_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if best_epoch >= 0:
        history["best_epoch"] = [float(best_epoch)]
        history["best_val_early_stopping_objective"] = [float(best_val)]
    if best_teacher_forced_kl_epoch >= 0:
        history["best_teacher_forced_kl_epoch"] = [float(best_teacher_forced_kl_epoch)]
        history["best_val_teacher_forced_kl"] = [float(best_teacher_forced_kl)]
    return model, history


def _epoch_learning_rate(
    *,
    peak_lr: float,
    epoch: int,
    epochs: int,
    schedule: str,
    warmup_epochs: int,
) -> float:
    if schedule == "constant":
        return float(peak_lr)
    if schedule != "warmup_cosine":
        raise ValueError(f"unknown learning-rate schedule: {schedule}")
    if warmup_epochs <= 0:
        raise ValueError("warmup_cosine learning-rate schedule requires positive warmup_epochs")
    if epoch < warmup_epochs:
        return float(peak_lr) * float(epoch + 1) / float(warmup_epochs)
    decay_steps = max(1, int(epochs) - int(warmup_epochs) - 1)
    progress = min(1.0, max(0.0, float(epoch - warmup_epochs) / float(decay_steps)))
    return float(peak_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _is_significant_improvement(value: float, best: float, *, relative_min_delta: float) -> bool:
    if best == float("inf"):
        return True
    if relative_min_delta <= 0.0:
        return value < best
    threshold = abs(best) * float(relative_min_delta)
    return value < best - threshold


def _eval_method_validation_loss(
    *,
    method: str,
    model: CausalTransformerLM,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    teacher_probs: torch.Tensor,
    role_ids: torch.Tensor | None,
    pad_id: int,
    batch_size: int,
    device: torch.device,
    lambda_soft: float,
) -> float:
    model.eval()
    if role_ids is None:
        role_ids = torch.zeros_like(target_ids, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(input_ids, target_ids, teacher_probs, role_ids),
        batch_size=batch_size,
        shuffle=False,
    )
    weighted_sum = 0.0
    token_count = 0.0
    with torch.no_grad():
        for batch_inp, batch_tgt, batch_probs, batch_roles in loader:
            batch_inp = batch_inp.to(device)
            batch_tgt = batch_tgt.to(device)
            batch_probs = batch_probs.to(device)
            batch_roles = batch_roles.to(device)
            logits = model(batch_inp)
            if method == "ce_ref":
                per_token = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch_tgt.reshape(-1),
                    ignore_index=pad_id,
                    reduction="none",
                ).reshape(batch_tgt.shape)
            elif method == "hard_kd":
                per_token = _hard_kd_per_token(logits, batch_probs)
            elif method == "soft_kd":
                per_token = _soft_kd_per_token(logits, batch_probs)
            elif method == "hybrid_kd":
                local_lambda = _hybrid_soft_weight(batch_probs, lambda_soft=lambda_soft, role_ids=batch_roles)
                per_token = (
                    local_lambda * _soft_kd_per_token(logits, batch_probs)
                    + (1.0 - local_lambda) * _hard_kd_per_token(logits, batch_probs)
                )
            else:
                raise ValueError(f"unknown oracle student method: {method}")
            mask = valid_target_mask(batch_tgt, pad_id)
            weighted_sum += float((per_token * mask).sum().detach().cpu())
            token_count += float(mask.sum().detach().cpu())
    return weighted_sum / max(1.0, token_count)


def _eval_teacher_forced_kl(
    *,
    model: CausalTransformerLM,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    teacher_probs: torch.Tensor,
    role_ids: torch.Tensor | None = None,
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    if role_ids is None:
        role_ids = torch.zeros_like(target_ids, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(input_ids, target_ids, teacher_probs, role_ids),
        batch_size=batch_size,
        shuffle=False,
    )
    weighted_sum = 0.0
    token_count = 0.0
    with torch.no_grad():
        for batch_inp, batch_tgt, batch_probs, _batch_roles in loader:
            batch_inp = batch_inp.to(device)
            batch_tgt = batch_tgt.to(device)
            batch_probs = batch_probs.to(device)
            logits = model(batch_inp)
            per_token = _soft_kd_per_token(logits, batch_probs)
            mask = valid_target_mask(batch_tgt, pad_id)
            weighted_sum += float((per_token * mask).sum().detach().cpu())
            token_count += float(mask.sum().detach().cpu())
    return weighted_sum / max(1.0, token_count)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def _teacher_ambiguity(teacher_probs: torch.Tensor) -> torch.Tensor:
    support = (teacher_probs > 0.0).float()
    support_size = support.sum(dim=-1).clamp(min=1.0)
    max_prob = teacher_probs.max(dim=-1).values
    denom = (1.0 - 1.0 / support_size).clamp(min=1e-6)
    ambiguity = (1.0 - max_prob) / denom
    return ambiguity.clamp(0.0, 1.0)


def _hybrid_soft_weight(
    teacher_probs: torch.Tensor,
    *,
    lambda_soft: float,
    role_ids: torch.Tensor | None,
) -> torch.Tensor:
    ambiguity_weight = lambda_soft * _teacher_ambiguity(teacher_probs)
    if role_ids is None:
        return ambiguity_weight
    role_ids = role_ids.to(device=teacher_probs.device)
    role_weight = ambiguity_weight
    role_weight = torch.where(role_ids == 1, torch.zeros_like(role_weight), role_weight)
    role_weight = torch.where(role_ids == 2, torch.ones_like(role_weight), role_weight)
    role_weight = torch.where(role_ids == 3, torch.ones_like(role_weight), role_weight)
    return role_weight
