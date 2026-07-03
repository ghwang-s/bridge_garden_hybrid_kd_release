from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PerStepMetric:
    values: np.ndarray
    counts: np.ndarray


def build_valid_target_mask(batch_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Mask for positions whose next-token target is real, not PAD."""
    return (batch_ids[:, 1:] != pad_id).float()


def teacher_forced_kl(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    batch_ids: torch.Tensor,
    pad_id: int,
) -> PerStepMetric:
    inp, tgt = batch_ids[:, :-1], batch_ids[:, 1:]
    tp = F.softmax(teacher(inp), dim=-1)
    sp = F.softmax(student(inp), dim=-1)
    kl = (tp * (tp.clamp(1e-10).log() - sp.clamp(1e-10).log())).sum(dim=-1)
    mask = (tgt != pad_id).float()
    return PerStepMetric(
        values=(kl * mask).sum(dim=0).detach().cpu().numpy(),
        counts=mask.sum(dim=0).detach().cpu().numpy(),
    )


def masked_mean(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.divide(values, counts, out=np.zeros_like(values), where=counts > 0)


def rollout_kl_full(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    gold_batch: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> PerStepMetric:
    """
    Full autoregressive rollout KL with EOS-aware stopping.

    This estimates E_{s~d_theta}[ell_soft(s;theta)] under the student-generated
    prefix distribution. A state contributes if the student trajectory is still alive.
    """
    device = gold_batch.device
    batch_size, full_len = gold_batch.shape
    t_pred = full_len - 1
    generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    alive = torch.ones(batch_size, dtype=torch.bool, device=device)
    sums = np.zeros(t_pred, dtype=np.float64)
    counts = np.zeros(t_pred, dtype=np.float64)

    for step in range(t_pred):
        tp = F.softmax(teacher(generated)[:, -1], dim=-1)
        sp = F.softmax(student(generated)[:, -1], dim=-1)
        kl = (tp * (tp.clamp(1e-10).log() - sp.clamp(1e-10).log())).sum(dim=-1)
        valid = alive
        if valid.any():
            kl_np = kl[valid].detach().cpu().numpy()
            sums[step] += kl_np.sum()
            counts[step] += kl_np.shape[0]
        next_tok = torch.multinomial(sp, 1)
        generated = torch.cat([generated, next_tok], dim=1)
        alive = alive & (next_tok[:, 0] != eos_id)

    return PerStepMetric(values=sums, counts=counts)


def rollout_kl_windowed(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    gold_batch: torch.Tensor,
    window: int,
    eos_id: int,
    pad_id: int,
) -> PerStepMetric:
    """
    Metric-only windowed rollout KL.

    For position p, use the gold prefix up to p-window, then let the student generate
    `window` steps. Teacher and student are compared at the resulting prefix.
    """
    device = gold_batch.device
    batch_size, full_len = gold_batch.shape
    t_pred = full_len - 1
    sums = np.zeros(t_pred, dtype=np.float64)
    counts = np.zeros(t_pred, dtype=np.float64)

    for pos in range(window, t_pred):
        prefix = gold_batch[:, : pos + 1 - window].clone()
        generated = prefix
        alive = torch.ones(batch_size, dtype=torch.bool, device=device)
        for _ in range(window):
            sp = F.softmax(student(generated)[:, -1], dim=-1)
            next_tok = torch.multinomial(sp, 1)
            generated = torch.cat([generated, next_tok], dim=1)
            alive = alive & (next_tok[:, 0] != eos_id)
        tp = F.softmax(teacher(generated)[:, -1], dim=-1)
        sp = F.softmax(student(generated)[:, -1], dim=-1)
        kl = (tp * (tp.clamp(1e-10).log() - sp.clamp(1e-10).log())).sum(dim=-1)
        valid = alive
        if valid.any():
            kl_np = kl[valid].detach().cpu().numpy()
            sums[pos] += kl_np.sum()
            counts[pos] += kl_np.shape[0]

    return PerStepMetric(values=sums, counts=counts)
