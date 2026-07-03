from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .domains import build_domain
from .io_utils import ensure_dir, save_json, save_pt
from .metrics import (
    PerStepMetric,
    masked_mean,
    rollout_kl_full,
    rollout_kl_windowed,
    teacher_forced_kl,
)
from .modeling import CausalTransformerLM
from .schema import (
    DomainDefinition,
    ExperimentConfig,
    GenerationConfig,
    ModelConfig,
    RegionMaskBundle,
    SequenceRecord,
)


def records_to_padded_tensor(records: List[SequenceRecord], pad_id: int) -> torch.Tensor:
    max_len = max(len(r.token_ids) for r in records)
    tensor = torch.full((len(records), max_len), pad_id, dtype=torch.long)
    for i, record in enumerate(records):
        seq = torch.tensor(record.token_ids, dtype=torch.long)
        tensor[i, : seq.numel()] = seq
    return tensor


def _smoke_check_records(records: List[SequenceRecord], domain: DomainDefinition) -> None:
    for idx, record in enumerate(records[:10]):
        if len(record.token_ids) != len(record.tokens):
            raise ValueError(f"{domain.name}: token/id length mismatch at sample {idx}")
        if len(record.token_ids) != len(record.oracle_roles):
            raise ValueError(f"{domain.name}: oracle role length mismatch at sample {idx}")
        if record.token_ids[0] != domain.special_ids.bos:
            raise ValueError(f"{domain.name}: sample {idx} missing BOS at position 0")
        if record.token_ids[-1] != domain.special_ids.eos:
            raise ValueError(f"{domain.name}: sample {idx} missing EOS at final position")


def generate_domain_splits(domain_name: str, generation: GenerationConfig) -> Tuple[DomainDefinition, Dict[str, List[SequenceRecord]]]:
    domain = build_domain(domain_name)
    splits = {
        "train": domain.generate_split("train", generation),
        "val": domain.generate_split("val", generation),
        "test": domain.generate_split("test", generation),
        "analysis": domain.generate_split("analysis", generation),
    }
    for split_records in splits.values():
        _smoke_check_records(split_records, domain)
    return domain, splits


def save_generation_artifacts(artifact_root: Path, config: ExperimentConfig, domain: DomainDefinition, splits: Dict[str, List[SequenceRecord]]) -> Dict[str, Path]:
    dataset_dir = ensure_dir(artifact_root / "datasets")
    paths = {}
    save_json(dataset_dir / "config.json", asdict(config))
    save_json(dataset_dir / "domain.json", {
        "name": domain.name,
        "vocab": domain.vocab,
        "special_ids": asdict(domain.special_ids),
        "default_config": dict(domain.default_config),
    })
    for split_name, records in splits.items():
        payload = [record.to_dict() for record in records]
        path = dataset_dir / f"{split_name}.json"
        save_json(path, payload)
        paths[split_name] = path
    return paths


def _make_loader(tensor: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    vocab = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        ignore_index=pad_id,
    )


def _masked_ce_loss_with_valid_targets(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_targets: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_labels = labels.reshape(-1)
    flat_valid = valid_targets.reshape(-1)
    per_token = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    mask = (flat_valid != pad_id).float()
    denom = mask.sum().clamp(min=1.0)
    return (per_token * mask).sum() / denom


def _masked_soft_kl(student_logits: torch.Tensor, teacher_probs: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    log_probs = F.log_softmax(student_logits, dim=-1)
    kl = F.kl_div(log_probs, teacher_probs, reduction="none").sum(dim=-1)
    mask = (targets != pad_id).float()
    denom = mask.sum().clamp(min=1.0)
    return (kl * mask).sum() / denom


def train_teacher(
    model: CausalTransformerLM,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    pad_id: int,
    device: torch.device,
    checkpoint_dir: Path,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}
    train_loader = _make_loader(train_ids, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(val_ids, batch_size=batch_size, shuffle=False)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            loss = _masked_ce_loss(model(inp), tgt, pad_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history["train_loss"].append(float(np.mean(losses)))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                inp, tgt = batch[:, :-1], batch[:, 1:]
                val_losses.append(float(_masked_ce_loss(model(inp), tgt, pad_id).detach().cpu()))
        history["val_loss"].append(float(np.mean(val_losses)))
        save_pt(checkpoint_dir / f"teacher_ep{epoch}.pt", model.state_dict())
    return history


def train_student_method(
    method_name: str,
    teacher: CausalTransformerLM,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    student_cfg: ModelConfig,
    model_max_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    pad_id: int,
    device: torch.device,
    checkpoint_dir: Path,
    lambda_soft: float = 0.5,
) -> Dict[str, object]:
    student = CausalTransformerLM(teacher.head.out_features, student_cfg, model_max_len, pad_id).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    teacher.eval()
    train_loader = _make_loader(train_ids, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(val_ids, batch_size=batch_size, shuffle=False)
    history: Dict[str, object] = {"train_loss": [], "val_loss": [], "checkpoint_paths": []}

    for epoch in range(1, epochs + 1):
        student.train()
        losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            with torch.no_grad():
                teacher_probs = F.softmax(teacher(inp), dim=-1)
                teacher_hard = teacher_probs.argmax(dim=-1)
            student_logits = student(inp)
            hard_loss = _masked_ce_loss_with_valid_targets(student_logits, teacher_hard, tgt, pad_id)
            soft_loss = _masked_soft_kl(student_logits, teacher_probs, tgt, pad_id)
            if method_name == "hard":
                loss = hard_loss
            elif method_name == "soft":
                loss = soft_loss
            else:
                loss = lambda_soft * soft_loss + (1.0 - lambda_soft) * hard_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history["train_loss"].append(float(np.mean(losses)))

        student.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                inp, tgt = batch[:, :-1], batch[:, 1:]
                with torch.no_grad():
                    teacher_probs = F.softmax(teacher(inp), dim=-1)
                    teacher_hard = teacher_probs.argmax(dim=-1)
                student_logits = student(inp)
                hard_loss = _masked_ce_loss_with_valid_targets(student_logits, teacher_hard, tgt, pad_id)
                soft_loss = _masked_soft_kl(student_logits, teacher_probs, tgt, pad_id)
                if method_name == "hard":
                    val_losses.append(float(hard_loss.detach().cpu()))
                elif method_name == "soft":
                    val_losses.append(float(soft_loss.detach().cpu()))
                else:
                    val_losses.append(float((lambda_soft * soft_loss + (1.0 - lambda_soft) * hard_loss).detach().cpu()))
        history["val_loss"].append(float(np.mean(val_losses)))
        ckpt_path = checkpoint_dir / f"student_{method_name}_ep{epoch}.pt"
        save_pt(ckpt_path, student.state_dict())
        history["checkpoint_paths"].append(str(ckpt_path))
    history["final_state_dict"] = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    return history


def compute_counterfactual_kappa(
    teacher: CausalTransformerLM,
    records: List[SequenceRecord],
    batch_ids: torch.Tensor,
    domain: DomainDefinition,
    future_len: int,
    n_alts: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Generic teacher-only kappa estimate.

    For a step p, replace the gold token with sampled alternatives and compare the
    teacher's future predictive distributions over the next `future_len` gold steps.
    """
    teacher.eval()
    batch_ids = batch_ids.to(device)
    n_seq, full_len = batch_ids.shape
    t_pred = full_len - 1
    kappas = np.full((n_seq, t_pred), np.nan, dtype=np.float64)
    coverage = np.zeros((n_seq, t_pred), dtype=np.int32)

    with torch.no_grad():
        for si, record in enumerate(records):
            seq_len = len(record.token_ids)
            for pos in range(0, seq_len - 2):
                alts = [alt for alt in domain.sample_alternatives(record, pos, n_alts) if alt != record.token_ids[pos + 1]]
                if not alts:
                    continue
                end = min(seq_len, pos + 2 + future_len)
                if end - (pos + 2) <= 0:
                    continue
                base_seq = batch_ids[si : si + 1, :end]
                base_logits = teacher(base_seq[:, :-1])
                start_idx = pos + 1
                base_future = F.softmax(base_logits[:, start_idx : end - 1], dim=-1)
                alt_scores = []
                for alt in alts:
                    alt_seq = base_seq.clone()
                    alt_seq[0, pos + 1] = alt
                    alt_logits = teacher(alt_seq[:, :-1])
                    alt_future = F.softmax(alt_logits[:, start_idx : end - 1], dim=-1)
                    kl = (base_future * (base_future.clamp(1e-10).log() - alt_future.clamp(1e-10).log())).sum(dim=-1)
                    alt_scores.append(float(kl.mean().detach().cpu()))
                if alt_scores:
                    kappas[si, pos] = float(np.mean(alt_scores))
                    coverage[si, pos] = len(alt_scores)
    return {"kappas": kappas, "coverage": coverage}


def _teacher_student_soft_kl_at_prefix(
    teacher: CausalTransformerLM,
    student: CausalTransformerLM,
    prefix_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute ℓ_soft(s;θ) = KL(π_T(.|s) || π_θ(.|s)) at one prefix state s."""
    teacher_probs = F.softmax(teacher(prefix_ids)[:, -1], dim=-1)
    student_probs = F.softmax(student(prefix_ids)[:, -1], dim=-1)
    return (teacher_probs * (teacher_probs.clamp(1e-10).log() - student_probs.clamp(1e-10).log())).sum(dim=-1)


def _rollout_teacher_continuation_loss_to_go(
    teacher: CausalTransformerLM,
    student: CausalTransformerLM,
    prefix_after_action: torch.Tensor,
    horizon: int,
    eos_id: int,
) -> float:
    """
    Approximate the loss-to-go Q(s,a) from the paper.

    Starting from the overridden prefix (state s followed by action a), follow the
    teacher policy for `horizon` future steps and accumulate future ℓ_soft.
    """
    generated = prefix_after_action.clone()
    total = 0.0
    for _ in range(horizon):
        if generated[0, -1].item() == eos_id:
            break
        step_loss = _teacher_student_soft_kl_at_prefix(teacher, student, generated)
        total += float(step_loss.item())
        teacher_probs = F.softmax(teacher(generated)[:, -1], dim=-1)
        next_tok = torch.multinomial(teacher_probs, 1)
        generated = torch.cat([generated, next_tok], dim=1)
    return total


def _rollout_teacher_continuation_loss_to_go_batched(
    teacher: CausalTransformerLM,
    student: CausalTransformerLM,
    prefixes_after_action: torch.Tensor,
    horizon: int,
    eos_id: int,
) -> np.ndarray:
    """
    Batched Q(s,a) estimation for all B actions simultaneously.

    prefixes_after_action: [B, seq_len]  — one row per action candidate
    Returns: [B] cumulative soft-KL loss-to-go for each action.
    """
    device = prefixes_after_action.device
    B = prefixes_after_action.shape[0]
    generated = prefixes_after_action.clone()
    totals = torch.zeros(B, dtype=torch.float64, device=device)
    alive = torch.ones(B, dtype=torch.bool, device=device)

    for _ in range(horizon):
        alive = alive & (generated[:, -1] != eos_id)
        if not alive.any():
            break
        tp = F.softmax(teacher(generated)[:, -1], dim=-1)   # [B, V]
        sp = F.softmax(student(generated)[:, -1], dim=-1)   # [B, V]
        kl = (tp * (tp.clamp(1e-10).log() - sp.clamp(1e-10).log())).sum(dim=-1)  # [B]
        totals += kl.double() * alive.double()
        next_toks = torch.multinomial(tp, 1)                 # [B, 1]
        generated = torch.cat([generated, next_toks], dim=1)

    return totals.cpu().numpy()


def compute_kappa_definition_reference(
    teacher: CausalTransformerLM,
    student: CausalTransformerLM,
    records: List[SequenceRecord],
    batch_ids: torch.Tensor,
    domain: DomainDefinition,
    special_ids,
    device: torch.device,
    *,
    horizon: int,
    n_rollouts: int,
    max_action_count: int,
) -> Dict[str, np.ndarray]:
    """

      kappa(a|s) = [L_{d^(s,a)}(pi_theta) - L_{dT}(pi_theta)] / dT(s)
                 = Q(s,a) - Qbar(s)

    We estimate Q(s,a) via a single-override action followed by teacher continuation

    Notes:
    - This is action-level and state-level, not a teacher-only future-logit proxy.
    - Earlier prefix losses cancel, so we only estimate the future loss-to-go from the
      overridden state onward.
    """
    teacher.eval()
    student.eval()
    batch_ids = batch_ids.to(device)
    n_seq, full_len = batch_ids.shape
    t_pred = full_len - 1
    state_kappas = np.full((n_seq, t_pred), np.nan, dtype=np.float64)
    qbars = np.full((n_seq, t_pred), np.nan, dtype=np.float64)
    action_coverage = np.zeros((n_seq, t_pred), dtype=np.int32)

    model_max_len = teacher.pos_emb.num_embeddings
    # prefix (pos+1 tokens) + override (1) + rollout (horizon) must not exceed model_max_len
    max_safe_pos = model_max_len - horizon - 2

    with torch.no_grad():
        for si, record in enumerate(records):
            seq_len = len(record.token_ids)
            for pos in range(0, min(seq_len - 2, max_safe_pos)):
                prefix = batch_ids[si : si + 1, : pos + 1]
                teacher_probs = F.softmax(teacher(prefix)[:, -1], dim=-1)[0]
                if domain.teacher_action_ids is not None:
                    candidate_action_ids = [
                        idx for idx in domain.teacher_action_ids(record, pos)
                        if idx not in {special_ids.pad, special_ids.bos, special_ids.eos}
                    ]
                else:
                    candidate_action_ids = [idx for idx in range(teacher.head.out_features) if idx not in {special_ids.pad, special_ids.bos, special_ids.eos}]
                candidate_action_ids = sorted(candidate_action_ids, key=lambda idx: float(teacher_probs[idx].item()), reverse=True)
                candidate_action_ids = candidate_action_ids[:max_action_count]
                if not candidate_action_ids:
                    continue
                B_act = len(candidate_action_ids)
                action_tensor = torch.tensor(candidate_action_ids, dtype=torch.long, device=device)
                # [B_act, pos+2]: all actions batched into one tensor
                prefixes_after_action = torch.cat(
                    [prefix.expand(B_act, -1), action_tensor.unsqueeze(1)], dim=1
                )
                # Average Q estimates over n_rollouts
                q_matrix = np.zeros((n_rollouts, B_act), dtype=np.float64)
                for r in range(n_rollouts):
                    q_matrix[r] = _rollout_teacher_continuation_loss_to_go_batched(
                        teacher=teacher,
                        student=student,
                        prefixes_after_action=prefixes_after_action,
                        horizon=horizon,
                        eos_id=special_ids.eos,
                    )
                q_values_np = q_matrix.mean(axis=0)  # [B_act]
                weights_np = np.array([float(teacher_probs[aid].item()) for aid in candidate_action_ids], dtype=np.float64)
                if weights_np.sum() <= 0:
                    continue
                weights_np = weights_np / weights_np.sum()
                qbar = float(np.sum(weights_np * q_values_np))
                kappas_a = q_values_np - qbar
                state_kappas[si, pos] = float(np.sum(kappas_a))
                qbars[si, pos] = qbar
                action_coverage[si, pos] = B_act

    return {
        "state_kappas": state_kappas,
        "qbar": qbars,
        "coverage": action_coverage,
    }


def build_region_masks_from_state_kappa(kappa_payload: Dict[str, np.ndarray], min_region_coverage: int) -> RegionMaskBundle:
    kappas = kappa_payload["state_kappas"]
    coverage = kappa_payload["coverage"]
    t_pred = kappas.shape[1]
    mean_kappa = np.full(t_pred, np.nan, dtype=np.float64)
    counts = coverage.sum(axis=0)
    valid_positions = []
    for pos in range(t_pred):
        vals = kappas[:, pos]
        finite = vals[np.isfinite(vals)]
        if finite.size and counts[pos] >= min_region_coverage:
            mean_kappa[pos] = float(finite.mean())
            valid_positions.append(pos)
    if not valid_positions:
        return RegionMaskBundle(
            coverage_counts=counts.astype(int).tolist(),
            valid_positions=[],
            bridge_positions=[],
            garden_positions=[],
            bridge_threshold=float("nan"),
            garden_threshold=float("nan"),
            position_mean_kappa=mean_kappa.tolist(),
        )
    valid_vals = mean_kappa[valid_positions]
    bridge_threshold = float(np.percentile(valid_vals, 70))
    garden_threshold = float(np.percentile(valid_vals, 30))
    bridge_positions = [pos for pos in valid_positions if mean_kappa[pos] >= bridge_threshold]
    garden_positions = [pos for pos in valid_positions if mean_kappa[pos] <= garden_threshold]
    return RegionMaskBundle(
        coverage_counts=counts.astype(int).tolist(),
        valid_positions=valid_positions,
        bridge_positions=bridge_positions,
        garden_positions=garden_positions,
        bridge_threshold=bridge_threshold,
        garden_threshold=garden_threshold,
        position_mean_kappa=mean_kappa.tolist(),
    )


def build_instance_kappa_masks(kappa_payload: Dict[str, np.ndarray]) -> Dict[str, object]:
    kappas = np.array(kappa_payload["state_kappas"], dtype=np.float64)
    finite = kappas[np.isfinite(kappas)]
    if finite.size == 0:
        return {
            "bridge_threshold": float("nan"),
            "garden_threshold": float("nan"),
            "bridge_mask": [],
            "garden_mask": [],
        }
    bridge_threshold = float(np.percentile(finite, 70))
    garden_threshold = float(np.percentile(finite, 30))
    bridge_mask = np.isfinite(kappas) & (kappas >= bridge_threshold)
    garden_mask = np.isfinite(kappas) & (kappas <= garden_threshold)
    return {
        "bridge_threshold": bridge_threshold,
        "garden_threshold": garden_threshold,
        "bridge_mask": bridge_mask.tolist(),
        "garden_mask": garden_mask.tolist(),
    }


def build_region_masks(kappa_payload: Dict[str, np.ndarray], min_region_coverage: int) -> RegionMaskBundle:
    kappas = kappa_payload["kappas"]
    coverage = kappa_payload["coverage"]
    t_pred = kappas.shape[1]
    mean_kappa = np.full(t_pred, np.nan, dtype=np.float64)
    counts = coverage.sum(axis=0)
    valid_positions = []
    for pos in range(t_pred):
        vals = kappas[:, pos]
        finite = vals[np.isfinite(vals)]
        if finite.size and counts[pos] >= min_region_coverage:
            mean_kappa[pos] = float(finite.mean())
            valid_positions.append(pos)
    if not valid_positions:
        return RegionMaskBundle(
            coverage_counts=counts.astype(int).tolist(),
            valid_positions=[],
            bridge_positions=[],
            garden_positions=[],
            bridge_threshold=float("nan"),
            garden_threshold=float("nan"),
            position_mean_kappa=mean_kappa.tolist(),
        )
    valid_vals = mean_kappa[valid_positions]
    bridge_threshold = float(np.percentile(valid_vals, 70))
    garden_threshold = float(np.percentile(valid_vals, 30))
    bridge_positions = [pos for pos in valid_positions if mean_kappa[pos] >= bridge_threshold]
    garden_positions = [pos for pos in valid_positions if mean_kappa[pos] <= garden_threshold]
    return RegionMaskBundle(
        coverage_counts=counts.astype(int).tolist(),
        valid_positions=valid_positions,
        bridge_positions=bridge_positions,
        garden_positions=garden_positions,
        bridge_threshold=bridge_threshold,
        garden_threshold=garden_threshold,
        position_mean_kappa=mean_kappa.tolist(),
    )


def build_oracle_role_positions(records: List[SequenceRecord], pad_id: int) -> Dict[str, object]:
    max_len = max(len(record.token_ids) for record in records)
    t_pred = max_len - 1
    role_names = ["commit", "bridge", "garden", "middle"]
    counts = {role: np.zeros(t_pred, dtype=np.int32) for role in role_names}
    coverage = np.zeros(t_pred, dtype=np.int32)

    for record in records:
        seq_len = len(record.token_ids)
        for pos in range(t_pred):
            token_index = pos + 1
            if token_index >= seq_len:
                continue
            token_id = record.token_ids[token_index]
            if token_id == pad_id:
                continue
            coverage[pos] += 1
            role = record.oracle_roles[token_index]
            if role in counts:
                counts[role][pos] += 1

    dominant = []
    role_positions = {role: [] for role in role_names}
    for pos in range(t_pred):
        if coverage[pos] == 0:
            dominant.append("none")
            continue
        best_role = max(role_names, key=lambda role: counts[role][pos])
        dominant.append(best_role)
        role_positions[best_role].append(pos)

    return {
        "coverage_counts": coverage.astype(int).tolist(),
        "dominant_role_by_position": dominant,
        "role_positions": role_positions,
        "raw_counts": {role: counts[role].astype(int).tolist() for role in role_names},
    }


def compute_region_overlap(region_masks: RegionMaskBundle, oracle_positions: Dict[str, object]) -> Dict[str, object]:
    def _stats(kappa_positions: List[int], oracle_role: str) -> Dict[str, object]:
        kappa_set = set(kappa_positions)
        oracle_set = set(oracle_positions["role_positions"].get(oracle_role, []))
        overlap = sorted(kappa_set & oracle_set)
        union = kappa_set | oracle_set
        return {
            "kappa_count": len(kappa_set),
            "oracle_count": len(oracle_set),
            "overlap_count": len(overlap),
            "overlap_positions": overlap,
            "jaccard": float(len(overlap) / len(union)) if union else float("nan"),
        }

    return {
        "bridge": _stats(region_masks.bridge_positions, "bridge"),
        "garden": _stats(region_masks.garden_positions, "garden"),
    }


def summarize_eb(
    tf_metric: PerStepMetric,
    rollout_metric: PerStepMetric,
    region_masks: RegionMaskBundle,
    oracle_positions: Dict[str, object] | None = None,
    region_overlap: Dict[str, object] | None = None,
) -> Dict[str, object]:
    tf = masked_mean(tf_metric.values, tf_metric.counts)
    ro = masked_mean(rollout_metric.values, rollout_metric.counts)
    eb = ro - tf
    counts = rollout_metric.counts.astype(np.float64)
    valid = counts > 0
    bridge = [pos for pos in region_masks.bridge_positions if pos < eb.shape[0] and valid[pos]]
    garden = [pos for pos in region_masks.garden_positions if pos < eb.shape[0] and valid[pos]]

    def _position_mean(values: np.ndarray, positions: List[int]) -> float:
        if not positions:
            return float("nan")
        return float(values[positions].mean())

    def _weighted_position_mean(values: np.ndarray, positions: List[int]) -> float:
        if not positions:
            return float("nan")
        pos_counts = counts[positions]
        denom = float(pos_counts.sum())
        if denom <= 0:
            return float("nan")
        return float(np.dot(values[positions], pos_counts) / denom)

    valid_positions = np.flatnonzero(valid).tolist()
    summary = {
        "overall_eb": _position_mean(eb, valid_positions),
        "bridge_eb": _position_mean(eb, bridge),
        "garden_eb": _position_mean(eb, garden),
        "overall_eb_weighted": _weighted_position_mean(eb, valid_positions),
        "bridge_eb_weighted": _weighted_position_mean(eb, bridge),
        "garden_eb_weighted": _weighted_position_mean(eb, garden),
        "overall_tf_kl": _position_mean(tf, valid_positions),
        "overall_rollout_kl": _position_mean(ro, valid_positions),
        "overall_tf_kl_weighted": _weighted_position_mean(tf, valid_positions),
        "overall_rollout_kl_weighted": _weighted_position_mean(ro, valid_positions),
        "valid_steps": int(valid.sum()),
        "per_step_eb": eb.tolist(),
        "per_step_tf_kl": tf.tolist(),
        "per_step_rollout_kl": ro.tolist(),
        "per_step_valid_counts": rollout_metric.counts.astype(int).tolist(),
    }
    if oracle_positions is not None:
        oracle_summary = {}
        for role in ["commit", "bridge", "garden", "middle"]:
            positions = [pos for pos in oracle_positions["role_positions"].get(role, []) if pos < eb.shape[0] and valid[pos]]
            oracle_summary[role] = {
                "count": len(positions),
                "eb": _position_mean(eb, positions),
                "eb_weighted": _weighted_position_mean(eb, positions),
            }
        summary["oracle_role_summary"] = oracle_summary
    if region_overlap is not None:
        summary["region_overlap"] = region_overlap
    return summary


def _aggregate_teacher_forced_dataset(
    teacher: CausalTransformerLM,
    student: CausalTransformerLM,
    dataset_ids: torch.Tensor,
    batch_size: int,
    pad_id: int,
    device: torch.device,
) -> PerStepMetric:
    total_values = np.zeros(dataset_ids.shape[1] - 1, dtype=np.float64)
    total_counts = np.zeros(dataset_ids.shape[1] - 1, dtype=np.float64)
    loader = _make_loader(dataset_ids, batch_size=batch_size, shuffle=False)
    teacher.eval()
    student.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            metric = teacher_forced_kl(teacher, student, batch, pad_id)
            total_values += metric.values
            total_counts += metric.counts
    return PerStepMetric(total_values, total_counts)


def _aggregate_rollout_dataset(
    teacher: CausalTransformerLM,
    student: CausalTransformerLM,
    dataset_ids: torch.Tensor,
    batch_size: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    device: torch.device,
) -> PerStepMetric:
    total_values = np.zeros(dataset_ids.shape[1] - 1, dtype=np.float64)
    total_counts = np.zeros(dataset_ids.shape[1] - 1, dtype=np.float64)
    loader = _make_loader(dataset_ids, batch_size=batch_size, shuffle=False)
    teacher.eval()
    student.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            metric = rollout_kl_full(teacher, student, batch, bos_id, eos_id, pad_id)
            total_values += metric.values
            total_counts += metric.counts
    return PerStepMetric(total_values, total_counts)


def _aggregate_windowed_dataset(
    teacher: CausalTransformerLM,
    student: CausalTransformerLM,
    dataset_ids: torch.Tensor,
    batch_size: int,
    window: int,
    eos_id: int,
    pad_id: int,
    device: torch.device,
) -> PerStepMetric:
    total_values = np.zeros(dataset_ids.shape[1] - 1, dtype=np.float64)
    total_counts = np.zeros(dataset_ids.shape[1] - 1, dtype=np.float64)
    loader = _make_loader(dataset_ids, batch_size=batch_size, shuffle=False)
    teacher.eval()
    student.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            metric = rollout_kl_windowed(teacher, student, batch, window, eos_id, pad_id)
            total_values += metric.values
            total_counts += metric.counts
    return PerStepMetric(total_values, total_counts)


def load_domain_tensors(domain_name: str, config: ExperimentConfig) -> Tuple[DomainDefinition, Dict[str, List[SequenceRecord]], Dict[str, torch.Tensor]]:
    domain, splits = generate_domain_splits(domain_name, config.generation)
    tensors = {
        split_name: records_to_padded_tensor(records, domain.special_ids.pad)
        for split_name, records in splits.items()
    }
    return domain, splits, tensors


def smoke_generate(domain_name: str, config: ExperimentConfig) -> Dict[str, object]:
    artifact_root = config.artifact_path() / domain_name
    domain, splits, tensors = load_domain_tensors(domain_name, config)
    save_generation_artifacts(artifact_root, config, domain, splits)
    save_pt(artifact_root / "datasets" / "tensors.pt", tensors)
    examples = []
    for record in splits["analysis"][:3]:
        payload = domain.build_example_payload(record) if domain.build_example_payload else record.to_dict()
        examples.append(payload)
    report = {
        "domain": domain_name,
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
        "analysis_size": len(splits["analysis"]),
        "tensor_shapes": {k: list(v.shape) for k, v in tensors.items()},
        "example_payloads": examples,
    }
    save_json(artifact_root / "reports" / "smoke_report.json", report)
    return report


def _shared_val_teacher_forced_kl(
    teacher: CausalTransformerLM,
    student_state_dict: Dict[str, torch.Tensor],
    student_cfg: ModelConfig,
    vocab_size: int,
    model_max_len: int,
    pad_id: int,
    val_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    student = CausalTransformerLM(vocab_size, student_cfg, model_max_len, pad_id).to(device)
    student.load_state_dict(student_state_dict)
    metric = _aggregate_teacher_forced_dataset(
        teacher=teacher,
        student=student,
        dataset_ids=val_ids,
        batch_size=batch_size,
        pad_id=pad_id,
        device=device,
    )
    tf = masked_mean(metric.values, metric.counts)
    valid = metric.counts > 0
    if not np.any(valid):
        return float("inf")
    counts = metric.counts.astype(np.float64)
    return float(np.dot(tf[valid], counts[valid]) / counts[valid].sum())


def train_experiment(domain_name: str, config: ExperimentConfig) -> Dict[str, object]:
    artifact_root = ensure_dir(config.artifact_path() / domain_name / f"seed{config.train.seed}")
    reports_dir = ensure_dir(artifact_root / "reports")
    ckpt_dir = ensure_dir(artifact_root / "checkpoints")
    metrics_dir = ensure_dir(artifact_root / "metrics")
    figures_dir = ensure_dir(artifact_root / "figures")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain, splits, tensors = load_domain_tensors(domain_name, config)
    model_max_len = max(tensor.shape[1] for tensor in tensors.values())
    save_generation_artifacts(artifact_root, config, domain, splits)
    save_pt(artifact_root / "datasets" / "tensors.pt", tensors)

    example_record = splits["analysis"][0]
    example_payload = domain.build_example_payload(example_record) if domain.build_example_payload else example_record.to_dict()
    save_json(reports_dir / "example_payload.json", example_payload)
    try:
        from .plots import plot_example_sequence
        plot_example_sequence(example_payload, figures_dir / "fig_example_sequence.png", title=f"{domain_name} example")
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        save_json(reports_dir / "example_figure_skipped.json", {
            "reason": "matplotlib_missing",
            "domain": domain_name,
        })

    teacher = CausalTransformerLM(len(domain.vocab), config.train.teacher, model_max_len, domain.special_ids.pad)
    teacher_history = train_teacher(
        model=teacher,
        train_ids=tensors["train"],
        val_ids=tensors["val"],
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        lr=config.train.lr,
        pad_id=domain.special_ids.pad,
        device=device,
        checkpoint_dir=ckpt_dir,
    )
    save_json(reports_dir / "teacher_history.json", teacher_history)

    method_histories: Dict[str, object] = {}
    summaries: Dict[str, object] = {
        "teacher_history": teacher_history,
        "methods": {},
    }
    method_specs = [("hard", None), ("soft", None)] + [(f"hybrid_{int(lam * 10):02d}", lam) for lam in config.train.hybrid_lambdas]

    for method_name, lam in method_specs:
        history = train_student_method(
            method_name=method_name,
            teacher=teacher,
            train_ids=tensors["train"],
            val_ids=tensors["val"],
            student_cfg=config.train.student,
            model_max_len=model_max_len,
            epochs=config.train.epochs,
            batch_size=config.train.batch_size,
            lr=config.train.lr,
            pad_id=domain.special_ids.pad,
            device=device,
            checkpoint_dir=ckpt_dir,
            lambda_soft=lam if lam is not None else 0.5,
        )
        method_key = method_name if lam is None else method_name
        method_histories[method_key] = history
        save_json(reports_dir / f"{method_key}_history.json", {k: v for k, v in history.items() if k != "final_state_dict"})

    # Fixed, shared kappa reference student: choose the trained method with the best validation teacher-forced KL under a shared criterion.
    ref_scores = {
        name: _shared_val_teacher_forced_kl(
            teacher=teacher,
            student_state_dict=history["final_state_dict"],
            student_cfg=config.train.student,
            vocab_size=len(domain.vocab),
            model_max_len=model_max_len,
            pad_id=domain.special_ids.pad,
            val_ids=tensors["val"],
            batch_size=config.train.batch_size,
            device=device,
        )
        for name, history in method_histories.items()
    }
    ref_method = min(ref_scores, key=ref_scores.get)
    ref_student = CausalTransformerLM(len(domain.vocab), config.train.student, model_max_len, domain.special_ids.pad).to(device)
    ref_student.load_state_dict(method_histories[ref_method]["final_state_dict"])
    kappa_payload = compute_kappa_definition_reference(
        teacher=teacher.to(device),
        student=ref_student,
        records=splits["analysis"],
        batch_ids=tensors["analysis"],
        domain=domain,
        special_ids=domain.special_ids,
        device=device,
        horizon=config.eval.kappa_future_len,
        n_rollouts=5,
        max_action_count=config.eval.kappa_n_alts,
    )
    save_pt(metrics_dir / "kappa.pt", kappa_payload)
    region_masks = build_region_masks_from_state_kappa(kappa_payload, config.eval.min_region_coverage)
    instance_kappa_masks = build_instance_kappa_masks(kappa_payload)
    save_json(metrics_dir / "region_masks.json", asdict(region_masks))
    save_json(metrics_dir / "instance_kappa_masks.json", instance_kappa_masks)
    oracle_positions = build_oracle_role_positions(splits["analysis"], domain.special_ids.pad)
    save_json(metrics_dir / "oracle_positions.json", oracle_positions)
    region_overlap = compute_region_overlap(region_masks, oracle_positions)
    save_json(metrics_dir / "region_overlap.json", region_overlap)
    summaries["region_masks"] = asdict(region_masks)
    summaries["instance_kappa_masks"] = instance_kappa_masks
    summaries["oracle_positions"] = oracle_positions
    summaries["region_overlap"] = region_overlap
    summaries["kappa_reference_method"] = ref_method
    summaries["kappa_reference_scores"] = ref_scores

    for method_name, lam in method_specs:
        method_key = method_name if lam is None else method_name
        history = method_histories[method_key]

        student = CausalTransformerLM(len(domain.vocab), config.train.student, model_max_len, domain.special_ids.pad).to(device)
        student.load_state_dict(history["final_state_dict"])
        tf_metric = _aggregate_teacher_forced_dataset(
            teacher=teacher,
            student=student,
            dataset_ids=tensors["test"],
            batch_size=config.train.batch_size,
            pad_id=domain.special_ids.pad,
            device=device,
        )
        rollout_metric = _aggregate_rollout_dataset(
            teacher=teacher,
            student=student,
            dataset_ids=tensors["test"],
            batch_size=config.train.batch_size,
            bos_id=domain.special_ids.bos,
            eos_id=domain.special_ids.eos,
            pad_id=domain.special_ids.pad,
            device=device,
        )
        summary = summarize_eb(
            tf_metric,
            rollout_metric,
            region_masks,
            oracle_positions=oracle_positions,
            region_overlap=region_overlap,
        )
        windowed = {}
        for window in config.eval.window_sizes:
            win_metric = _aggregate_windowed_dataset(
                teacher=teacher,
                student=student,
                dataset_ids=tensors["test"],
                batch_size=config.train.batch_size,
                window=window,
                eos_id=domain.special_ids.eos,
                pad_id=domain.special_ids.pad,
                device=device,
            )
            win_summary = summarize_eb(
                tf_metric,
                win_metric,
                region_masks,
                oracle_positions=oracle_positions,
                region_overlap=region_overlap,
            )
            windowed[f"W{window}"] = win_summary
        summary["windowed"] = windowed
        summaries["methods"][method_key] = summary
        save_json(metrics_dir / f"{method_key}_summary.json", summary)

    save_json(reports_dir / "train_summary.json", summaries)
    return {
        "domain": domain_name,
        "artifact_root": str(artifact_root),
        "region_bridge_count": len(region_masks.bridge_positions),
        "region_garden_count": len(region_masks.garden_positions),
        "methods": {k: {"overall_eb": v["overall_eb"], "bridge_eb": v["bridge_eb"], "garden_eb": v["garden_eb"]} for k, v in summaries["methods"].items()},
    }
