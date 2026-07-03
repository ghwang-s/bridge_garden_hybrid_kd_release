from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .exact_oracle import ExactOracle, exact_loss_to_go
from .exact_oracle import ExactKappaStateResult
from .oracle_student import NeuralStudentLoss


@dataclass(frozen=True)
class OracleEBSummary:
    teacher_forced_kl: float
    rollout_kl: float
    exposure_bias: float
    teacher_forced_steps: int
    rollout_steps: int
    rollout_kl_se: float = float("nan")
    exposure_bias_se: float = float("nan")
    rollout_repeats: int = 1


@dataclass(frozen=True)
class OracleRegionEBSummary:
    overall: OracleEBSummary
    bridge_eb: float
    garden_eb: float
    bridge_teacher_forced_kl: float
    bridge_rollout_kl: float
    garden_teacher_forced_kl: float
    garden_rollout_kl: float
    bridge_eb_se: float = float("nan")
    garden_eb_se: float = float("nan")


@dataclass(frozen=True)
class OracleLocalEBSummary:
    teacher_mode_loss: float
    student_action_loss: float
    exposure_bias: float
    states: int
    student_in_teacher_support_rate: float
    student_violation_rate: float


@dataclass(frozen=True)
class OracleLocalRegionEBSummary:
    overall: OracleLocalEBSummary
    bridge: OracleLocalEBSummary
    garden: OracleLocalEBSummary


@dataclass(frozen=True)
class OracleKappaContributionSummary:
    contribution: float
    absolute_contribution: float
    states: int
    mean_l1_delta: float
    mean_teacher_kl: float


@dataclass(frozen=True)
class OracleKappaRegionContributionSummary:
    overall: OracleKappaContributionSummary
    bridge: OracleKappaContributionSummary
    garden: OracleKappaContributionSummary


@dataclass(frozen=True)
class OracleExpectedInterventionSummary:
    expected_eb: float
    teacher_expected_loss: float
    student_expected_loss: float
    states: int
    mean_student_eval_mass: float


@dataclass(frozen=True)
class OracleExpectedRegionInterventionSummary:
    overall: OracleExpectedInterventionSummary
    bridge: OracleExpectedInterventionSummary
    garden: OracleExpectedInterventionSummary


def summarize_oracle_eb(
    *,
    oracle: ExactOracle,
    model: torch.nn.Module,
    vocab_size: int,
    sample_ids: Sequence[int],
    max_steps: int | None = None,
    rollout_repeats: int = 1,
    rollout_seed: int = 0,
    device: torch.device | None = None,
) -> OracleEBSummary:
    loss = NeuralStudentLoss(model, vocab_size=vocab_size, device=device)
    teacher_values = []
    student_values = []
    student_means = []
    eb_means = []
    for sample_id in sample_ids:
        for repeat in range(max(1, int(rollout_repeats))):
            teacher_generator = _rollout_generator(device, int(rollout_seed) + 50_000_019, int(sample_id), repeat)
            student_generator = _rollout_generator(device, int(rollout_seed), int(sample_id), repeat)
            teacher_rep = _teacher_rollout_values(
                oracle,
                loss,
                int(sample_id),
                max_steps=max_steps,
                generator=teacher_generator,
            )
            student_rep = _student_rollout_values(
                oracle,
                model,
                loss,
                int(sample_id),
                max_steps=max_steps,
                device=device,
                generator=student_generator,
            )
            teacher_values.extend(teacher_rep)
            student_values.extend(student_rep)
            if student_rep:
                student_means.append(float(np.mean(student_rep)))
            if teacher_rep and student_rep:
                eb_means.append(float(np.mean(student_rep) - np.mean(teacher_rep)))
    tf = float(np.mean(teacher_values)) if teacher_values else float("nan")
    ro = float(np.mean(student_values)) if student_values else float("nan")
    eb_se = _standard_error(eb_means)
    return OracleEBSummary(
        teacher_forced_kl=tf,
        rollout_kl=ro,
        exposure_bias=ro - tf,
        teacher_forced_steps=len(teacher_values),
        rollout_steps=len(student_values),
        rollout_kl_se=_standard_error(student_means),
        exposure_bias_se=eb_se,
        rollout_repeats=max(1, int(rollout_repeats)),
    )


def summarize_oracle_eb_by_positions(
    *,
    oracle: ExactOracle,
    model: torch.nn.Module,
    vocab_size: int,
    sample_ids: Sequence[int],
    bridge_positions: set[int],
    garden_positions: set[int],
    max_steps: int | None = None,
    rollout_repeats: int = 1,
    rollout_seed: int = 0,
    device: torch.device | None = None,
) -> OracleRegionEBSummary:
    loss = NeuralStudentLoss(model, vocab_size=vocab_size, device=device)
    teacher_rows = []
    rollout_rows = []
    teacher_rep_rows = []
    rollout_rep_rows = []
    for sample_id in sample_ids:
        for repeat in range(max(1, int(rollout_repeats))):
            teacher_generator = _rollout_generator(device, int(rollout_seed) + 50_000_019, int(sample_id), repeat)
            student_generator = _rollout_generator(device, int(rollout_seed), int(sample_id), repeat)
            teacher_rows_rep = _teacher_rollout_position_values(
                oracle,
                loss,
                int(sample_id),
                max_steps=max_steps,
                generator=teacher_generator,
            )
            student_rows_rep = _student_rollout_position_values(
                oracle,
                model,
                loss,
                int(sample_id),
                max_steps=max_steps,
                device=device,
                generator=student_generator,
            )
            teacher_rows.extend(teacher_rows_rep)
            rollout_rows.extend(student_rows_rep)
            teacher_rep_rows.append(teacher_rows_rep)
            rollout_rep_rows.append(student_rows_rep)

    def _mean(rows: list[tuple[int, float]], positions: set[int]) -> float:
        vals = [value for pos, value in rows if pos in positions]
        return float(np.mean(vals)) if vals else float("nan")

    overall = OracleEBSummary(
        teacher_forced_kl=float(np.mean([value for _, value in teacher_rows])) if teacher_rows else float("nan"),
        rollout_kl=float(np.mean([value for _, value in rollout_rows])) if rollout_rows else float("nan"),
        exposure_bias=(
            float(np.mean([value for _, value in rollout_rows]) - np.mean([value for _, value in teacher_rows]))
            if teacher_rows and rollout_rows else float("nan")
        ),
        teacher_forced_steps=len(teacher_rows),
        rollout_steps=len(rollout_rows),
        rollout_kl_se=_standard_error([
            float(np.mean([value for _, value in rows]))
            for rows in rollout_rep_rows
            if rows
        ]),
        exposure_bias_se=_paired_position_eb_se(teacher_rep_rows, rollout_rep_rows),
        rollout_repeats=max(1, int(rollout_repeats)),
    )
    bridge_tf = _mean(teacher_rows, bridge_positions)
    bridge_ro = _mean(rollout_rows, bridge_positions)
    garden_tf = _mean(teacher_rows, garden_positions)
    garden_ro = _mean(rollout_rows, garden_positions)
    bridge_ro_se = _paired_region_eb_se(teacher_rep_rows, rollout_rep_rows, bridge_positions)
    garden_ro_se = _paired_region_eb_se(teacher_rep_rows, rollout_rep_rows, garden_positions)
    return OracleRegionEBSummary(
        overall=overall,
        bridge_eb=bridge_ro - bridge_tf,
        garden_eb=garden_ro - garden_tf,
        bridge_teacher_forced_kl=bridge_tf,
        bridge_rollout_kl=bridge_ro,
        garden_teacher_forced_kl=garden_tf,
        garden_rollout_kl=garden_ro,
        bridge_eb_se=bridge_ro_se,
        garden_eb_se=garden_ro_se,
    )


def trace_oracle_rollouts(
    *,
    oracle: ExactOracle,
    model: torch.nn.Module,
    sample_ids: Sequence[int],
    max_steps: int | None = None,
    device: torch.device | None = None,
) -> list[dict]:
    """Trace greedy student rollouts against the oracle for failure diagnosis."""
    device = device or next(model.parameters()).device
    rows = []
    model.eval()
    for sample_id in sample_ids:
        state = oracle.initial_state(int(sample_id))  # type: ignore[attr-defined]
        steps = 0
        while not oracle.is_terminal(state) and (max_steps is None or steps < max_steps):
            dist = oracle.next_dist(state)
            if not dist:
                break
            teacher_action = max(dist, key=lambda token_id: (dist[token_id], -int(token_id)))
            prefix = tuple(int(x) for x in getattr(state, "prefix"))
            with torch.no_grad():
                input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
                logits = model(input_ids)[:, -1, :]
                probs = F.softmax(logits, dim=-1)[0]
                student_action = int(torch.argmax(probs).detach().cpu())
                student_prob = float(probs[student_action].detach().cpu())
                teacher_action_prob = float(probs[int(teacher_action)].detach().cpu())
            next_state = oracle.step(state, student_action)
            realized_action = int(tuple(getattr(next_state, "prefix", ()))[-1])
            rows.append({
                "sample_id": int(sample_id),
                "position": int(getattr(state, "position")),
                "status_before": str(getattr(state, "status", "")),
                "status_after": str(getattr(next_state, "status", "")),
                "style_before": int(getattr(state, "style", 0)),
                "style_after": int(getattr(next_state, "style", 0)),
                "semantic_tag": _semantic_tag(oracle, state),
                "expected_role": _expected_role(oracle, state),
                "teacher_action": int(teacher_action),
                "student_action": int(student_action),
                "realized_student_action": realized_action,
                "student_action_canonicalized": realized_action != int(student_action),
                "student_action_prob": student_prob,
                "teacher_action_student_prob": teacher_action_prob,
                "student_action_teacher_prob": float(dist.get(student_action, 0.0)),
                "student_action_in_teacher_support": student_action in dist,
            })
            state = next_state
            steps += 1
    return rows


def summarize_oracle_sampled_rollout_statuses(
    *,
    oracle: ExactOracle,
    model: torch.nn.Module,
    sample_ids: Sequence[int],
    max_steps: int | None = None,
    rollout_repeats: int = 1,
    rollout_seed: int = 0,
    device: torch.device | None = None,
) -> dict:
    """Summarize stochastic student rollout status transitions."""
    if not hasattr(oracle, "initial_state"):
        raise TypeError("oracle must expose initial_state(sample_id)")
    device = device or next(model.parameters()).device
    model.eval()
    transitions: Counter[tuple[str, str]] = Counter()
    off_support_by_tag: Counter[str] = Counter()
    first_violations: Counter[tuple[int, str]] = Counter()
    style_transitions: Counter[tuple[int, int]] = Counter()
    steps = 0
    off_support = 0
    first_violation_count = 0
    for sample_id in sample_ids:
        for repeat in range(max(1, int(rollout_repeats))):
            generator = _rollout_generator(device, int(rollout_seed), int(sample_id), repeat)
            state = oracle.initial_state(int(sample_id))  # type: ignore[attr-defined]
            seen_violation = False
            local_steps = 0
            while not oracle.is_terminal(state) and (max_steps is None or local_steps < max_steps):
                dist = oracle.next_dist(state)
                if not dist:
                    break
                prefix = tuple(int(x) for x in getattr(state, "prefix"))
                with torch.no_grad():
                    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
                    logits = model(input_ids)[:, -1, :]
                    probs = F.softmax(logits, dim=-1)[0]
                    action = int(torch.multinomial(probs.detach().cpu(), 1, generator=generator).item())
                tag = _semantic_tag(oracle, state)
                if action not in dist:
                    off_support += 1
                    off_support_by_tag[tag] += 1
                next_state = oracle.step(state, action)
                before = str(getattr(state, "status", ""))
                after = str(getattr(next_state, "status", ""))
                transitions[(before, after)] += 1
                style_transitions[(int(getattr(state, "style", 0)), int(getattr(next_state, "style", 0)))] += 1
                if before == "clean" and after == "violation" and not seen_violation:
                    first_violation_count += 1
                    first_violations[(int(getattr(state, "position")), tag)] += 1
                    seen_violation = True
                state = next_state
                steps += 1
                local_steps += 1
    return {
        "steps": int(steps),
        "rollout_repeats": max(1, int(rollout_repeats)),
        "transition_counts": {f"{before}->{after}": int(count) for (before, after), count in sorted(transitions.items())},
        "off_teacher_support_count": int(off_support),
        "off_teacher_support_rate": float(off_support / steps) if steps else float("nan"),
        "off_teacher_support_by_tag": dict(sorted((tag, int(count)) for tag, count in off_support_by_tag.items())),
        "first_violation_count": int(first_violation_count),
        "first_violation_top": [
            {"position": int(position), "semantic_tag": tag, "count": int(count)}
            for (position, tag), count in first_violations.most_common(10)
        ],
        "style_transition_counts": {f"{before}->{after}": int(count) for (before, after), count in sorted(style_transitions.items())},
    }


def summarize_oracle_local_intervention_eb(
    *,
    oracle: ExactOracle,
    model: torch.nn.Module,
    vocab_size: int,
    states: Sequence[Hashable],
    bridge_states: Sequence[Hashable],
    garden_states: Sequence[Hashable],
    device: torch.device | None = None,
) -> OracleLocalRegionEBSummary:
    """Clean-prefix local EB for κ-defined state regions.

    For each clean state s, compare the teacher-mode continuation after the
    student's greedy action against the continuation after the teacher-mode
    action. This isolates the local state risk from earlier rollout drift.
    """
    loss = NeuralStudentLoss(model, vocab_size=vocab_size, device=device)
    bridge_set = set(bridge_states)
    garden_set = set(garden_states)
    rows = _local_intervention_rows(oracle, model, loss, states, device=device)
    return OracleLocalRegionEBSummary(
        overall=_summarize_local_rows(rows),
        bridge=_summarize_local_rows([row for row in rows if row["state"] in bridge_set]),
        garden=_summarize_local_rows([row for row in rows if row["state"] in garden_set]),
    )


def summarize_oracle_kappa_contribution(
    *,
    oracle: ExactOracle,
    model: torch.nn.Module,
    vocab_size: int,
    kappa_results: Sequence[ExactKappaStateResult],
    bridge_states: Sequence[Hashable],
    garden_states: Sequence[Hashable],
    device: torch.device | None = None,
) -> OracleKappaRegionContributionSummary:
    """Theorem-aligned local contribution over teacher-prefix states.

    Uses the paper's single-override sensitivities and signed local
    distribution deviation from the teacher. The κ values are supplied by the
    caller, so this can be evaluated on κ_ref or κ_m.
    """
    device = device or next(model.parameters()).device
    rows = _kappa_contribution_rows(
        oracle=oracle,
        model=model,
        vocab_size=vocab_size,
        kappa_results=kappa_results,
        device=device,
    )
    bridge_set = set(bridge_states)
    garden_set = set(garden_states)
    return OracleKappaRegionContributionSummary(
        overall=_summarize_kappa_contribution_rows(rows),
        bridge=_summarize_kappa_contribution_rows([row for row in rows if row["state"] in bridge_set]),
        garden=_summarize_kappa_contribution_rows([row for row in rows if row["state"] in garden_set]),
    )


def summarize_oracle_expected_intervention(
    *,
    model: torch.nn.Module,
    kappa_results: Sequence[ExactKappaStateResult],
    bridge_states: Sequence[Hashable],
    garden_states: Sequence[Hashable],
    device: torch.device | None = None,
) -> OracleExpectedRegionInterventionSummary:
    """Exact single-step stochastic intervention over the evaluated vocabulary."""
    device = device or next(model.parameters()).device
    rows = _expected_intervention_rows(model=model, kappa_results=kappa_results, device=device)
    bridge_set = set(bridge_states)
    garden_set = set(garden_states)
    return OracleExpectedRegionInterventionSummary(
        overall=_summarize_expected_intervention_rows(rows),
        bridge=_summarize_expected_intervention_rows([row for row in rows if row["state"] in bridge_set]),
        garden=_summarize_expected_intervention_rows([row for row in rows if row["state"] in garden_set]),
    )


def _teacher_forced_values(
    oracle: ExactOracle,
    loss: NeuralStudentLoss,
    sample_id: int,
    *,
    max_steps: int | None,
) -> list[float]:
    if not hasattr(oracle, "initial_state"):
        raise TypeError("oracle must expose initial_state(sample_id)")
    state = oracle.initial_state(sample_id)  # type: ignore[attr-defined]
    values = []
    steps = 0
    while not oracle.is_terminal(state) and (max_steps is None or steps < max_steps):
        values.append(loss.loss_at_state(state, oracle))
        dist = oracle.next_dist(state)
        if not dist:
            break
        action = max(dist, key=lambda token_id: (dist[token_id], -int(token_id)))
        state = oracle.step(state, int(action))
        steps += 1
    return values


def _local_intervention_rows(
    oracle: ExactOracle,
    model: torch.nn.Module,
    loss: NeuralStudentLoss,
    states: Sequence[Hashable],
    *,
    device: torch.device | None,
) -> list[dict]:
    device = device or next(model.parameters()).device
    model.eval()
    rows = []
    for state in states:
        if oracle.is_terminal(state):
            continue
        dist = oracle.next_dist(state)
        if not dist:
            continue
        teacher_action = max(dist, key=lambda token_id: (dist[token_id], -int(token_id)))
        prefix = tuple(int(x) for x in getattr(state, "prefix"))
        with torch.no_grad():
            input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
            logits = model(input_ids)[:, -1, :]
            student_action = int(torch.argmax(logits, dim=-1).detach().cpu())
        teacher_next = oracle.step(state, int(teacher_action))
        student_next = oracle.step(state, int(student_action))
        teacher_loss = exact_loss_to_go(oracle, loss, teacher_next, continuation="mode")
        student_loss = exact_loss_to_go(oracle, loss, student_next, continuation="mode")
        rows.append({
            "state": state,
            "teacher_loss": teacher_loss,
            "student_loss": student_loss,
            "student_in_teacher_support": student_action in dist,
            "student_violates": getattr(student_next, "status", "") == "violation",
        })
    return rows


def _kappa_contribution_rows(
    *,
    oracle: ExactOracle,
    model: torch.nn.Module,
    vocab_size: int,
    kappa_results: Sequence[ExactKappaStateResult],
    device: torch.device,
) -> list[dict]:
    model.eval()
    rows = []
    for result in kappa_results:
        state = result.state
        if oracle.is_terminal(state):
            continue
        prefix = tuple(int(x) for x in getattr(state, "prefix"))
        with torch.no_grad():
            input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
            logits = model(input_ids)[:, -1, :]
            student_probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        teacher = np.zeros(vocab_size, dtype=np.float64)
        for token_id, prob in oracle.next_dist(state).items():
            teacher[int(token_id)] = float(prob)
        action_ids = np.array(result.action_ids, dtype=np.int64)
        kappa = np.array(result.action_kappa, dtype=np.float64)
        signed_delta = student_probs[action_ids] - teacher[action_ids]
        l1_delta = np.abs(signed_delta)
        clipped = np.clip(student_probs[action_ids], 1e-12, 1.0)
        teacher_on_actions = teacher[action_ids]
        teacher_kl = float(np.sum(teacher_on_actions * (np.log(np.clip(teacher_on_actions, 1e-12, 1.0)) - np.log(clipped))))
        rows.append({
            "state": state,
            "contribution": float(np.sum(kappa * signed_delta)),
            "absolute_contribution": float(np.sum(np.abs(kappa) * l1_delta)),
            "l1_delta": float(np.sum(l1_delta)),
            "teacher_kl": teacher_kl,
        })
    return rows


def _expected_intervention_rows(
    *,
    model: torch.nn.Module,
    kappa_results: Sequence[ExactKappaStateResult],
    device: torch.device,
) -> list[dict]:
    model.eval()
    rows = []
    for result in kappa_results:
        state = result.state
        prefix = tuple(int(x) for x in getattr(state, "prefix"))
        action_ids = np.array(result.action_ids, dtype=np.int64)
        q_values = np.array(result.q_values, dtype=np.float64)
        with torch.no_grad():
            input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
            logits = model(input_ids)[:, -1, :]
            probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        student_probs = probs[action_ids].astype(np.float64)
        eval_mass = float(student_probs.sum())
        off_eval_mass = max(0.0, 1.0 - eval_mass)
        off_eval_loss = float(np.max(q_values)) if q_values.size else 0.0
        student_loss = float(np.dot(student_probs, q_values) + off_eval_mass * off_eval_loss)
        rows.append({
            "state": state,
            "teacher_loss": float(result.qbar),
            "student_loss": student_loss,
            "expected_eb": student_loss - float(result.qbar),
            "student_eval_mass": eval_mass,
        })
    return rows


def _summarize_expected_intervention_rows(rows: list[dict]) -> OracleExpectedInterventionSummary:
    if not rows:
        return OracleExpectedInterventionSummary(
            expected_eb=float("nan"),
            teacher_expected_loss=float("nan"),
            student_expected_loss=float("nan"),
            states=0,
            mean_student_eval_mass=float("nan"),
        )
    teacher = float(np.mean([row["teacher_loss"] for row in rows]))
    student = float(np.mean([row["student_loss"] for row in rows]))
    return OracleExpectedInterventionSummary(
        expected_eb=student - teacher,
        teacher_expected_loss=teacher,
        student_expected_loss=student,
        states=len(rows),
        mean_student_eval_mass=float(np.mean([row["student_eval_mass"] for row in rows])),
    )


def _summarize_kappa_contribution_rows(rows: list[dict]) -> OracleKappaContributionSummary:
    if not rows:
        return OracleKappaContributionSummary(
            contribution=float("nan"),
            absolute_contribution=float("nan"),
            states=0,
            mean_l1_delta=float("nan"),
            mean_teacher_kl=float("nan"),
        )
    return OracleKappaContributionSummary(
        contribution=float(np.mean([row["contribution"] for row in rows])),
        absolute_contribution=float(np.mean([row["absolute_contribution"] for row in rows])),
        states=len(rows),
        mean_l1_delta=float(np.mean([row["l1_delta"] for row in rows])),
        mean_teacher_kl=float(np.mean([row["teacher_kl"] for row in rows])),
    )


def _summarize_local_rows(rows: list[dict]) -> OracleLocalEBSummary:
    if not rows:
        return OracleLocalEBSummary(
            teacher_mode_loss=float("nan"),
            student_action_loss=float("nan"),
            exposure_bias=float("nan"),
            states=0,
            student_in_teacher_support_rate=float("nan"),
            student_violation_rate=float("nan"),
        )
    teacher = float(np.mean([row["teacher_loss"] for row in rows]))
    student = float(np.mean([row["student_loss"] for row in rows]))
    return OracleLocalEBSummary(
        teacher_mode_loss=teacher,
        student_action_loss=student,
        exposure_bias=student - teacher,
        states=len(rows),
        student_in_teacher_support_rate=float(np.mean([row["student_in_teacher_support"] for row in rows])),
        student_violation_rate=float(np.mean([row["student_violates"] for row in rows])),
    )


def _student_rollout_values(
    oracle: ExactOracle,
    model: torch.nn.Module,
    loss: NeuralStudentLoss,
    sample_id: int,
    *,
    max_steps: int | None,
    device: torch.device | None,
    generator: torch.Generator,
) -> list[float]:
    if not hasattr(oracle, "initial_state"):
        raise TypeError("oracle must expose initial_state(sample_id)")
    device = device or next(model.parameters()).device
    state = oracle.initial_state(sample_id)  # type: ignore[attr-defined]
    values = []
    steps = 0
    model.eval()
    while not oracle.is_terminal(state) and (max_steps is None or steps < max_steps):
        values.append(loss.loss_at_state(state, oracle))
        prefix = tuple(int(x) for x in getattr(state, "prefix"))
        with torch.no_grad():
            input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
            logits = model(input_ids)[:, -1, :]
            probs = F.softmax(logits, dim=-1)[0]
            action = int(torch.multinomial(probs.detach().cpu(), 1, generator=generator).item())
        state = oracle.step(state, action)
        steps += 1
    return values


def _teacher_rollout_values(
    oracle: ExactOracle,
    loss: NeuralStudentLoss,
    sample_id: int,
    *,
    max_steps: int | None,
    generator: torch.Generator,
) -> list[float]:
    if not hasattr(oracle, "initial_state"):
        raise TypeError("oracle must expose initial_state(sample_id)")
    state = oracle.initial_state(sample_id)  # type: ignore[attr-defined]
    values = []
    steps = 0
    while not oracle.is_terminal(state) and (max_steps is None or steps < max_steps):
        values.append(loss.loss_at_state(state, oracle))
        dist = oracle.next_dist(state)
        if not dist:
            break
        action_ids = list(dist)
        probs = torch.tensor([float(dist[action]) for action in action_ids], dtype=torch.float32)
        sampled = int(torch.multinomial(probs, 1, generator=generator).item())
        state = oracle.step(state, int(action_ids[sampled]))
        steps += 1
    return values


def _teacher_forced_position_values(
    oracle: ExactOracle,
    loss: NeuralStudentLoss,
    sample_id: int,
    *,
    max_steps: int | None,
) -> list[tuple[int, float]]:
    state = oracle.initial_state(sample_id)  # type: ignore[attr-defined]
    rows = []
    steps = 0
    while not oracle.is_terminal(state) and (max_steps is None or steps < max_steps):
        rows.append((int(getattr(state, "position")), loss.loss_at_state(state, oracle)))
        dist = oracle.next_dist(state)
        if not dist:
            break
        action = max(dist, key=lambda token_id: (dist[token_id], -int(token_id)))
        state = oracle.step(state, int(action))
        steps += 1
    return rows


def _teacher_rollout_position_values(
    oracle: ExactOracle,
    loss: NeuralStudentLoss,
    sample_id: int,
    *,
    max_steps: int | None,
    generator: torch.Generator,
) -> list[tuple[int, float]]:
    state = oracle.initial_state(sample_id)  # type: ignore[attr-defined]
    rows = []
    steps = 0
    while not oracle.is_terminal(state) and (max_steps is None or steps < max_steps):
        rows.append((int(getattr(state, "position")), loss.loss_at_state(state, oracle)))
        dist = oracle.next_dist(state)
        if not dist:
            break
        action_ids = list(dist)
        probs = torch.tensor([float(dist[action]) for action in action_ids], dtype=torch.float32)
        sampled = int(torch.multinomial(probs, 1, generator=generator).item())
        state = oracle.step(state, int(action_ids[sampled]))
        steps += 1
    return rows


def _semantic_tag(oracle: ExactOracle, state: Hashable) -> str:
    if hasattr(oracle, "semantic_tag"):
        return str(oracle.semantic_tag(state))  # type: ignore[attr-defined]
    return ""


def _expected_role(oracle: ExactOracle, state: Hashable) -> str:
    if hasattr(oracle, "expected_role"):
        return str(oracle.expected_role(state))  # type: ignore[attr-defined]
    return ""


def _student_rollout_position_values(
    oracle: ExactOracle,
    model: torch.nn.Module,
    loss: NeuralStudentLoss,
    sample_id: int,
    *,
    max_steps: int | None,
    device: torch.device | None,
    generator: torch.Generator,
) -> list[tuple[int, float]]:
    device = device or next(model.parameters()).device
    state = oracle.initial_state(sample_id)  # type: ignore[attr-defined]
    rows = []
    steps = 0
    model.eval()
    while not oracle.is_terminal(state) and (max_steps is None or steps < max_steps):
        rows.append((int(getattr(state, "position")), loss.loss_at_state(state, oracle)))
        prefix = tuple(int(x) for x in getattr(state, "prefix"))
        with torch.no_grad():
            input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
            logits = model(input_ids)[:, -1, :]
            probs = F.softmax(logits, dim=-1)[0]
            action = int(torch.multinomial(probs.detach().cpu(), 1, generator=generator).item())
        state = oracle.step(state, action)
        steps += 1
    return rows


def _rollout_generator(device: torch.device | None, rollout_seed: int, sample_id: int, repeat: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(rollout_seed) + 1_000_003 * int(sample_id) + 9_176 * int(repeat))
    return generator


def _standard_error(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return float("nan")
    arr = np.array(values, dtype=np.float64)
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _region_rollout_se(rows_by_repeat: Sequence[list[tuple[int, float]]], positions: set[int]) -> float:
    means = []
    for rows in rows_by_repeat:
        values = [value for pos, value in rows if pos in positions]
        if values:
            means.append(float(np.mean(values)))
    return _standard_error(means)


def _paired_position_eb_se(
    teacher_rows_by_repeat: Sequence[list[tuple[int, float]]],
    student_rows_by_repeat: Sequence[list[tuple[int, float]]],
) -> float:
    diffs = []
    for teacher_rows, student_rows in zip(teacher_rows_by_repeat, student_rows_by_repeat):
        teacher_values = [value for _, value in teacher_rows]
        student_values = [value for _, value in student_rows]
        if teacher_values and student_values:
            diffs.append(float(np.mean(student_values) - np.mean(teacher_values)))
    return _standard_error(diffs)


def _paired_region_eb_se(
    teacher_rows_by_repeat: Sequence[list[tuple[int, float]]],
    student_rows_by_repeat: Sequence[list[tuple[int, float]]],
    positions: set[int],
) -> float:
    diffs = []
    for teacher_rows, student_rows in zip(teacher_rows_by_repeat, student_rows_by_repeat):
        teacher_values = [value for pos, value in teacher_rows if pos in positions]
        student_values = [value for pos, value in student_rows if pos in positions]
        if teacher_values and student_values:
            diffs.append(float(np.mean(student_values) - np.mean(teacher_values)))
    return _standard_error(diffs)
