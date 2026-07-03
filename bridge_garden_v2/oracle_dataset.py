from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Hashable, Sequence

import torch

from .exact_oracle import ExactOracle


@dataclass(frozen=True)
class OraclePathBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    teacher_probs: torch.Tensor
    semantic_tags: list[list[str]]
    expected_roles: list[list[str]]
    role_ids: torch.Tensor


def materialize_mode_paths(
    oracle: ExactOracle,
    *,
    sample_ids: Sequence[int],
    vocab_size: int,
) -> OraclePathBatch:
    """Materialize teacher-mode paths with exact per-state teacher distributions."""
    paths: list[list[int]] = []
    prob_rows: list[torch.Tensor] = []
    tags: list[list[str]] = []
    roles: list[list[str]] = []
    max_steps = 0
    per_sample_probs: list[list[torch.Tensor]] = []

    for sample_id in sample_ids:
        if not hasattr(oracle, "initial_state"):
            raise TypeError("oracle must expose initial_state(sample_id)")
        state = oracle.initial_state(int(sample_id))  # type: ignore[attr-defined]
        tokens = [oracle.bos_id]
        sample_probs: list[torch.Tensor] = []
        sample_tags: list[str] = []
        sample_roles: list[str] = []
        while not oracle.is_terminal(state):
            dist = oracle.next_dist(state)
            probs = torch.zeros(vocab_size, dtype=torch.float32)
            for token_id, prob in dist.items():
                probs[int(token_id)] = float(prob)
            sample_probs.append(probs)
            sample_tags.append(_semantic_tag(oracle, state))
            sample_roles.append(_expected_role(oracle, state))
            action = max(dist, key=lambda token_id: (dist[token_id], -int(token_id)))
            tokens.append(int(action))
            state = oracle.step(state, int(action))
        paths.append(tokens)
        per_sample_probs.append(sample_probs)
        tags.append(sample_tags)
        roles.append(sample_roles)
        max_steps = max(max_steps, len(sample_probs))

    batch = len(paths)
    input_ids = torch.full((batch, max_steps), oracle.pad_id, dtype=torch.long)
    target_ids = torch.full((batch, max_steps), oracle.pad_id, dtype=torch.long)
    teacher_probs = torch.zeros((batch, max_steps, vocab_size), dtype=torch.float32)
    role_ids = torch.zeros((batch, max_steps), dtype=torch.long)

    for row, tokens in enumerate(paths):
        steps = len(tokens) - 1
        input_ids[row, :steps] = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids[row, :steps] = torch.tensor(tokens[1:], dtype=torch.long)
        if steps:
            teacher_probs[row, :steps] = torch.stack(per_sample_probs[row], dim=0)
            role_ids[row, :steps] = torch.tensor([_role_id(role) for role in roles[row]], dtype=torch.long)

    return OraclePathBatch(
        input_ids=input_ids,
        target_ids=target_ids,
        teacher_probs=teacher_probs,
        semantic_tags=tags,
        expected_roles=roles,
        role_ids=role_ids,
    )


def materialize_sampled_paths(
    oracle: ExactOracle,
    *,
    sample_ids: Sequence[int],
    vocab_size: int,
    seed: int = 0,
) -> OraclePathBatch:
    """Materialize teacher-sampled paths with exact per-state distributions."""
    rng = random.Random(seed)
    paths: list[list[int]] = []
    tags: list[list[str]] = []
    roles: list[list[str]] = []
    max_steps = 0
    per_sample_probs: list[list[torch.Tensor]] = []

    for sample_id in sample_ids:
        if not hasattr(oracle, "initial_state"):
            raise TypeError("oracle must expose initial_state(sample_id)")
        state = oracle.initial_state(int(sample_id))  # type: ignore[attr-defined]
        tokens = [oracle.bos_id]
        sample_probs: list[torch.Tensor] = []
        sample_tags: list[str] = []
        sample_roles: list[str] = []
        while not oracle.is_terminal(state):
            dist = oracle.next_dist(state)
            probs = torch.zeros(vocab_size, dtype=torch.float32)
            for token_id, prob in dist.items():
                probs[int(token_id)] = float(prob)
            sample_probs.append(probs)
            sample_tags.append(_semantic_tag(oracle, state))
            sample_roles.append(_expected_role(oracle, state))
            action = _sample_from_dist(dist, rng)
            tokens.append(int(action))
            state = oracle.step(state, int(action))
        paths.append(tokens)
        per_sample_probs.append(sample_probs)
        tags.append(sample_tags)
        roles.append(sample_roles)
        max_steps = max(max_steps, len(sample_probs))

    return _batch_from_paths(
        oracle=oracle,
        paths=paths,
        per_sample_probs=per_sample_probs,
        tags=tags,
        roles=roles,
        max_steps=max_steps,
        vocab_size=vocab_size,
    )


def _batch_from_paths(
    *,
    oracle: ExactOracle,
    paths: list[list[int]],
    per_sample_probs: list[list[torch.Tensor]],
    tags: list[list[str]],
    roles: list[list[str]],
    max_steps: int,
    vocab_size: int,
) -> OraclePathBatch:
    batch = len(paths)
    input_ids = torch.full((batch, max_steps), oracle.pad_id, dtype=torch.long)
    target_ids = torch.full((batch, max_steps), oracle.pad_id, dtype=torch.long)
    teacher_probs = torch.zeros((batch, max_steps, vocab_size), dtype=torch.float32)
    role_ids = torch.zeros((batch, max_steps), dtype=torch.long)

    for row, tokens in enumerate(paths):
        steps = len(tokens) - 1
        input_ids[row, :steps] = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids[row, :steps] = torch.tensor(tokens[1:], dtype=torch.long)
        if steps:
            teacher_probs[row, :steps] = torch.stack(per_sample_probs[row], dim=0)
            role_ids[row, :steps] = torch.tensor([_role_id(role) for role in roles[row]], dtype=torch.long)

    return OraclePathBatch(
        input_ids=input_ids,
        target_ids=target_ids,
        teacher_probs=teacher_probs,
        semantic_tags=tags,
        expected_roles=roles,
        role_ids=role_ids,
    )


def _sample_from_dist(dist, rng: random.Random) -> int:
    total = float(sum(dist.values()))
    if total <= 0.0:
        raise ValueError("cannot sample from empty distribution")
    threshold = rng.random() * total
    cdf = 0.0
    last = None
    for token_id, prob in sorted(dist.items()):
        last = int(token_id)
        cdf += float(prob)
        if threshold <= cdf:
            return int(token_id)
    if last is None:
        raise ValueError("cannot sample from empty distribution")
    return last


def _semantic_tag(oracle: ExactOracle, state: Hashable) -> str:
    if hasattr(oracle, "semantic_tag"):
        return str(oracle.semantic_tag(state))  # type: ignore[attr-defined]
    return ""


def _expected_role(oracle: ExactOracle, state: Hashable) -> str:
    if hasattr(oracle, "expected_role"):
        return str(oracle.expected_role(state))  # type: ignore[attr-defined]
    return ""


def _role_id(role: str) -> int:
    return {
        "high_risk": 1,
        "flexible": 2,
        "control": 3,
    }.get(role, 0)
