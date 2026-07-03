from __future__ import annotations

from functools import lru_cache
from typing import Hashable, Mapping

import torch
import torch.nn.functional as F

from .exact_oracle import ExactOracle


class NeuralStudentLoss:
    """StudentLoss adapter for exact oracle states with token prefixes."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        vocab_size: int,
        violation_penalty: float = 0.0,
        device: torch.device | None = None,
        max_batch_rows: int = 8192,
    ) -> None:
        self.model = model
        self.vocab_size = int(vocab_size)
        self.violation_penalty = float(violation_penalty)
        self.max_batch_rows = int(max_batch_rows)
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()
        self._batch_loss_cache: dict[tuple[tuple[int, ...], tuple[tuple[int, float], ...]], float] = {}
        self.batch_forward_calls = 0
        self.batch_forward_rows = 0
        self.cache_miss_entries = 0

    def loss_at_state(self, state: Hashable, oracle: ExactOracle) -> float:
        prefix = _prefix_tuple(state)
        dist = _dist_tuple(oracle.next_dist(state))
        key = (prefix, dist)
        cached = self._batch_loss_cache.get(key)
        if cached is None:
            cached = self._loss_for_prefix(prefix, dist)
            self._batch_loss_cache[key] = cached
        return cached + _state_penalty(state, self.violation_penalty)

    def loss_many(self, states: list[Hashable], oracle: ExactOracle) -> list[float]:
        keys = [(_prefix_tuple(state), _dist_tuple(oracle.next_dist(state))) for state in states]
        misses = list(dict.fromkeys(key for key in keys if key not in self._batch_loss_cache))
        for group in _group_by_prefix_len(misses).values():
            self._fill_batch_cache(group)
        return [
            self._batch_loss_cache[key] + _state_penalty(state, self.violation_penalty)
            for state, key in zip(states, keys)
        ]

    @lru_cache(maxsize=200_000)
    def _loss_for_prefix(self, prefix: tuple[int, ...], dist_items: tuple[tuple[int, float], ...]) -> float:
        if not dist_items:
            return 0.0
        with torch.inference_mode():
            input_ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
            logits = self.model(input_ids)[:, -1, :]
            return _sparse_teacher_kl(logits, [dist_items], self.device)[0]

    def _fill_batch_cache(self, keys: list[tuple[tuple[int, ...], tuple[tuple[int, float], ...]]]) -> None:
        if not keys:
            return
        nonempty = [key for key in keys if key[1]]
        for key in keys:
            if not key[1]:
                self._batch_loss_cache[key] = 0.0
        if not nonempty:
            return
        for chunk_start in range(0, len(nonempty), self.max_batch_rows):
            chunk = nonempty[chunk_start : chunk_start + self.max_batch_rows]
            self.batch_forward_calls += 1
            self.batch_forward_rows += len(chunk)
            self.cache_miss_entries += len(chunk)
            with torch.inference_mode():
                input_ids = torch.tensor([key[0] for key in chunk], dtype=torch.long, device=self.device)
                logits = self.model(input_ids)[:, -1, :]
                values = _sparse_teacher_kl(logits, [key[1] for key in chunk], self.device)
                for key, value in zip(chunk, values):
                    self._batch_loss_cache[key] = float(value)


def _prefix_tuple(state: Hashable) -> tuple[int, ...]:
    prefix = getattr(state, "prefix", None)
    if prefix is None:
        raise TypeError("oracle state must expose a token prefix for neural student loss")
    return tuple(int(x) for x in prefix)


def _dist_tuple(dist: Mapping[int, float]) -> tuple[tuple[int, float], ...]:
    return tuple(sorted((int(token_id), float(prob)) for token_id, prob in dist.items() if prob > 0.0))


def _state_penalty(state: Hashable, violation_penalty: float) -> float:
    if violation_penalty <= 0.0:
        return 0.0
    return violation_penalty if getattr(state, "status", "") == "violation" else 0.0


def _group_by_prefix_len(
    keys: list[tuple[tuple[int, ...], tuple[tuple[int, float], ...]]],
) -> dict[int, list[tuple[tuple[int, ...], tuple[tuple[int, float], ...]]]]:
    groups: dict[int, list[tuple[tuple[int, ...], tuple[tuple[int, float], ...]]]] = {}
    for key in keys:
        groups.setdefault(len(key[0]), []).append(key)
    return groups


def _sparse_teacher_kl(
    logits: torch.Tensor,
    dists: list[tuple[tuple[int, float], ...]],
    device: torch.device,
) -> list[float]:
    log_probs = F.log_softmax(logits, dim=-1)
    values = []
    for row, dist_items in enumerate(dists):
        if not dist_items:
            values.append(0.0)
            continue
        token_ids = torch.tensor([int(token_id) for token_id, _ in dist_items], dtype=torch.long, device=device)
        probs = torch.tensor([float(prob) for _, prob in dist_items], dtype=torch.float32, device=device)
        selected = log_probs[row, token_ids]
        kl = (probs * (probs.clamp_min(1e-12).log() - selected)).sum()
        values.append(float(kl.detach().cpu()))
    return values
