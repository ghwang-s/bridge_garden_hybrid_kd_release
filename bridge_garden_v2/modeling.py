from __future__ import annotations

import math

import torch
import torch.nn as nn

from .schema import ModelConfig


class CausalTransformerLM(nn.Module):
    """Shared causal LM used by all v2 synthetic domains."""

    def __init__(self, vocab_size: int, config: ModelConfig, max_len: int, pad_id: int) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, config.d_model)
        layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, config.n_layers)
        self.head = nn.Linear(config.d_model, vocab_size)
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device)
        hidden = self.tok_emb(input_ids) * math.sqrt(self.d_model) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device)
        return self.head(self.transformer(hidden, mask, is_causal=True))
