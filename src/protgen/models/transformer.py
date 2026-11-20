"""Decoder-only transformer with a causal mask.

Built from TransformerEncoderLayer, since "decoder-only" is just an encoder stack with the
mask and no cross-attention.
"""

import torch
from torch import nn


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_layers=2,
                 dropout=0.1, max_len=129):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)   # attention has no order of its own
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True,
            norm_first=True,   # pre-LN, so no warm-up schedule needed
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        T = x.size(1)
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_embed(x) + self.pos_embed(pos))
        # without this mask position t sees the answer and the loss is meaningless
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))
