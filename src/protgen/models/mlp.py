"""MLP language model (Bengio et al. 2003): next token from the previous `context` residues.

One-hot the context window, then a few linear layers. Based on the version from feature/mlp.
"""

import torch
import torch.nn.functional as F
from torch import nn

from protgen.data import AA

NUM_AA = len(AA)
FIRST_RESIDUE = 3  # ids 0-2 are <PAD>, <BOS>, <EOS>


def one_hot_context(windows):
    """(..., context) token ids -> (..., context * 20) one-hot. Specials become zero blocks."""
    is_residue = windows >= FIRST_RESIDUE
    residue_idx = (windows - FIRST_RESIDUE).clamp(min=0)
    one_hot = F.one_hot(residue_idx, NUM_AA) * is_residue.unsqueeze(-1)
    return one_hot.flatten(start_dim=-2).float()


class MLPLM(nn.Module):
    def __init__(self, vocab_size, context=3, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.context = context
        layers, in_dim = [], context * NUM_AA
        for _ in range(num_layers):
            # dropout stays a layer at p=0 so checkpoints keep the same keys
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # left-pad so early positions still get a full window; the window at t ends at t,
        # so the model never sees t+1
        padded = F.pad(x, (self.context - 1, 0), value=0)
        windows = padded.unfold(dimension=1, size=self.context, step=1)  # (B, T, context)
        return self.net(one_hot_context(windows))
