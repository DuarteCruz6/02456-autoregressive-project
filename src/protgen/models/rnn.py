"""Vanilla RNN, GRU and LSTM. Same class, different cell, so only the cell differs."""

from torch import nn

CELLS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


class RNNLM(nn.Module):
    def __init__(self, vocab_size, cell="gru", embed_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.1):
        super().__init__()
        if cell not in CELLS:
            raise ValueError(f"cell must be one of {list(CELLS)}, got {cell!r}")
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = CELLS[cell](
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # only applies between layers
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        out, _ = self.rnn(self.embed(x))   # final state unused: we always feed whole sequences
        return self.head(self.drop(out))
