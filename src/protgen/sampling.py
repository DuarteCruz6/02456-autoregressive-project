"""Sampling, shared by every model.

Feeds the whole prefix back in at each step instead of keeping a hidden state. That is O(T^2),
but T <= 129 here and it keeps the models free of state-passing.
"""

import torch

from protgen.data import AA, total_tokens

VOCAB = {token: i for i, token in enumerate(total_tokens)}
PAD, BOS, EOS = VOCAB["<PAD>"], VOCAB["<BOS>"], VOCAB["<EOS>"]
RESIDUE = {i: token for i, token in enumerate(total_tokens) if token in AA}


def decode(tokens):
    """Token ids -> residue string, stopping at the first <EOS>."""
    out = []
    for token in tokens:
        token = int(token)
        if token == EOS:
            break
        if token in RESIDUE:
            out.append(RESIDUE[token])
    return "".join(out)


@torch.no_grad()
def sample(model, num_samples, max_len=129, temperature=1.0, device="cpu"):
    """Generate `num_samples` sequences, returned as residue strings."""
    model.eval()
    x = torch.full((num_samples, 1), BOS, dtype=torch.long, device=device)
    done = torch.zeros(num_samples, dtype=torch.bool, device=device)
    for step in range(max_len - 1):     # max_len counts <BOS> and <EOS>
        logits = model(x)[:, -1] / temperature
        logits[:, [PAD, BOS]] = float("-inf")   # never targets in training
        if step == 0:
            logits[:, EOS] = float("-inf")      # no empty proteins
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        nxt[done] = EOS                 # finished rows keep emitting <EOS>
        x = torch.cat([x, nxt], dim=1)
        done |= nxt.squeeze(1) == EOS
        if done.all():
            break
    return [decode(row[1:]) for row in x.cpu()]   # row[0] is <BOS>
