import os

import torch
import torch.nn.functional as F
from torch import nn
from protgen.data import AA, IGNORE_INDEX, ProteinDataset, total_tokens

CONTEXT_WIN = 3             # context window, 3 amino acids
NUM_AA = len(AA)            # total number of amino acids
START_AA_ID = 3             # amino acids start id 0-2 are used for other tokens

# X_train, Y_train = ProteinDataset("/zhome/3b/a/229154/02456-autoregressive-project/data/train.csv")
# X_val, Y_val = ProteinDataset("/zhome/3b/a/229154/02456-autoregressive-project/data/val.csv")
# X_test, Y_test = ProteinDataset("/zhome/3b/a/229154/02456-autoregressive-project/data/test.csv")
train_data = ProteinDataset("/zhome/3b/a/229154/02456-autoregressive-project/data/train.csv")
val_dats = ProteinDataset("/zhome/3b/a/229154/02456-autoregressive-project/data/val.csv")
test_data = ProteinDataset("/zhome/3b/a/229154/02456-autoregressive-project/data/test.csv")


def one_hot_context(context_tokens):
    """(..., CONTEXT) token ids -> (..., CONTEXT * NUM_AA) float one-hot.

    Only residues get a one: <BOS> and <PAD> (the positions before the protein starts) become
    all-zero blocks, so the first residue is predicted from a context of zeros.
    """
    is_residue = context_tokens >= START_AA_ID
    residue_idx = (context_tokens - START_AA_ID).clamp(min=0)
    one_hot = F.one_hot(residue_idx, NUM_AA) * is_residue.unsqueeze(-1)
    return one_hot.flatten(start_dim=-2).float()


def make_windows(dataset, context=CONTEXT_WIN):
    """Every (context, next token) pair in `dataset`.

    Returns X: (N, context * NUM_AA) float, the one-hot of the `context` tokens before each
    position, and Y: (N,) long, the token that follows - a residue id or <EOS>.
    """
    tokens, targets = dataset.tokens, dataset.tokens_shifted          # (S, 129) each
    # Left-pad so the earliest positions also have `context` tokens to look back on.
    padded = F.pad(tokens, (context - 1, 0), value=0)                  # (S, 129 + context - 1)
    windows = padded.unfold(dimension=1, size=context, step=1)        # (S, 129, context)
    keep = targets != IGNORE_INDEX                                    # drop positions past <EOS>
    return one_hot_context(windows[keep]), targets[keep]

class MLP(nn.Module):
    """One-hot context (B, context * NUM_AA) -> next-token logits (B, vocab_size).

    Returns raw logits, no softmax: F.cross_entropy applies log-softmax itself.
    """

    def __init__(self, context=CONTEXT_WIN, hidden_dim=256, num_layers=2, dropout=0.0,
                 vocab_size=len(total_tokens)):
        super().__init__()
        self.context = context      # the sampler reads it, so it always matches the training windows
        layers, in_dim = [], context * NUM_AA
        for _ in range(num_layers):
            # Dropout is always a layer, even at p=0, so checkpoints share keys whatever p was.
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, vocab_size))   # one score per token id, 0..22
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, X, Y, batch_size=4096):
    """Mean cross-entropy (nats per token) of `model` on all of X, Y."""
    model.eval()
    device = next(model.parameters()).device
    total = 0.0
    for i in range(0, len(X), batch_size):
        xb, yb = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
        total += F.cross_entropy(model(xb), yb, reduction="sum").item()
    return total / len(X)


def train(model, X_train, Y_train, X_val, Y_val, *, epochs=10, batch_size=512, lr=1e-3,
          weight_decay=0.01, device=None):
    """Fit `model` with AdamW; returns the per-epoch history and keeps the best-val weights.

    The windows are already tensors in memory, so batches are slices of a shuffled index rather
    than a DataLoader, which would build every batch one row at a time.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history, best_val, best_state = [], float("inf"), None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train))
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx].to(device), Y_train[idx].to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        train_loss, val_loss = total / len(X_train), evaluate(model, X_val, Y_val)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)   # end on the epoch that did best on validation
    return history


PAD_ID, BOS_ID, EOS_ID = (total_tokens.index(t) for t in ("<PAD>", "<BOS>", "<EOS>"))
MAX_RESIDUES = 127          # the longest protein in the data


@torch.no_grad()
def sample(model, n, *, temperature=1.0, max_residues=MAX_RESIDUES):
    """Generate `n` proteins at once; returns them as residue strings.

    Every sequence starts from the context <PAD> ... <PAD> <BOS> (all zeros, as in training) and
    stops at its first <EOS>. One that reaches `max_residues` without an <EOS> is cut there.
    """
    model.eval()
    device = next(model.parameters()).device
    ctx = torch.full((n, model.context), PAD_ID, dtype=torch.long, device=device)
    ctx[:, -1] = BOS_ID
    out = torch.full((n, max_residues), PAD_ID, dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)

    for t in range(max_residues + 1):
        logits = model(one_hot_context(ctx)) / temperature
        logits[:, [PAD_ID, BOS_ID]] = float("-inf")   # never targets in training, never sampled
        if t == 0:
            logits[:, EOS_ID] = float("-inf")         # no empty proteins
        if t == max_residues:
            break
        nxt = torch.multinomial(logits.softmax(dim=-1), 1).squeeze(1)
        nxt[done] = PAD_ID                            # finished rows just pad
        done |= nxt == EOS_ID
        out[:, t] = torch.where(nxt == EOS_ID, PAD_ID, nxt)
        ctx = torch.cat([ctx[:, 1:], nxt[:, None]], dim=1)
        if done.all():
            break

    return ["".join(AA[i - START_AA_ID] for i in row.tolist() if i >= START_AA_ID) for row in out]


if __name__ == "__main__":
    import argparse
    from protgen.metrics.sequences import write_sequences

    parser = argparse.ArgumentParser(description="Train the MLP, or sample from a trained one.")
    parser.add_argument("--sample", type=int, metavar="N", help="sample N proteins instead of training")
    parser.add_argument("--context", type=int, default=CONTEXT_WIN, help="residues of context")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout after each hidden layer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--checkpoint", help="default: runs/<run name>.pt")
    parser.add_argument("--out", help="default: results/sequences/<run name>.csv")
    args = parser.parse_args()
    # e.g. mlp_ctx10, mlp_ctx10_do0.2 - the settings that change the model name the run
    name = (f"mlp_ctx{args.context}" + (f"_do{args.dropout:g}" if args.dropout else "")
            + (f"_ep{args.epochs}" if args.epochs != 10 else ""))
    args.checkpoint = args.checkpoint or f"runs/{name}.pt"
    args.out = args.out or f"results/sequences/{name}.csv"
    torch.manual_seed(0)

    if args.sample:
        model = MLP(context=args.context)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        seqs = sample(model, args.sample, temperature=args.temperature)
        write_sequences(args.out, [{"sequence": s} for s in seqs])
        print(f"wrote {len(seqs)} sequences to {args.out}")
    else:
        X_train, Y_train = make_windows(train_data, context=args.context)
        X_val, Y_val = make_windows(val_dats, context=args.context)
        model = MLP(context=args.context, dropout=args.dropout)
        history = train(model, X_train, Y_train, X_val, Y_val, epochs=args.epochs)
        os.makedirs("runs", exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)   # runs/ is gitignored
