"""Training loop, imported by both train.py and the notebooks so they run the same code.

    splits = load_splits()                  # tokenising takes a moment, so do it once
    result = train_model("gru", splits=splits, epochs=30, lr=1e-3)
"""

import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from protgen.data import IGNORE_INDEX, ProteinDataset
from protgen.models import build_model

DEFAULTS = dict(epochs=30, batch_size=64, lr=1e-3, dropout=0.1, hidden_dim=256,
                num_layers=2, num_heads=4, context=3, seed=0)


def model_kwargs(name, *, hidden_dim, num_layers, dropout, num_heads, context, **ignored):
    """Translate the flat hyperparameters into each model's constructor arguments."""
    if name == "mlp":
        return dict(context=context, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if name == "transformer":
        return dict(d_model=hidden_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
    return dict(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)


def run_epoch(model, loader, device, optimizer=None):
    """One pass over the data; trains if given an optimizer. Returns mean NLL per token."""
    training = optimizer is not None
    model.train(training)
    total_nll, total_tokens = 0.0, 0
    with torch.set_grad_enabled(training):
        for x, target, _, _ in loader:
            x, target = x.to(device), target.to(device)
            logits = model(x)
            # reduction="sum": batches hold different numbers of real tokens, so the per-token
            # mean has to be taken over the whole epoch
            loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(),
                                   ignore_index=IGNORE_INDEX, reduction="sum")
            n_tokens = int((target != IGNORE_INDEX).sum())
            if training:
                optimizer.zero_grad()
                (loss / n_tokens).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # the vanilla RNN needs it
                optimizer.step()
            total_nll += loss.item()
            total_tokens += n_tokens
    return total_nll / total_tokens


def load_model(run_name, out_dir="runs", device=None):
    """Rebuild a run's best checkpoint. Returns (model in eval mode, checkpoint dict)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(out_dir) / run_name / "best.pt", map_location=device)
    model = build_model(checkpoint["model"], checkpoint["vocab_size"], **checkpoint["kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def load_results(out_dir="runs"):
    """Read finished runs back, in the same shape train_model returns.

    Lets the notebooks report runs trained elsewhere (hpc/train.sh, say) without retraining.
    """
    results = []
    for path in sorted(Path(out_dir).glob("*/best.pt")):
        checkpoint = torch.load(path, map_location="cpu")
        model = build_model(checkpoint["model"], checkpoint["vocab_size"], **checkpoint["kwargs"])
        results.append(dict(
            run=path.parent.name, model=checkpoint["model"],
            params=sum(p.numel() for p in model.parameters()),
            best_val_nll=checkpoint["val_nll"], best_val_ppl=math.exp(checkpoint["val_nll"]),
            best_epoch=checkpoint["epoch"],
            history=pd.read_csv(path.parent / "history.csv"),
            **checkpoint.get("config", {}),
        ))
    return results


def load_splits(data_dir="data"):
    """(train, val) datasets."""
    data_dir = Path(data_dir)
    return ProteinDataset(data_dir / "train.csv"), ProteinDataset(data_dir / "val.csv")


def train_model(name, *, splits=None, run_name=None, out_dir="runs", device=None, log=print, **hp):
    """Train one configuration. Any of DEFAULTS can be overridden by keyword.

    Writes <out_dir>/<run_name>/best.pt and history.csv. Returns a dict with the config, the
    parameter count, the best validation loss, and `history` as a DataFrame.
    """
    config = {**DEFAULTS, **hp}
    unknown = set(hp) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown hyperparameter(s): {sorted(unknown)}")

    run_name = run_name or name
    out = Path(out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])

    train_data, val_data = splits or load_splits()
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256)

    kwargs = model_kwargs(name, **config)
    model = build_model(name, train_data.vocab_size, **kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"{run_name}: {name} {kwargs}")
    log(f"  {n_params:,} parameters on {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    best_val, best_epoch, history = math.inf, 0, []
    for epoch in range(1, config["epochs"] + 1):
        train_nll = run_epoch(model, train_loader, device, optimizer)
        val_nll = run_epoch(model, val_loader, device)
        history.append(dict(epoch=epoch, train_nll=train_nll, val_nll=val_nll,
                            train_ppl=math.exp(train_nll), val_ppl=math.exp(val_nll)))
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

        improved = val_nll < best_val
        log(f"  epoch {epoch:3d} | train {train_nll:.4f} (ppl {math.exp(train_nll):5.2f})"
            f" | val {val_nll:.4f} (ppl {math.exp(val_nll):5.2f})" + ("  *" if improved else ""))
        if improved:
            best_val, best_epoch = val_nll, epoch
            # store enough to rebuild the model and to report the run without retraining it
            torch.save(dict(model=name, kwargs=kwargs, vocab_size=train_data.vocab_size,
                            state_dict=model.state_dict(), epoch=epoch, val_nll=val_nll,
                            config=config),
                       out / "best.pt")

    log(f"  best val {best_val:.4f} (ppl {math.exp(best_val):.2f}) at epoch {best_epoch}")
    return dict(run=run_name, model=name, params=n_params, best_val_nll=best_val,
                best_val_ppl=math.exp(best_val), best_epoch=best_epoch,
                history=pd.DataFrame(history), **config)
