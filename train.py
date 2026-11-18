"""Train one model.

    uv run python train.py --model gru
    uv run python train.py --model transformer --epochs 40 --dropout 0.2
    uv run python train.py --model gru --lr 3e-4 --run-name gru_lr3e-4

Next-token cross-entropy with teacher forcing. Picks the checkpoint with the lowest validation
loss; the test split is not touched here. Loss is mean NLL per token, so perplexity = exp(loss).
"""

import argparse

from protgen.models import MODEL_NAMES
from protgen.training import DEFAULTS, train_model


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    p.add_argument("--hidden-dim", type=int, default=DEFAULTS["hidden_dim"],
                   help="RNN hidden size, or the transformer's d_model")
    p.add_argument("--num-layers", type=int, default=DEFAULTS["num_layers"])
    p.add_argument("--num-heads", type=int, default=DEFAULTS["num_heads"], help="transformer only")
    p.add_argument("--context", type=int, default=DEFAULTS["context"], help="mlp only")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--run-name", default=None, help="output folder (default: the model name)")
    args = vars(p.parse_args())

    name, run_name = args.pop("model"), args.pop("run_name")
    train_model(name, run_name=run_name, **args)


if __name__ == "__main__":
    main()
