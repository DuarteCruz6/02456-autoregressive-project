"""Sample from a trained checkpoint into a CSV that eval/ can read.

    uv run python generate.py --run gru
    uv run python generate.py --run gru --temperature 0.8 --out results/sequences/gru_T0.8.csv

Use the same --num as eval/make_controls.py, since some metrics depend on set size.
"""

import argparse

import torch

from protgen.metrics import write_sequences
from protgen.sampling import sample
from protgen.training import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="folder name under runs/")
    p.add_argument("--num", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default: results/sequences/<run>.csv")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.run, device=device)
    seqs = sample(model, args.num, temperature=args.temperature, device=device)

    out = write_sequences(args.out or f"results/sequences/{args.run}.csv",
                          [{"sequence": s} for s in seqs])

    lengths = [len(s) for s in seqs]
    print(f"{len(seqs)} sequences -> {out}")
    print(f"length min/mean/max: {min(lengths)} / {sum(lengths) / len(lengths):.1f} /"
          f" {max(lengths)}   (data: 68 / 95 / 127)")


if __name__ == "__main__":
    main()
