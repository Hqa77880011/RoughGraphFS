import argparse

from .data import DATASETS, download
from .gnn import BACKBONES
from .report import plot, summarize
from .run import run
from .selector import SELECTORS


def main():
    parser = argparse.ArgumentParser(prog="python -m rgfs")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("download")
    data.add_argument("--dataset", choices=DATASETS, required=True)
    data.add_argument("--root", default="data")
    train = commands.add_parser("train")
    train.add_argument("--dataset", choices=DATASETS, required=True)
    train.add_argument("--root", default="data")
    train.add_argument("--output", required=True)
    train.add_argument("--fold", type=int, default=0)
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--split-seed", type=int, default=2024)
    train.add_argument("--selector", choices=SELECTORS, default="rgfs")
    train.add_argument("--backbone", choices=BACKBONES, default="gin")
    train.add_argument("--retained", type=float, default=0.5)
    train.add_argument("--views", type=int, default=3)
    train.add_argument("--anchors", type=int, default=512)
    train.add_argument("--radius-quantile", type=float, default=0.2)
    train.add_argument("--alpha", type=float, default=0.1)
    train.add_argument("--beta", type=float, default=0.1)
    train.add_argument("--redundancy", type=float, default=0.05)
    train.add_argument("--stability", type=float, default=0.05)
    train.add_argument("--minimum-weight", type=float, default=0.5)
    train.add_argument("--inner-rate", type=float, default=0.1)
    train.add_argument("--nuisance-ratio", type=float, default=0.3)
    train.add_argument("--label-noise", type=float, default=0)
    train.add_argument("--mask")
    train.add_argument("--hidden", type=int, default=64)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--patience", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=0.001)
    train.add_argument("--weight-decay", type=float, default=0.0001)
    train.add_argument("--device", default="cpu")
    stats = commands.add_parser("summarize")
    stats.add_argument("--metrics", nargs="+", required=True)
    stats.add_argument("--output", default="outputs/statistics")
    stats.add_argument("--resamples", type=int, default=10000)
    visual = commands.add_parser("plot")
    visual.add_argument("--summary", required=True)
    visual.add_argument("--output", default="outputs/figures")
    args = parser.parse_args()
    if args.command == "download":
        download(args.dataset, args.root)
    elif args.command == "train":
        run(args)
    elif args.command == "summarize":
        summarize(args.metrics, args.output, args.resamples)
    else:
        plot(args.summary, args.output)


if __name__ == "__main__":
    main()
