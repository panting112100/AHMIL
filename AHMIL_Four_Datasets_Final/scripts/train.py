"""Train AHMIL or one of its controlled ablations with subject-wise LOSO."""

import argparse
from pathlib import Path

from ahmil.data import load_dataset_bundle
from ahmil.engine import run_loso


def build_parser():
    parser = argparse.ArgumentParser(description="AHMIL four-dataset LOSO runner")
    parser.add_argument("--dataset", required=True,
                        choices=["seed", "seed_iv", "dreamer", "hbued"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--task", default="emotion",
                        choices=["emotion", "valence", "arousal"])
    parser.add_argument("--feature_key", default="rb_data", choices=["rb_data", "data"])
    parser.add_argument("--output_dir", default="./outputs/ahmil")
    parser.add_argument("--variant", default="full",
                        choices=["baseline", "mlp", "fuzzy", "hyper", "fuzzy_epc", "full"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--fuzzynum", type=int, default=3)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default=None)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm; manuscript does not specify the threshold.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Fixed seed for fold-wise train/validation partitions across model random seeds.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Set -1 to force CPU.")
    parser.add_argument("--interference_ratio", type=float, default=0.0,
                        help="Irrelevant-instance fraction in each resulting bag (0 to 0.75 for the paper experiment).")
    parser.add_argument("--electrode_occlusion", action="store_true")
    parser.add_argument("--validation_only", action="store_true",
                        help="Run model selection on validation subjects only; do not evaluate held-out test subjects.")
    parser.add_argument("--log_interval", type=int, default=5)
    return parser


def main():
    args = build_parser().parse_args()
    args.dataset = args.dataset.lower().replace("-", "_")
    if args.dataset in {"seed", "seed_iv"} and args.task != "emotion":
        raise SystemExit("SEED and SEED-IV require --task emotion.")
    if args.dataset in {"dreamer", "hbued"} and args.task == "emotion":
        raise SystemExit("DREAMER and HBUED require --task valence or --task arousal.")
    if args.fuzzynum < 1 or args.fuzzynum % 2 == 0:
        raise SystemExit("--fuzzynum must be a positive odd integer (e.g. 3, 5, 7, 9, 11).")
    if args.curvature <= 0:
        raise SystemExit("--curvature must be positive.")

    bundle = load_dataset_bundle(
        args.dataset, args.data_dir, args.task, feature_key=args.feature_key
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_loso(bundle, args)


if __name__ == "__main__":
    main()
