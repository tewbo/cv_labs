import argparse
from pathlib import Path

from lab1.data import prepare_dvm_manifests
from lab1.infer import predict_samples
# from lab1.report import build_report
from lab1.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lab 1: car color classification on DVM-CAR front views."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-dvm")
    prepare.add_argument("--images-root", required=True, type=Path)
    prepare.add_argument("--out", required=True, type=Path)
    prepare.add_argument("--image-table", type=Path)
    prepare.add_argument("--front-viewpoint", default="0")
    prepare.add_argument("--seed", default=42, type=int)
    prepare.add_argument("--val-size", default=0.15, type=float)
    prepare.add_argument("--test-size", default=0.15, type=float)
    prepare.add_argument("--min-per-class", default=20, type=int)
    prepare.add_argument("--limit-per-class", type=int)
    prepare.add_argument(
        "--exclude-label",
        action="append",
        default=["Unlisted"],
        help="Label to exclude; can be passed multiple times. Defaults to Unlisted.",
    )

    train = subparsers.add_parser("train")
    _add_train_args(train)

    run_all = subparsers.add_parser("run-all")
    _add_train_args(run_all)

    report = subparsers.add_parser("report")
    report.add_argument("--runs-dir", default=Path("runs"), type=Path)
    report.add_argument("--out", default=Path("report.md"), type=Path)

    predict = subparsers.add_parser("predict-samples")
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--manifest-dir", required=True, type=Path)
    predict.add_argument("--out-dir", default=Path("runs/inference"), type=Path)
    predict.add_argument("--model", default="resnet18")
    predict.add_argument("--num-samples", default=16, type=int)
    predict.add_argument("--image-size", default=224, type=int)
    predict.add_argument("--seed", default=42, type=int)
    predict.add_argument("--device", default="auto")

    args = parser.parse_args()

    if args.command == "prepare-dvm":
        prepare_dvm_manifests(
            images_root=args.images_root,
            out_dir=args.out,
            image_table=args.image_table,
            front_viewpoint=args.front_viewpoint,
            seed=args.seed,
            val_size=args.val_size,
            test_size=args.test_size,
            min_per_class=args.min_per_class,
            limit_per_class=args.limit_per_class,
            exclude_labels=args.exclude_label,
        )
        return

    if args.command == "train":
        train_model(args)
        return

    if args.command == "run-all":
        for model_name in ("scratch_resnet", "resnet18", "mobilenet_v3_small"):
            args.model = model_name
            train_model(args)
        return

    if args.command == "report":
        # build_report(args.runs_dir, args.out)
        return

    if args.command == "predict-samples":
        predict_samples(args)
        return


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=("scratch_resnet", "resnet18", "mobilenet_v3_small"),
        default="scratch_resnet",
    )
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--runs-dir", default=Path("runs"), type=Path)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--freeze-backbone-epochs", default=2, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-eval-batches", type=int)
