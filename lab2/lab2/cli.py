import argparse
from pathlib import Path

from lab2.metrics import evaluate_yolo
from lab2.svhn32 import prepare_svhn32
from lab2.yolo import predict, train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lab 2: house/street number detection with YOLO."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    svhn32_parser = subparsers.add_parser("prepare-svhn32")
    svhn32_parser.add_argument("--src", default=Path("data/svhn"), type=Path)
    svhn32_parser.add_argument("--out", default=Path("data/svhn32-yolo"), type=Path)
    svhn32_parser.add_argument(
        "--yaml-out", default=Path("config/svhn32.yaml"), type=Path
    )
    svhn32_parser.add_argument("--val-fraction", default=0.15, type=float)
    svhn32_parser.add_argument("--seed", default=42, type=int)

    train_parser = subparsers.add_parser("train")
    _add_common_yolo_args(train_parser)
    train_parser.add_argument("--model", default="yolov8n.pt")
    train_parser.add_argument("--epochs", default=80, type=int)
    train_parser.add_argument("--batch", default=16, type=int)
    train_parser.add_argument("--patience", default=20, type=int)
    train_parser.add_argument("--workers", default=4, type=int)
    train_parser.add_argument("--project", default=Path("runs"), type=Path)
    train_parser.add_argument("--name", default="svhn_yolo")
    train_parser.add_argument("--resume", action="store_true")

    eval_parser = subparsers.add_parser("evaluate")
    _add_common_yolo_args(eval_parser)
    eval_parser.add_argument("--model", required=True, type=Path)
    eval_parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    eval_parser.add_argument(
        "--out", default=Path("runs/eval_metrics.json"), type=Path
    )

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", required=True, type=Path)
    predict_parser.add_argument("--source", required=True, type=Path)
    predict_parser.add_argument("--imgsz", default=640, type=int)
    predict_parser.add_argument("--conf", default=0.25, type=float)
    predict_parser.add_argument("--iou", default=0.7, type=float)
    predict_parser.add_argument("--device", default=None)
    predict_parser.add_argument("--project", default=Path("runs"), type=Path)
    predict_parser.add_argument("--name", default="street_photos")

    args = parser.parse_args()

    if args.command == "prepare-svhn32":
        prepare_svhn32(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate_yolo(args)
    elif args.command == "predict":
        predict(args)


def _add_common_yolo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--conf", default=0.001, type=float)
    parser.add_argument("--iou", default=0.7, type=float)
    parser.add_argument("--device", default=None)
