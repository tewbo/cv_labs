from __future__ import annotations

import json
import random
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from lab1.dataset import ManifestImageDataset
from lab1.models import build_model, set_backbone_trainable


def train_model(args: Namespace) -> None:
    set_seed(args.seed)
    manifest_dir = Path(args.manifest_dir)
    class_to_idx = json.loads((manifest_dir / "class_to_idx.json").read_text(encoding="utf-8"))
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    num_classes = len(class_to_idx)

    device = _select_device(args.device)
    run_dir = Path(args.runs_dir) / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = _build_loaders(args, manifest_dir)
    model = build_model(args.model, num_classes=num_classes, pretrained=args.model != "scratch_resnet")
    model.to(device)
    checkpoint_path = run_dir / "best.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("class_to_idx") != class_to_idx:
            raise RuntimeError("Cannot resume: checkpoint classes differ from current manifest classes")
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Resumed {args.model} from {checkpoint_path}")

    criterion = nn.CrossEntropyLoss(weight=_class_weights(manifest_dir / "train.csv", num_classes).to(device))
    best_val_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        freeze = args.model != "scratch_resnet" and epoch <= args.freeze_backbone_epochs
        set_backbone_trainable(model, args.model, trainable=not freeze)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        train_loss = _run_train_epoch(
            model, train_loader, criterion, optimizer, device, args.limit_train_batches
        )
        val_metrics = _evaluate(
            model, val_loader, criterion, device, args.limit_eval_batches
        )
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"{args.model} epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            torch.save(
                {
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "image_size": args.image_size,
                },
                run_dir / "best.pt",
            )

    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics, y_true, y_pred = _evaluate(
        model,
        test_loader,
        criterion,
        device,
        args.limit_eval_batches,
        return_predictions=True,
    )

    metrics = {
        "model": args.model,
        "num_classes": num_classes,
        "best_val_f1_macro": best_val_f1,
        "test": test_metrics,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    target_names = [idx_to_class[i] for i in range(num_classes)]
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=target_names,
        zero_division=0,
    )
    (run_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    _plot_history(history, run_dir / "learning_curves.png")
    _plot_confusion(y_true, y_pred, target_names, run_dir / "confusion_matrix.png")
    print(f"{args.model} test F1_macro={test_metrics['f1_macro']:.4f}; artifacts: {run_dir}")


def _build_loaders(args: Namespace, manifest_dir: Path):
    train_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=5, translate=(0.03, 0.03)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    train_ds = ManifestImageDataset(manifest_dir / "train.csv", transform=train_tf)
    val_ds = ManifestImageDataset(manifest_dir / "val.csv", transform=eval_tf)
    test_ds = ManifestImageDataset(manifest_dir / "test.csv", transform=eval_tf)
    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
    )


def _class_weights(train_csv: Path, num_classes: int) -> torch.Tensor:
    frame = pd.read_csv(train_csv)
    frame = frame[frame["path"].map(lambda path: Path(path).exists())]
    targets = frame["target"].to_numpy()
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    limit_batches: int | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        total_count += images.size(0)
        if limit_batches is not None and batch_idx >= limit_batches:
            break
    return total_loss / max(total_count, 1)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    limit_batches: int | None,
    return_predictions: bool = False,
):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_targets = []
    all_preds = []
    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="eval", leave=False), start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)
        total_loss += float(loss.item()) * images.size(0)
        total_count += images.size(0)
        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        if limit_batches is not None and batch_idx >= limit_batches:
            break

    metrics = {
        "loss": total_loss / max(total_count, 1),
        "f1_macro": f1_score(all_targets, all_preds, average="macro", zero_division=0),
    }
    if return_predictions:
        return metrics, all_targets, all_preds
    return metrics


def _plot_history(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["val_f1_macro"] for row in history], label="val F1_macro")
    plt.xlabel("epoch")
    plt.ylabel("F1_macro")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_confusion(y_true: list[int], y_pred: list[int], target_names: list[str], path: Path) -> None:
    fig_size = max(7, len(target_names) * 0.55)
    _, ax = plt.subplots(figsize=(fig_size, fig_size))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=target_names,
        xticks_rotation=45,
        normalize="true",
        ax=ax,
        colorbar=False,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _select_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
