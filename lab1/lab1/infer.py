from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from lab1.models import build_model


def predict_samples(args) -> None:
    checkpoint_path = Path(args.checkpoint)
    manifest_dir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _select_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    image_size = int(checkpoint.get("image_size", args.image_size))
    model_name = checkpoint.get("model_name", args.model)

    model = build_model(model_name, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    images_root = manifest_dir.parent / "confirmed_fronts"
    frame = pd.read_csv(manifest_dir / "test.csv")
    frame["path"] = frame["path"].map(lambda path: str(_resolve_image_path(Path(path), images_root)))
    frame = frame[frame["path"].map(lambda path: Path(path).exists())].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("No existing images found in test.csv")
    sample = frame.sample(n=min(args.num_samples, len(frame)), random_state=args.seed)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    rows = []
    images_for_grid = []
    with torch.no_grad():
        for _, row in sample.iterrows():
            image_path = Path(row["path"])
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            probabilities = torch.softmax(model(tensor), dim=1)[0]
            confidence, pred_idx = probabilities.max(dim=0)
            true_label = str(row["label"])
            pred_label = idx_to_class[int(pred_idx.item())]
            rows.append(
                {
                    "path": str(image_path),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": float(confidence.item()),
                    "correct": true_label == pred_label,
                }
            )
            images_for_grid.append((image.copy(), true_label, pred_label, float(confidence.item())))

    predictions_csv = out_dir / "predictions.csv"
    with predictions_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["path", "true_label", "pred_label", "confidence", "correct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    grid_path = out_dir / "predictions_grid.png"
    _save_grid(images_for_grid, grid_path)
    accuracy = sum(row["correct"] for row in rows) / len(rows)
    print(f"Model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {len(rows)}")
    print(f"Sample accuracy: {accuracy:.4f}")
    print(f"Saved predictions: {predictions_csv}")
    print(f"Saved visualization: {grid_path}")


def _save_grid(items: list[tuple[Image.Image, str, str, float]], path: Path) -> None:
    cols = 4
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    if rows == 1:
        axes = [axes]
    flat_axes = [ax for row_axes in axes for ax in row_axes]
    for ax in flat_axes:
        ax.axis("off")

    for ax, (image, true_label, pred_label, confidence) in zip(flat_axes, items):
        ax.imshow(image)
        color = "green" if true_label == pred_label else "red"
        ax.set_title(
            f"true: {true_label}\npred: {pred_label} ({confidence:.2f})",
            color=color,
            fontsize=9,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def _resolve_image_path(path: Path, images_root: Path) -> Path:
    if path.exists():
        return path
    parts = path.parts
    if "confirmed_fronts" in parts:
        index = parts.index("confirmed_fronts")
        candidate = images_root.joinpath(*parts[index + 1 :])
        if candidate.exists():
            return candidate
    return path


def _select_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
