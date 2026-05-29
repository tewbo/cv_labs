import json
import os
from pathlib import Path

from PIL import Image

from lab2.yolo import _load_yolo


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def evaluate_yolo(args) -> None:
    YOLO = _load_yolo()
    model = YOLO(str(args.model))
    results = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_json=True,
        plots=True,
    )

    split_paths = _resolve_split_paths(args.data, args.split)
    mean_iou = None
    if split_paths is not None:
        image_dir, label_dir = split_paths
        mean_iou = _mean_best_iou(model, image_dir, label_dir, args)

    box = results.box
    metrics = {
        "split": args.split,
        "model": str(args.model),
        "data": str(args.data),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "mean_best_iou": mean_iou,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def _resolve_split_paths(data_yaml: Path, split: str):
    try:
        import yaml
    except ImportError:
        return None

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    split_value = config.get(split)
    if split_value is None:
        return None
    image_dir = Path(split_value)
    if not image_dir.is_absolute():
        image_dir = root / image_dir

    label_dir = Path(
        str(image_dir).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
    )
    if image_dir.name == "images":
        label_dir = image_dir.parent / "labels"
    return image_dir, label_dir


def _mean_best_iou(model, image_dir: Path, label_dir: Path, args) -> float | None:
    image_paths = [
        p for p in sorted(image_dir.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_paths:
        return None

    ious: list[float] = []
    predictions = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=args.imgsz,
        conf=0.25,
        iou=args.iou,
        device=args.device,
        verbose=False,
        stream=True,
    )
    for image_path, result in zip(image_paths, predictions):
        label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
        if not label_path.exists():
            continue

        with Image.open(image_path) as image:
            width, height = image.size
        gt_boxes = _read_yolo_boxes(label_path, width, height)
        if not gt_boxes:
            continue

        pred_boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
        for gt in gt_boxes:
            best = max((_iou_xyxy(gt, pred) for pred in pred_boxes), default=0.0)
            ious.append(best)

    if not ious:
        return None
    return float(sum(ious) / len(ious))


def _read_yolo_boxes(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, x_center, y_center, box_width, box_height = parts[:5]
        x_center = float(x_center) * width
        y_center = float(y_center) * height
        box_width = float(box_width) * width
        box_height = float(box_height) * height
        boxes.append(
            [
                x_center - box_width / 2,
                y_center - box_height / 2,
                x_center + box_width / 2,
                y_center + box_height / 2,
            ]
        )
    return boxes


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
