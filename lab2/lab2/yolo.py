from pathlib import Path


def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Package 'ultralytics' is not installed. Run: "
            ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc
    return YOLO


def train(args) -> None:
    YOLO = _load_yolo()
    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        resume=args.resume,
        exist_ok=True,
    )


def predict(args) -> None:
    YOLO = _load_yolo()
    model = YOLO(str(args.model))
    model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(args.project),
        name=args.name,
        save=True,
        save_txt=True,
        save_conf=True,
        exist_ok=True,
    )
    print(f"Predictions saved to {Path(args.project) / args.name}")
