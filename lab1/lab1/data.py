import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prepare_dvm_manifests(
    images_root: Path,
    out_dir: Path,
    image_table: Path | None = None,
    front_viewpoint: str = "0",
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
    min_per_class: int = 20,
    limit_per_class: int | None = None,
    exclude_labels: list[str] | None = None,
) -> None:
    images_root = images_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _rows_from_image_table(images_root, image_table, front_viewpoint) if image_table else _rows_from_tree(images_root)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {images_root}")

    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"].ne("")]
    if exclude_labels:
        df = df[~df["label"].isin(exclude_labels)]

    counts = df["label"].value_counts()
    keep_labels = counts[counts >= min_per_class].index
    df = df[df["label"].isin(keep_labels)].copy()
    if df.empty:
        raise RuntimeError("No classes left after --min-per-class filtering")

    if limit_per_class is not None:
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda part: part.sample(min(len(part), limit_per_class), random_state=seed))
            .reset_index(drop=True)
        )

    labels = sorted(df["label"].unique())
    class_to_idx = {label: idx for idx, label in enumerate(labels)}
    df["target"] = df["label"].map(class_to_idx)

    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=df["target"],
    )
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_df["target"],
    )

    _write_split(train_df, out_dir / "train.csv")
    _write_split(val_df, out_dir / "val.csv")
    _write_split(test_df, out_dir / "test.csv")
    (out_dir / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "images_root": str(images_root),
        "num_classes": len(class_to_idx),
        "classes": class_to_idx,
        "num_train": len(train_df),
        "num_val": len(val_df),
        "num_test": len(test_df),
        "class_counts": df["label"].value_counts().sort_index().to_dict(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Prepared {len(df)} images and {len(class_to_idx)} classes in {out_dir}")


def _rows_from_tree(images_root: Path) -> list[dict[str, str]]:
    rows = []
    for image_path in images_root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = _label_from_path_or_name(image_path)
        rows.append({"path": str(image_path.resolve()), "label": label})
    return rows


def _rows_from_image_table(
    images_root: Path,
    image_table: Path,
    front_viewpoint: str,
) -> list[dict[str, str]]:
    table = pd.read_csv(image_table)
    required = {"Image_name", "Predicted_viewpoint"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Image table is missing columns: {sorted(missing)}")

    front = table[table["Predicted_viewpoint"].astype(str) == str(front_viewpoint)]
    rows = []
    for image_name in front["Image_name"].dropna().astype(str):
        parts = image_name.split("$$")
        if len(parts) < 4:
            continue
        label = parts[3].strip()
        image_path = images_root.joinpath(*parts[:4], image_name)
        if image_path.exists():
            rows.append({"path": str(image_path.resolve()), "label": label})
    return rows


def _label_from_path_or_name(image_path: Path) -> str:
    parts = image_path.name.split("$$")
    if len(parts) >= 4:
        return parts[3].strip()
    return image_path.parent.name.strip()


def _write_split(df: pd.DataFrame, path: Path) -> None:
    df = df[["path", "label", "target"]].sample(frac=1.0, random_state=0)
    df.to_csv(path, index=False)
