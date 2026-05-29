from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


def prepare_svhn32(args) -> None:
    src_dir = args.src.resolve()
    out_dir = args.out.resolve()
    train_mat = src_dir / "train_32x32.mat"
    test_mat = src_dir / "test_32x32.mat"

    if not train_mat.exists() or not test_mat.exists():
        raise SystemExit(
            f"Expected train_32x32.mat and test_32x32.mat in {src_dir}"
        )

    train_images, train_labels = _load_svhn_mat(train_mat)
    test_images, test_labels = _load_svhn_mat(test_mat)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(train_labels))
    val_count = max(1, int(len(indices) * args.val_fraction))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    _write_split(out_dir, "train", train_images, train_labels, train_indices)
    _write_split(out_dir, "valid", train_images, train_labels, val_indices)
    _write_split(out_dir, "test", test_images, test_labels, np.arange(len(test_labels)))
    _write_yaml(args.yaml_out.resolve(), out_dir)

    print(f"SVHN 32x32 YOLO dataset written to {out_dir}")
    print(f"YOLO config written to {args.yaml_out}")


def _load_svhn_mat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(path)
    images = data["X"]
    labels = data["y"].reshape(-1).astype(int)
    labels[labels == 10] = 0
    return images, labels


def _write_split(
    out_dir: Path,
    split: str,
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> None:
    image_dir = out_dir / split / "images"
    label_dir = out_dir / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for counter, idx in enumerate(indices):
        cls = int(labels[idx])
        image = images[:, :, :, idx]
        stem = f"{split}_{counter:06d}"
        Image.fromarray(image).save(image_dir / f"{stem}.png")
        (label_dir / f"{stem}.txt").write_text(
            f"{cls} 0.5 0.5 1.0 1.0\n", encoding="utf-8"
        )


def _write_yaml(path: Path, dataset_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"path: {dataset_dir.as_posix()}",
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "",
                "nc: 10",
                "names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']",
                "",
            ]
        ),
        encoding="utf-8",
    )
