from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ManifestImageDataset(Dataset):
    def __init__(self, csv_path: Path, transform=None, check_files: bool = True):
        self.frame = pd.read_csv(csv_path)
        self.transform = transform
        images_root = csv_path.parent.parent / "confirmed_fronts"
        self.frame["path"] = self.frame["path"].map(
            lambda path: str(_resolve_image_path(Path(path), images_root))
        )
        if check_files:
            existing = self.frame["path"].map(lambda path: Path(path).exists())
            missing_count = int((~existing).sum())
            if missing_count:
                print(f"{csv_path}: skipped {missing_count} missing files")
            self.frame = self.frame[existing].reset_index(drop=True)
        if self.frame.empty:
            raise RuntimeError(f"No existing images left in {csv_path}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["target"])


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
