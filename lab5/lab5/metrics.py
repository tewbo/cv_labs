from itertools import combinations
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

from .utils import get_device, list_images, load_json, open_rgb, save_json


def _load_prompts_from_metadata(image_dir: Path) -> list[str]:
    metadata_path = image_dir / "metadata.json"
    if not metadata_path.exists():
        return ["a high quality realistic portrait photo"] * len(list_images(image_dir))
    metadata = load_json(metadata_path)
    return [item["prompt"] for item in metadata["images"]]


@torch.no_grad()
def clip_score(image_dir: Path, device: torch.device) -> float:
    paths = list_images(image_dir)
    prompts = _load_prompts_from_metadata(image_dir)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    scores: list[float] = []
    for path, prompt in zip(paths, prompts):
        inputs = processor(text=[prompt], images=open_rgb(path), return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        image_features = F.normalize(outputs.image_embeds, dim=-1)
        text_features = F.normalize(outputs.text_embeds, dim=-1)
        scores.append(float((image_features * text_features).sum(dim=-1).cpu()))
    return float(np.mean(scores))


@torch.no_grad()
def face_similarity(reference_dir: Path, generated_dir: Path, device: torch.device) -> dict:
    reference_paths = list_images(reference_dir)
    generated_paths = list_images(generated_dir)
    if not reference_paths or not generated_paths:
        return {"mean": None, "reference_images": len(reference_paths), "generated_images": len(generated_paths)}

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def embed(paths: list[Path]) -> torch.Tensor:
        images = [open_rgb(path) for path in paths]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        embeddings = model.get_image_features(**inputs)
        return F.normalize(embeddings, dim=-1)

    reference_center = F.normalize(embed(reference_paths).mean(dim=0, keepdim=True), dim=-1)
    generated_embeddings = embed(generated_paths)
    values = (generated_embeddings @ reference_center.T).squeeze(1).detach().cpu().numpy()
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "reference_images": len(reference_paths),
        "generated_images": len(generated_paths),
    }


@torch.no_grad()
def lpips_diversity(image_dir: Path, device: torch.device) -> float | None:
    paths = list_images(image_dir)
    if len(paths) < 2:
        return None
    metric = lpips.LPIPS(net="alex").to(device)
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    tensors = [transform(open_rgb(path)).unsqueeze(0).to(device) for path in paths]
    values = [float(metric(a, b).cpu()) for a, b in combinations(tensors, 2)]
    return float(np.mean(values))


def evaluate(
    reference_dir: str | Path,
    token_dir: str | Path,
    class_dir: str | Path | None,
    output_path: str | Path,
    device_name: str | None = None,
) -> dict:
    device = get_device(device_name)
    reference_dir = Path(reference_dir)
    token_dir = Path(token_dir)
    result = {
        "token_images": {
            "clip_score": clip_score(token_dir, device),
            "face_similarity": face_similarity(reference_dir, token_dir, device),
            "lpips_diversity": lpips_diversity(token_dir, device),
        }
    }
    if class_dir:
        class_dir = Path(class_dir)
        result["class_images"] = {
            "clip_score": clip_score(class_dir, device),
            "lpips_diversity": lpips_diversity(class_dir, device),
        }
    save_json(result, output_path)
    return result
