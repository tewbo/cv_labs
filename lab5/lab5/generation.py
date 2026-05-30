from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

from .utils import dtype_for_device, get_device, save_json, seed_everything


def load_pipeline(model_id: str, learned_embeds: str | Path | None, device_name: str | None):
    device = get_device(device_name)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype_for_device(device),
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    if learned_embeds:
        pipe.load_textual_inversion(str(learned_embeds))
    if device.type == "cuda":
        pipe.enable_attention_slicing()
    return pipe, device


def generate_images(
    model_id: str,
    prompts: list[str],
    output_dir: str | Path,
    learned_embeds: str | Path | None = None,
    device: str | None = None,
    seed: int = 42,
    steps: int = 40,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    images_per_prompt: int = 1,
) -> list[Path]:
    seed_everything(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe, torch_device = load_pipeline(model_id, learned_embeds, device)
    generator = torch.Generator(device=torch_device).manual_seed(seed)

    saved: list[Path] = []
    metadata: list[dict] = []
    for prompt_index, prompt in enumerate(prompts, start=1):
        result = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_images_per_prompt=images_per_prompt,
            generator=generator,
        )
        for image_index, image in enumerate(result.images, start=1):
            name = f"{prompt_index:02d}_{image_index:02d}.png"
            path = output_dir / name
            image.save(path)
            saved.append(path)
            metadata.append({"file": name, "prompt": prompt})

    save_json({"images": metadata}, output_dir / "metadata.json")
    return saved
