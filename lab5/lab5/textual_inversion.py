from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

from .utils import list_images, open_rgb, save_json, seed_everything


@dataclass
class TrainConfig:
    model_id: str
    train_dir: Path
    output_dir: Path
    token: str
    initializer_token: str = "person"
    resolution: int = 512
    max_steps: int = 1200
    batch_size: int = 1
    learning_rate: float = 5e-4
    gradient_accumulation_steps: int = 4
    seed: int = 42
    device: str | None = None


class SelfPortraitDataset(Dataset):
    def __init__(self, image_dir: Path, token: str, resolution: int) -> None:
        self.paths = list_images(image_dir)
        if len(self.paths) < 5:
            raise ValueError(
                f"Expected at least 5 portrait images in {image_dir}, found {len(self.paths)}."
            )
        self.caption = f"a high quality realistic portrait photo of {token}"
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        image = open_rgb(self.paths[index])
        return {"pixel_values": self.transform(image), "caption": self.caption}


def _add_placeholder_token(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    token: str,
    initializer_token: str,
) -> int:
    if tokenizer.add_tokens(token) != 1:
        raise ValueError(f"Token {token!r} already exists in tokenizer. Choose a unique token.")

    token_id = tokenizer.convert_tokens_to_ids(token)
    initializer_id = tokenizer.encode(initializer_token, add_special_tokens=False)
    if len(initializer_id) != 1:
        raise ValueError(f"Initializer token {initializer_token!r} must map to one token.")

    text_encoder.resize_token_embeddings(len(tokenizer))
    embeddings = text_encoder.get_input_embeddings().weight.data
    embeddings[token_id] = embeddings[initializer_id[0]].clone()
    return token_id


def train_textual_inversion(config: TrainConfig) -> dict:
    seed_everything(config.seed)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = CLIPTokenizer.from_pretrained(config.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.model_id, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(config.model_id, subfolder="vae").to(device, dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(config.model_id, subfolder="unet").to(device, dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_id, subfolder="scheduler")

    placeholder_id = _add_placeholder_token(
        tokenizer, text_encoder, config.token, config.initializer_token
    )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    text_encoder.train()

    dataset = SelfPortraitDataset(config.train_dir, config.token, config.resolution)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(), lr=config.learning_rate
    )

    original_embeddings = text_encoder.get_input_embeddings().weight.detach().clone()
    global_step = 0
    losses: list[float] = []

    while global_step < config.max_steps:
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            input_ids = tokenizer(
                batch["caption"],
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(input_ids)[0].to(dtype=dtype)
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float())
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    embeddings = text_encoder.get_input_embeddings().weight
                    mask = torch.ones(len(tokenizer), dtype=torch.bool, device=device)
                    mask[placeholder_id] = False
                    embeddings[mask] = original_embeddings.to(device)[mask]

            losses.append(float(loss.detach().cpu()) * config.gradient_accumulation_steps)
            global_step += 1
            if global_step >= config.max_steps:
                break

    config.output_dir.mkdir(parents=True, exist_ok=True)
    learned_embedding = text_encoder.get_input_embeddings().weight[placeholder_id].detach().cpu()
    torch.save({config.token: learned_embedding}, config.output_dir / "learned_embeds.bin")
    tokenizer.save_pretrained(config.output_dir / "tokenizer")

    summary = {
        "model_id": config.model_id,
        "token": config.token,
        "initializer_token": config.initializer_token,
        "train_images": len(dataset),
        "resolution": config.resolution,
        "max_steps": config.max_steps,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "final_loss": losses[-1],
        "mean_last_50_loss": sum(losses[-50:]) / min(50, len(losses)),
    }
    save_json(summary, config.output_dir / "training_summary.json")
    return summary
