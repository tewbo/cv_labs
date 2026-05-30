import argparse
from pathlib import Path

from .utils import list_images, load_json


DEFAULT_PROMPTS = Path("prompts.json")
DEFAULT_SELF_DIR = Path("data/self")
DEFAULT_RUN_DIR = Path("runs/textual_inversion")
DEFAULT_TOKEN_DIR = Path("outputs/token")
DEFAULT_CLASS_DIR = Path("outputs/class")
DEFAULT_METRICS = Path("outputs/metrics.json")


def _load_prompt_config(path: Path) -> dict:
    config = load_json(path)
    required = {"token", "class_word", "seed", "base_model", "token_prompts", "class_prompts"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing keys in {path}: {', '.join(missing)}")
    return config


def _learned_embeds_path(run_dir: Path) -> Path:
    return run_dir / "learned_embeds.bin"


def _check_training_images(train_dir: Path) -> None:
    images = list_images(train_dir)
    if len(images) < 5:
        raise ValueError(
            f"Put at least 5 portrait photos into {train_dir} before training. "
            f"Found {len(images)} image(s)."
        )


def command_train(args: argparse.Namespace) -> None:
    prompt_config = _load_prompt_config(args.prompts)
    _check_training_images(args.train_dir)
    from .textual_inversion import TrainConfig, train_textual_inversion

    summary = train_textual_inversion(
        TrainConfig(
            model_id=args.model or prompt_config["base_model"],
            train_dir=args.train_dir,
            output_dir=args.output_dir,
            token=prompt_config["token"],
            initializer_token=args.initializer_token or prompt_config["class_word"],
            resolution=args.resolution,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            seed=prompt_config["seed"],
            device=args.device,
        )
    )
    print(f"Saved learned embedding to {_learned_embeds_path(args.output_dir)}")
    print(f"Final loss: {summary['final_loss']:.6f}")


def command_generate_token(args: argparse.Namespace) -> None:
    prompt_config = _load_prompt_config(args.prompts)
    learned_embeds = args.learned_embeds or _learned_embeds_path(args.run_dir)
    if not learned_embeds.exists():
        raise FileNotFoundError(f"Learned embedding not found: {learned_embeds}")
    from .generation import generate_images

    paths = generate_images(
        model_id=args.model or prompt_config["base_model"],
        prompts=prompt_config["token_prompts"],
        output_dir=args.output_dir,
        learned_embeds=learned_embeds,
        device=args.device,
        seed=prompt_config["seed"],
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        images_per_prompt=args.images_per_prompt,
    )
    print(f"Generated {len(paths)} personalized image(s) in {args.output_dir}")


def command_generate_class(args: argparse.Namespace) -> None:
    from .generation import generate_images

    prompt_config = _load_prompt_config(args.prompts)
    paths = generate_images(
        model_id=args.model or prompt_config["base_model"],
        prompts=prompt_config["class_prompts"],
        output_dir=args.output_dir,
        learned_embeds=None,
        device=args.device,
        seed=prompt_config["seed"],
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        images_per_prompt=args.images_per_prompt,
    )
    print(f"Generated {len(paths)} class baseline image(s) in {args.output_dir}")


def command_evaluate(args: argparse.Namespace) -> None:
    from .metrics import evaluate

    result = evaluate(
        reference_dir=args.reference_dir,
        token_dir=args.token_dir,
        class_dir=args.class_dir,
        output_path=args.output,
        device_name=args.device,
    )
    print(f"Saved metrics to {args.output}")
    print(result)


def command_run_all(args: argparse.Namespace) -> None:
    train_args = argparse.Namespace(
        prompts=args.prompts,
        train_dir=args.train_dir,
        output_dir=args.run_dir,
        model=args.model,
        device=args.device,
        initializer_token=args.initializer_token,
        resolution=args.resolution,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    token_args = argparse.Namespace(
        prompts=args.prompts,
        model=args.model,
        device=args.device,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        images_per_prompt=args.images_per_prompt,
        run_dir=args.run_dir,
        learned_embeds=args.learned_embeds,
        output_dir=args.token_dir,
    )
    class_args = argparse.Namespace(
        prompts=args.prompts,
        model=args.model,
        device=args.device,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        images_per_prompt=args.images_per_prompt,
        output_dir=args.class_dir,
    )
    evaluate_args = argparse.Namespace(
        reference_dir=args.train_dir,
        token_dir=args.token_dir,
        class_dir=args.class_dir,
        output=args.metrics,
        device=args.device,
    )
    command_train(train_args)
    command_generate_token(token_args)
    command_generate_class(class_args)
    command_evaluate(evaluate_args)


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--images-per-prompt", type=int, default=1)


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_SELF_DIR)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--initializer-token", default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab 5 text-to-image personalization pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train textual inversion token")
    _add_train_args(train)
    train.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR)
    train.set_defaults(func=command_train)

    token = subparsers.add_parser("generate-token", help="generate images with learned token")
    _add_generation_args(token)
    token.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    token.add_argument("--learned-embeds", type=Path, default=None)
    token.add_argument("--output-dir", type=Path, default=DEFAULT_TOKEN_DIR)
    token.set_defaults(func=command_generate_token)

    klass = subparsers.add_parser("generate-class", help="generate class baseline images")
    _add_generation_args(klass)
    klass.add_argument("--output-dir", type=Path, default=DEFAULT_CLASS_DIR)
    klass.set_defaults(func=command_generate_class)

    metrics = subparsers.add_parser("evaluate", help="compute CLIP, portrait similarity and LPIPS metrics")
    metrics.add_argument("--reference-dir", type=Path, default=DEFAULT_SELF_DIR)
    metrics.add_argument("--token-dir", type=Path, default=DEFAULT_TOKEN_DIR)
    metrics.add_argument("--class-dir", type=Path, default=DEFAULT_CLASS_DIR)
    metrics.add_argument("--output", type=Path, default=DEFAULT_METRICS)
    metrics.add_argument("--device", default=None)
    metrics.set_defaults(func=command_evaluate)

    run_all = subparsers.add_parser("run-all", help="train, generate and evaluate")
    _add_train_args(run_all)
    run_all.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    run_all.add_argument("--learned-embeds", type=Path, default=None)
    run_all.add_argument("--token-dir", type=Path, default=DEFAULT_TOKEN_DIR)
    run_all.add_argument("--class-dir", type=Path, default=DEFAULT_CLASS_DIR)
    run_all.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    run_all.add_argument("--steps", type=int, default=40)
    run_all.add_argument("--guidance-scale", type=float, default=7.5)
    run_all.add_argument("--width", type=int, default=512)
    run_all.add_argument("--height", type=int, default=512)
    run_all.add_argument("--images-per-prompt", type=int, default=1)
    run_all.set_defaults(func=command_run_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(status=1, message=f"error: {error}\n")
