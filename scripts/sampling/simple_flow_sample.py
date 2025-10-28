"""
Simple sampling script for Flow-based Diffusion Models
Generate samples from a trained Flow-based diffusion model
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sgm.util import instantiate_from_config


def get_parser():
    parser = argparse.ArgumentParser(description="Sample from Flow-based Diffusion Model")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.002,
        help="Minimum noise level",
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=80.0,
        help="Maximum noise level",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/flow_samples",
        help="Output directory for samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=None,
        help="Class label for conditional generation (optional)",
    )

    return parser


def save_images(samples, output_dir, prefix="sample"):
    """Save samples as images"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        # Convert to numpy and rescale to [0, 255]
        sample_np = sample.cpu().numpy()

        if sample_np.ndim == 3:
            # Image format: CHW -> HWC
            sample_np = sample_np.transpose(1, 2, 0)

        # Normalize to [0, 1]
        sample_np = (sample_np - sample_np.min()) / (sample_np.max() - sample_np.min() + 1e-8)
        sample_np = (sample_np * 255).astype(np.uint8)

        # Save
        if sample_np.shape[-1] == 1:
            sample_np = sample_np.squeeze(-1)
            img = Image.fromarray(sample_np, mode='L')
        else:
            img = Image.fromarray(sample_np, mode='RGB')

        img.save(output_dir / f"{prefix}_{i:04d}.png")

    print(f"Saved {len(samples)} samples to {output_dir}")


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # Load config
    print(f"Loading config from {args.config}")
    config = OmegaConf.load(args.config)

    # Instantiate model
    print("Instantiating model...")
    model_config = config.model
    model = instantiate_from_config(model_config)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")

    # Prepare conditioning if needed
    cond = None
    if args.class_label is not None:
        print(f"Generating class-conditional samples for class {args.class_label}")
        cond = {
            "cls": torch.tensor([args.class_label] * args.num_samples, device=device)
        }

    # Generate samples
    print(f"Generating {args.num_samples} samples with {args.num_steps} steps...")
    with torch.no_grad():
        samples = model.sample(
            batch_size=args.num_samples,
            num_steps=args.num_steps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            cond=cond,
        )

    # Decode if using VAE
    if hasattr(model, 'decode_first_stage'):
        print("Decoding samples from latent space...")
        samples = model.decode_first_stage(samples)

    # Save samples
    save_images(samples, args.output_dir)

    # Create grid
    print("Creating sample grid...")
    try:
        from torchvision.utils import make_grid
        grid = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
        save_images([grid], args.output_dir, prefix="grid")
    except Exception as e:
        print(f"Could not create grid: {e}")

    print("Sampling completed!")


if __name__ == "__main__":
    main()
