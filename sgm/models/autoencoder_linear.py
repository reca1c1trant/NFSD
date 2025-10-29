"""
Simple Linear Autoencoder for MNIST latent space compression
"""

import torch
import torch.nn as nn


class AutoencoderKLLinear(nn.Module):
    """
    Simple linear autoencoder for compressing MNIST (784) to latent space
    Used to enable latent space training for Flow-based diffusion
    """

    def __init__(
        self,
        embed_dim: int = 128,
        in_channels: int = 1,
        z_channels: int = 128,
        resolution: int = 28,
    ):
        """
        Args:
            embed_dim: Latent dimension
            in_channels: Input channels (1 for MNIST)
            z_channels: Same as embed_dim for compatibility
            resolution: Input resolution (28 for MNIST)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.in_dim = in_channels * resolution * resolution  # 1 * 28 * 28 = 784
        self.resolution = resolution
        self.in_channels = in_channels

        # Encoder: 784 -> 512 -> 256 -> 128
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

        # Decoder: 128 -> 256 -> 512 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.in_dim),
            nn.Tanh(),  # Output in [-1, 1] to match input
        )

    def encode(self, x):
        """
        Encode image to latent space

        Args:
            x: Input images [B, C, H, W] or [B, D]

        Returns:
            z: Latent codes [B, embed_dim]
        """
        batch_size = x.shape[0]

        # Flatten if needed
        if x.dim() > 2:
            x = x.reshape(batch_size, -1)

        z = self.encoder(x)
        return z

    def decode(self, z):
        """
        Decode latent codes to images

        Args:
            z: Latent codes [B, embed_dim]

        Returns:
            x: Reconstructed images [B, C, H, W]
        """
        batch_size = z.shape[0]

        # Decode
        x_flat = self.decoder(z)

        # Reshape to image
        x = x_flat.reshape(batch_size, self.in_channels, self.resolution, self.resolution)

        return x

    def forward(self, x):
        """
        Full forward pass: encode then decode

        Args:
            x: Input images [B, C, H, W]

        Returns:
            recon: Reconstructed images [B, C, H, W]
            z: Latent codes [B, embed_dim]
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class AutoencoderKLLinearPreTrained(AutoencoderKLLinear):
    """
    Pre-trained version - just use the architecture but train it
    For now, this is the same as AutoencoderKLLinear
    You can train it separately and load weights if needed
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Will be loaded from checkpoint if provided
