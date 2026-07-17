import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .layer import (
    PatchEmbedding,
    PatchReconstruction,
    SpatioTemporalProcessor,
    SpatioTemporalReconstructor
)


class Encoder(nn.Module):
    """
    基于时空patch的编码器：
    1. PatchEmbedding: (B,1,2235,168) -> (B,num_spatial_patches,7,16,24)
    2. SpatioTemporalProcessor: patch内部attention + patch间attention -> tokens
    """
    def __init__(self,
                 num_regions: int = 2235,
                 time_steps: int = 168,
                 patch_spatial: int = 16,
                 patch_temporal: int = 24,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 tokens_per_patch: int = 7,
                 token_dim: int = 64):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            num_regions=num_regions,
            time_steps=time_steps,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            tokens_per_patch=tokens_per_patch,
        )
        self.spatiotemporal_processor = SpatioTemporalProcessor(
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            tokens_per_patch=tokens_per_patch,
            token_dim=token_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        tokens = self.spatiotemporal_processor(x)
        return tokens


class Decoder(nn.Module):
    """
    基于时空patch的解码器：
    1. SpatioTemporalReconstructor: tokens -> (B,num_spatial_patches,7,16,24)
    2. PatchReconstruction: (B,num_spatial_patches,7,16,24) -> (B,1,2235,168)
    """
    def __init__(self,
                 num_regions: int = 2235,
                 time_steps: int = 168,
                 patch_spatial: int = 16,
                 patch_temporal: int = 24,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 tokens_per_patch: int = 7,
                 token_dim: int = 64):
        super().__init__()
        self.spatiotemporal_reconstructor = SpatioTemporalReconstructor(
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            tokens_per_patch=tokens_per_patch,
            token_dim=token_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.patch_reconstruction = PatchReconstruction(
            num_regions=num_regions,
            time_steps=time_steps,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            tokens_per_patch=tokens_per_patch,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.spatiotemporal_reconstructor(tokens)
        x_reconstructed = self.patch_reconstruction(x)
        return x_reconstructed


class VAE(nn.Module):
    """
    基于patch的VAE模型
    """
    def __init__(self,
                 num_regions: int = 2235,
                 time_steps: int = 168,
                 patch_spatial: int = 16,
                 patch_temporal: int = 24,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 tokens_per_patch: int = 7,
                 token_dim: int = 64):
        super().__init__()
        self.tokens_per_patch = tokens_per_patch
        self.token_dim = token_dim

        self.encoder = Encoder(
            num_regions=num_regions,
            time_steps=time_steps,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            tokens_per_patch=tokens_per_patch,
            token_dim=token_dim
        )

        self.to_mu = nn.Linear(token_dim, token_dim)
        self.to_logvar = nn.Linear(token_dim, token_dim)

        self.decoder = Decoder(
            num_regions=num_regions,
            time_steps=time_steps,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            tokens_per_patch=tokens_per_patch,
            token_dim=token_dim
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encoder(x)
        mu = self.to_mu(tokens)
        logvar = self.to_logvar(tokens)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl.mean()
        return x_hat, kl_loss


