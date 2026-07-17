import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple





class SpatioTemporalProcessor(nn.Module):
    """
    时空patch处理器：
    1. 对每个时空patch (16×24) 在16个区域维度上做attention和FFN (patch内部)
    2. 将时空patch展平为384维向量
    3. 在7个时间patch之间做attention和FFN (patch间)
    4. 还原为16×24维度

    输入: (B, num_spatial_patches, tokens_per_patch, patch_spatial, patch_temporal)
    输出: (B, num_spatial_patches, tokens_per_patch, token_dim)
    """
    def __init__(self,
                 patch_spatial: int = 16,
                 patch_temporal: int = 24,
                 tokens_per_patch: int = 7,
                 token_dim: int = 64,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.0,
                 num_layers: int = 2):
        super().__init__()
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.tokens_per_patch = tokens_per_patch
        self.token_dim = token_dim
        self.st_patch_dim = patch_spatial * patch_temporal
        self.num_layers = num_layers


        self.spatial_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=tokens_per_patch * patch_temporal,
                nhead=num_heads,
                dropout=dropout,
                dim_feedforward=tokens_per_patch * patch_temporal * mlp_ratio,
                batch_first=True,
                activation='gelu',
                norm_first=True
            )
            for _ in range(num_layers)
        ])


        self.temporal_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.st_patch_dim,
                nhead=num_heads,
                dropout=dropout,
                dim_feedforward=self.st_patch_dim * mlp_ratio,
                batch_first=True,
                activation='gelu',
                norm_first=True
            )
            for _ in range(num_layers)
        ])


        self.to_token_dim = nn.Linear(self.st_patch_dim, token_dim)

        print(f"SpatioTemporalProcessor:")
        print(f"  时空patch: {patch_spatial}×{patch_temporal} -> {token_dim}维token")
        print(f"  每个空间patch有 {tokens_per_patch} 个时间patch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        期望输入:
            x: (B, num_spatial_patches, patch_spatial, tokens_per_patch * patch_temporal)
        输出:
            (B, num_spatial_patches, tokens_per_patch, token_dim)
        """
        B, num_spatial_patches, patch_spatial, total_temporal = x.shape
        assert patch_spatial == self.patch_spatial
        assert total_temporal == self.tokens_per_patch * self.patch_temporal, \
            f"最后一维应为 tokens_per_patch * patch_temporal = {self.tokens_per_patch * self.patch_temporal}, " \
            f"但得到 {total_temporal}"


        x = x.reshape(B * num_spatial_patches, patch_spatial, total_temporal)

        for layer_idx in range(self.num_layers):

            x = self.spatial_blocks[layer_idx](x)


            x = x.reshape(
                B * num_spatial_patches,
                self.patch_spatial,
                self.tokens_per_patch,
                self.patch_temporal,
            )


            x = x.permute(0, 2, 1, 3)


            x = x.reshape(
                B * num_spatial_patches,
                self.tokens_per_patch,
                self.st_patch_dim,
            )


            x = self.temporal_blocks[layer_idx](x)


            if layer_idx < self.num_layers - 1:

                x = x.reshape(
                    B * num_spatial_patches,
                    self.tokens_per_patch,
                    self.patch_spatial,
                    self.patch_temporal,
                )

                x = x.permute(0, 2, 1, 3)

                x = x.reshape(
                    B * num_spatial_patches,
                    self.patch_spatial,
                    self.tokens_per_patch * self.patch_temporal,
                )




        x = self.to_token_dim(x)


        x = x.reshape(B, num_spatial_patches, self.tokens_per_patch, self.token_dim)

        return x


class SpatioTemporalReconstructor(nn.Module):
    """
    时空patch重构器：
    1. 在7个时间patch之间做attention和FFN (patch间)
    2. 从token维度重构为384维向量
    3. 还原为16×24时空patch
    4. 在16个区域维度上做attention和FFN (patch内部)

    输入: (B, num_spatial_patches, tokens_per_patch, token_dim)
    输出: (B, num_spatial_patches, tokens_per_patch, patch_spatial, patch_temporal)
    """
    def __init__(self,
                 patch_spatial: int = 16,
                 patch_temporal: int = 24,
                 tokens_per_patch: int = 7,
                 token_dim: int = 64,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.0,
                 num_layers: int = 2):
        """
        这是与 SpatioTemporalProcessor 对称的重构器：
        - Processor:  时空 patch -> 时序 token
        - Reconstructor: 时序 token -> 时空 patch
        """
        super().__init__()
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.tokens_per_patch = tokens_per_patch
        self.token_dim = token_dim
        self.st_patch_dim = patch_spatial * patch_temporal
        self.num_layers = num_layers


        self.from_token_dim = nn.Linear(token_dim, self.st_patch_dim)


        self.temporal_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.st_patch_dim,
                nhead=num_heads,
                dropout=dropout,
                dim_feedforward=self.st_patch_dim * mlp_ratio,
                batch_first=True,
                activation='gelu',
                norm_first=True
            )
            for _ in range(num_layers)
        ])


        self.spatial_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=tokens_per_patch * patch_temporal,
                nhead=num_heads,
                dropout=dropout,
                dim_feedforward=tokens_per_patch * patch_temporal * mlp_ratio,
                batch_first=True,
                activation='gelu',
                norm_first=True
            )
            for _ in range(num_layers)
        ])
















        print("SpatioTemporalReconstructor:")
        print(f"  token_dim {token_dim} -> 时空patch {patch_spatial}×{patch_temporal}")
        print(f"  每个空间patch有 {tokens_per_patch} 个时间patch")
        print(f"  时间位置编码: 粗粒度({tokens_per_patch}) + 细粒度({patch_temporal})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: (B, num_spatial_patches, tokens_per_patch, token_dim)
        输出:
            (B, num_spatial_patches, tokens_per_patch, patch_spatial, patch_temporal)
        """
        B, num_spatial_patches, tokens_per_patch, token_dim = x.shape
        assert tokens_per_patch == self.tokens_per_patch
        assert token_dim == self.token_dim



        x = self.from_token_dim(x)


        x = x.reshape(B * num_spatial_patches, self.tokens_per_patch, self.st_patch_dim)




        for layer_idx in range(self.num_layers):


            x = self.temporal_blocks[layer_idx](x)


            x_sp = x.reshape(
                B * num_spatial_patches,
                self.tokens_per_patch,
                self.patch_spatial,
                self.patch_temporal,
            )





            x_sp = x_sp.permute(0, 2, 1, 3)


            x_sp = x_sp.reshape(
                B * num_spatial_patches,
                self.patch_spatial,
                self.tokens_per_patch * self.patch_temporal,
            )


            x_sp = self.spatial_blocks[layer_idx](x_sp)

            if layer_idx < self.num_layers - 1:

                x_sp = x_sp.reshape(
                    B * num_spatial_patches,
                    self.patch_spatial,
                    self.tokens_per_patch,
                    self.patch_temporal,
                )
                x_sp = x_sp.permute(0, 2, 1, 3)
                x = x_sp.reshape(
                    B * num_spatial_patches,
                    self.tokens_per_patch,
                    self.st_patch_dim,
                )
            else:

                x = x_sp



        x = x.reshape(
            B,
            num_spatial_patches,
            self.patch_spatial,
            self.tokens_per_patch * self.patch_temporal,
        )


        x = x.reshape(
            B,
            num_spatial_patches,
            self.patch_spatial,
            self.tokens_per_patch,
            self.patch_temporal,
        )
        x = x.permute(0, 1, 3, 2, 4).contiguous()

        return x




























































































class PatchEmbedding(nn.Module):
    """
    将输入数据分割为时空patches
    输入: (B, 1, N, T) where N=2235, T=168
    输出: (B, num_spatial_patches, tokens_per_patch, patch_spatial, patch_temporal)

    流程:
    1. 2235个区域 -> 分成16个区域的空间patch
    2. 168个时间步 -> 分成7个24时间步的时间patch
    3. 每个空间patch与每个时间patch组合 -> 时空patch (16×24)
    """
    def __init__(self,
                 num_regions: int = 2235,
                 time_steps: int = 168,
                 patch_spatial: int = 16,
                 patch_temporal: int = 24,
                 tokens_per_patch: int = 7):
        super().__init__()
        self.num_regions = num_regions
        self.time_steps = time_steps
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.tokens_per_patch = tokens_per_patch


        self.num_spatial_patches = (num_regions + patch_spatial - 1) // patch_spatial
        self.padded_regions = self.num_spatial_patches * patch_spatial


        assert time_steps % tokens_per_patch == 0, f"time_steps ({time_steps}) 必须能被 tokens_per_patch ({tokens_per_patch}) 整除"
        assert time_steps // tokens_per_patch == patch_temporal, f"每个时间patch应该是 {patch_temporal} 步，但计算得到 {time_steps // tokens_per_patch}"

        print(f"PatchEmbedding: {num_regions}x{time_steps}")
        print(f"  空间patch: {self.num_spatial_patches} 个，每个 {patch_spatial} 区域")
        print(f"  时间patch: {tokens_per_patch} 个，每个 {patch_temporal} 时间步")
        print(f"  总时空patch: {self.num_spatial_patches} × {tokens_per_patch} = {self.num_spatial_patches * tokens_per_patch}")
        print(f"  每个时空patch: {patch_spatial}×{patch_temporal}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, N, T = x.shape
        assert C == 1 and N == self.num_regions and T == self.time_steps


        x = x.squeeze(1)


        if N < self.padded_regions:
            x = F.pad(x, (0, 0, 0, self.padded_regions - N), mode='constant', value=0.0)


        x = x.reshape(B, self.num_spatial_patches, self.patch_spatial, T)







        return x


class PatchReconstruction(nn.Module):
    """
    从时空patches重构原始数据
    输入: (B, num_spatial_patches, tokens_per_patch, patch_spatial, patch_temporal)
    输出: (B, 1, N, T)
    """
    def __init__(self,
                 num_regions: int = 2235,
                 time_steps: int = 168,
                 patch_spatial: int = 16,
                 patch_temporal: int = 24,
                 tokens_per_patch: int = 7):
        super().__init__()
        self.num_regions = num_regions
        self.time_steps = time_steps
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.tokens_per_patch = tokens_per_patch


        self.num_spatial_patches = (num_regions + patch_spatial - 1) // patch_spatial
        self.padded_regions = self.num_spatial_patches * patch_spatial


        assert time_steps % tokens_per_patch == 0, f"time_steps ({time_steps}) 必须能被 tokens_per_patch ({tokens_per_patch}) 整除"
        assert time_steps // tokens_per_patch == patch_temporal, f"每个时间patch应该是 {patch_temporal} 步"

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, num_spatial_patches, tokens_per_patch, patch_spatial, patch_temporal = x.shape
        assert num_spatial_patches == self.num_spatial_patches
        assert tokens_per_patch == self.tokens_per_patch
        assert patch_spatial == self.patch_spatial
        assert patch_temporal == self.patch_temporal


        x = x.permute(0, 1, 3, 2, 4)


        x = x.reshape(B, num_spatial_patches, patch_spatial, self.time_steps)


        x = x.reshape(B, self.padded_regions, self.time_steps)


        x = x[:, :self.num_regions, :self.time_steps]


        x = x.unsqueeze(1)

        return x