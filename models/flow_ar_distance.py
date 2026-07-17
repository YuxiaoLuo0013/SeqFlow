import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinCosPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(base) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        seq_len = x.shape[1]
        end = position_offset + seq_len
        if end <= self.max_seq_len:
            return self.pe[:, position_offset:end, :]
        device = x.device
        pe = torch.zeros(seq_len, self.dim, device=device)
        position = torch.arange(position_offset, end, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2).float().to(device) * (-math.log(self.base) / self.dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


class FlowARDistance(nn.Module):
    def __init__(
        self,
        token_dim: int,
        model_dim: int = 768,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_tokens_per_patch: int = 7,
        patch_spatial: int = 16,
        cond_channels: int = 266,
        spatial_channels: int = 5,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.model_dim = model_dim
        self.patch_spatial = patch_spatial
        self.num_tokens_per_patch = num_tokens_per_patch
        self.cond_channels = cond_channels
        self.spatial_channels = spatial_channels
        self.non_spatial_channels = self.cond_channels - self.spatial_channels
        if self.non_spatial_channels <= 0:
            raise ValueError("cond_channels must be larger than spatial_channels")

        self.z_proj = nn.Linear(token_dim, model_dim)
        self.pos_embed = SinCosPositionalEmbedding(model_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=int(model_dim * mlp_ratio),
                dropout=proj_drop,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = nn.LayerNorm(model_dim)
        self.token_head = nn.Linear(model_dim, token_dim)

        self.cond_proj = nn.Sequential(
            nn.Linear(patch_spatial * self.non_spatial_channels, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.spatial_proj = nn.Linear(patch_spatial * self.spatial_channels, model_dim)
        self.start_token = nn.Sequential(
            nn.Linear(self.non_spatial_channels, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )

    def pad_to_patch_size(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        current_size = tensor.shape[dim]
        padded_size = ((current_size + self.patch_spatial - 1) // self.patch_spatial) * self.patch_spatial
        padding_size = padded_size - current_size
        if padding_size <= 0:
            return tensor
        padding_shape = list(tensor.shape)
        padding_shape[dim] = padding_size
        padding = torch.zeros(padding_shape, device=tensor.device, dtype=tensor.dtype)
        return torch.cat([tensor, padding], dim=dim)

    def _prepare_condition(self, pop_condition: torch.Tensor, groups: List[int]):
        B = pop_condition.shape[0]
        total_L = sum(groups)
        origin_cond = pop_condition[:, :self.non_spatial_channels, 0, 0]
        buffer_tokens = self.start_token(origin_cond).unsqueeze(1)
        buffer_tokens = buffer_tokens.expand(B, self.num_tokens_per_patch, self.token_dim)

        dest_condition = pop_condition[:, :, 1:, 0]
        dest_condition = self.pad_to_patch_size(dest_condition, dim=2)
        L_patches = dest_condition.shape[2] // self.patch_spatial
        if L_patches != len(groups):
            raise ValueError(f"L_patches={L_patches}, groups={len(groups)}")

        dest_condition = dest_condition.transpose(1, 2)
        dest_condition = dest_condition.reshape(B, L_patches, -1, self.cond_channels)
        non_spatial = dest_condition[:, :, :, :self.non_spatial_channels].reshape(B, L_patches, -1)
        spatial = dest_condition[:, :, :, self.non_spatial_channels:].reshape(B, L_patches, -1)
        cond = self.cond_proj(non_spatial)
        spatial = self.spatial_proj(spatial)
        cond = cond.unsqueeze(2).expand(B, L_patches, self.num_tokens_per_patch, self.model_dim)
        spatial = spatial.unsqueeze(2).expand(B, L_patches, self.num_tokens_per_patch, self.model_dim)
        cond = cond.reshape(B, total_L, self.model_dim)
        spatial = spatial.reshape(B, total_L, self.model_dim)
        return buffer_tokens, cond + spatial

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((length, length), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def _encode(self, input_tokens: torch.Tensor, condition_tokens: torch.Tensor) -> torch.Tensor:
        length = input_tokens.shape[1]
        x = self.z_proj(input_tokens)
        x = x + self.pos_embed(x) + condition_tokens[:, :length]
        x = self.attn_drop(x)
        mask = self._causal_mask(length, x.device)
        for block in self.blocks:
            x = block(x, src_mask=mask)
        return self.token_head(self.norm(x))

    def forward(
        self,
        gt_tokens: torch.Tensor,
        groups: List[int],
        pop_condition: torch.Tensor,
        buffer_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, _, _, Dz = gt_tokens.shape
        if Dz != self.token_dim:
            raise ValueError(f"token_dim mismatch: got {Dz}, expected {self.token_dim}")
        total_L = sum(groups)
        gt_tokens = gt_tokens.reshape(B, total_L, Dz)
        default_buffer, condition_tokens = self._prepare_condition(pop_condition, groups)
        if buffer_tokens is None:
            buffer_tokens = default_buffer
        buffer_len = buffer_tokens.shape[1]
        inputs = torch.cat([buffer_tokens, gt_tokens[:, :-buffer_len]], dim=1)
        pred_tokens = self._encode(inputs, condition_tokens)
        loss = F.huber_loss(pred_tokens, gt_tokens)
        return loss, pred_tokens

    @torch.no_grad()
    def sample(
        self,
        pop_condition: torch.Tensor,
        groups: List[int],
        buffer_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        default_buffer, condition_tokens = self._prepare_condition(pop_condition, groups)
        if buffer_tokens is None:
            buffer_tokens = default_buffer
        B = pop_condition.shape[0]
        total_L = sum(groups)
        tokens = torch.zeros(B, total_L, self.token_dim, device=pop_condition.device, dtype=buffer_tokens.dtype)
        start = 0
        for k in groups:
            end = start + k
            inputs = torch.cat([buffer_tokens, tokens[:, :start]], dim=1)
            pred = self._encode(inputs, condition_tokens)
            tokens[:, start:end] = pred[:, -k:]
            start = end
        return tokens
