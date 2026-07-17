import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, weight: bool = False):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x.to(orig_dtype)
        if self.weight is None:
            return x
        return x * self.weight


class SinCosPositionalEmbedding(nn.Module):
    """
    Standard sin/cos positional embedding from "Attention is All You Need".
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
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


        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """
        Apply positional embedding to input tensor.
        Args:
            x: [B, N, D] input tensor
            position_offset: 在预计算表中从该下标开始取 PE（用于 KV-cache 增量解码）
        Returns:
            pos_emb: [1, N, D] positional embedding (broadcastable to x)
        """
        seq_len = x.shape[1]
        offset = position_offset
        max_use = offset + seq_len
        if max_use <= self.max_seq_len:
            return self.pe[:, offset:max_use, :]
        else:

            device = x.device
            pe = torch.zeros(seq_len, self.dim, device=device)
            position = torch.arange(offset, offset + seq_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.dim, 2).float().to(device) * (-math.log(self.base) / self.dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe.unsqueeze(0)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        is_causal: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv.unbind(0)
        past_len = 0
        if past_key_value is not None:
            pk, pv = past_key_value
            past_len = pk.shape[2]
            k = torch.cat([pk, k_new], dim=2)
            v = torch.cat([pv, v_new], dim=2)
        else:
            k, v = k_new, v_new
        attn = (q @ k.transpose(-2, -1)) * self.scale

        total_k = past_len + N
        if attn_mask is not None:
            attn = attn + attn_mask
        elif is_causal:
            qi = torch.arange(past_len, past_len + N, device=q.device, dtype=torch.long).view(N, 1)
            kj = torch.arange(total_k, device=q.device, dtype=torch.long).view(1, total_k)
            causal = torch.where(kj <= qi, 0.0, torch.finfo(attn.dtype).min)
            attn = attn + causal.view(1, 1, N, total_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        if use_cache:
            return out, (k, v)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, attn_drop: float = 0.0, proj_drop: float = 0.0, use_ada_ln: bool = True):
        super().__init__()
        self.use_ada_ln = use_ada_ln
        self.norm1 = RMSNorm(dim, weight=True)
        self.attn = SelfAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(dim, weight=True)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=proj_drop)

        if use_ada_ln:


            self.ada_ln = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 6 * dim, bias=True)
            )
            self.dim = dim

            nn.init.constant_(self.ada_ln[1].weight, 0)
            nn.init.constant_(self.ada_ln[1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        condition: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        is_causal: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        if self.use_ada_ln and condition is not None:


            B, N, _ = x.shape



            T=int(N/7)

            ada_params = self.ada_ln(condition[:, :T])
            ada_params = ada_params.view(B, T, 6, self.dim)
            attn_gamma, mlp_gamma, attn_scale, mlp_scale, attn_shift, mlp_shift = ada_params.unbind(2)


            x_norm1 = self.norm1(x)
            x_norm1 = x_norm1.reshape(B, T,-1, self.dim)
            x_norm1 = x_norm1 * (attn_scale.unsqueeze(2)) + attn_shift.unsqueeze(2)
            x_norm1 = x_norm1.reshape(B, N, -1)
            if use_cache:
                attn_out, layer_present = self.attn(
                    x_norm1, attn_mask,
                    past_key_value=layer_past, use_cache=True, is_causal=is_causal,
                )
            else:
                attn_out = self.attn(
                    x_norm1, attn_mask,
                    past_key_value=layer_past, use_cache=False, is_causal=is_causal,
                )
                layer_present = None
            attn_out = attn_out.reshape(B, T,-1, self.dim)* (attn_gamma.unsqueeze(2))
            x = x + attn_out.reshape(B, N, -1)


            x_norm2 = self.norm2(x)
            x_norm2 = x_norm2.reshape(B, T,-1, self.dim)
            x_norm2 = x_norm2 * (mlp_scale.unsqueeze(2)) + mlp_shift.unsqueeze(2)
            x_norm2 = x_norm2.reshape(B, N, -1)
            mlp_out = self.mlp(x_norm2)
            mlp_out = mlp_out.reshape(B, T,-1, self.dim)* (mlp_gamma.unsqueeze(2))
            mlp_out = mlp_out.reshape(B, N, -1)
            x = x + mlp_out
            if use_cache:
                return x, layer_present
            return x
        else:

            h = self.norm1(x)
            if use_cache:
                attn_out, layer_present = self.attn(
                    h, attn_mask,
                    past_key_value=layer_past, use_cache=True, is_causal=is_causal,
                )
            else:
                attn_out = self.attn(
                    h, attn_mask,
                    past_key_value=layer_past, use_cache=False, is_causal=is_causal,
                )
                layer_present = None
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            if use_cache:
                return x, layer_present
            return x



class TimestepEmbedder(nn.Module):
    """时间步嵌入器"""
    def __init__(self, model_channels):
        super().__init__()
        self.model_channels = model_channels
        self.mlp = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels),
        )

    def forward(self, t):

        t = t.float()
        half_dim = self.model_channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Block_v1(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(
            dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=proj_drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ada_lin = nn.Linear(dim, 6*dim)
        self.dim = dim

    def forward(self, x: torch.Tensor, pop_condition) -> torch.Tensor:
        B, N, C = pop_condition.shape
        gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(nn.SiLU()(pop_condition)).view(B, N, 6, self.dim).unbind(2)
        x = x + self.drop_path1(self.attn(self.norm1(x).mul(scale1.add(1)).add_(shift1)).mul_(gamma1))
        x = x + self.drop_path2(self.mlp(self.norm2(x).mul(scale2.add(1)).add_(shift2)).mul_(gamma2))
        return x



class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(model_channels)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )


    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift


class SimpleTransformerAdaLN(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        cross=False,
        num_tokens_per_patch=7,
        patch_spatial=16,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.pop_proj = nn.Linear(128, model_channels, bias=True)
        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(Block_v1(
                model_channels, model_channels//64
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()
        self.num_tokens_per_patch = num_tokens_per_patch
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)




        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x,pop_condition):

        B, N, C = x.shape
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C x ...] Tensor of outputs.
        """
        x = self.input_proj(x)




        x=x.reshape(-1, self.num_tokens_per_patch, x.shape[-1])


        pop_condition=pop_condition.reshape(-1, 1, pop_condition.shape[-1])
        for block in self.res_blocks:
            x = block(x,pop_condition)

        return self.final_layer(x,pop_condition)
