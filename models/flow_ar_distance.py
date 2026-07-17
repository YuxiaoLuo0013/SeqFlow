import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


from .flowar_layer import *
from .flowloss import SILoss
from .sampler import euler_sampler


class FlowARDistance(nn.Module):
    """
    单尺度、基于距离由近及远的自回归Flow匹配模型框架。

    约定：
    - 输入tokens来自预训练VAE编码器的潜变量（例如`mu`），形状 [B, L, D_z]
    - 本模型对tokens做投影与Transformer编码，产生条件`z_cond`，
      使用SDE/Flow网络`SimpleTransformerAdaLN`进行匹配损失（SILoss）。
    - 自回归顺序通过外部传入的分组`groups`与其构造的mask体现。
    """

    def __init__(
        self,
        token_dim: int,
        model_dim: int = 768,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        flow_width: int = 1024,
        flow_depth: int = 3,
        cross: bool = False,
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


        self.z_proj = nn.Linear(token_dim, model_dim, bias=True)
        self.z_ln = RMSNorm(model_dim, weight=True)


        self.pos_embed = SinCosPositionalEmbedding(model_dim, max_seq_len=4096, base=10000)


        self.blocks = nn.ModuleList([
            TransformerBlock(model_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
            for _ in range(depth)
        ])
        self.norm = RMSNorm(model_dim, weight=True)


        self.flownet = SimpleTransformerAdaLN(
            in_channels=token_dim,
            model_channels=flow_width,
            out_channels=token_dim,
            z_channels=model_dim,
            num_res_blocks=flow_depth,
            cross=cross,
            num_tokens_per_patch=num_tokens_per_patch,
        )
        self.flow_loss = SILoss()

        self.cond_channels = cond_channels
        self.spatial_channels = spatial_channels
        self.non_spatial_channels = self.cond_channels - self.spatial_channels
        assert self.non_spatial_channels > 0, (
            f"cond_channels({self.cond_channels}) 必须大于 spatial_channels({self.spatial_channels})"
        )

        self.pop_to_start_token = nn.Sequential(
            nn.Linear(self.non_spatial_channels, token_dim, bias=True),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim, bias=True),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(patch_spatial * self.non_spatial_channels, model_dim, bias=True),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, bias=True),
        )
        self.spatial_embedding_linear = nn.Linear(
            patch_spatial * self.spatial_channels, token_dim, bias=True
        )
    @staticmethod
    def build_causal_mask_from_groups(groups: List[int], device: torch.device, buffer: int = 0) -> torch.Tensor:
        """
        根据分组（每一步解锁的token数量）构造分块下三角注意力mask。
        返回形状 [1, 1, L_all, L_all] 的mask，允许当前及以往组可见。
        buffer: 头部可选的额外tokens（如类别token等），默认0。
        """
        seq_parts = [buffer] + list(groups)
        cum = [0]
        for n in seq_parts:
            cum.append(cum[-1] + n)
        total = cum[-1]
        mask = torch.full((total, total), float('-inf'), device=device)

        for i in range(len(seq_parts)):
            start = cum[i]
            end = cum[i + 1]
            mask[start:end, : end] = 0.0
        return mask.unsqueeze(0).unsqueeze(0)
    def pad_to_patch_size(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        将tensor在指定维度填充到patch_size的倍数
        """
        patch_size = self.patch_spatial
        current_size = tensor.shape[dim]
        padded_size = ((current_size + patch_size - 1) // patch_size) * patch_size
        padding_size = padded_size - current_size

        if padding_size > 0:
            padding_shape = list(tensor.shape)
            padding_shape[dim] = padding_size
            padding = torch.zeros(padding_shape, device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=dim)

        return tensor

    def forward(
        self,
        gt_tokens: torch.Tensor,
        groups: List[int],
        pop_condition: torch.Tensor,
        buffer_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        训练前向：使用 Teacher Forcing，逻辑与 sample() 中的自回归保持一致。
        """
        B, L, S, Dz = gt_tokens.shape
        assert Dz == self.token_dim
        device = gt_tokens.device
        total_L = sum(groups)
        gt_tokens = gt_tokens.reshape(B, total_L, Dz)



        origin_cond = pop_condition[:, :self.non_spatial_channels, 0, 0]
        origin_cond = origin_cond.unsqueeze(1)


        start_tokens = self.pop_to_start_token(origin_cond)
        start_tokens = start_tokens.expand(B, self.num_tokens_per_patch, self.token_dim)


        dest_pop_condition = pop_condition[:, :, 1:, 0]
        dest_pop_condition = self.pad_to_patch_size(dest_pop_condition, dim=2)
        L_patches = dest_pop_condition.shape[2] // self.patch_spatial
        assert L_patches == len(groups), f"L_patches={L_patches}, groups={len(groups)}"

        dest_pop_condition = dest_pop_condition.transpose(1, 2)
        dest_pop_condition = dest_pop_condition.reshape(B, L_patches, -1,self.cond_channels)
        pop_condition_d = dest_pop_condition[:, :, :, :self.non_spatial_channels].reshape(B, L_patches, -1)
        spatial_cond = dest_pop_condition[:, :, :, self.non_spatial_channels:].reshape(B, L_patches, -1)
        spatial_embedding = self.spatial_embedding_linear(spatial_cond)
        spatial_embedding = spatial_embedding.unsqueeze(2).expand(
            B, L_patches, self.num_tokens_per_patch, self.token_dim
        )
        spatial_embedding = spatial_embedding.reshape(B, total_L, self.token_dim)

        if buffer_tokens is None:
            buffer_tokens = start_tokens
        buffer = buffer_tokens.shape[1]


        input_tokens = gt_tokens
        x = torch.cat([buffer_tokens, input_tokens[:,:-buffer]], dim=1)


        x = self.z_proj(x)
        x = self.z_ln(x)
        x = x + self.pos_embed(x) + spatial_embedding.expand(B, total_L, self.token_dim)

        attn_mask = self.build_causal_mask_from_groups(groups, device=device, buffer=0)
        pop_condition = self.cond_proj(pop_condition_d)


        for blk in self.blocks:
            x = blk(x, attn_mask,None)

        z_slice = x
        z_slice=self.flownet(z_slice,pop_condition)
        z_slice=z_slice.reshape(B, total_L, self.token_dim)


        loss = torch.nn.HuberLoss()(z_slice, gt_tokens)
        return loss.mean(),z_slice

    @torch.no_grad()


















































































    @torch.no_grad()
    def sample(
        self,
        pop_condition: torch.Tensor,
        num_steps: int,
        groups: List[int],
        sampler_fn=None,
        buffer_tokens: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        guidance_low: float = 0.0,
        guidance_high: float = 1.0,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        """
        基于由近及远的顺序进行自回归采样，返回 [B, L, D_z] 的token序列。
        默认使用KV cache；use_kv_cache=False时运行非cache因果参考路径。
        注意：这里的 L = sum(groups)，不包含 7 个 CLS 起始 token。
        """
        del num_steps, sampler_fn, guidance_scale, guidance_low, guidance_high

        device = pop_condition.device
        B = pop_condition.shape[0]
        groups = [int(g) for g in groups]
        total_L = sum(groups)



        origin_cond = pop_condition[:, :self.non_spatial_channels, 0, 0]
        origin_cond = origin_cond.unsqueeze(1)
        origin_cond_embed = self.pop_to_start_token(origin_cond)
        start_tokens = origin_cond_embed.expand(B, self.num_tokens_per_patch, self.token_dim)


        dest_pop_condition = pop_condition[:, :, 1:, 0]
        dest_pop_condition = self.pad_to_patch_size(dest_pop_condition, dim=2)
        L_patches = dest_pop_condition.shape[2] // self.patch_spatial
        assert L_patches == len(groups), f"L_patches={L_patches}, groups={len(groups)}"
        dest_pop_condition = dest_pop_condition.transpose(1, 2)
        dest_pop_condition = dest_pop_condition.reshape(B, L_patches, -1, self.cond_channels)
        pop_condition_d = dest_pop_condition[:, :, :, :self.non_spatial_channels].reshape(B, L_patches, -1)
        spatial_cond = dest_pop_condition[:, :, :, self.non_spatial_channels:].reshape(B, L_patches, -1)
        spatial_embedding = self.spatial_embedding_linear(spatial_cond)
        spatial_embedding = spatial_embedding.unsqueeze(2).expand(
            B, L_patches, self.num_tokens_per_patch, self.token_dim
        )
        spatial_embedding = spatial_embedding.reshape(B, total_L, self.token_dim)

        if buffer_tokens is None:
            buffer_tokens = start_tokens
        buffer_len = buffer_tokens.shape[1]
        if buffer_len <= 0:
            raise ValueError("buffer_tokens must contain at least one token")
        if any(k != buffer_len for k in groups):
            raise ValueError(
                "KV-cache sampling expects one fixed-size input group per output group; "
                f"got buffer_len={buffer_len}, groups={groups[:8]}..."
            )

        pop_condition_encoded = self.cond_proj(pop_condition_d)
        tokens = torch.zeros(B, total_L, self.token_dim, device=device, dtype=buffer_tokens.dtype)

        def encode_inputs(
            input_tokens: torch.Tensor,
            token_offset: int,
            attn_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
            cache: bool = False,
        ):
            x = self.z_proj(input_tokens)
            x = self.z_ln(x)
            seq_len = x.shape[1]
            spatial_slice = spatial_embedding[:, token_offset:token_offset + seq_len, :]
            x = x + self.pos_embed(x, position_offset=token_offset) + spatial_slice

            if cache:
                next_key_values = []
                if past_key_values is None:
                    past_key_values = [None] * len(self.blocks)
                for layer_idx, blk in enumerate(self.blocks):
                    x, layer_present = blk(
                        x,
                        attn_mask=None,
                        condition=None,
                        layer_past=past_key_values[layer_idx],
                        use_cache=True,
                        is_causal=False,
                    )
                    next_key_values.append(layer_present)
                return x, next_key_values

            for blk in self.blocks:
                x = blk(x, attn_mask, None)
            return x, None

        def flow_step(z_cond: torch.Tensor, step: int) -> torch.Tensor:
            if step < pop_condition_encoded.shape[1]:
                pop_slice = pop_condition_encoded[:, step:step + 1]
            else:
                pop_slice = pop_condition_encoded[:, -1:]
            return self.flownet(z_cond, pop_slice)

        if not use_kv_cache:



            start = 0
            for step, k in enumerate(groups):
                end = start + k
                local_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(self.blocks)
                hist_start = 0
                z_cond = None
                for hist_step, hist_k in enumerate(groups[:step + 1]):
                    if hist_step == 0:
                        hist_inputs = buffer_tokens
                    else:
                        hist_inputs = tokens[:, hist_start - hist_k:hist_start]
                    z_cond, local_key_values = encode_inputs(
                        hist_inputs,
                        token_offset=hist_start,
                        past_key_values=local_key_values,
                        cache=True,
                    )
                    hist_start += hist_k
                assert z_cond is not None
                tokens[:, start:end] = flow_step(z_cond, step)
                start = end
            return tokens

        past_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(self.blocks)
        start = 0
        for step, k in enumerate(groups):
            end = start + k
            if step == 0:
                step_inputs = buffer_tokens
            else:
                step_inputs = tokens[:, start - k:start]
            z_cond, past_key_values = encode_inputs(
                step_inputs,
                token_offset=start,
                past_key_values=past_key_values,
                cache=True,
            )
            tokens[:, start:end] = flow_step(z_cond, step)
            start = end

        return tokens




if __name__ == "__main__":
    """
    测试FlowARDistance模型是否能正常运行
    按照OD流数据的形式进行测试
    """
    import numpy as np
    import sys
    import os


    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)

    print("=" * 60)
    print("测试FlowARDistance模型")
    print("=" * 60)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")


    num_regions = 2235
    time_steps = 168
    batch_size = 4


    print(f"\n1. 创建模拟OD数据: [{batch_size}, 1, {num_regions}, {time_steps}]")
    od_data = torch.randn(batch_size, 1, num_regions, time_steps, device=device)
    print(f"   OD数据形状: {od_data.shape}")
    print(f"   OD数据范围: [{od_data.min():.3f}, {od_data.max():.3f}]")


    print(f"\n2. 模拟VAE编码过程")



    token_dim = 64

    patch_spatial = 16
    patch_temporal = 24
    tokens_per_patch = 7
    num_spatial_patches = (num_regions + patch_spatial - 1) // patch_spatial
    seq_len = num_spatial_patches * tokens_per_patch

    print(f"   空间patch数量: {num_spatial_patches}")
    print(f"   每个空间patch的时间token数: {tokens_per_patch}")
    print(f"   总序列长度: {seq_len}")


    gt_tokens = torch.randn(batch_size, seq_len, token_dim, device=device)
    print(f"   VAE tokens形状: {gt_tokens.shape}")


    print(f"\n3. 创建FlowARDistance模型")
    model = FlowARDistance(
        token_dim=token_dim,
        model_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        flow_width=768,
        flow_depth=3,
        cross=False,
        num_tokens_per_patch=tokens_per_patch,
    ).to(device)

    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


    print(f"\n4. 定义分组策略")

    groups_per_patch = 7
    groups = [groups_per_patch] * num_spatial_patches
    print(f"   每个空间patch的时间token数: {groups_per_patch}")
    print(f"   空间patch数量: {num_spatial_patches}")
    print(f"   分组策略: {groups}")
    print(f"   分组总和: {sum(groups)} (应该等于序列长度 {seq_len})")





















    print(f"\n6. 测试自回归采样")
    model.eval()
    try:
        with torch.no_grad():
            sampled_tokens = model.sample(
                num_steps=10,
                groups=groups,
                sampler_fn=None,
                buffer_tokens=None,
                guidance_scale=1.0,
                guidance_low=0.0,
                guidance_high=1.0,
            )
        print(f"   ✓ 采样成功")
        print(f"   采样tokens形状: {sampled_tokens.shape}")
        print(f"   采样tokens范围: [{sampled_tokens.min():.3f}, {sampled_tokens.max():.3f}]")


        if sampled_tokens.shape == gt_tokens.shape:
            print(f"   ✓ 采样形状与输入一致")
        else:
            print(f"   ✗ 采样形状不匹配: {sampled_tokens.shape} vs {gt_tokens.shape}")

    except Exception as e:
        print(f"   ✗ 采样失败: {e}")
        raise


    print(f"\n7. 测试带buffer的采样")
    try:
        buffer_tokens = torch.randn(batch_size, 4, token_dim, device=device)
        with torch.no_grad():
            sampled_tokens_with_buffer = model.sample(
                num_steps=10,
                groups=groups,
                sampler_fn=None,
                buffer_tokens=buffer_tokens,
                guidance_scale=1.0,
                guidance_low=0.0,
                guidance_high=1.0,
            )
        print(f"   ✓ 带buffer采样成功")
        print(f"   采样tokens形状: {sampled_tokens_with_buffer.shape}")

    except Exception as e:
        print(f"   ✗ 带buffer采样失败: {e}")
        raise


    print(f"\n8. 测试注意力掩码生成")
    try:
        mask = FlowARDistance.build_causal_mask_from_groups(groups, device=device, buffer=0)
        print(f"   ✓ 注意力掩码生成成功")
        print(f"   掩码形状: {mask.shape}")
        print(f"   掩码范围: [{mask.min():.3f}, {mask.max():.3f}]")


        mask_np = mask.squeeze().cpu().numpy()
        is_causal = True
        for i in range(mask_np.shape[0]):
            for j in range(i+1, mask_np.shape[1]):
                if mask_np[i, j] != float('-inf'):
                    is_causal = False
                    break
        if is_causal:
            print(f"   ✓ 掩码具有正确的因果性")
        else:
            print(f"   ✗ 掩码因果性检查失败")

    except Exception as e:
        print(f"   ✗ 注意力掩码生成失败: {e}")
        raise


    print(f"\n9. 测试梯度计算")
    model.train()
    try:

        model.zero_grad()


        loss = model(gt_tokens=gt_tokens, groups=groups)
        loss.backward()


        grad_norm = 0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
                param_count += 1

        grad_norm = grad_norm ** 0.5
        print(f"   ✓ Teacher Forcing梯度计算成功")
        print(f"   梯度范数: {grad_norm:.6f}")
        print(f"   有梯度的参数数量: {param_count}")

    except Exception as e:
        print(f"   ✗ 梯度计算失败: {e}")
        raise


    print(f"\n10. 内存使用情况")
    if torch.cuda.is_available():
        print(f"   GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"   GPU内存缓存: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

    print(f"\n" + "=" * 60)
    print("✓ 所有测试通过！FlowARDistance模型可以正常运行")
    print("=" * 60)

    print(f"\n📝 Teacher Forcing训练说明:")
    print(f"   - 训练时（固定使用Teacher Forcing）:")
    print(f"     * 输入: [起始符(7) + gt_tokens[:-7]]")
    print(f"     * 目标: gt_tokens (完整序列)")
    print(f"     * 分组: [7+7, 7, 7, ...] (第一组包含起始符)")
    print(f"   - 推理时:")
    print(f"     * 使用sample()方法进行自回归生成")
    print(f"     * 从噪声开始，逐步生成每个patch的token")
    print(f"   - 优势:")
    print(f"     * 训练稳定，收敛更快")
    print(f"     * 避免训练-推理不一致问题")
    print(f"     * 符合自回归生成的最佳实践")
