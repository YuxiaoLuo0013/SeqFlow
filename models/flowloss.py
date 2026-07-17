import torch
import torch.distributions as dist
import numpy as np
from torch.distributions import LogisticNormal, Beta
import torch.nn.functional as F


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    """对除batch维之外的所有维度取均值，返回形状 [B]。"""
    if x.dim() <= 1:
        return x
    dims = list(range(1, x.dim()))
    return torch.mean(x, dim=dims)


def sum_flat(x: torch.Tensor) -> torch.Tensor:
    """对除batch维之外的所有维度取求和，返回形状 [B]。"""
    if x.dim() <= 1:
        return x
    dims = list(range(1, x.dim()))
    return torch.sum(x, dim=dims)


class SILoss:
    def __init__(
        self,
        prediction: str = 'v',
        path_type: str = 'linear',
        weighting: str = 'uniform',
        encoders=None,
        accelerator=None,
        latents_scale=None,
        latents_bias=None,
    ) -> None:
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders if encoders is not None else []
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t: torch.Tensor):
        if self.path_type == 'linear':
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -torch.ones_like(t)
            d_sigma_t = torch.ones_like(t)
        elif self.path_type == 'cosine':
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -(np.pi / 2) * torch.sin(t * np.pi / 2)
            d_sigma_t = (np.pi / 2) * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError(f'Unknown path_type: {self.path_type}')
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, tokens: torch.Tensor, condition: torch.Tensor,pop_condition: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, L, D] 真实目标token（来自VAE latent，例如mu）
        condition: [B, L, Z] 条件编码（来自AR编码器输出）
        返回：形状 [B] 的逐样本loss，以及预测的 x0（形状 [B, L, D]）
        """
        assert tokens.dim() == 3, 'tokens must be [B, L, D]'
        assert condition.dim() == 3, 'condition must be [B, L, Z]'
        B, L, D = tokens.shape


        if self.weighting == 'uniform':
            time_input = torch.rand((B, 1, 1), device=tokens.device, dtype=tokens.dtype)
        else:
            raise NotImplementedError(f'weighting {self.weighting} not implemented')

        noises = torch.randn_like(tokens)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        model_input = alpha_t * tokens + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * tokens + d_sigma_t * noises
        else:
            raise NotImplementedError('Only v-prediction is supported currently')


        t_vec = time_input.flatten()
        model_output = model(model_input, t_vec, condition, pop_condition)
        model_output = model_output.reshape(B, L, -1)
        denoising_loss = mean_flat((model_output - model_target)**2)







        x_t = model_input
        v_t_pred = model_output
        det = alpha_t * d_sigma_t - sigma_t * d_alpha_t
        eps = 1e-6
        det_safe = det.sign() * det.abs().clamp(min=eps)
        x0_pred = (d_sigma_t * x_t - sigma_t * v_t_pred) / det_safe
        return denoising_loss, x0_pred


