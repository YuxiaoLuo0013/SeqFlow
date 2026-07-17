import torch
import numpy as np


def expand_t_like_x(t: torch.Tensor, x_cur: torch.Tensor) -> torch.Tensor:
    dims = [1] * (len(x_cur.size()) - 1)
    return t.view(t.size(0), *dims)


def get_score_from_velocity(vt: torch.Tensor, xt: torch.Tensor, t: torch.Tensor, path_type: str = "linear") -> torch.Tensor:
    t = expand_t_like_x(t, xt)
    if path_type == "linear":

        alpha_t, d_alpha_t = t, torch.ones_like(xt, device=xt.device)
        sigma_t, d_sigma_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t ** 2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var
    return score


def compute_diffusion(t_cur: torch.Tensor) -> torch.Tensor:
    return 2 * t_cur


@torch.no_grad()
def euler_sampler(
    model,
    latents: torch.Tensor,
    condition: torch.Tensor,
    pop_condition: torch.Tensor,
    num_steps: int = 20,
    heun: bool = False,
    cfg_scale: float = 1.0,
    guidance_low: float = 0.0,
    guidance_high: float = 1.0,
    path_type: str = "linear",
):
    _dtype = latents.dtype
    t_steps = torch.cat([
        torch.linspace(1, 0.3, num_steps // 2, dtype=torch.float64),
        torch.linspace(0.25, 0, num_steps // 2 + 1, dtype=torch.float64),
    ])
    if num_steps == 50:
        t_steps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64)
    x_next = latents.to(torch.float64).reshape(-1,7,latents.shape[-1])
    device = x_next.device

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
            model_input = torch.cat([x_cur[: x_cur.shape[0] // 2]] * 2, dim=0)
        else:
            model_input = x_cur
        time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
        d_cur = model(
            model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), condition.to(dtype=_dtype),pop_condition.to(dtype=_dtype)
        ).to(torch.float64)
        if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
            d_cur = torch.cat([d_cur, d_cur], dim=0)
        x_next = x_cur + (t_next - t_cur) * d_cur
    return x_next.to(_dtype)




































































