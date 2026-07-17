from __future__ import annotations

from typing import Literal, Union, Dict, Tuple

import numpy as np
from scipy.stats import entropy

try:
    import torch

    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _to_numpy(arr: ArrayLike) -> np.ndarray:
    """
    将输入数据转换为 numpy.ndarray。

    - 支持 numpy 数组与 PyTorch 张量。
    - 不修改原数据；若为 torch 张量则拷贝到 CPU。
    """
    if _HAS_TORCH and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    if isinstance(arr, np.ndarray):
        return arr
    raise TypeError("输入必须是 numpy.ndarray 或 torch.Tensor")


def _validate_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true 与 y_pred 形状不一致: {y_true.shape} vs {y_pred.shape}"
        )
    if y_true.ndim not in (2, 3):
        raise ValueError(
            "期望输入形状为 (T, N, N) 或 (N, N)；即时间×起点×终点。"
        )


def _ensure_time_axis(y: np.ndarray) -> np.ndarray:
    """
    将 (N, N) 扩展为 (1, N, N)，便于统一按时间维度计算。
    """
    if y.ndim == 2:
        return y[None, ...]
    return y


def compute_cpc(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    mode: Literal["global", "per_origin", "per_destination"] = "global",
    eps: float = 1e-12,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    计算 CPC(Common Part of Commuters)。

    定义：先对时间维度求和，然后计算空间OD的CPC
    CPC(x, y) = 2 * sum(min(x_ij, y_ij)) / (sum(x_ij) + sum(y_ij) + eps)

    参数:
    - y_true/y_pred: 形状为 (T, N, N) 或 (N, N) 的 OD 流量矩阵/序列
    - mode:
        - "global": 全局空间CPC，输出标量
        - "per_origin": 每个起点的CPC，输出形状 (N,)
        - "per_destination": 每个终点的CPC，输出形状 (N,)
    - eps: 防止除零
    - clip_negative: 是否将负值裁剪为0（推荐用于OD流量数据）

    返回:
    - numpy.ndarray，见上述形状说明。
    """
    t = _to_numpy(y_true)
    p = _to_numpy(y_pred)
    _validate_shapes(t, p)

    t = _ensure_time_axis(t)
    p = _ensure_time_axis(p)


    if clip_negative:
        t = np.maximum(t, 0.0)
        p = np.maximum(p, 0.0)


    t_spatial = t.sum(axis=0)
    p_spatial = p.sum(axis=0)

    if mode == "global":

        numerator = 2.0 * np.minimum(t_spatial, p_spatial).sum()
        denominator = (t_spatial.sum() + p_spatial.sum() + eps)
        return np.array([numerator / denominator])

    if mode == "per_origin":

        numerator = 2.0 * np.minimum(t_spatial, p_spatial).sum(axis=1)
        denominator = (t_spatial.sum(axis=1) + p_spatial.sum(axis=1) + eps)
        return numerator / denominator

    if mode == "per_destination":

        numerator = 2.0 * np.minimum(t_spatial, p_spatial).sum(axis=0)
        denominator = (t_spatial.sum(axis=0) + p_spatial.sum(axis=0) + eps)
        return numerator / denominator

    raise ValueError("mode 仅支持 'global' | 'per_origin' | 'per_destination'")


def compute_mse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    mode: Literal["global", "per_origin", "per_destination"] = "global",
    eps: float = 0.0,
) -> np.ndarray:
    """
    计算 MSE（均方误差）。

    定义：MSE(x, y) = mean((x_tij - y_tij)^2)
    先计算每个时间点、每个 OD 对的平方误差，然后对所有求平均。

    - "global": 对所有 (t, i, j) 求平均，输出标量
    - "per_origin": 对每个起点 i，沿 (t, j) 聚合，输出 (N,)
    - "per_destination": 对每个终点 j，沿 (t, i) 聚合，输出 (N,)
    """
    t = _to_numpy(y_true)
    p = _to_numpy(y_pred)
    _validate_shapes(t, p)

    t = _ensure_time_axis(t)
    p = _ensure_time_axis(p)


    sq = (t - p) ** 2

    if mode == "global":

        return np.array([sq.mean() + eps])

    if mode == "per_origin":


        return sq.mean(axis=(0, 2)) + eps

    if mode == "per_destination":


        return sq.mean(axis=(0, 1)) + eps

    raise ValueError("mode 仅支持 'global' | 'per_origin' | 'per_destination'")


def compute_rmse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    mode: Literal["global", "per_origin", "per_destination"] = "global",
    eps: float = 0.0,
) -> np.ndarray:
    """
    计算 RMSE（均方根误差）。

    定义：RMSE(x, y) = sqrt(mean((x_tij - y_tij)^2))
    先计算每个时间点、每个 OD 对的平方误差，然后对所有求平均并开平方根。

    - "global": 对所有 (t, i, j) 求平均，输出标量
    - "per_origin": 对每个起点 i，沿 (t, j) 聚合，输出 (N,)
    - "per_destination": 对每个终点 j，沿 (t, i) 聚合，输出 (N,)
    """
    mse = compute_mse(y_true, y_pred, mode=mode, eps=eps)
    return np.sqrt(mse)


def compute_nrmse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    mode: Literal["global", "per_origin", "per_destination"] = "global",
    normalization: Literal["mean", "std", "range", "rms"] = "rms",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    计算 NRMSE（归一化 RMSE）。

    按照公式：NRMSE = RMSE / normalization_factor

    归一化选项（分母基于 y_true）：
    - rms: 用RMS（均方根）进行归一化（默认，符合公式定义）
    - mean: 用均值进行归一化
    - std: 用标准差进行归一化
    - range: 用 (max - min) 进行归一化

    输出形状同 compute_rmse。
    """
    t = _to_numpy(y_true)
    p = _to_numpy(y_pred)
    _validate_shapes(t, p)

    t = _ensure_time_axis(t)
    p = _ensure_time_axis(p)


    rmse = compute_rmse(t, p, mode=mode, eps=eps)

    if mode == "global":

        if normalization == "rms":

            denom = np.sqrt(np.mean((t - t.mean()) ** 2))
        elif normalization == "mean":
            denom = t.mean()
        elif normalization == "std":
            denom = t.std()
        elif normalization == "range":
            denom = t.max() - t.min()
        else:
            raise ValueError("normalization 仅支持 'rms' | 'mean' | 'std' | 'range'")
        return rmse / (denom + eps)

    if mode == "per_origin":

        if normalization == "rms":
            mean_true = t.mean(axis=(0, 2), keepdims=True)
            denom = np.sqrt(np.mean((t - mean_true) ** 2, axis=(0, 2)))
        elif normalization == "mean":
            denom = t.mean(axis=(0, 2))
        elif normalization == "std":
            denom = t.std(axis=(0, 2))
        elif normalization == "range":
            denom = t.max(axis=(0, 2)) - t.min(axis=(0, 2))
        else:
            raise ValueError("normalization 仅支持 'rms' | 'mean' | 'std' | 'range'")
        return rmse / (denom + eps)

    if mode == "per_destination":

        if normalization == "rms":
            mean_true = t.mean(axis=(0, 1), keepdims=True)
            denom = np.sqrt(np.mean((t - mean_true) ** 2, axis=(0, 1)))
        elif normalization == "mean":
            denom = t.mean(axis=(0, 1))
        elif normalization == "std":
            denom = t.std(axis=(0, 1))
        elif normalization == "range":
            denom = t.max(axis=(0, 1)) - t.min(axis=(0, 1))
        else:
            raise ValueError("normalization 仅支持 'rms' | 'mean' | 'std' | 'range'")
        return rmse / (denom + eps)

    raise ValueError("mode 仅支持 'global' | 'per_origin' | 'per_destination'")




def _values_to_bucket(values: np.ndarray) -> Tuple[list, list]:
    max_ = float(values.max()) if values.size else 0.0
    i = 0
    sections = []
    nums = []
    while True:
        if i == 0:
            left, right = 0, 1
            sections.append(left)
            sections.append(right)
            i = 1
        else:
            left, right = i, i * 2
            sections.append(right)
            i = i * 2
        nums.append(int(((values >= left) & (values < right)).sum()))
        if right > max_:
            break
    return sections, nums


def _normalize_bucket_counts(counts: list[int]) -> np.ndarray:
    arr = np.array(counts, dtype=float)
    total = arr.sum()
    if total <= 0:
        return np.ones_like(arr, dtype=float) / max(len(arr), 1)
    return arr / total


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2.0
    return float(0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2))


def compute_jsd_inflow(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    t = _to_numpy(y_true)
    p = _to_numpy(y_pred)
    _validate_shapes(t, p)
    t = _ensure_time_axis(t).sum(axis=0)
    p = _ensure_time_axis(p).sum(axis=0)

    b_in = np.maximum(t.sum(axis=0).astype(float), 0.0)
    a_in = np.maximum(p.sum(axis=0).astype(float), 0.0)
    sections, b_dist = _values_to_bucket(b_in)
    a_dist = []
    for i in range(len(sections) - 1):
        low, high = sections[i], sections[i + 1]
        a_dist.append(int(np.sum((a_in >= low) & (a_in < high))))
    return _js_divergence(_normalize_bucket_counts(a_dist), _normalize_bucket_counts(b_dist))


def compute_jsd_outflow(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    t = _to_numpy(y_true)
    p = _to_numpy(y_pred)
    _validate_shapes(t, p)
    t = _ensure_time_axis(t).sum(axis=0)
    p = _ensure_time_axis(p).sum(axis=0)

    b_out = np.maximum(t.sum(axis=1).astype(float), 0.0)
    a_out = np.maximum(p.sum(axis=1).astype(float), 0.0)
    sections, b_dist = _values_to_bucket(b_out)
    a_dist = []
    for i in range(len(sections) - 1):
        low, high = sections[i], sections[i + 1]
        a_dist.append(int(np.sum((a_out >= low) & (a_out < high))))
    return _js_divergence(_normalize_bucket_counts(a_dist), _normalize_bucket_counts(b_dist))


def compute_jsd_odflow(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    t = _to_numpy(y_true)
    p = _to_numpy(y_pred)
    _validate_shapes(t, p)
    b_flat = np.maximum(_ensure_time_axis(t).sum(axis=0).reshape(-1).astype(float), 0.0)
    a_flat = np.maximum(_ensure_time_axis(p).sum(axis=0).reshape(-1).astype(float), 0.0)
    sections, b_dist = _values_to_bucket(b_flat)
    a_dist = []
    for i in range(len(sections) - 1):
        low, high = sections[i], sections[i + 1]
        a_dist.append(int(np.sum((a_flat >= low) & (a_flat < high))))
    return _js_divergence(_normalize_bucket_counts(a_dist), _normalize_bucket_counts(b_dist))


def compute_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    mode: Literal["global", "per_origin", "per_destination"] = "global",
    nrmse_norm: Literal["mean", "std", "range", "rms"] = "rms",
    eps: float = 1e-12,
    clip_negative: bool = True,
) -> Dict[str, np.ndarray]:
    """
    便捷接口：同时计算 CPC、MSE、RMSE、NRMSE。

    返回字典：{"cpc": ..., "mse": ..., "rmse": ..., "nrmse": ...}
    """
    return {
        "cpc": compute_cpc(y_true, y_pred, mode=mode, eps=eps, clip_negative=clip_negative),
        "mse": compute_mse(y_true, y_pred, mode=mode, eps=0.0),
        "rmse": compute_rmse(y_true, y_pred, mode=mode, eps=0.0),
        "nrmse": compute_nrmse(
            y_true, y_pred, mode=mode, normalization=nrmse_norm, eps=eps
        ),
        "jsd_inflow": np.array([compute_jsd_inflow(y_true, y_pred)]),
        "jsd_outflow": np.array([compute_jsd_outflow(y_true, y_pred)]),
        "jsd_odflow": np.array([compute_jsd_odflow(y_true, y_pred)]),
    }


__all__ = [
    "compute_cpc",
    "compute_mse",
    "compute_rmse",
    "compute_nrmse",
    "compute_jsd_inflow",
    "compute_jsd_outflow",
    "compute_jsd_odflow",
    "compute_metrics",
]


