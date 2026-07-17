from typing import List, Tuple
import numpy as np


def ring_groups_by_distance(distance_matrix: np.ndarray, center_index: int, thresholds: List[float]) -> List[List[int]]:
    """
    基于距离阈值将节点按由近及远分配到环形分组。
    - distance_matrix: [N, N] 对称距离矩阵，distance_matrix[i,j] 为 i->j 距离
    - center_index: 以该索引为中心，向外扩展
    - thresholds: 单调递增的阈值列表，例如 [1.0, 2.0, 3.0]
    返回：List[groups]，每组是索引列表；不包含 center 本身。
    """
    dists = distance_matrix[center_index]
    order = np.argsort(dists)
    groups: List[List[int]] = []
    prev_t = 0.0
    for t in thresholds:
        mask = (dists > prev_t) & (dists <= t)
        grp = [int(i) for i in np.where(mask)[0] if int(i) != center_index]
        groups.append(grp)
        prev_t = t

    mask_far = dists > prev_t
    grp_far = [int(i) for i in np.where(mask_far)[0] if int(i) != center_index]
    if len(grp_far) > 0:
        groups.append(grp_far)
    return groups


def linearize_groups(groups: List[List[int]]) -> List[int]:
    """
    将分组按顺序线性展开为生成序列（由近及远）。
    返回线性索引序列。
    """
    seq: List[int] = []
    for g in groups:
        seq.extend(g)
    return seq


def group_lengths(groups: List[List[int]]) -> List[int]:
    """返回每个分组的长度列表。"""
    return [len(g) for g in groups]


