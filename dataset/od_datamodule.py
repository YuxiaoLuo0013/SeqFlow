import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import pytorch_lightning as pl
from typing import Optional, List


def custom_collate_fn(batch):
    """
    自定义collate函数，正确处理condition为None的情况
    """

    od_list = [item['od'] for item in batch]
    condition_list = [item['condition'] for item in batch]


    od_batch = default_collate(od_list)


    if all(c is None for c in condition_list):
        condition_batch = None
    else:
        condition_batch = default_collate(condition_list)

    return {'od': od_batch, 'condition': condition_batch}


class LogTransformer:
    """对数变换器：使用log1p和expm1进行对数变换"""
    def __init__(self):
        pass

    def fit_transform(self, x):
        """应用log1p变换（纯Torch实现）"""
        if isinstance(x, torch.Tensor):
            return torch.log1p(x)
        else:

            x_t = torch.from_numpy(np.asarray(x))
            return torch.log1p(x_t)

    def inverse_transform(self, x):
        """应用expm1逆变换（纯Torch实现）"""
        if isinstance(x, torch.Tensor):
            return torch.expm1(x)
        else:
            x_t = torch.from_numpy(np.asarray(x))
            return torch.expm1(x_t)


class ODMatrixDataset(Dataset):
    def __init__(self,
                 od_path: str,
                 neighbors_path: str,
                 hours: Optional[List[int]] = None,
                 normalize: bool = True,
                 clip_max: Optional[float] = None,
                 compute_stats: bool = True,
                 population_path: Optional[str] = None,
                 tfidf_path: Optional[str] = None,
                 spatial_features_path: Optional[str] = None,
                 indices: Optional[List[int]] = None,
):
        """
        构建以"区域"为样本的OD数据：
        - 对于每个 origin 区域 i，按照 neighbors 序列重排其 168 小时的 OD 向量，得到形状 (2235, 168)
        - 返回给模型时，增加通道维度 -> (1, 2235, 168)
        - 如果提供population_path，则加载人口数据作为条件，形状为 (2236,) 包含origin和所有destination的人口
        - 如果提供tfidf_path，则加载tfidf特征作为额外条件，形状为 (2236, 13) 包含origin和所有destination的13个poi特征
        - 如果提供spatial_features_path，则加载空间特征作为条件，形状 (N, N, 5)，
          特征顺序为 [dx, dy, distance, sin_theta, cos_theta]，其中 dx/dy/distance 需归一化，
          sin_theta/cos_theta 保持原值
        - 如果提供indices，则只使用指定索引的区域作为样本
        """
        super().__init__()
        self.normalize = normalize
        self.clip_max = clip_max
        self.hours = list(range(168)) if hours is None else list(hours)
        self.population_path = population_path
        self.tfidf_path = tfidf_path
        self.spatial_features_path = spatial_features_path
        self.indices = indices


        self.global_mean = None
        self.global_std = None
        self.global_min = None
        self.global_max = None


        self.population_mean = None
        self.population_std = None



        self.spatial_mean = None
        self.spatial_std = None


        self.log_transformer = LogTransformer()


        self.population_data = None
        if self.population_path is not None:
            self.population_data = np.load(self.population_path)
            print(f"加载人口数据: {self.population_data.shape}")


        self.tfidf_data = None
        self.tfidf_dim = 0
        if self.tfidf_path is not None:
            tfidf_df = pd.read_csv(self.tfidf_path)

            tfidf_df = tfidf_df.sort_values('ID').reset_index(drop=True)

            poi_cols = [col for col in tfidf_df.columns if col != 'ID']
            self.tfidf_data = tfidf_df[poi_cols].values.astype(np.float32)
            self.tfidf_dim = self.tfidf_data.shape[1]
            print(f"加载tfidf数据: {self.tfidf_data.shape}, 特征维度: {self.tfidf_dim}")


        self.spatial_data = None
        if self.spatial_features_path is not None:
            self.spatial_data = np.load(self.spatial_features_path).astype(np.float32)
            assert self.spatial_data.ndim == 3 and self.spatial_data.shape[2] == 5, \
                f"spatial_features 应为 (N, N, 5)，实际形状: {self.spatial_data.shape}"
            print(f"加载空间特征: {self.spatial_data.shape} [dx, dy, distance, sin_theta, cos_theta]")


        self._load_raw_data_and_compute_stats(od_path, neighbors_path, compute_stats)

    def _load_od_matrices(self, path: str, hours: List[int]) -> List[np.ndarray]:
        ext = os.path.splitext(path)[1].lower()
        mats: List[np.ndarray] = []
        if ext == '.npz':
            data = np.load(path)

            keys = [f'hour_{h}' for h in hours]
            if not all(k in data for k in keys):

                keys = [str(h) for h in hours if str(h) in data]
                if len(keys) != len(hours):

                    keys = sorted(list(data.keys()), key=lambda x: int(x) if str(x).isdigit() else str(x))[:len(hours)]
            for k in keys:
                mats.append(np.asarray(data[k], dtype=np.float32))
        elif ext == '.pkl' or ext == '.pickle':
            import pickle
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                for h in hours:
                    key_candidates = [f'hour_{h}', h]
                    key = None
                    for kc in key_candidates:
                        if kc in obj:
                            key = kc
                            break
                    if key is None:
                        raise ValueError(f'未找到小时{h}对应的OD矩阵')
                    mats.append(np.asarray(obj[key], dtype=np.float32))
            else:
                raise ValueError('pickle格式需为按小时键的dict')
        else:
            raise ValueError('仅支持 .npz 或 .pkl 的OD矩阵文件')
        return mats

    def _load_raw_data_and_compute_stats(self, od_path: str, neighbors_path: str, compute_stats: bool):
        """加载原始数据并计算统计信息（原有逻辑）"""
        print("使用原始数据模式")


        neigh = np.load(neighbors_path)
        self.tile_ids: np.ndarray = neigh['tile_ids'].astype(np.int32)
        self.neighbors: np.ndarray = neigh['neighbors'].astype(np.int32)
        self.N = int(self.tile_ids.shape[0])
        assert self.neighbors.shape == (self.N, self.N - 1), 'neighbors形状应为 (N, N-1)'



        self.tile_id_to_index = np.arange(self.N, dtype=np.int32)


        self.od_mats = self._load_od_matrices(od_path, self.hours)
        for h, mat in zip(self.hours, self.od_mats):
            assert mat.shape[0] == self.N and mat.shape[1] == self.N, f'小时{h}的OD矩阵尺寸与N不一致'


        if self.normalize and compute_stats:
            self._compute_global_stats()

            if self.population_data is not None:
                self._compute_population_stats()

        if self.spatial_data is not None:
                self._compute_spatial_stats()


    def _compute_global_stats(self):
        """计算整个数据集的全局统计信息（Log变换 + z-score，纯Torch张量）"""
        print("计算全局统计信息（Log变换 + z-score）...")

        all_values_tensors = []


        for idx in range(self.N):

            origin_tid = int(self.tile_ids[idx])
            dest_tids = self.neighbors[idx]

            cols = dest_tids.astype(np.int64)
            rows = origin_tid

            for mat in self.od_mats:
                vec_np = mat[rows, cols]
                vec_t = torch.from_numpy(vec_np.astype(np.float32))

                if self.clip_max is not None:
                    vec_t = torch.clamp(vec_t, max=float(self.clip_max))

                vec_t = self.log_transformer.fit_transform(vec_t)
                all_values_tensors.append(vec_t)


        if len(all_values_tensors) == 0:
            raise ValueError("未收集到用于统计的数据")
        all_values = torch.cat(all_values_tensors, dim=0)


        total_count = int(all_values.numel())
        print(f"原始数据统计: 总样本数={total_count}")
        print(f"Log变换后范围: [{float(torch.min(all_values)):.6f}, {float(torch.max(all_values)):.6f}]")


        self.global_mean = float(torch.mean(all_values))
        self.global_std = float(torch.std(all_values, unbiased=False))
        if self.global_std < 1e-8:
            self.global_std = 1.0


        normalized_values = (all_values - self.global_mean) / self.global_std
        print(f"z-score统计: 均值={self.global_mean:.6f}, 标准差={self.global_std:.6f}")
        print(f"z后范围: [{float(torch.min(normalized_values)):.6f}, {float(torch.max(normalized_values)):.6f}]")


        print(f"Log变换后是否有NaN: {bool(torch.isnan(all_values).any())}")
        print(f"Log变换后是否有Inf: {bool(torch.isinf(all_values).any())}")
        print(f"归一化后是否有NaN: {bool(torch.isnan(normalized_values).any())}")
        print(f"归一化后是否有Inf: {bool(torch.isinf(normalized_values).any())}")

    def _compute_population_stats(self):
        """计算人口数据的统计信息（z-score归一化）"""
        print("计算人口数据统计信息（z-score）...")

        all_population_values = []


        for idx in range(self.N):
            origin_tid = int(self.tile_ids[idx])
            dest_tids = self.neighbors[idx]


            origin_pop = self.population_data[origin_tid]
            dest_pops = self.population_data[dest_tids]
            population_vec = np.concatenate([[origin_pop], dest_pops])


            pop_tensor = torch.from_numpy(population_vec.astype(np.float32))
            all_population_values.append(pop_tensor)


        if len(all_population_values) == 0:
            raise ValueError("未收集到用于统计的人口数据")
        all_population = torch.cat(all_population_values, dim=0)


        total_count = int(all_population.numel())
        print(f"人口数据统计: 总样本数={total_count}")
        print(f"人口数据范围: [{float(torch.min(all_population)):.6f}, {float(torch.max(all_population)):.6f}]")


        self.population_mean = float(torch.tensor(np.mean(self.population_data)))
        self.population_std = float(torch.tensor(np.std(self.population_data)))
        if self.population_std < 1e-8:
            self.population_std = 1.0


        normalized_population = (all_population - self.population_mean) / self.population_std
        print(f"人口数据z-score统计: 均值={self.population_mean:.6f}, 标准差={self.population_std:.6f}")
        print(f"人口数据z后范围: [{float(torch.min(normalized_population)):.6f}, {float(torch.max(normalized_population)):.6f}]")


        print(f"人口数据是否有NaN: {bool(torch.isnan(all_population).any())}")
        print(f"人口数据是否有Inf: {bool(torch.isinf(all_population).any())}")
        print(f"人口数据归一化后是否有NaN: {bool(torch.isnan(normalized_population).any())}")
        print(f"人口数据归一化后是否有Inf: {bool(torch.isinf(normalized_population).any())}")

    def _compute_spatial_stats(self):
        """计算空间特征 [dx, dy, distance] 三个通道的 z-score 统计信息。
        sin_theta 和 cos_theta（通道3、4）无需归一化，保持原值。
        统计结果存储在 self.spatial_mean / self.spatial_std，形状均为 (3,)。
        """
        print("计算空间特征统计信息（对 dx/dy/distance 做 z-score）...")

        all_dx, all_dy, all_dist = [], [], []

        for idx in range(self.N):
            origin_tid = int(self.tile_ids[idx])
            dest_tids = self.neighbors[idx].astype(np.int64)


            feats = self.spatial_data[origin_tid, dest_tids]
            all_dx.append(feats[:, 0])
            all_dy.append(feats[:, 1])
            all_dist.append(feats[:, 2])

        all_dx   = np.concatenate(all_dx)
        all_dy   = np.concatenate(all_dy)
        all_dist = np.concatenate(all_dist)


        dx_mean  = float(np.mean(all_dx));   dx_std  = float(np.std(all_dx))
        dy_mean  = float(np.mean(all_dy));   dy_std  = float(np.std(all_dy))

        nonzero_dist = all_dist[all_dist > 0]
        dist_mean = float(np.mean(nonzero_dist)); dist_std = float(np.std(nonzero_dist))

        self.spatial_mean = np.array([dx_mean, dy_mean, dist_mean], dtype=np.float32)
        self.spatial_std  = np.array([max(dx_std, 1e-8), max(dy_std, 1e-8), max(dist_std, 1e-8)], dtype=np.float32)

        print(f"  dx:       mean={dx_mean:.4f}, std={dx_std:.4f}")
        print(f"  dy:       mean={dy_mean:.4f}, std={dy_std:.4f}")
        print(f"  distance: mean={dist_mean:.4f}, std={dist_std:.4f}")
        print(f"  sin_theta, cos_theta: 不归一化，保持原值")

    def __len__(self):

        if self.indices is not None:
            return len(self.indices)
        return self.N

    def __getitem__(self, idx: int):
        """
        返回字典格式数据，分离OD和条件特征以节省内存：
        - 'od': OD数据 [1, 2235, T]
        - 'condition': 条件数据 [C, 2236, 1]，C = 1人口 + tfidf_dim + 5空间特征，时间维度只有1
          空间特征通道顺序：[dx, dy, distance, sin_theta, cos_theta]
          其中 dx/dy/distance 已做 z-score 归一化，sin/cos 保持原值
        """

        if self.indices is not None:
            actual_idx = self.indices[idx]
        else:
            actual_idx = idx


        origin_tid = int(self.tile_ids[actual_idx])
        dest_tids = self.neighbors[actual_idx]


        cols = dest_tids.astype(np.int64)
        rows = origin_tid
        od_stack = []
        for mat in self.od_mats:

            vec_np = mat[rows, cols]
            od_stack.append(vec_np)
        od_array = np.stack(od_stack, axis=1).astype(np.float32)

        x = torch.from_numpy(od_array)
        if self.clip_max is not None:
            x = torch.clamp(x, max=float(self.clip_max))

        if self.normalize:

            x = self.log_transformer.fit_transform(x)

            x = (x - self.global_mean) / self.global_std


        x = x.unsqueeze(0)


        condition = None
        if self.population_data is not None or self.tfidf_data is not None or self.spatial_data is not None:
            condition_tensors = []


            if self.population_data is not None:

                origin_pop = self.population_data[origin_tid]
                dest_pops = self.population_data[dest_tids]
                population_vec = np.concatenate([[origin_pop], dest_pops])
                population_tensor = torch.from_numpy(population_vec.astype(np.float32))


                if self.normalize and self.population_mean is not None and self.population_std is not None:
                    population_tensor = (population_tensor - self.population_mean) / self.population_std


                condition_tensors.append(population_tensor.unsqueeze(0))


            if self.tfidf_data is not None:

                origin_tfidf = self.tfidf_data[origin_tid]
                dest_tfidf = self.tfidf_data[dest_tids]
                tfidf_mat = np.concatenate([[origin_tfidf], dest_tfidf], axis=0)
                tfidf_tensor = torch.from_numpy(tfidf_mat.astype(np.float32))



                condition_tensors.append(tfidf_tensor.T)


            if self.spatial_data is not None:


                dest_spatial = self.spatial_data[origin_tid, dest_tids].astype(np.float32)
                origin_spatial = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
                spatial_mat = np.concatenate([origin_spatial, dest_spatial], axis=0)
                spatial_tensor = torch.from_numpy(spatial_mat)


                if self.normalize and self.spatial_mean is not None and self.spatial_std is not None:
                    mean_t = torch.as_tensor(self.spatial_mean, dtype=spatial_tensor.dtype)
                    std_t = torch.as_tensor(self.spatial_std, dtype=spatial_tensor.dtype)
                    spatial_tensor[:, :3] = (spatial_tensor[:, :3] - mean_t) / std_t


                condition_tensors.append(spatial_tensor.T)


            condition = torch.cat(condition_tensors, dim=0).unsqueeze(-1)

        return {'od': x, 'condition': condition}


class MultiCityODDataset(Dataset):
    """
    Multi-city OD dataset: concatenate regions from multiple cities, pad to a
    common spatial dimension, and apply pre-computed normalization.

    Each sample is one *origin* region from one city.  The returned OD tensor
    is padded to (1, max_N-1, T) with zeros (after normalization), and a
    boolean ``mask`` of shape (max_N-1,) indicates valid positions.
    """

    def __init__(
        self,
        data_dir: str,
        city_names: List[str],
        norm_stats: dict,
        hours: Optional[List[int]] = None,
        normalize: bool = True,
        use_population: bool = True,
        use_tfidf: bool = False,
        use_spatial_features: bool = True,
        indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.normalize = normalize
        self.hours = list(range(168)) if hours is None else list(hours)
        self.use_population = use_population
        self.use_tfidf = use_tfidf
        self.use_spatial_features = use_spatial_features
        self.indices = indices
        self.log_transformer = LogTransformer()


        norm = norm_stats['normalization']
        self.global_mean = norm['od']['mean']
        self.global_std = norm['od']['std']
        pop_norm = norm.get('population', {})
        self.population_mean = pop_norm.get('mean')
        self.population_std = pop_norm.get('std')
        spatial_norm = norm.get('spatial_features', {})
        dx_norm = spatial_norm.get('dx', {})
        dy_norm = spatial_norm.get('dy', {})
        dist_norm = spatial_norm.get('distance', {})
        self.spatial_mean = np.array([
            dx_norm.get('mean', 0.0),
            dy_norm.get('mean', 0.0),
            dist_norm.get('mean', 0.0),
        ], dtype=np.float32)
        self.spatial_std = np.array([
            max(dx_norm.get('std', 1.0), 1e-8),
            max(dy_norm.get('std', 1.0), 1e-8),
            max(dist_norm.get('std', 1.0), 1e-8),
        ], dtype=np.float32)
        self.clip_max = norm_stats.get('clip_max')
        self.max_num_regions = norm_stats['max_num_regions']
        self.common_tfidf_columns = norm_stats.get('common_tfidf_columns', [])
        self.tfidf_dim = len(self.common_tfidf_columns) if use_tfidf else 0


        self.cities: List[dict] = []
        self.cumulative_sizes = [0]
        for name in city_names:
            cdata = self._load_city(data_dir, name)
            self.cities.append(cdata)
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + cdata['N'])
        self.total_samples = self.cumulative_sizes[-1]


        self.N = self.total_samples
        if len(self.cities) == 1:
            self.neighbors = self.cities[0]['neighbors']
        else:
            import itertools
            self.neighbors = list(itertools.chain.from_iterable(
                c['neighbors'] for c in self.cities
            ))

        print(f"MultiCityODDataset: {len(city_names)} cities, {self.total_samples} total regions, "
              f"max_N={self.max_num_regions}")
        for i, name in enumerate(city_names):
            print(f"  {name}: N={self.cities[i]['N']}")


    def _load_city(self, data_dir: str, city_name: str) -> dict:
        city_dir = os.path.join(data_dir, city_name)


        neigh = np.load(os.path.join(city_dir, f'{city_name}_neighbors_sorted.npz'))
        tile_ids = neigh['tile_ids'].astype(np.int32)
        neighbors = neigh['neighbors'].astype(np.int32)
        N = int(tile_ids.shape[0])


        od_data = np.load(os.path.join(city_dir, f'{city_name}_od_matrix_week_168.npz'))
        od_keys = sorted(od_data.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        od_mats = [np.asarray(od_data[od_keys[h]], dtype=np.float32) for h in self.hours]

        city: dict = {
            'name': city_name, 'N': N,
            'tile_ids': tile_ids, 'neighbors': neighbors,
            'od_mats': od_mats,
            'population': None, 'spatial': None, 'tfidf': None,
        }


        if self.use_population:
            pop_path = os.path.join(city_dir, f'{city_name}_pop_filtered.csv')
            if os.path.exists(pop_path):
                pop_df = pd.read_csv(pop_path)
                pop_arr = np.zeros(N, dtype=np.float32)
                tids = pop_df['tile_ID'].values.astype(np.int64)
                valid = (tids >= 0) & (tids < N)
                pop_arr[tids[valid]] = pop_df['pop'].values[valid].astype(np.float32)
                city['population'] = pop_arr


        if self.use_spatial_features:
            spatial_path = os.path.join(city_dir, f'{city_name}_spatial_features.npy')
            if os.path.exists(spatial_path):
                spatial = np.load(spatial_path).astype(np.float32)
                if spatial.ndim == 3 and spatial.shape[2] == 5:
                    city['spatial'] = spatial


        if self.use_tfidf and self.common_tfidf_columns:
            tfidf_path = os.path.join(city_dir, f'{city_name}_kind_tfidf.csv')
            if os.path.exists(tfidf_path):
                tfidf_df = pd.read_csv(tfidf_path)
                K = len(self.common_tfidf_columns)
                tfidf_arr = np.zeros((N, K), dtype=np.float32)
                tids = tfidf_df['tile_ID'].values.astype(np.int64)
                valid = (tids >= 0) & (tids < N)
                avail_cols = [c for c in self.common_tfidf_columns if c in tfidf_df.columns]
                avail_idx = [self.common_tfidf_columns.index(c) for c in avail_cols]
                vals = tfidf_df[avail_cols].values[valid].astype(np.float32)
                tfidf_arr[np.ix_(tids[valid], avail_idx)] = vals
                city['tfidf'] = tfidf_arr

        return city


    def _locate(self, global_idx: int):
        """Map global sample index -> (city_index, local_region_index)."""
        for c in range(len(self.cities)):
            if global_idx < self.cumulative_sizes[c + 1]:
                return c, global_idx - self.cumulative_sizes[c]
        raise IndexError(f"index {global_idx} out of range [0, {self.total_samples})")


    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return self.total_samples


    def __getitem__(self, idx: int):
        if self.indices is not None:
            actual_idx = self.indices[idx]
        else:
            actual_idx = idx

        city_idx, local_idx = self._locate(actual_idx)
        city = self.cities[city_idx]
        N = city['N']
        max_N = self.max_num_regions

        origin_tid = int(city['tile_ids'][local_idx])
        dest_tids = city['neighbors'][local_idx]
        cols = dest_tids.astype(np.int64)


        od_stack = [mat[origin_tid, cols] for mat in city['od_mats']]
        x = torch.from_numpy(np.stack(od_stack, axis=1).astype(np.float32))

        if self.clip_max is not None:
            x = torch.clamp(x, max=float(self.clip_max))
        if self.normalize:
            x = self.log_transformer.fit_transform(x)
            x = (x - self.global_mean) / self.global_std


        pad_h = (max_N - 1) - (N - 1)
        if pad_h > 0:
            x = torch.cat([x, torch.zeros(pad_h, x.shape[1], dtype=x.dtype)], dim=0)
        x = x.unsqueeze(0)


        mask = torch.zeros(max_N - 1, dtype=torch.bool)
        mask[:N - 1] = True


        cond_parts: List[torch.Tensor] = []
        pad_cond = max_N - N

        if self.use_population and city['population'] is not None:
            pop_vec = np.concatenate(
                [[city['population'][origin_tid]], city['population'][dest_tids]]
            ).astype(np.float32)
            pop_t = torch.from_numpy(pop_vec)
            if self.normalize and self.population_mean is not None:
                pop_t = (pop_t - self.population_mean) / self.population_std
            if pad_cond > 0:
                pop_t = torch.cat([pop_t, torch.zeros(pad_cond)])
            cond_parts.append(pop_t.unsqueeze(0))

        if self.use_tfidf and city['tfidf'] is not None:
            tfidf_mat = np.concatenate(
                [[city['tfidf'][origin_tid]], city['tfidf'][dest_tids]], axis=0
            ).astype(np.float32)
            tfidf_t = torch.from_numpy(tfidf_mat)
            if pad_cond > 0:
                tfidf_t = torch.cat([tfidf_t, torch.zeros(pad_cond, tfidf_t.shape[1])], dim=0)
            cond_parts.append(tfidf_t.T)

        if self.use_spatial_features and city['spatial'] is not None:
            dest_spatial = city['spatial'][origin_tid, dest_tids].astype(np.float32)
            origin_spatial = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            spatial_mat = np.concatenate([origin_spatial, dest_spatial], axis=0)
            spatial_t = torch.from_numpy(spatial_mat)
            if self.normalize:
                mean_t = torch.as_tensor(self.spatial_mean, dtype=spatial_t.dtype)
                std_t = torch.as_tensor(self.spatial_std, dtype=spatial_t.dtype)
                spatial_t[:, :3] = (spatial_t[:, :3] - mean_t) / std_t
            if pad_cond > 0:
                spatial_t = torch.cat([spatial_t, torch.zeros(pad_cond, spatial_t.shape[1])], dim=0)
            cond_parts.append(spatial_t.T)

        condition = None
        if cond_parts:
            condition = torch.cat(cond_parts, dim=0).unsqueeze(-1)

        return {'od': x, 'condition': condition, 'mask': mask}


def multi_city_collate_fn(batch):
    """Collate that additionally handles the ``mask`` key."""
    od_batch = default_collate([item['od'] for item in batch])
    mask_batch = default_collate([item['mask'] for item in batch])

    cond_list = [item['condition'] for item in batch]
    if all(c is None for c in cond_list):
        cond_batch = None
    else:
        cond_batch = default_collate(cond_list)

    return {'od': od_batch, 'condition': cond_batch, 'mask': mask_batch}






class MultiCityODDataModule(pl.LightningDataModule):
    """
    Lightning DataModule that trains on multiple cities and validates/tests
    on another set of cities.

    - train_set: regions sampled from ``train_cities``
    - val_set: held-out regions sampled from ``train_cities``
    - test_set: all regions from ``test_cities``

    Normalization stats are loaded from a pre-computed YAML file generated
    by ``scripts/compute_norm_stats.py``.
    """

    def __init__(
        self,
        data_dir: str,
        train_cities: List[str],
        test_cities: List[str],
        norm_stats_path: str,
        hours: Optional[List[int]] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        normalize: bool = True,
        use_population: bool = True,
        use_tfidf: bool = False,
        use_spatial_features: bool = True,
        train_val_split: float = 0.8,
        val_seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_cities = train_cities
        self.test_cities = test_cities
        self.norm_stats_path = norm_stats_path
        self.hours = list(range(168)) if hours is None else list(hours)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.use_population = use_population
        self.use_tfidf = use_tfidf
        self.use_spatial_features = use_spatial_features
        if not 0.0 < float(train_val_split) < 1.0:
            raise ValueError("train_val_split must be in (0, 1)")
        self.train_val_split = float(train_val_split)
        self.val_seed = int(val_seed)

    def setup(self, stage: Optional[str] = None):
        if hasattr(self, 'train_set') and hasattr(self, 'val_set'):
            return

        import yaml

        with open(self.norm_stats_path, 'r', encoding='utf-8') as f:
            self.norm_stats = yaml.safe_load(f)

        common_kwargs = dict(
            data_dir=self.data_dir,
            norm_stats=self.norm_stats,
            hours=self.hours,
            normalize=self.normalize,
            use_population=self.use_population,
            use_tfidf=self.use_tfidf,
            use_spatial_features=self.use_spatial_features,
        )

        full_train_set = MultiCityODDataset(city_names=self.train_cities, **common_kwargs)
        total = len(full_train_set)
        train_size = int(total * self.train_val_split)
        train_size = max(1, min(train_size, total - 1))
        rng = np.random.default_rng(self.val_seed)
        indices = np.arange(total, dtype=np.int64)
        rng.shuffle(indices)
        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:].tolist()

        self.train_set = MultiCityODDataset(
            city_names=self.train_cities,
            indices=train_indices,
            **common_kwargs,
        )
        self.val_set = MultiCityODDataset(
            city_names=self.train_cities,
            indices=val_indices,
            **common_kwargs,
        )
        print(
            f"Train/Val split: total={total}, train={len(self.train_set)}, "
            f"val={len(self.val_set)}, split={self.train_val_split:.2f}, seed={self.val_seed}"
        )

        if self.test_cities:
            self.test_set = MultiCityODDataset(city_names=self.test_cities, **common_kwargs)
            print(f"Test set: {len(self.test_set)} regions from {self.test_cities}")
        else:
            raise ValueError("test_cities must be specified for testing")

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=multi_city_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=multi_city_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=multi_city_collate_fn,
        )






class ODDataModule(pl.LightningDataModule):
    def __init__(self,
                 od_path: str,
                 neighbors_path: str,
                 hours: Optional[List[int]] = None,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 normalize: bool = True,
                 clip_max: Optional[float] = None,
                 population_path: Optional[str] = None,
                 tfidf_path: Optional[str] = None,
                 spatial_features_path: Optional[str] = None,
                 train_split: float = 0.7,
                 ):
        super().__init__()
        self.od_path = od_path
        self.neighbors_path = neighbors_path
        self.hours = list(range(168)) if hours is None else list(hours)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.clip_max = clip_max
        self.population_path = population_path
        self.tfidf_path = tfidf_path
        self.spatial_features_path = spatial_features_path
        self.train_split = train_split

    def setup(self, stage: Optional[str] = None):

        if hasattr(self, 'train_set') and hasattr(self, 'val_set'):
            return


        temp_dataset = ODMatrixDataset(self.od_path, self.neighbors_path, self.hours,
                                      self.normalize, self.clip_max, compute_stats=False,
                                      population_path=self.population_path,
                                      tfidf_path=self.tfidf_path,
                                      spatial_features_path=self.spatial_features_path)
        total_regions = temp_dataset.N


        train_size = int(total_regions * self.train_split)
        val_size = total_regions - train_size


        import random
        all_indices = list(range(total_regions))
        random.shuffle(all_indices)

        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:]

        print(f"数据分割: 总区域数={total_regions}, 训练集={len(train_indices)}, 验证集={len(val_indices)}")
        print(f"训练集比例: {len(train_indices)/total_regions:.2%}, 验证集比例: {len(val_indices)/total_regions:.2%}")


        self.train_set = ODMatrixDataset(self.od_path, self.neighbors_path, self.hours,
                                         self.normalize, self.clip_max, compute_stats=True,
                                         population_path=self.population_path,
                                         tfidf_path=self.tfidf_path,
                                         spatial_features_path=self.spatial_features_path,
                                         indices=train_indices)


        self.val_set = ODMatrixDataset(self.od_path, self.neighbors_path, self.hours,
                                       self.normalize, self.clip_max, compute_stats=False,
                                       population_path=self.population_path,
                                       tfidf_path=self.tfidf_path,
                                       spatial_features_path=self.spatial_features_path,
                                       indices=val_indices)


        if self.normalize:
            self.val_set.global_mean = self.train_set.global_mean
            self.val_set.global_std = self.train_set.global_std
            print(f"验证集使用训练集的统计信息: 均值={self.val_set.global_mean:.6f}, 标准差={self.val_set.global_std:.6f}")


            if self.population_path is not None:
                self.val_set.population_mean = self.train_set.population_mean
                self.val_set.population_std = self.train_set.population_std
                print(f"验证集使用训练集的人口统计信息: 均值={self.val_set.population_mean:.6f}, 标准差={self.val_set.population_std:.6f}")


            if self.spatial_features_path is not None:
                self.val_set.spatial_mean = self.train_set.spatial_mean
                self.val_set.spatial_std = self.train_set.spatial_std
                print(
                    "验证集使用训练集的空间特征统计信息: "
                    f"mean={self.val_set.spatial_mean}, std={self.val_set.spatial_std}"
                )


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=custom_collate_fn)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'multi'],
                        help='single: 单城市模式(原始), multi: 多城市模式')


    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--train_cities', type=str, nargs='+', default=['Shenzhen', 'Wuhan'])
    parser.add_argument('--test_cities', type=str, nargs='+', default=['Shanghai'])
    parser.add_argument('--norm_stats_path', type=str, default='./configs/norm_stats.yaml')
    parser.add_argument('--use_population', action='store_true', default=True)
    parser.add_argument('--use_tfidf', action='store_true', default=False)
    parser.add_argument('--use_spatial_features', action='store_true', default=True)
    parser.add_argument('--train_val_split', type=float, default=0.85)


    parser.add_argument('--od_path', type=str, default='')
    parser.add_argument('--neighbors_path', type=str, default='')
    parser.add_argument('--population_path', type=str, default=None)
    parser.add_argument('--tfidf_path', type=str, default=None)
    parser.add_argument('--spatial_features_path', type=str, default=None)
    parser.add_argument('--train_split', type=float, default=0.7)


    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hours', type=int, nargs='*', default=None)
    parser.add_argument('--no_normalize', action='store_true')
    parser.add_argument('--clip_max', type=float, default=None)
    args = parser.parse_args()

    if args.mode == 'multi':

        dm = MultiCityODDataModule(
            data_dir=args.data_dir,
            train_cities=args.train_cities,
            test_cities=args.test_cities,
            norm_stats_path=args.norm_stats_path,
            hours=args.hours,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            normalize=not args.no_normalize,
            use_population=args.use_population,
            use_tfidf=args.use_tfidf,
            use_spatial_features=args.use_spatial_features,
            train_val_split=args.train_val_split,
        )
        dm.setup()

        train_dl = dm.train_dataloader()
        train_batch = next(iter(train_dl))

        print(f"\n=== Multi-city DataModule ===")
        print(f"训练集样本数: {len(dm.train_set)}")
        print(f"验证集样本数: {len(dm.val_set)}")
        if hasattr(dm, 'test_set'):
            print(f"测试集样本数: {len(dm.test_set)}")

        sample = dm.train_set[0]
        max_N = dm.norm_stats['max_num_regions']
        T = len(dm.hours)
        print(f"OD 形状: {sample['od'].shape}  (应为 (1, {max_N - 1}, {T}))")
        print(f"Mask 形状: {sample['mask'].shape}, valid={sample['mask'].sum().item()}")
        if sample['condition'] is not None:
            print(f"Condition 形状: {sample['condition'].shape}")
        else:
            print("Condition: None")

        print(f"\n批次 OD 形状: {train_batch['od'].shape}")
        print(f"批次 Mask 形状: {train_batch['mask'].shape}")
        if train_batch['condition'] is not None:
            print(f"批次 Condition 形状: {train_batch['condition'].shape}")

    else:

        dm = ODDataModule(
            od_path=args.od_path,
            neighbors_path=args.neighbors_path,
            hours=args.hours,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            normalize=not args.no_normalize,
            clip_max=args.clip_max,
            population_path=args.population_path,
            tfidf_path=args.tfidf_path,
            spatial_features_path=args.spatial_features_path,
            train_split=args.train_split,
        )
        dm.setup()

        train_dl = dm.train_dataloader()
        train_batch = next(iter(train_dl))
        print(f"训练集样本数: {len(dm.train_set)}")
        print(f"验证集样本数: {len(dm.val_set)}")
        sample = dm.train_set[0]
        print(f"训练集单样本 OD 形状: {sample['od'].shape}")
        if sample['condition'] is not None:
            print(f"训练集单样本 条件 形状: {sample['condition'].shape}")
        print(f"训练集批次 OD 形状: {train_batch['od'].shape}")