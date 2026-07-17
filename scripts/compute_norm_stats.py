
"""
Pre-compute normalization statistics for multi-city OD dataset training.

Usage:
    python scripts/compute_norm_stats.py \
        --data_dir ./data \
        --train_cities Shenzhen Wuhan \
        --test_cities Shanghai \
        --output ./configs/norm_stats.yaml



Normalization pipeline (consistent with ODMatrixDataset):
    OD:         clip_max -> log1p -> z-score (mean, std)
    Population: z-score (mean, std)
    Spatial:    [dx, dy, distance] z-score; sin_theta/cos_theta keep raw values
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import yaml
from typing import List, Dict, Optional, Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

NON_TFIDF_COLS: Set[str] = {
    'tile_ID', 'lat', 'lon', 'area', 'pop', 'pop_per',
    'total_poi', 'geometry', 'ID',
}


def get_city_paths(data_dir: str, city: str) -> Dict[str, str]:
    city_dir = os.path.join(data_dir, city)
    return {
        'od': os.path.join(city_dir, f'{city}_od_matrix_week_168.npz'),
        'neighbors': os.path.join(city_dir, f'{city}_neighbors_sorted.npz'),
        'spatial_features': os.path.join(city_dir, f'{city}_spatial_features.npy'),
        'population': os.path.join(city_dir, f'{city}_pop_filtered.csv'),
        'tfidf': os.path.join(city_dir, f'{city}_kind_tfidf.csv'),
    }


def get_poi_columns(tfidf_path: str) -> List[str]:
    df = pd.read_csv(tfidf_path, nrows=0)
    return [c for c in df.columns if c not in NON_TFIDF_COLS]


def compute_norm_stats(
    data_dir: str,
    train_cities: List[str],
    test_cities: List[str],
    clip_max: Optional[float] = None,
    hours: Optional[List[int]] = None,
) -> dict:
    if hours is None:
        hours = list(range(168))

    all_cities = train_cities + test_cities


    city_info: Dict[str, dict] = {}
    all_poi_col_sets: List[Set[str]] = []

    for city in all_cities:
        paths = get_city_paths(data_dir, city)
        neigh = np.load(paths['neighbors'])
        N = int(neigh['tile_ids'].shape[0])
        city_info[city] = {'N': N, 'paths': paths}

        if os.path.exists(paths['tfidf']):
            cols = get_poi_columns(paths['tfidf'])
            all_poi_col_sets.append(set(cols))
            city_info[city]['poi_cols'] = cols
            print(f"  {city}: N={N}, POI features={len(cols)}")
        else:
            print(f"  {city}: N={N}, no tfidf file")

    common_tfidf_cols = sorted(set.intersection(*all_poi_col_sets)) if all_poi_col_sets else []
    max_N = max(info['N'] for info in city_info.values())

    print(f"\nMax N across all cities: {max_N}")
    print(f"Common tfidf POI columns: {len(common_tfidf_cols)}")


    print("\n--- Computing OD statistics (training cities) ---")
    all_od_vals: List[torch.Tensor] = []

    for city in train_cities:
        info = city_info[city]
        N = info['N']
        paths = info['paths']

        neigh = np.load(paths['neighbors'])
        tile_ids = neigh['tile_ids'].astype(np.int32)
        neighbors = neigh['neighbors'].astype(np.int32)

        od_data = np.load(paths['od'])
        od_keys = sorted(od_data.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        od_mats = [np.asarray(od_data[od_keys[h]], dtype=np.float32) for h in hours]

        for idx in range(N):
            origin = int(tile_ids[idx])
            dests = neighbors[idx].astype(np.int64)
            for mat in od_mats:
                vec = torch.from_numpy(mat[origin, dests])
                if clip_max is not None:
                    vec = torch.clamp(vec, max=float(clip_max))
                vec = torch.log1p(vec)
                all_od_vals.append(vec)

        print(f"  {city}: {N} regions processed")

    all_od = torch.cat(all_od_vals, dim=0)
    od_mean = float(torch.mean(all_od))
    od_std = float(torch.std(all_od, unbiased=False))
    if od_std < 1e-8:
        od_std = 1.0

    print(f"  OD mean={od_mean:.6f}, std={od_std:.6f}")
    print(f"  OD log1p range: [{float(all_od.min()):.4f}, {float(all_od.max()):.4f}]")
    z_od = (all_od - od_mean) / od_std
    print(f"  OD z-score range: [{float(z_od.min()):.4f}, {float(z_od.max()):.4f}]")
    del all_od, z_od, all_od_vals


    print("\n--- Computing population statistics (training cities) ---")
    all_pop: List[torch.Tensor] = []
    for city in train_cities:
        pop_path = city_info[city]['paths']['population']
        if os.path.exists(pop_path):
            pop_df = pd.read_csv(pop_path).sort_values('tile_ID').reset_index(drop=True)
            all_pop.append(torch.from_numpy(pop_df['pop'].values.astype(np.float32)))

    pop_mean = pop_std = None
    if all_pop:
        pop_cat = torch.cat(all_pop, dim=0)
        non_zero_pop = pop_cat[pop_cat > 0]
        pop_mean = float(torch.mean(non_zero_pop))
        pop_std = float(torch.std(non_zero_pop, unbiased=False))
        if pop_std < 1e-8:
            pop_std = 1.0
        print(f"  Population mean={pop_mean:.4f}, std={pop_std:.4f}")
        print(f"  Population range: [{float(pop_cat.min()):.4f}, {float(pop_cat.max()):.4f}]")
        del pop_cat
    else:
        print("  No population data found")
    del all_pop


    print("\n--- Computing spatial feature statistics (training cities) ---")
    all_dx: List[torch.Tensor] = []
    all_dy: List[torch.Tensor] = []
    all_dist: List[torch.Tensor] = []
    for city in train_cities:
        info = city_info[city]
        spatial_path = info['paths']['spatial_features']
        if not os.path.exists(spatial_path):
            continue
        spatial = np.load(spatial_path).astype(np.float32)
        if spatial.ndim != 3 or spatial.shape[2] != 5:
            raise ValueError(f"{city} spatial_features shape expected (N,N,5), got {spatial.shape}")
        neigh = np.load(info['paths']['neighbors'])
        tile_ids = neigh['tile_ids'].astype(np.int32)
        neighbors = neigh['neighbors'].astype(np.int32)
        for idx in range(info['N']):
            origin = int(tile_ids[idx])
            dests = neighbors[idx].astype(np.int64)
            feats = spatial[origin, dests, :]
            all_dx.append(torch.from_numpy(feats[:, 0]))
            all_dy.append(torch.from_numpy(feats[:, 1]))
            all_dist.append(torch.from_numpy(feats[:, 2]))

    spatial_stats = None
    if all_dx and all_dy and all_dist:
        dx_cat = torch.cat(all_dx, dim=0)
        dy_cat = torch.cat(all_dy, dim=0)
        dist_cat = torch.cat(all_dist, dim=0)
        non_zero = dist_cat[dist_cat > 0]

        dx_mean = float(torch.mean(dx_cat))
        dy_mean = float(torch.mean(dy_cat))
        dist_mean = float(torch.mean(non_zero))
        dx_std = float(torch.std(dx_cat, unbiased=False))
        dy_std = float(torch.std(dy_cat, unbiased=False))
        dist_std = float(torch.std(non_zero, unbiased=False))

        dx_std = max(dx_std, 1e-8)
        dy_std = max(dy_std, 1e-8)
        dist_std = max(dist_std, 1e-8)

        spatial_stats = {
            'dx': {'mean': dx_mean, 'std': dx_std},
            'dy': {'mean': dy_mean, 'std': dy_std},
            'distance': {'mean': dist_mean, 'std': dist_std},
            'angle': {
                'sin_theta': {'normalized': False},
                'cos_theta': {'normalized': False},
            },
        }

        print(f"  dx mean={dx_mean:.4f}, std={dx_std:.4f}")
        print(f"  dy mean={dy_mean:.4f}, std={dy_std:.4f}")
        print(f"  distance mean={dist_mean:.4f}, std={dist_std:.4f}")
        print("  sin_theta/cos_theta: keep raw values")
        del dx_cat, dy_cat, dist_cat, non_zero
    else:
        print("  No spatial_features data found")
    del all_dx, all_dy, all_dist


    config = {
        'train_cities': train_cities,
        'test_cities': test_cities,
        'cities': {
            city: {'num_regions': info['N']}
            for city, info in city_info.items()
        },
        'max_num_regions': int(max_N),
        'normalization': {
            'od': {'mean': od_mean, 'std': od_std},
        },
        'common_tfidf_columns': common_tfidf_cols,
    }

    if pop_mean is not None:
        config['normalization']['population'] = {'mean': pop_mean, 'std': pop_std}
    if spatial_stats is not None:
        config['normalization']['spatial_features'] = spatial_stats
    if clip_max is not None:
        config['clip_max'] = float(clip_max)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute normalization statistics for multi-city OD training"
    )
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--train_cities', type=str, nargs='+', required=True)
    parser.add_argument('--test_cities', type=str, nargs='+', default=[])
    parser.add_argument('--clip_max', type=float, default=None)
    parser.add_argument('--hours', type=int, nargs='*', default=None)
    parser.add_argument('--output', type=str, default='./configs/norm_stats.yaml')
    args = parser.parse_args()

    print(f"Train cities: {args.train_cities}")
    print(f"Test cities:  {args.test_cities}")
    print(f"Data dir:     {args.data_dir}")
    print(f"Clip max:     {args.clip_max}")
    print()

    config = compute_norm_stats(
        data_dir=args.data_dir,
        train_cities=args.train_cities,
        test_cities=args.test_cities,
        clip_max=args.clip_max,
        hours=args.hours,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nNormalization stats saved to: {args.output}")


if __name__ == '__main__':
    main()
