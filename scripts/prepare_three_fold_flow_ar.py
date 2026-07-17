
import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml

CITIES = ["Shenzhen", "Wuhan", "Shanghai"]


def fold_name(test_city):
    return f"test_{test_city}"


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/multi_city_flow_ar_config.yaml")
    parser.add_argument("--base_vae_config", default="configs/multi_city_vae_config.yaml")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="outputs/three_fold_flow_ar")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--vae_epochs", type=int, default=500)
    parser.add_argument("--ar_epochs", type=int, default=1000)
    parser.add_argument("--vae_patience", type=int, default=30)
    parser.add_argument("--ar_patience", type=int, default=10)
    parser.add_argument("--kl_weight", type=float, default=None)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    base_ar = load_yaml(args.base_config)
    base_vae = load_yaml(args.base_vae_config)

    manifest = {"cities": CITIES, "folds": []}
    for test_city in CITIES:
        train_cities = [c for c in CITIES if c != test_city]
        name = fold_name(test_city)
        fold_dir = out_root / name
        fold_dir.mkdir(parents=True, exist_ok=True)

        norm_path = fold_dir / "norm_stats.yaml"
        ar_config_path = fold_dir / "config.yaml"
        vae_config_path = fold_dir / "vae_config.yaml"
        vae_output_dir = fold_dir / "pretrained_vae"

        cmd = [
            args.python,
            "scripts/compute_norm_stats.py",
            "--data_dir",
            args.data_dir,
            "--train_cities",
            *train_cities,
            "--test_cities",
            test_city,
            "--output",
            str(norm_path),
        ]
        print("[prepare] computing norm stats:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        with open(norm_path, "r", encoding="utf-8") as f:
            norm_stats = yaml.safe_load(f)
        num_regions = int(norm_stats["max_num_regions"]) - 1

        vae_cfg = copy.deepcopy(base_vae)
        vae_cfg.setdefault("data", {})
        vae_cfg["data"]["data_dir"] = args.data_dir
        vae_cfg["data"]["train_cities"] = train_cities
        vae_cfg["data"]["test_cities"] = [test_city]
        vae_cfg["data"]["norm_stats_path"] = str(norm_path)
        vae_cfg["data"]["train_val_split"] = 0.8
        vae_cfg.setdefault("model", {})["num_regions"] = num_regions
        vae_cfg.setdefault("training", {})
        vae_cfg["training"]["max_epochs"] = args.vae_epochs
        vae_cfg["training"]["max_steps"] = -1
        if args.kl_weight is not None:
            vae_cfg["training"]["kl_weight"] = float(args.kl_weight)
        vae_cfg.setdefault("trainer", {})
        vae_cfg["trainer"]["check_val_every_n_epoch"] = 10
        vae_cfg["trainer"]["num_sanity_val_steps"] = 0
        vae_cfg["trainer"]["enable_progress_bar"] = False
        vae_cfg.setdefault("callbacks", {})
        vae_cfg["callbacks"]["model_checkpoint"] = {
            "monitor": "val/cpc",
            "mode": "max",
            "save_top_k": 1,
            "save_last": False,
            "auto_insert_metric_name": False,
            "save_on_train_epoch_end": False,
        }
        vae_cfg["callbacks"]["early_stopping"] = {
            "monitor": "val/cpc",
            "mode": "max",
            "patience": args.vae_patience,
            "min_delta": 0.0,
            "check_on_train_epoch_end": False,
        }
        vae_cfg.setdefault("logging", {})
        vae_cfg["logging"]["save_dir"] = str(fold_dir / "vae_logs")
        vae_cfg["logging"]["name"] = f"vae_{name}"
        vae_cfg["logging"]["version"] = None
        write_yaml(vae_config_path, vae_cfg)

        ar_cfg = copy.deepcopy(base_ar)
        ar_cfg.setdefault("data", {})
        ar_cfg["data"]["data_dir"] = args.data_dir
        ar_cfg["data"]["train_cities"] = train_cities
        ar_cfg["data"]["test_cities"] = [test_city]
        ar_cfg["data"]["norm_stats_path"] = str(norm_path)
        ar_cfg["data"]["train_val_split"] = 0.8
        ar_cfg.setdefault("vae", {})
        ar_cfg["vae"]["ckpt"] = "__FOLD_VAE_CKPT_TO_BE_FILLED__"
        ar_cfg["vae"]["num_regions"] = num_regions
        ar_cfg.setdefault("training", {})
        ar_cfg["training"]["epochs"] = args.ar_epochs
        ar_cfg["training"]["min_epochs"] = 0
        ar_cfg["training"]["min_checkpoint_epoch"] = 0
        ar_cfg["training"]["max_steps"] = -1
        ar_cfg.setdefault("callbacks", {})
        ar_cfg["callbacks"]["model_checkpoint"] = {
            "monitor": "val/cpc",
            "mode": "max",
            "save_top_k": 1,
            "save_last": False,
            "auto_insert_metric_name": False,
            "save_on_train_epoch_end": False,
        }
        ar_cfg["callbacks"]["early_stopping"] = {
            "monitor": "val/cpc",
            "mode": "max",
            "patience": args.ar_patience,
            "min_delta": 0.0,
            "check_on_train_epoch_end": False,
            "min_monitor_epoch": 0,
        }
        ar_cfg.setdefault("trainer", {})
        ar_cfg["trainer"]["check_val_every_n_epoch"] = 10
        ar_cfg["trainer"]["limit_val_batches"] = 1.0
        ar_cfg["trainer"]["num_sanity_val_steps"] = 0
        ar_cfg.setdefault("logging", {})
        ar_cfg["logging"]["save_dir"] = str(fold_dir / "logs")
        ar_cfg["logging"]["name"] = name
        ar_cfg["logging"]["version"] = None
        write_yaml(ar_config_path, ar_cfg)

        manifest["folds"].append(
            {
                "name": name,
                "train_cities": train_cities,
                "val_city": test_city,
                "test_city": test_city,
                "config": str(ar_config_path),
                "vae_config": str(vae_config_path),
                "vae_output_dir": str(vae_output_dir),
                "kl_weight": args.kl_weight,
                "norm_stats": str(norm_path),
                "output_dir": str(fold_dir),
            }
        )
        print(f"[prepare] wrote {vae_config_path}", flush=True)
        print(f"[prepare] wrote {ar_config_path}", flush=True)

    manifest_path = out_root / "manifest.yaml"
    write_yaml(manifest_path, manifest)
    print(f"[prepare] manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
