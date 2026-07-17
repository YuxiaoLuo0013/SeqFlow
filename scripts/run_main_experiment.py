
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

SEED = 1000
KL_WEIGHT = 0.001
FOLDS = ["test_Shenzhen", "test_Wuhan", "test_Shanghai"]
METRICS = [
    "cpc",
    "mae",
    "mse",
    "rmse",
    "nrmse",
    "jsd_inflow",
    "jsd_outflow",
    "jsd_odflow",
    "t_mse",
    "generated_sum",
    "ground_truth_sum",
]


def run(cmd, log_path=None):
    print("[cmd]", " ".join(map(str, cmd)), flush=True)
    if log_path is None:
        subprocess.run(cmd, check=True)
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
    return proc, handle


def wait_all(processes):
    failed = []
    handles = []
    for value in processes.values():
        if isinstance(value, tuple):
            handles.append(value[1])
    try:
        while processes:
            for name, value in list(processes.items()):
                proc = value[0] if isinstance(value, tuple) else value
                ret = proc.poll()
                if ret is None:
                    continue
                print(f"[done] {name}: exit={ret}", flush=True)
                if ret != 0:
                    failed.append((name, ret))
                del processes[name]
            if processes:
                time.sleep(30)
    finally:
        for handle in handles:
            handle.close()
    if failed:
        raise RuntimeError(f"failed processes: {failed}")


def patch_yaml(path: Path, ckpt_path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("vae", {})["ckpt"] = ckpt_path
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def patch_fold_vae_ckpts(run_dir: Path):
    for fold in FOLDS:
        fold_dir = run_dir / fold
        best_file = fold_dir / "pretrained_vae" / "best_checkpoint.txt"
        ckpt = best_file.read_text(encoding="utf-8").strip()
        if not ckpt or not Path(ckpt).exists():
            raise FileNotFoundError(f"VAE checkpoint not found: {ckpt}")
        patch_yaml(fold_dir / "config.yaml", ckpt)
        print(f"[vae] {fold}: {ckpt}", flush=True)


def summarize(run_dir: Path):
    rows = []
    for fold in FOLDS:
        metrics_path = run_dir / fold / "metrics.json"
        with open(metrics_path, "r", encoding="utf-8") as f:
            rows.append({"seed": SEED, "fold": fold, **json.load(f)})

    fieldnames = ["seed", "fold", "city", "checkpoint", "matrix_path", *METRICS]
    with open(run_dir / "main_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    import numpy as np

    summary = {"seed": SEED, "kl_weight": KL_WEIGHT, "num_folds": len(rows), "mean": {}, "std": {}, "folds": rows}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        summary["mean"][metric] = float(values.mean())
        summary["std"][metric] = float(values.std(ddof=0))

    with open(run_dir / "main_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    print(json.dumps(summary["mean"], indent=2), flush=True)
    print(f"[summary] wrote {run_dir / 'main_summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs/main_seqflow_seed1000_kl0p001")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip_vae", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.output_root) / f"seed_{SEED}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run([
        args.python,
        "scripts/prepare_three_fold_flow_ar.py",
        "--output_dir",
        str(run_dir),
        "--python",
        args.python,
        "--kl_weight",
        str(KL_WEIGHT),
    ])

    if not args.skip_vae:
        vae_processes = {}
        for fold in FOLDS:
            fold_dir = run_dir / fold
            vae_processes[fold] = run([
                args.python,
                "train_vae.py",
                "--config",
                str(fold_dir / "vae_config.yaml"),
                "--output_dir",
                str(fold_dir / "pretrained_vae"),
                "--seed",
                str(SEED),
            ], fold_dir / "train_vae.log")
        wait_all(vae_processes)

    patch_fold_vae_ckpts(run_dir)

    ar_processes = {}
    for fold in FOLDS:
        fold_dir = run_dir / fold
        ar_processes[fold] = run([
            args.python,
            "scripts/run_fold_flow_ar.py",
            "--config",
            str(fold_dir / "config.yaml"),
            "--output_dir",
            str(fold_dir),
            "--seed",
            str(SEED),
        ], fold_dir / "run.log")
    wait_all(ar_processes)
    summarize(run_dir)


if __name__ == "__main__":
    main()
