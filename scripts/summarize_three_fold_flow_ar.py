
import argparse
import csv
import json
import time
from pathlib import Path

METRICS = ["cpc", "mae", "mse", "rmse", "nrmse", "jsd_inflow", "jsd_outflow", "jsd_odflow", "t_mse", "generated_sum", "ground_truth_sum"]


def load_metrics(output_dir: Path):
    rows = []
    for path in sorted(output_dir.glob("test_*/metrics.json")):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def summarize(rows):
    summary = {"num_folds": len(rows), "folds": rows, "mean": {}, "std": {}}
    for key in METRICS:
        vals = [float(r[key]) for r in rows if key in r]
        if vals:
            import numpy as np
            arr = np.asarray(vals, dtype=float)
            summary["mean"][key] = float(arr.mean())
            summary["std"][key] = float(arr.std(ddof=0))
    return summary


def write_outputs(output_dir: Path, summary):
    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    with open(output_dir / "summary_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        for key in METRICS:
            writer.writerow([key, summary["mean"].get(key, ""), summary["std"].get(key, "")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/three_fold_flow_ar")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--expected", type=int, default=3)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    while True:
        rows = load_metrics(output_dir)
        print(f"[summary] found {len(rows)}/{args.expected} fold metric files", flush=True)
        if len(rows) >= args.expected or not args.wait:
            if len(rows) == 0:
                raise SystemExit("No metrics.json files found.")
            summary = summarize(rows)
            write_outputs(output_dir, summary)
            print(json.dumps(summary["mean"], indent=2), flush=True)
            print(f"[summary] wrote {output_dir / 'summary_metrics.json'}", flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
