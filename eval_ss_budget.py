import argparse
import csv
import glob
import json
import os
import sys

import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import metrics as cosp_metrics  # noqa: E402


DEFAULT_ASSET_DIRS = [
    "assets/camoscan500",
    "assets/cosp_hat_baseline_len8",
    "assets/cosp_camo_memory_p4_a0p05_len8",
    "assets/cosp_camo_align_p4_a0p05_w0p02_len8_15k",
    "assets/cosp_camo_align_p4_a0p05_w0p05_len8_15k",
    "assets/cosp_camo_memory_p4_a0p05_uncert_alpha002_len8_15k",
    "assets/camo_align_gate_w0p01_frozen_15k",
    "assets/cosp_camo_align_p4_a0p05_w0p02_free_15k",
    "assets/cosp_camo_align_p4_a0p05_w0p02_len8_30k_v2",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def truncate_scanpaths(scanpaths, max_len):
    out = []
    for sp in scanpaths:
        xs = list(sp.get("X", []))
        ys = list(sp.get("Y", []))
        ts = list(sp.get("T", [300.0] * len(xs)))
        n = min(len(xs), len(ys), len(ts), max_len)
        if n < 1:
            continue
        item = dict(sp)
        item["X"] = xs[:n]
        item["Y"] = ys[:n]
        item["T"] = ts[:n]
        item["length"] = n
        out.append(item)
    return out


def mean_len(scanpaths):
    lens = [min(len(sp.get("X", [])), len(sp.get("Y", []))) for sp in scanpaths]
    return float(np.mean(lens)) if lens else float("nan")


def safe_score(scanpaths, clusters, max_step, truncate_gt):
    try:
        return float(cosp_metrics.get_seq_score(
            scanpaths, clusters, max_step=max_step, truncate_gt=truncate_gt))
    except Exception as exc:
        print(f"[warn] SS failed: {exc}")
        return float("nan")


def collect_asset_dirs(args):
    if args.asset_dirs:
        return args.asset_dirs
    if args.glob:
        return sorted(glob.glob(args.glob))
    return DEFAULT_ASSET_DIRS


def evaluate_asset(asset_dir, clusters, max_len):
    pred_path = os.path.join(asset_dir, "predictions_FV.json")
    if not os.path.exists(pred_path):
        print(f"[skip] {asset_dir}: predictions_FV.json not found")
        return None

    preds_full = load_json(pred_path)
    preds_budget = truncate_scanpaths(preds_full, max_len)
    metric_path = os.path.join(asset_dir, "metrics_FV.json")
    old_metrics = load_json(metric_path) if os.path.exists(metric_path) else {}

    row = {
        "method": os.path.basename(os.path.normpath(asset_dir)),
        "asset_dir": asset_dir,
        "num_preds": len(preds_full),
        "avg_pred_len_raw": mean_len(preds_full),
        "avg_pred_len_budget": mean_len(preds_budget),
        "SS_pred8_fullGT": safe_score(
            preds_budget, clusters, max_step=max_len, truncate_gt=False),
        "SS_pred8_GT8": safe_score(
            preds_budget, clusters, max_step=max_len, truncate_gt=True),
        "SS_raw_fullGT": safe_score(
            preds_full, clusters, max_step=20, truncate_gt=False),
        "old_seq_score_max": old_metrics.get("Greedy_FV_seq_score_max"),
        "old_seq_score_8steps": old_metrics.get("Greedy_FV_seq_score_8steps"),
        "old_cIG": old_metrics.get("Greedy_FV_cIG"),
        "old_cNSS": old_metrics.get("Greedy_FV_cNSS"),
        "old_cAUC": old_metrics.get("Greedy_FV_cAUC"),
        "old_MM_Avg": old_metrics.get("Greedy_FV_MM_Avg"),
        "old_MM_Direction": old_metrics.get("Greedy_FV_MM_Direction"),
        "old_MM_Position": old_metrics.get("Greedy_FV_MM_Position"),
    }
    return row


def fmt(value):
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def print_table(rows):
    cols = [
        "method",
        "SS_pred8_fullGT",
        "SS_pred8_GT8",
        "SS_raw_fullGT",
        "old_seq_score_max",
        "old_seq_score_8steps",
        "old_cNSS",
        "old_cIG",
        "old_MM_Avg",
        "avg_pred_len_raw",
    ]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|")
    for row in rows:
        print("| " + " | ".join(fmt(row.get(c, "")) for c in cols) + " |")


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate FV scanpath SS under a fixed prediction budget.")
    parser.add_argument(
        "--cluster-path",
        default="datasets/CamoScan-500/clusters.npy",
        help="Path to CamoScan-500 clusters.npy.")
    parser.add_argument(
        "--max-len",
        type=int,
        default=8,
        help="Prediction budget used for Pred@K metrics.")
    parser.add_argument(
        "--asset-dirs",
        nargs="*",
        help="Asset dirs containing predictions_FV.json.")
    parser.add_argument(
        "--glob",
        help="Glob pattern for asset dirs. Ignored when --asset-dirs is set.")
    parser.add_argument(
        "--out-csv",
        default="assets/ss_budget_eval.csv",
        help="CSV path to save results.")
    args = parser.parse_args()

    clusters = np.load(args.cluster_path, allow_pickle=True).item()
    rows = []
    for asset_dir in collect_asset_dirs(args):
        row = evaluate_asset(asset_dir, clusters, args.max_len)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda r: (
        -np.nan_to_num(r["SS_pred8_fullGT"], nan=-1.0),
        -np.nan_to_num(r["old_cNSS"], nan=-1.0),
    ))
    print_table(rows)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    if rows:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved -> {args.out_csv}")


if __name__ == "__main__":
    main()
