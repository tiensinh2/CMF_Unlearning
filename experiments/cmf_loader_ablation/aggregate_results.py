#!/usr/bin/env python3
"""
experiments/cmf_loader_ablation/aggregate_results.py

Aggregates the raw per-seed CSV produced by run_ablation.py into a final
results table (mean +/- std across seeds) and writes:
  - results_aggregated.csv      - machine-readable
  - results_table.md            - Markdown summary table
  - findings_summary.txt        - written analysis per method

Usage:
    python experiments/cmf_loader_ablation/aggregate_results.py \
        --input_csv  ./experiments/cmf_loader_ablation/results/results_summary.csv \
        --output_dir ./experiments/cmf_loader_ablation/results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, List


# Metrics to aggregate (column names matching run_ablation.py output)
METRIC_COLS = [
    "output_retain_acc",
    "output_forget_acc",
    "probe_retain_acc",
    "probe_forget_acc",
    "ncc_retain_acc",
    "ncc_forget_acc",
    "mean_mu_drift_avg",
    "mean_mu_drift_max",
]

# Shorter names for the Markdown table header
METRIC_SHORT = {
    "output_retain_acc":  "OutR",
    "output_forget_acc":  "OutF",
    "probe_retain_acc":   "LPR",
    "probe_forget_acc":   "LPF",
    "ncc_retain_acc":     "NCCR",
    "ncc_forget_acc":     "NCCF",
    "mean_mu_drift_avg":  "muDriftAvg",
    "mean_mu_drift_max":  "muDriftMax",
}


def _mean_std(vals: List[float]) -> tuple[float, float]:
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    mu = sum(vals) / n
    if n == 1:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, math.sqrt(var)


def load_csv(path: str) -> List[Dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            typed = {}
            for k, v in row.items():
                try:
                    typed[k] = float(v)
                except (ValueError, TypeError):
                    typed[k] = v
            rows.append(typed)
    return rows


def aggregate(rows: List[Dict]) -> List[Dict]:
    """
    Group rows by (method, mean_source) and compute mean +/- std across seeds.
    Returns a list of summary dicts.
    """
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        key = (str(row["method"]), str(row["mean_source"]))
        groups[key].append(row)

    summary = []
    for (method, mean_source), group_rows in sorted(groups.items()):
        entry: Dict = {"method": method, "mean_source": mean_source,
                       "n_seeds": len(group_rows)}
        for col in METRIC_COLS:
            vals = [float(r[col]) for r in group_rows if col in r]
            mu, std = _mean_std(vals)
            entry[f"{col}_mean"] = round(mu, 4)
            entry[f"{col}_std"]  = round(std, 4)
        summary.append(entry)
    return summary


def write_aggregated_csv(summary: List[Dict], path: str):
    if not summary:
        return
    fields = list(summary[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    print(f"[Saved] {path}")


def write_markdown_table(summary: List[Dict], path: str):
    """Produce a Markdown table formatted as mean+/-std for each metric."""
    lines = []
    lines.append("# CMF mu-Source Ablation Results - 3/7 Split (CIFAR-10)")
    lines.append("")
    lines.append(
        "Metrics are mean +/- std across seeds.  "
        "**OutR/OutF** = output-level retain/forget accuracy.  "
        "**LPR/LPF** = linear-probe retain/forget accuracy.  "
        "**NCCR/NCCF** = NCC retain/forget accuracy.  "
        "**muDriftAvg/Max** = epoch-mean/max ||mu_e - mu_{e-1}||_2."
    )
    lines.append("")

    # Header
    col_headers = ["Method", "mu-Source", "N"] + [METRIC_SHORT[m] for m in METRIC_COLS]
    sep = [":---"] + [":---"] + ["---:"] + ["---:"] * len(METRIC_COLS)
    lines.append("| " + " | ".join(col_headers) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    for row in summary:
        cells = [
            str(row["method"]),
            str(row["mean_source"]),
            str(row["n_seeds"]),
        ]
        for col in METRIC_COLS:
            mu_val = row.get(f"{col}_mean", float("nan"))
            std    = row.get(f"{col}_std",  float("nan"))
            if math.isnan(mu_val):
                cells.append("n/a")
            else:
                cells.append(f"{mu_val:.4f}+/-{std:.4f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Saved] {path}")


def write_findings(summary: List[Dict], path: str):
    """
    Write a brief automated findings summary comparing mean_source=train vs retain
    for each method, focussing on the three key questions:
      1. Does retain-only mu improve retain accuracy?
      2. Does retain-only mu change forget accuracy? (should be minimal)
      3. Does retain-only mu reduce mu drift?
    """
    # Index by method
    by_method: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for row in summary:
        by_method[row["method"]][row["mean_source"]] = row

    lines = []
    lines.append("CMF mu-Source Ablation - Findings Summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        "Dataset: CIFAR-10, 3/7 forget/retain split "
        "(forget classes 0,1,2 = airplane/automobile/bird)."
    )
    lines.append(
        "Hypothesis: when 3 of 10 CIFAR-10 classes are forgotten (30% of training data), "
        "contamination of the global mean mu by forget-class features may distort "
        "the CMF classifier weights for retain classes, reducing retain accuracy and "
        "increasing epoch-to-epoch mu instability.  Using a retain-only mu "
        "(mean_source='retain') should isolate this effect."
    )
    lines.append("")

    for method, sources in sorted(by_method.items()):
        lines.append(f"--- {method} ---")

        train_r  = sources.get("train",  {})
        retain_r = sources.get("retain", {})

        def _val(d, col):
            v = d.get(f"{col}_mean", float("nan"))
            s = d.get(f"{col}_std",  float("nan"))
            if math.isnan(v):
                return "N/A"
            return f"{v:.4f}+/-{s:.4f}"

        if method in ("original", "retrain"):
            lines.append(f"  Baseline (no mean_source split). "
                         f"OutR={_val(train_r,'output_retain_acc')}  "
                         f"OutF={_val(train_r,'output_forget_acc')}")
            lines.append("")
            continue

        if not train_r or not retain_r:
            lines.append("  Incomplete data - skipping comparison.")
            lines.append("")
            continue

        def _delta(col):
            a = train_r.get(f"{col}_mean", float("nan"))
            b = retain_r.get(f"{col}_mean", float("nan"))
            if math.isnan(a) or math.isnan(b):
                return float("nan")
            return b - a   # positive = retain better

        delta_r   = _delta("output_retain_acc")
        delta_f   = _delta("output_forget_acc")
        delta_lpr = _delta("probe_retain_acc")
        delta_lrf = _delta("probe_forget_acc")
        delta_mu  = _delta("mean_mu_drift_avg")   # negative = retain reduces drift

        lines.append(f"  Train  : OutR={_val(train_r,'output_retain_acc')}  "
                     f"OutF={_val(train_r,'output_forget_acc')}  "
                     f"muDriftAvg={_val(train_r,'mean_mu_drift_avg')}")
        lines.append(f"  Retain : OutR={_val(retain_r,'output_retain_acc')}  "
                     f"OutF={_val(retain_r,'output_forget_acc')}  "
                     f"muDriftAvg={_val(retain_r,'mean_mu_drift_avg')}")

        # Interpretation
        thr = 0.005  # 0.5 pp threshold for "meaningful"
        retain_improved = delta_r  >  thr
        retain_hurt     = delta_r  < -thr
        forget_changed  = abs(delta_f) > thr
        drift_reduced   = delta_mu < -1e-4

        if retain_improved:
            lines.append(f"  [OK] retain-only mu IMPROVED retain accuracy by {delta_r:+.4f} (output).")
        elif retain_hurt:
            lines.append(f"  [!!] retain-only mu HURT retain accuracy by {delta_r:+.4f} (output).")
        else:
            lines.append(f"  [~~] retain-only mu had negligible effect on retain accuracy ({delta_r:+.4f}).")

        if forget_changed:
            lines.append(f"  [!!] forget accuracy changed by {delta_f:+.4f} - may indicate "
                         "coupling between mu and forget-class classifier geometry.")
        else:
            lines.append(f"  [OK] forget accuracy was stable ({delta_f:+.4f} change) - "
                         "forget classes' own m_c are unaffected by mu source, as expected.")

        if drift_reduced:
            lines.append(f"  [OK] mu drift was lower under retain-only source "
                         f"(delta_avg={delta_mu:+.4f}), consistent with the hypothesis.")
        else:
            lines.append(f"  [~~] mu drift showed no clear reduction (delta_avg={delta_mu:+.4f}).")

        lines.append("")

    lines.append(
        "Overall conclusion: see results_table.md for the full numbers.\n"
        "If retain-only mu consistently improves retain accuracy and reduces mu drift\n"
        "without changing forget accuracy under the 3/7 split, it is a recommended\n"
        "default for high-forget-ratio settings.  If the effect is negligible here,\n"
        "it confirms the original 1/9-split assumption that mu contamination is minor."
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Saved] {path}")


def main():
    p = argparse.ArgumentParser(description="Aggregate ablation results")
    p.add_argument("--input_csv",  type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows    = load_csv(args.input_csv)
    summary = aggregate(rows)

    write_aggregated_csv(
        summary,
        os.path.join(args.output_dir, "results_aggregated.csv"),
    )
    write_markdown_table(
        summary,
        os.path.join(args.output_dir, "results_table.md"),
    )
    write_findings(
        summary,
        os.path.join(args.output_dir, "findings_summary.txt"),
    )


if __name__ == "__main__":
    main()
