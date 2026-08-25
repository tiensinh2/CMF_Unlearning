"""
Smoke test for aggregate_results (no torch/sklearn required).
Run: python experiments/cmf_loader_ablation/smoke_test_aggregate.py
"""
import sys, os, math, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.cmf_loader_ablation.aggregate_results import (
    aggregate, write_markdown_table, write_findings, write_aggregated_csv, _mean_std,
)

rows = [
    {"method": "scrub", "mean_source": "train",  "seed": 0,
     "output_retain_acc": 0.80, "output_forget_acc": 0.20,
     "probe_retain_acc":  0.82, "probe_forget_acc":  0.18,
     "ncc_retain_acc":    0.78, "ncc_forget_acc":    0.22,
     "mean_mu_drift_avg": 0.05, "mean_mu_drift_max": 0.10, "wall_time_s": 10},
    {"method": "scrub", "mean_source": "train",  "seed": 1,
     "output_retain_acc": 0.81, "output_forget_acc": 0.19,
     "probe_retain_acc":  0.83, "probe_forget_acc":  0.17,
     "ncc_retain_acc":    0.79, "ncc_forget_acc":    0.21,
     "mean_mu_drift_avg": 0.06, "mean_mu_drift_max": 0.11, "wall_time_s": 11},
    {"method": "scrub", "mean_source": "retain", "seed": 0,
     "output_retain_acc": 0.83, "output_forget_acc": 0.21,
     "probe_retain_acc":  0.84, "probe_forget_acc":  0.19,
     "ncc_retain_acc":    0.80, "ncc_forget_acc":    0.23,
     "mean_mu_drift_avg": 0.02, "mean_mu_drift_max": 0.04, "wall_time_s": 12},
    {"method": "scrub", "mean_source": "retain", "seed": 1,
     "output_retain_acc": 0.84, "output_forget_acc": 0.20,
     "probe_retain_acc":  0.85, "probe_forget_acc":  0.18,
     "ncc_retain_acc":    0.81, "ncc_forget_acc":    0.22,
     "mean_mu_drift_avg": 0.03, "mean_mu_drift_max": 0.05, "wall_time_s": 13},
    {"method": "original", "mean_source": "train", "seed": 0,
     "output_retain_acc": 0.90, "output_forget_acc": 0.90,
     "probe_retain_acc":  0.91, "probe_forget_acc":  0.91,
     "ncc_retain_acc":    0.88, "ncc_forget_acc":    0.88,
     "mean_mu_drift_avg": float("nan"), "mean_mu_drift_max": float("nan"),
     "wall_time_s": 5},
]

summary = aggregate(rows)
print(f"aggregate OK: {len(summary)} groups")
for row in summary:
    print(f"  {row['method']:12s} | {row['mean_source']:6s} | n={row['n_seeds']} | "
          f"retainMean={row['output_retain_acc_mean']:.4f} +/- {row['output_retain_acc_std']:.4f}")

with tempfile.TemporaryDirectory() as td:
    write_aggregated_csv(summary, os.path.join(td, "agg.csv"))
    write_markdown_table(summary, os.path.join(td, "table.md"))
    write_findings(summary, os.path.join(td, "findings.txt"))

    with open(os.path.join(td, "table.md")) as fh:
        print()
        print(fh.read())
    with open(os.path.join(td, "findings.txt")) as fh:
        print(fh.read())

# Verify retain > train in this synthetic data
scrub_train  = next(r for r in summary if r["method"] == "scrub" and r["mean_source"] == "train")
scrub_retain = next(r for r in summary if r["method"] == "scrub" and r["mean_source"] == "retain")
assert scrub_retain["output_retain_acc_mean"] > scrub_train["output_retain_acc_mean"], "retain should be higher in synthetic data"
assert scrub_retain["mean_mu_drift_avg_mean"] < scrub_train["mean_mu_drift_avg_mean"], "drift should be lower under retain source"
print("ASSERTIONS PASSED")
