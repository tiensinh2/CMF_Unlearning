#!/usr/bin/env python3
"""
notebooks/reeval_nc_ncc.py — Part A-8: Re-evaluate existing CMF checkpoints
with the fixed evaluation/nc.py and produce a before/after comparison table.

Usage:
    python notebooks/reeval_nc_ncc.py \
        --dataset cifar10 \
        --arch resnet18 \
        --data-path ./data \
        --methods grad_ascent_descent_CMF_RemoveFC random_label_CMF_RemoveFC \
        --unlearn-classes 0 1 2 3 4 5 6 7 8 9

The script:
  1. Loads each checkpoint from checkpoints/<method>/<dataset>_<arch>/<class>.pt
  2. Runs nc_metrics() with the NEW code (CMF-aware W + geometry)
  3. Loads any previously saved .json result and reads the old NC3/NCC values
  4. Emits a CSV:  method | unlearn_class | NC3_old | NC3_new | NCC_ret_old |
                   NCC_ret_new | NCC_for_old | NCC_for_new | delta_NCC_ret |
                   delta_NCC_for | flagged
  5. Flags rows where |NCC_retain_new − NCC_retain_old| > 5 pp
     or          |NCC_forget_new  − NCC_forget_old|  > 5 pp
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import get_dataset, get_model, get_retain_forget_partition
from evaluation.nc import nc_metrics, get_classifier_weights


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--arch",    default="resnet18")
    p.add_argument("--data-path", default="./data")
    p.add_argument("--methods", nargs="+",
                   default=["grad_ascent_descent_CMF_RemoveFC",
                             "random_label_CMF_RemoveFC",
                             "salun_CMF_RemoveFC",
                             "scrub_CMF_RemoveFC"])
    p.add_argument("--unlearn-classes", nargs="+", type=int,
                   default=list(range(10)))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gpu-id",     type=int, default=0)
    p.add_argument("--out-csv",    default="notebooks/reeval_nc_ncc_comparison.csv")
    p.add_argument("--seed",       type=int, default=1234)
    return p.parse_args()


def load_checkpoint(ckpt_path: str, model, device):
    if not os.path.exists(ckpt_path):
        return False
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
    return True


def read_old_metrics(json_path: str) -> dict:
    """Read previously saved metrics from .json result file."""
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        data = json.load(f)
    # The old format may store nc_metrics_penultimate or nc_metrics
    nc = data.get("nc_metrics_penultimate") or data.get("nc_metrics") or {}
    # Try to find NCC values in history_log (from unlearn run)
    if not nc and "history_log" in data:
        pass  # no old NC in run log
    return nc


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Minimal args namespace for get_dataset / get_model
    import types
    ns = types.SimpleNamespace(
        dataset=args.dataset,
        arch=args.arch,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_classes=10 if args.dataset == "cifar10" else (
            100 if args.dataset == "cifar100" else 200),
        train_transform=False,
        no_train_transform=True,
        sub_set_mode=False,
        seed=args.seed,
        unlearn_class=[],
        # CMF settings
        CMFClassifier=True,
        remove_FC=True,
        CMF_momentum=0.9,
        temperature=1.0,
        no_normalization=False,
        pretrained=False,
        loss="CE",
        optimizer="SGD",
        learning_rate=1e-4,
        weight_decay=5e-4,
        max_epochs=200,
    )
    if args.dataset == "cifar10":
        ns.class_label_names = [str(i) for i in range(10)]
    elif args.dataset == "cifar100":
        ns.class_label_names = [str(i) for i in range(100)]
    else:
        ns.class_label_names = [str(i) for i in range(200)]

    dataset_train, dataset_test = get_dataset(ns)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    rows = []
    flagged_count = 0

    for method in args.methods:
        for cls in args.unlearn_classes:
            ckpt_path = (
                f"./checkpoints/{method}/{args.dataset}_{args.arch}/{cls}.pt"
            )
            json_path = (
                f"./checkpoints/{method}/{args.dataset}_{args.arch}/{cls}.json"
            )

            ns.unlearn_class = [cls]
            ns.unlearn_method = method

            # Build model
            model = get_model(ns, device)

            # Load checkpoint
            if not load_checkpoint(ckpt_path, model, device):
                print(f"  [SKIP] {ckpt_path} not found")
                continue

            model.eval()
            # Recompute CMF geometry (needed for _preprocess_feats_for_cmf)
            retain_dataset, _ = get_retain_forget_partition(
                ns, dataset_train, [cls]
            )
            retain_loader = torch.utils.data.DataLoader(
                retain_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=2
            )
            if hasattr(model, "recompute_cmf"):
                model.recompute_cmf(retain_loader, device=device)

            retain_classes = [c for c in range(ns.num_classes) if c != cls]
            forget_classes = [cls]

            # NEW metrics
            new_m = nc_metrics(
                model, test_loader, device,
                retain_classes, forget_classes,
                args=ns,
            )

            # OLD metrics (from saved json)
            old_m = read_old_metrics(json_path)

            # Extract comparable values
            nc3_old  = old_m.get("NC3_all") or old_m.get("NC3") or old_m.get("NC3_duality")
            nc3_new  = new_m.get("NC3_duality")
            ncc_ret_old = old_m.get("NCC_retain") or old_m.get("ncc_retain")
            ncc_for_old = old_m.get("NCC_forget") or old_m.get("ncc_forget")
            ncc_ret_new = new_m.get("NCC_retain")
            ncc_for_new = new_m.get("NCC_forget")

            def _safe_delta(a, b):
                if a is None or b is None:
                    return None
                return round(float(b) - float(a), 4)

            d_ret = _safe_delta(ncc_ret_old, ncc_ret_new)
            d_for = _safe_delta(ncc_for_old, ncc_for_new)
            flagged = (
                (d_ret is not None and abs(d_ret) > 0.05) or
                (d_for is not None and abs(d_for) > 0.05)
            )
            if flagged:
                flagged_count += 1

            row = {
                "method":         method,
                "unlearn_class":  cls,
                "NC3_old":        round(float(nc3_old), 6) if nc3_old is not None else "N/A",
                "NC3_new":        round(float(nc3_new), 6) if nc3_new is not None else "N/A",
                "NCC_retain_old": round(float(ncc_ret_old), 4) if ncc_ret_old is not None else "N/A",
                "NCC_retain_new": round(float(ncc_ret_new), 4) if ncc_ret_new is not None else "N/A",
                "NCC_forget_old": round(float(ncc_for_old), 4) if ncc_for_old is not None else "N/A",
                "NCC_forget_new": round(float(ncc_for_new), 4) if ncc_for_new is not None else "N/A",
                "delta_NCC_ret":  d_ret if d_ret is not None else "N/A",
                "delta_NCC_for":  d_for if d_for is not None else "N/A",
                "flagged":        "YES" if flagged else "",
            }
            rows.append(row)
            status = "⚠ FLAGGED" if flagged else "ok"
            print(f"  [{status}] {method} cls={cls}: "
                  f"NCC_ret {ncc_ret_old} → {ncc_ret_new}  "
                  f"NCC_for {ncc_for_old} → {ncc_for_new}")

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    if rows:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[Done] {len(rows)} rows written to {args.out_csv}")
        print(f"       {flagged_count} rows flagged (|delta| > 5 pp)")
    else:
        print("[Done] No checkpoints found.")


if __name__ == "__main__":
    main()
