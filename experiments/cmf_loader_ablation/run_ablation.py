#!/usr/bin/env python3
"""
experiments/cmf_loader_ablation/run_ablation.py

Top-level runner for the CMF μ-source ablation experiment.

Runs the full 2 × N_method experiment matrix on the 3/7 forget/retain split:
  - mean_source="train"  (legacy: μ includes forget-class samples)
  - mean_source="retain" (fixed:  μ computed from retain set only)

Supports SCRUB, NegGrad+, SalUn, and Random-label CMF variants.
Also supports "original" and "retrain" (retain-only) baselines.

Usage examples
--------------
# Full matrix, 3 seeds, default forget_classes=[0,1,2]
python experiments/cmf_loader_ablation/run_ablation.py \
    --arch resnet18 --dataset cifar10 --num_classes 10 \
    --epochs 10 --batch_size 128 \
    --methods scrub random_label salun naive \
    --mean_sources train retain \
    --seeds 0 1 2 \
    --checkpoint_path ./checkpoints/pre_train/cifar10_resnet18.pt \
    --output_dir ./experiments/cmf_loader_ablation/results

# Quick smoke-test (1 epoch, 1 seed, dry_run)
python experiments/cmf_loader_ablation/run_ablation.py \
    --arch resnet18 --dataset cifar10 --num_classes 10 \
    --epochs 1 --batch_size 128 \
    --methods scrub \
    --mean_sources train retain \
    --seeds 0 \
    --checkpoint_path ./checkpoints/pre_train/cifar10_resnet18.pt \
    --output_dir ./experiments/cmf_loader_ablation/results \
    --dry_run
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import types

# Repo root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import numpy as np

from utils import get_dataset, get_retain_forget_partition, test, load_encoder_ckpt_safely
import evaluation
from evaluation.nc_cmf import ncc_mismatch

from experiments.cmf_loader_ablation.ablation_model import AblationModelModule
from experiments.cmf_loader_ablation.ablation_partition import (
    make_37_partition,
    log_partition_stats,
    make_retain_full_loader,
    DEFAULT_FORGET_CLASSES_37,
)
from experiments.cmf_loader_ablation.ablation_unlearn import (
    ablation_naive_CMF,
    ablation_random_label_CMF,
    ablation_salun_CMF,
    ablation_scrub_CMF,
)

# ---------------------------------------------------------------------------
# Registry of ablation unlearn functions
# ---------------------------------------------------------------------------
ABLATION_METHODS = {
    "naive":        ablation_naive_CMF,
    "random_label": ablation_random_label_CMF,
    "salun":        ablation_salun_CMF,
    "scrub":        ablation_scrub_CMF,
}

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CMF μ-source ablation: train vs retain loader for global mean"
    )

    # Model / dataset
    p.add_argument("--arch",        type=str, default="resnet18")
    p.add_argument("--dataset",     type=str, default="cifar10")
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--data_path",   type=str, default="./data")
    p.add_argument("--gpu_id",      type=int, default=0)

    # Forget partition
    p.add_argument("--forget_classes", type=int, nargs="+",
                   default=DEFAULT_FORGET_CLASSES_37,
                   help="Class indices to forget. Default: 0 1 2 (3/7 split on CIFAR-10).")

    # Checkpoint to load as starting model
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="Path to pre-trained encoder checkpoint (.pt).")

    # Experiment matrix
    p.add_argument("--methods", type=str, nargs="+",
                   choices=list(ABLATION_METHODS.keys()) + ["original", "retrain"],
                   default=["scrub", "random_label", "naive"],
                   help="Unlearning methods to run.")
    p.add_argument("--mean_sources", type=str, nargs="+",
                   choices=["train", "retain"],
                   default=["train", "retain"],
                   help="μ source modes to test.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    # Training hyperparams (mirror production defaults)
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--momentum",     type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)

    # Sample limits (match production defaults for 1/9 benchmarks)
    p.add_argument("--num_forget_samples", type=int, default=15000,
                   help="Training subset size for forget set. Default 15000 (full 3/7 forget).")
    p.add_argument("--num_retain_samples", type=int, default=35000,
                   help="Training subset size for retain set. Default 35000 (full 3/7 retain).")

    # CMF hyperparams
    p.add_argument("--CMF_momentum",  type=float, default=0.9)
    p.add_argument("--temperature",   type=float, default=1.0)
    p.add_argument("--remove_FC",     action="store_true", default=True,
                   help="Remove FC and use CMF classifier (recommended).")
    p.add_argument("--no_normalization", action="store_true", default=False)

    # Evaluation flags
    p.add_argument("--lp_every",          type=int, default=1)
    p.add_argument("--ncc_every",         type=int, default=1)
    p.add_argument("--prob_batch_size",   type=int, default=256)
    p.add_argument("--nc_pool_mode",      type=str, default="avg")

    # SCRUB-specific
    p.add_argument("--scrub_del_bsz",  type=int,   default=512)
    p.add_argument("--scrub_sgda_bsz", type=int,   default=64)
    p.add_argument("--scrub_msteps",   type=int,   default=2)
    p.add_argument("--scrub_epochs",   type=int,   default=3)

    # SalUn-specific
    p.add_argument("--salun_threshold", type=float, default=0.5)

    # Misc
    p.add_argument("--output_dir", type=str, default="./experiments/cmf_loader_ablation/results")
    p.add_argument("--dry_run",    action="store_true", default=False)
    p.add_argument("--no_train_transform", action="store_true", default=False)
    p.add_argument("--val_ratio",  type=float, default=0.1)
    p.add_argument("--grad_norm_clip", type=float, default=None)
    p.add_argument("--sub_set_mode",   action="store_true", default=False)
    p.add_argument("--sub_set_samples", type=int, default=50000)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Build a clean args namespace for each run
# ---------------------------------------------------------------------------

def _make_run_args(base_args: argparse.Namespace, method: str, mean_source: str,
                   seed: int, unlearn_class: list[int]) -> types.SimpleNamespace:
    """Clone base_args and inject run-specific fields."""
    d = vars(base_args).copy()
    d.update({
        "unlearn_method":  method,
        "mean_source":     mean_source,
        "seed":            seed,
        "unlearn_class":   unlearn_class,
        "CMFClassifier":   True,
        # Production defaults that unlearn methods might need:
        "align_coef":      0.0,
        "forget_scale":    0.5,
        "use_margin_forget": False,
        "forget_margin":   0.0,
        "beta_margin":     0.0,
        "lp_every":        base_args.lp_every,
        "ncc_every":       base_args.ncc_every,
        # for linear probe
        "prob_batch_size": base_args.prob_batch_size,
        "probe_batch_size": base_args.prob_batch_size,
        # Some evaluation paths need class_label_names
        "class_label_names": [str(i) for i in range(base_args.num_classes)],
        "pretrained":      False,
        "epochs_or_steps": base_args.epochs,
        "train_transform": not base_args.no_train_transform,
        # eval_every_iter, lp_every_iter, ncc_every_iter — disable iter-based evals
        "eval_every_iter": 0,
        "lp_every_iter":   0,
        "ncc_every_iter":  0,
    })
    return types.SimpleNamespace(**d)


# ---------------------------------------------------------------------------
# Load a fresh model from checkpoint
# ---------------------------------------------------------------------------

def _load_model(args, device) -> AblationModelModule:
    model = AblationModelModule(args).to(device)
    load_encoder_ckpt_safely(model, args.checkpoint_path, device=str(device))
    return model


# ---------------------------------------------------------------------------
# Single run: one method × one mean_source × one seed
# ---------------------------------------------------------------------------

def _single_run(
    base_args: argparse.Namespace,
    method: str,
    mean_source: str,
    seed: int,
    device: torch.device,
    train_dataset,
    test_dataset,
    retain_dataset,
    forget_dataset,
    train_loader,
    test_loader,
    retain_loader,
    forget_loader,
    retain_loader_full,
    test_forget_loader,
    forget_classes: list[int],
) -> dict:
    """Execute a single (method, mean_source, seed) configuration and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_args = _make_run_args(base_args, method, mean_source, seed, forget_classes)

    t_start = time.time()
    print(f"\n{'#'*70}")
    print(f"# method={method}  mean_source={mean_source}  seed={seed}")
    print(f"{'#'*70}")

    model = _load_model(run_args, device)

    if method == "original":
        # No unlearning — just evaluate the pre-trained model as-is
        model.eval()
        model.recompute_cmf(
            loader_full=train_loader,
            loader_retain=retain_loader_full,
            device=device,
            mean_source=mean_source,
        )
        model.history_log = {}

    elif method == "retrain":
        # Retain-only retraining baseline (fine-tune on retain_loader only)
        model = _retrain_baseline(run_args, model, device,
                                  retain_loader, train_loader, retain_loader_full,
                                  test_loader, mean_source)

    else:
        fn = ABLATION_METHODS[method]
        optimizer = torch.optim.SGD(model.parameters(), lr=run_args.lr)
        model = fn(
            args=run_args,
            model=model,
            device=device,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            train_loader=train_loader,
            retain_loader_full=retain_loader_full,
            test_loader=test_loader,
            optimizer=optimizer,
            epochs=run_args.epochs,
            test_forget_loader=test_forget_loader,
            mean_source=mean_source,
        )

    t_end = time.time()

    # --- Final evaluation ---
    model.eval()
    model.recompute_cmf(
        loader_full=train_loader,
        loader_retain=retain_loader_full,
        device=device,
        mean_source=mean_source,
    )

    # Forward accuracy
    r_out, f_out, _ = test(
        model, device, test_loader,
        forget_classes, run_args.class_label_names, run_args.num_classes,
        job_name=f"ablation_{method}_{mean_source}", set_name="Final Test Set"
    )

    # Linear probe
    from utils import get_model as _base_get_model
    def _get_model_fn(a): return AblationModelModule(a).to(device)

    lp_outs = evaluation.run_linear_probe_on_fresh_clone(
        args=run_args,
        get_model_fn=_get_model_fn,
        device_probe=device,
        src_model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=run_args.num_classes,
        bs_probe=run_args.prob_batch_size,
    )

    # NCC
    pool_mode = getattr(run_args, "nc_pool_mode", "avg")
    ncc_r = ncc_mismatch(args=run_args, model=model, train_loader=train_loader,
                         eval_loader=test_loader,        device=device, pool_mode=pool_mode)
    ncc_f = ncc_mismatch(args=run_args, model=model, train_loader=train_loader,
                         eval_loader=test_forget_loader, device=device, pool_mode=pool_mode)

    # μ drift stats from history_log (accumulated during training)
    history = getattr(model, "history_log", {})
    mu_drift = history.get("mu_drift", [])
    mu_drift_avg = float(np.mean(mu_drift))   if mu_drift else float("nan")
    mu_drift_max = float(np.max(mu_drift))    if mu_drift else float("nan")

    row = {
        "method":            method,
        "mean_source":       mean_source,
        "seed":              seed,
        "output_retain_acc": float(r_out),
        "output_forget_acc": float(f_out),
        "probe_retain_acc":  float(lp_outs["acc_test_retain"]),
        "probe_forget_acc":  float(lp_outs["acc_test_forget"]),
        "ncc_retain_acc":    float(ncc_r["ncc_acc"]),
        "ncc_forget_acc":    float(ncc_f["ncc_acc"]),
        "mean_mu_drift_avg": mu_drift_avg,
        "mean_mu_drift_max": mu_drift_max,
        "wall_time_s":       round(t_end - t_start, 1),
        "history":           history,
    }

    print(f"\n[RESULT] {method} | {mean_source} | seed={seed}")
    print(f"  output  : retain={r_out:.4f}  forget={f_out:.4f}")
    print(f"  LP      : retain={lp_outs['acc_test_retain']:.4f}  forget={lp_outs['acc_test_forget']:.4f}")
    print(f"  NCC     : retain={ncc_r['ncc_acc']:.4f}  forget={ncc_f['ncc_acc']:.4f}")
    print(f"  μ drift : avg={mu_drift_avg:.4f}  max={mu_drift_max:.4f}")

    return row


# ---------------------------------------------------------------------------
# Retain-only retraining baseline
# ---------------------------------------------------------------------------

def _retrain_baseline(run_args, model, device, retain_loader, train_loader,
                       retain_loader_full, test_loader, mean_source) -> nn.Module:
    """Fine-tune only on retain set for `run_args.epochs` epochs."""
    from utils import test

    opt = torch.optim.SGD(model.parameters(), lr=run_args.lr,
                          momentum=run_args.momentum, weight_decay=run_args.weight_decay,
                          nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=run_args.epochs)
    mu_prev = None

    mu_drift_list: list[float] = []
    history: dict = {"retain_acc": [], "forget_acc": [], "epoch": [], "mu_drift": []}

    for epoch in range(1, run_args.epochs + 1):
        model.train()
        for x, y in retain_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss   = nn.functional.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            if run_args.dry_run:
                break

        model.eval()
        model.recompute_cmf(
            loader_full=train_loader,
            loader_retain=retain_loader_full,
            device=device,
            mean_source=mean_source,
        )
        mu_curr = model.CMFweights.mu.clone().detach()
        if mu_prev is not None:
            mu_drift_list.append(float((mu_curr - mu_prev).norm().item()))
        mu_prev = mu_curr

        r, f, _ = test(model, device, test_loader,
                       run_args.unlearn_class, run_args.class_label_names, run_args.num_classes,
                       job_name="retrain", set_name=f"Test Set (Epoch {epoch})")
        history["retain_acc"].append(r)
        history["forget_acc"].append(f)
        history["epoch"].append(epoch)
        sched.step()

    history["mu_drift"] = mu_drift_list
    model.history_log   = history
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------
    # Build datasets once (shared across all runs)
    # -------------------------------------------------------------------
    train_dataset, test_dataset = get_dataset(args)

    # Strip validation split so we get the full training set
    if hasattr(train_dataset, "dataset"):
        # Subset from a val-split — unwrap to get the underlying full Dataset
        pass  # keep as-is; retains the training split

    forget_classes = args.forget_classes

    # Run-args stub for partition (needs .unlearn_class, .dataset, .num_classes)
    _stub = types.SimpleNamespace(
        unlearn_class=forget_classes,
        dataset=args.dataset,
        num_classes=args.num_classes,
        arch=args.arch,
        remove_FC=args.remove_FC,
        no_normalization=args.no_normalization,
        CMFClassifier=True,
        CMF_momentum=args.CMF_momentum,
        temperature=args.temperature,
        pretrained=False,
        train_transform=not args.no_train_transform,
    )

    retain_dataset, forget_dataset = make_37_partition(_stub, train_dataset, forget_classes)
    _, test_forget_dataset         = make_37_partition(_stub, test_dataset,   forget_classes)

    stats = log_partition_stats(
        retain_dataset, forget_dataset,
        args.num_classes,
        [str(i) for i in range(args.num_classes)],
        forget_classes,
    )

    # DataLoaders
    train_kw = dict(batch_size=args.batch_size, shuffle=True,
                    num_workers=2, pin_memory=True)
    test_kw  = dict(batch_size=args.batch_size, shuffle=False,
                    num_workers=2, pin_memory=True)

    train_loader        = torch.utils.data.DataLoader(train_dataset,        **train_kw)
    test_loader         = torch.utils.data.DataLoader(test_dataset,         **test_kw)
    retain_loader       = torch.utils.data.DataLoader(retain_dataset,       **train_kw)
    forget_loader       = torch.utils.data.DataLoader(forget_dataset,       **train_kw)
    test_forget_loader  = torch.utils.data.DataLoader(test_forget_dataset,  **test_kw)
    retain_loader_full  = make_retain_full_loader(retain_dataset,
                                                  batch_size=args.batch_size,
                                                  num_workers=2)

    print(f"Partition: forget={len(forget_dataset)} retain={len(retain_dataset)} "
          f"ratio={len(forget_dataset)/len(train_dataset)*100:.1f}%/{len(retain_dataset)/len(train_dataset)*100:.1f}%")

    # -------------------------------------------------------------------
    # Experiment matrix
    # -------------------------------------------------------------------
    all_rows: list[dict] = []

    for method in args.methods:
        for mean_source in (args.mean_sources if method not in ("original", "retrain") else ["train"]):
            for seed in args.seeds:
                row = _single_run(
                    base_args=args,
                    method=method,
                    mean_source=mean_source,
                    seed=seed,
                    device=device,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    retain_dataset=retain_dataset,
                    forget_dataset=forget_dataset,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    retain_loader=retain_loader,
                    forget_loader=forget_loader,
                    retain_loader_full=retain_loader_full,
                    test_forget_loader=test_forget_loader,
                    forget_classes=forget_classes,
                )
                all_rows.append(row)

                # Save incrementally so partial results survive crashes
                _save_results(all_rows, args.output_dir, stats)

    print(f"\n{'='*70}")
    print(f"  All runs complete. Results saved to {args.output_dir}")
    print(f"{'='*70}\n")


def _save_results(rows: list[dict], output_dir: str, stats: dict):
    """Save results to JSON (full) and CSV (summary)."""
    import csv

    # Full JSON (includes history)
    full_path = os.path.join(output_dir, "results_full.json")
    with open(full_path, "w") as f:
        json.dump({"partition_stats": stats, "runs": rows}, f, indent=2, default=str)

    # CSV (summary columns only — matches the spec)
    csv_path = os.path.join(output_dir, "results_summary.csv")
    fields = [
        "method", "mean_source", "seed",
        "output_retain_acc", "output_forget_acc",
        "probe_retain_acc",  "probe_forget_acc",
        "ncc_retain_acc",    "ncc_forget_acc",
        "mean_mu_drift_avg", "mean_mu_drift_max",
        "wall_time_s",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"[Saved] {csv_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
