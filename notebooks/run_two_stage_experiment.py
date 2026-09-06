#!/usr/bin/env python3
"""
notebooks/run_two_stage_experiment.py — Part B experiment matrix runner.

Runs cmf_two_stage_unlearn over the 4-config matrix:
  1. cmf_static (E=50)                         [baseline]
  2. cmf_two_stage, k=2, retain_only
  3. cmf_two_stage, k=3, retain_only
  4. cmf_two_stage, k=3, retain_plus_forget

For each base_method in [scrub, random_label, neggrad_plus, salun].
3 seeds: 1234, 42, 7.

Output:
  notebooks/results/two_stage_results.csv
  notebooks/results/two_stage_summary.md

Usage:
  python notebooks/run_two_stage_experiment.py \
      --dataset cifar10 \
      --arch resnet18 \
      --data-path ./data \
      --unlearn-class 0 \
      --base-methods scrub \
      --seeds 1234
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OFFICIAL_REPO_COMMIT
from utils import get_dataset, get_model, get_retain_forget_partition, test
import evaluation
from unlearn import unlear_func


# ─────────────────────────────────────────────────────────────────────────────
# Configs
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_MATRIX = [
    # (config_name, total_epochs, k, phase2_data)
    ("cmf_static_E50",         50, 0,  "retain_only"),         # baseline: k=0 → all Stage1
    ("cmf_two_stage_k2_ret",   50, 2,  "retain_only"),
    ("cmf_two_stage_k3_ret",   50, 3,  "retain_only"),
    ("cmf_two_stage_k3_r_f",   50, 3,  "retain_plus_forget"),
]

BASE_METHOD_TO_UNLEARN_KEY = {
    "scrub":        "scrub_CMF_RemoveFC",
    "random_label": "random_label_CMF_RemoveFC",
    "neggrad_plus": "grad_ascent_descent_CMF_RemoveFC",
    "salun":        "salun_CMF_RemoveFC",
}

SEEDS = [1234, 42, 7]

LR_MAP = {
    "scrub_CMF_RemoveFC":               5e-3,
    "random_label_CMF_RemoveFC":        1e-4,
    "grad_ascent_descent_CMF_RemoveFC": 1e-4,
    "salun_CMF_RemoveFC":               2e-4,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",       default="cifar10")
    p.add_argument("--arch",          default="resnet18")
    p.add_argument("--data-path",     default="./data")
    p.add_argument("--unlearn-class", type=int, default=0)
    p.add_argument("--base-methods",  nargs="+",
                   default=["scrub", "random_label", "neggrad_plus", "salun"])
    p.add_argument("--seeds",         nargs="+", type=int, default=SEEDS)
    p.add_argument("--gpu-id",        type=int, default=0)
    p.add_argument("--dry-run",       action="store_true")
    p.add_argument("--out-dir",       default="notebooks/results")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Build args namespace
# ─────────────────────────────────────────────────────────────────────────────

def build_args(dataset, arch, data_path, unlearn_class, method_key, seed,
               total_epochs, k, phase2_data, device, dry_run=False):
    num_classes = {"cifar10": 10, "cifar100": 100, "tinyimagenet": 200}[dataset]
    samples_per_class = {"cifar10": 5000, "cifar100": 500, "tinyimagenet": 500}[dataset]
    total = {"cifar10": 50000, "cifar100": 50000, "tinyimagenet": 100000}[dataset]
    retain_n = total - samples_per_class
    forget_n = samples_per_class

    ns = types.SimpleNamespace(
        dataset=dataset,
        arch=arch,
        data_path=data_path,
        batch_size=128,
        test_batch_size=256,
        unlearn_class=[unlearn_class],
        unlearn_method=method_key,
        seed=seed,
        epochs_or_steps=total_epochs,
        # CMF
        CMFClassifier=True,
        remove_FC=True,
        CMF_momentum=0.9,
        temperature=1.0,
        no_normalization=False,
        pretrained=False,
        loss="CE",
        optimizer="SGD",
        learning_rate=LR_MAP.get(method_key, 1e-4),
        lr=LR_MAP.get(method_key, 1e-4),
        momentum=0.9,
        weight_decay=5e-4,
        max_epochs=total_epochs,
        nesterov=True,
        # Subset sizes
        num_retain_samples=retain_n,
        num_forget_samples=forget_n,
        # Misc
        grad_norm_clip=1.0,
        num_classes=num_classes,
        class_label_names=[str(i) for i in range(num_classes)],
        train_transform=True,
        no_train_transform=False,
        sub_set_mode=False,
        # LP
        lp_every=total_epochs // 5,    # run LP 5x per experiment
        prob_batch_size=256,
        # Two-stage specific
        total_epochs=total_epochs,
        final_stage_epochs=k,
        phase2_data=phase2_data,
        mean_source="train",
        base_method=method_key,
        # Misc
        dry_run=dry_run,
        # scrub params
        scrub_del_bsz=64,
        scrub_sgda_bsz=64,
        scrub_msteps=2,
        scrub_epochs=3,
        # salun params
        salun_threshold=0.5,
        # tarun params (Table 4 line 2366/2426: impair_lr = same as main lr)
        tarun_impair_lr=5e-5,   # Table 4 CIFAR-10 UNSIR full-model lr=5e-5
        tarun_samples_per_class=1000,
        # official repo tracking
        official_repo_commit=OFFICIAL_REPO_COMMIT,
    )
    return ns


# ─────────────────────────────────────────────────────────────────────────────
# Run one experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_one(args_ns, device, dataset_train, dataset_test, is_two_stage: bool):
    torch.manual_seed(args_ns.seed)

    model = get_model(args_ns, device)

    # Load CMF pretrained checkpoint
    ckpt_base = (
        f"./checkpoints/CMF_FT_RemoveFC/"
        f"{args_ns.dataset}_{args_ns.arch}.pt"
    )
    if os.path.exists(ckpt_base):
        state = torch.load(ckpt_base, map_location=device)
        model.load_state_dict(state)
        print(f"[run_one] Loaded base CMF ckpt: {ckpt_base}")
    else:
        print(f"[run_one] WARNING: CMF base ckpt not found at {ckpt_base}")

    retain_dataset, forget_dataset = get_retain_forget_partition(
        args_ns, dataset_train, args_ns.unlearn_class
    )
    _, test_forget_dataset = get_retain_forget_partition(
        args_ns, dataset_test, args_ns.unlearn_class
    )

    kw = {"num_workers": 2, "pin_memory": True}
    retain_loader = torch.utils.data.DataLoader(
        retain_dataset, batch_size=args_ns.batch_size, shuffle=True, **kw
    )
    forget_loader = torch.utils.data.DataLoader(
        forget_dataset, batch_size=args_ns.batch_size, shuffle=True, **kw
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args_ns.batch_size, shuffle=True, **kw
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args_ns.test_batch_size, shuffle=False, **kw
    )
    test_forget_loader = torch.utils.data.DataLoader(
        test_forget_dataset, batch_size=args_ns.test_batch_size, shuffle=False
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args_ns.lr,
        momentum=args_ns.momentum, weight_decay=args_ns.weight_decay,
        nesterov=True,
    )

    t0 = time.time()
    method_key = "cmf_two_stage" if is_two_stage else args_ns.unlearn_method
    fn = unlear_func[method_key]
    model = fn(
        args=args_ns, model=model, device=device,
        retain_loader=retain_loader, forget_loader=forget_loader,
        train_loader=train_loader, test_loader=test_loader,
        optimizer=optimizer, epochs=args_ns.epochs_or_steps,
        test_forget_loader=test_forget_loader,
    )
    wall = (time.time() - t0) / 60.0
    return model, wall


# ─────────────────────────────────────────────────────────────────────────────
# Extract result rows from history_log
# ─────────────────────────────────────────────────────────────────────────────

def extract_rows(model, config_name, base_method, k, phase2_data,
                 mean_source, seed, wall):
    rows = []
    h = getattr(model, "history_log", {})

    if "two_stage_log" in h:
        for entry in h["two_stage_log"]:
            rows.append({
                "config": config_name,
                "base_method": base_method,
                "k": k,
                "phase2_data": phase2_data,
                "mean_source": mean_source,
                "seed": seed,
                "epoch_stage": entry.get("epoch_stage"),
                "output_retain_acc": entry.get("output_retain_acc"),
                "output_forget_acc": entry.get("output_forget_acc"),
                "probe_retain_acc": entry.get("probe_retain_acc"),
                "probe_forget_acc": entry.get("probe_forget_acc"),
                "ncc_retain_acc": entry.get("ncc_retain_acc"),
                "ncc_forget_acc": entry.get("ncc_forget_acc"),
                "wall_clock_minutes": entry.get("wall_clock_minutes", wall),
            })
    else:
        # Static CMF: history in retain_acc / forget_acc lists
        epochs = h.get("epoch", [])
        ret_accs = h.get("retain_acc", [])
        for_accs = h.get("forget_acc", [])
        lp_ret = h.get("LP_retain_acc", [])
        lp_for = h.get("LP_forget_acc", [])
        for i, ep in enumerate(epochs):
            rows.append({
                "config": config_name,
                "base_method": base_method,
                "k": k,
                "phase2_data": phase2_data,
                "mean_source": mean_source,
                "seed": seed,
                "epoch_stage": f"S1_ep{ep}",
                "output_retain_acc": ret_accs[i] if i < len(ret_accs) else None,
                "output_forget_acc": for_accs[i] if i < len(for_accs) else None,
                "probe_retain_acc": lp_ret[i] if i < len(lp_ret) else None,
                "probe_forget_acc": lp_for[i] if i < len(lp_for) else None,
                "ncc_retain_acc": None,
                "ncc_forget_acc": None,
                "wall_clock_minutes": wall,
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Markdown summary
# ─────────────────────────────────────────────────────────────────────────────

def make_markdown_summary(all_rows: list, out_dir: str):
    import statistics

    # Group: (config, base_method, k, phase2_data) → final-epoch rows
    groups: dict = {}
    for r in all_rows:
        key = (r["config"], r["base_method"], r["k"], r["phase2_data"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    lines = [
        "# CMF Two-Stage Experiment Results",
        "",
        f"Official repo commit: `{OFFICIAL_REPO_COMMIT}`",
        "",
        "## Summary Table (mean ± std over seeds, final epoch)",
        "",
        "| config | base_method | k | phase2 | output_ret | output_for | probe_ret | probe_for | ncc_ret | ncc_for |",
        "|--------|-------------|---|--------|-----------|-----------|-----------|-----------|---------|---------|",
    ]

    def _stat(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        if len(vals) == 1:
            return f"{vals[0]:.3f}"
        return f"{statistics.mean(vals):.3f}±{statistics.stdev(vals):.3f}"

    for (cfg, bm, k, p2), rows in sorted(groups.items()):
        # Get last S2 epoch or last S1 epoch for each seed
        seed_finals: dict = {}
        for r in rows:
            s = r["seed"]
            ep = r["epoch_stage"]
            if s not in seed_finals or (ep > seed_finals[s]["epoch_stage"]):
                seed_finals[s] = r
        finals = list(seed_finals.values())
        lines.append(
            f"| {cfg} | {bm} | {k} | {p2} "
            f"| {_stat([r['output_retain_acc'] for r in finals])}"
            f"| {_stat([r['output_forget_acc'] for r in finals])}"
            f"| {_stat([r['probe_retain_acc'] for r in finals])}"
            f"| {_stat([r['probe_forget_acc'] for r in finals])}"
            f"| {_stat([r['ncc_retain_acc'] for r in finals])}"
            f"| {_stat([r['ncc_forget_acc'] for r in finals])} |"
        )

    lines += [
        "",
        "## Answers",
        "",
        "### (a) Does Stage 2 recover retain accuracy for k=2 vs k=3?",
        "> *(Fill in after running experiments — compare output_retain_acc at*",
        "> *end of Stage 1 vs end of Stage 2 for k=2 and k=3.)*",
        "",
        "### (b) Does forget accuracy rise during Stage 2?",
        "> *(Fill in: compare NCC_forget and probe_forget at Stage-1 end vs each*",
        "> *Stage-2 epoch, and note the gain per epoch.)*",
        "",
        "### (c) Is retain_plus_forget better or worse than retain_only in Stage 2?",
        "> *(Fill in: compare NCC_retain/forget between phase2_data conditions.)*",
        "",
    ]

    md_path = os.path.join(out_dir, "two_stage_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Summary] written → {md_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cli = parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)

    device = torch.device(
        f"cuda:{cli.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}  |  official_repo_commit={OFFICIAL_REPO_COMMIT}")

    # Datasets (loaded once)
    _tmp = types.SimpleNamespace(
        dataset=cli.dataset, arch=cli.arch, data_path=cli.data_path,
        batch_size=128, train_transform=True, no_train_transform=False,
        sub_set_mode=False, seed=1234, unlearn_class=[cli.unlearn_class],
        num_classes={"cifar10":10,"cifar100":100,"tinyimagenet":200}[cli.dataset],
        CMFClassifier=True, remove_FC=True, CMF_momentum=0.9,
        temperature=1.0, no_normalization=False, pretrained=False,
        loss="CE", optimizer="SGD", learning_rate=1e-4, lr=1e-4,
        momentum=0.9, weight_decay=5e-4, max_epochs=200,
    )
    dataset_train, dataset_test = get_dataset(_tmp)

    all_rows = []
    csv_path = os.path.join(cli.out_dir, "two_stage_results.csv")
    csv_fields = [
        "config", "base_method", "k", "phase2_data", "mean_source",
        "seed", "epoch_stage",
        "output_retain_acc", "output_forget_acc",
        "probe_retain_acc",  "probe_forget_acc",
        "ncc_retain_acc",    "ncc_forget_acc",
        "wall_clock_minutes",
    ]

    for base_method_short in cli.base_methods:
        method_key = BASE_METHOD_TO_UNLEARN_KEY.get(base_method_short, base_method_short)

        for (cfg_name, E, k, p2) in EXPERIMENT_MATRIX:
            is_two_stage = (k > 0)
            # cmf_static uses the CMF method key directly; two_stage wraps it
            run_method_key = "cmf_two_stage" if is_two_stage else method_key

            for seed in cli.seeds:
                print(f"\n{'─'*70}")
                print(f"Config={cfg_name}  base={base_method_short}  k={k}  "
                      f"p2={p2}  seed={seed}")
                print(f"{'─'*70}")

                args_ns = build_args(
                    dataset=cli.dataset, arch=cli.arch,
                    data_path=cli.data_path,
                    unlearn_class=cli.unlearn_class,
                    method_key=method_key,
                    seed=seed,
                    total_epochs=E,
                    k=k,
                    phase2_data=p2,
                    device=device,
                    dry_run=cli.dry_run,
                )

                try:
                    model, wall = run_one(
                        args_ns, device,
                        dataset_train, dataset_test,
                        is_two_stage=is_two_stage,
                    )
                    rows = extract_rows(
                        model, cfg_name, base_method_short,
                        k, p2, "train", seed, wall,
                    )
                    all_rows.extend(rows)
                    print(f"  → {len(rows)} log entries, wall={wall:.1f} min")
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    import traceback; traceback.print_exc()

                # Incremental CSV write
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(all_rows)

    print(f"\n[Done] {len(all_rows)} total rows → {csv_path}")
    make_markdown_summary(all_rows, cli.out_dir)


if __name__ == "__main__":
    main()
