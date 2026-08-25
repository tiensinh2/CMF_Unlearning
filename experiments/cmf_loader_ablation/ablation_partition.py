"""
experiments/cmf_loader_ablation/ablation_partition.py

Dataset partition helper for the 3/7 CMF loader ablation experiment.

Provides:
  - make_37_partition(dataset, forget_classes)  →  (retain_dataset, forget_dataset)
  - log_partition_stats(retain_ds, forget_ds, num_classes, class_names)

Design choices (documented for reproducibility):
  - Default forget_classes = [0, 1, 2]  on CIFAR-10.
    These correspond to classes: airplane (0), automobile (1), bird (2).
    Three out of ten classes → 15 000 / 50 000 = 30 % forget (3:7 split).
  - Partition is deterministic: no random shuffling, same as the production
    get_retain_forget_partition() in utils.py.

CIFAR-10 class mapping (for documentation):
    0: airplane   1: automobile  2: bird       3: cat        4: deer
    5: dog        6: frog        7: horse      8: ship       9: truck
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from torch.utils.data import Dataset, DataLoader

# Re-use the production SubSet and partition logic
from utils import get_retain_forget_partition


# Default 3/7 forget classes for CIFAR-10
# Classes: airplane(0), automobile(1), bird(2) — first 3 of 10 alphabetically
DEFAULT_FORGET_CLASSES_37 = [0, 1, 2]


def make_37_partition(
    args,
    dataset: Dataset,
    forget_classes: list[int] | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Partition `dataset` into retain and forget subsets using `forget_classes`.

    Args:
        args:           Namespace passed to get_retain_forget_partition() (needs .targets attribute on dataset).
        dataset:        A torchvision-style Dataset with a .targets attribute.
        forget_classes: List of class indices to forget.  Defaults to [0, 1, 2].

    Returns:
        (retain_dataset, forget_dataset) — same types as production partition.

    Sample counts for CIFAR-10 (50 000 train, balanced 5 000/class):
        forget_classes=[0,1,2] → forget=15 000, retain=35 000  (ratio 3:7, 30%:70%)
    """
    if forget_classes is None:
        forget_classes = DEFAULT_FORGET_CLASSES_37

    # Temporarily override unlearn_class so production code sees the right list
    original_unlearn = getattr(args, "unlearn_class", [])
    args.unlearn_class = forget_classes

    retain_dataset, forget_dataset = get_retain_forget_partition(
        args, dataset, forget_classes
    )

    args.unlearn_class = original_unlearn  # restore
    return retain_dataset, forget_dataset


def log_partition_stats(
    retain_ds: Dataset,
    forget_ds: Dataset,
    num_classes: int,
    class_names: list[str] | None = None,
    forget_classes: list[int] | None = None,
) -> dict:
    """
    Print and return a summary of the partition statistics.

    Returns a dict with keys:
        n_forget, n_retain, n_total, ratio_forget, ratio_retain,
        forget_class_counts, retain_class_counts
    """
    if forget_classes is None:
        forget_classes = DEFAULT_FORGET_CLASSES_37

    n_forget = len(forget_ds)
    n_retain = len(retain_ds)
    n_total  = n_forget + n_retain

    ratio_f = n_forget / n_total if n_total else 0.0
    ratio_r = n_retain / n_total if n_total else 0.0

    # Per-class counts in forget set
    forget_counts: dict[int, int] = {c: 0 for c in range(num_classes)}
    for i in range(len(forget_ds)):
        _, label = forget_ds[i]
        lbl = int(label.item()) if torch.is_tensor(label) else int(label)
        forget_counts[lbl] = forget_counts.get(lbl, 0) + 1

    # Per-class counts in retain set
    retain_counts: dict[int, int] = {c: 0 for c in range(num_classes)}
    for i in range(len(retain_ds)):
        _, label = retain_ds[i]
        lbl = int(label.item()) if torch.is_tensor(label) else int(label)
        retain_counts[lbl] = retain_counts.get(lbl, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Partition statistics (3/7 ablation split)")
    print(f"{'='*60}")
    print(f"  Total samples : {n_total:>6d}")
    print(f"  Forget set    : {n_forget:>6d}  ({ratio_f*100:.1f}%)")
    print(f"  Retain set    : {n_retain:>6d}  ({ratio_r*100:.1f}%)")
    print(f"  Exact ratio   : {n_forget}:{n_retain} ≈ {n_forget//(n_forget+n_retain-n_retain) if n_retain else '?'}:7")
    print(f"\n  Forget classes: {forget_classes}")
    if class_names:
        named = {c: class_names[c] for c in forget_classes if c < len(class_names)}
        print(f"  Forget names  : {named}")
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':>6}  {'Name':15}  {'Forget':>8}  {'Retain':>8}")
    for c in range(num_classes):
        name = class_names[c] if class_names and c < len(class_names) else f"cls{c}"
        marker = " ← FORGET" if c in forget_classes else ""
        print(f"  {c:>6}  {name:15}  {forget_counts[c]:>8}  {retain_counts[c]:>8}{marker}")
    print(f"{'='*60}\n")

    return {
        "n_forget": n_forget,
        "n_retain": n_retain,
        "n_total": n_total,
        "ratio_forget": ratio_f,
        "ratio_retain": ratio_r,
        "forget_class_counts": forget_counts,
        "retain_class_counts": retain_counts,
    }


def make_retain_full_loader(retain_dataset: Dataset, batch_size: int = 256,
                            num_workers: int = 2) -> DataLoader:
    """
    Return a DataLoader over the FULL retain dataset (no subsampling).
    This is the loader passed to recompute_cmf() when mean_source="retain".
    """
    return DataLoader(
        retain_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
