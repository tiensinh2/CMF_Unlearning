"""
config.py — Centralised experiment configuration for CMF_Unlearning.

Paper: "An Illusion of Unlearning? Assessing Machine Unlearning Through
        Internal Representations"  (Gao, Unal, Rangamani, Zhu — AISTATS 2026
        arXiv:2604.08271v1)

Official GitHub repo: https://github.com/ycgao1/CMF_Unlearning

SOURCE PRIORITY (matches paper_hparams.py — single source of truth)
=====================================================================
  1. Table 4 (Appendix D.1, PDF lines 2307–2434) — primary for every
     method's LR, epochs, batch size, and method-specific flags.
  2. §A.4 text — only for parameters Table 4 does not cover.
  3. Standard ResNet-18/CIFAR defaults — only when neither source specifies.

PREVIOUS DECISION REVERSED (2026-09)
=====================================
  config.py previously documented "use shell-script values where they disagree
  with Table 4", with PRETRAIN_LR=5e-2 and SVD_alpha_r=100.  That decision is
  reversed.  All values here now match Table 4 exactly, consistent with
  paper_hparams.py (which passes the 72/72 _validate_table4.py check).

  Stale checkpoints produced under the old shell-derived values have been
  archived to archive_stale_shell_hparams/checkpoints_shell_derived/.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Official repo commit — log in every experiment config dump
OFFICIAL_REPO_COMMIT = "main"  # TODO: replace with pinned SHA after first run


# ─────────────────────────────────────────────────────────────
# Value-change log  (old shell-derived → new Table-4)
# Matches SHELL_TO_TABLE4_CHANGES in paper_hparams.py
# ─────────────────────────────────────────────────────────────
SHELL_TO_TABLE4_CHANGES = {
    "pretrain_lr_cifar10": {
        "old_shell": 5e-2,
        "new_table4": 1e-2,   # Table 4 line 2308
        "impact": "All pre-train checkpoints regenerated",
    },
    "random_label_lr_cifar10": {
        "old_shell": 1e-2,
        "new_table4": 1e-4,   # Table 4 line 2330
    },
    "salun_lr_cifar10": {
        "old_shell": 1e-2,
        "new_table4": 1e-4,   # Table 4 line 2339
    },
    "neggrad_lr_cifar10": {
        "old_shell": 1e-3,
        "new_table4": 1e-4,   # Table 4 line 2348
    },
    "unsir_lr_cifar10": {
        "old_shell": "2e-3 (classifier-only)",
        "new_table4": 5e-5,   # Table 4 line 2366 (full-model)
        "mode_change": "classifier-only → full-model",
    },
    "svd_alpha_r_cifar10": {
        "old_shell": 100,
        "new_table4": 1000,   # Table 4 line 2375
    },
    "svd_alpha_f_cifar10": {
        "old_shell": 3,
        "new_table4": 30,     # Table 4 line 2375
    },
    "retrain_epochs": {
        "old_shell": 50,
        "new_table4": 200,    # Table 4 line 2314
    },
    "random_label_cmf_lr_cifar10": {
        "old_shell": "1e-4 (single) / 2e-3 (multi)",
        "new_table4": 2e-3,   # Table 4 line 2390 (no single/multi split)
    },
    "salun_cmf_lr_cifar10": {
        "old_shell": "2e-4 (single) / 2e-3 (multi)",
        "new_table4": 2e-3,   # Table 4 line 2399
    },
    "scrub_cmf_cifar100_lr": {
        "old_shell": "1e-3 (multi)",
        "new_table4": 5e-3,   # Table 4 line 2420 (no single/multi split)
    },
}


# ─────────────────────────────────────────────────────────────
# Experiment environment  (paper §A.1)
# ─────────────────────────────────────────────────────────────
ENV = {
    "os":          "Ubuntu 20.04 (kernel 5.4)",
    "gpus":        "8× NVIDIA RTX A5000 (24 GB each)",
    "cpu_cores":   64,
    "ram_gb":      252,
    "python":      "3.11",
    "pytorch":     "2.5.1",
    "cuda":        "12.1",
}


# ─────────────────────────────────────────────────────────────
# Datasets  (paper §A.2)
# ─────────────────────────────────────────────────────────────
DATASETS: Dict[str, dict] = {
    "cifar10": {
        "num_classes":        10,
        "train_samples":      50_000,
        "test_samples":       10_000,
        "samples_per_class":  5_000,
        "arch":               "resnet18",
        "input_size":         32,
    },
    "cifar100": {
        "num_classes":        100,
        "train_samples":      50_000,
        "test_samples":       10_000,
        "samples_per_class":  500,
        "arch":               "resnet18",
        "input_size":         32,
    },
    "tinyimagenet": {
        "num_classes":        200,
        "train_samples":      100_000,
        "val_samples":        10_000,
        "samples_per_class":  500,
        "arch":               "resnet50",
        "input_size":         64,
    },
}


# ─────────────────────────────────────────────────────────────
# Unlearning scenarios  (paper §A.3)
# ─────────────────────────────────────────────────────────────
UNLEARN_SCENARIOS: Dict[str, Dict[str, List]] = {
    "cifar10": {
        "single": [[c] for c in range(10)],
        "multi": [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 5, 9],
            [2, 4, 8],
        ],
    },
    "cifar100": {
        "single": [[0], [1], [2], [3], [5]],
        "multi": [
            [3, 15, 19, 21, 31, 38, 42, 43, 88, 97],
            [47, 52, 54, 56, 59, 62, 70, 82, 92, 96],
            [5, 20, 22, 25, 39, 40, 84, 86, 87, 94],
            [8, 13, 41, 48, 59, 69, 81, 85, 89, 90],
            [1, 4, 30, 32, 55, 67, 72, 73, 91, 95],
        ],
    },
    "tinyimagenet": {
        "single": [[2], [3], [5], [7], [9]],
        "multi": [
            list(range(0,  20)),
            list(range(20, 40)),
            list(range(40, 60)),
            list(range(60, 80)),
            list(range(80, 100)),
        ],
    },
}


# ─────────────────────────────────────────────────────────────
# Original model training  (paper §A.4 + Table 4)
# CIFAR-10 ResNet-18: Table 4 line 2308
# ─────────────────────────────────────────────────────────────
TRAIN_RESNET = {
    "batch_size":          128,         # §A.4 text
    "epochs":              300,         # Table 4 line 2308
    "early_stop_patience":  50,         # §A.4 text
    "optimizer":           "SGD",
    "momentum":            0.9,         # Table 4 line 2308
    "weight_decay":        5e-4,        # Table 4 line 2308
    "nesterov":            True,        # §A.4 text
    # CIFAR-10 ResNet-18 lr = 1e-2 (Table 4 line 2308)
    # NOTE: previously this was 5e-2 (shell script value) — corrected to Table 4.
    "lr_init":             1e-2,        # Table 4 line 2308: lr=0.01 for CIFAR-10 ResNet-18
    "warmup_epochs":        5,          # §A.4 text
    "lr_scheduler":        "cosine_with_warmup",
    "min_lr":              1e-5,        # §A.4 text
    "augmentation":        ["random_crop", "random_horizontal_flip"],
    "val_ratio":           0.1,         # §A.4 text: 10% of train → validation
}

TRAIN_VIT = {
    "arch":            "vit_s_16",
    "pretrained":       True,           # Table 5: pretrained backbone
    "input_size":       224,            # §A.4 text
    "batch_size":       128,            # Table 5 line 2447
    "epochs":           10,             # Table 5 line 2447
    "lr_init":          3e-4,           # Table 5 line 2447: lr=3×10⁻⁴ for CIFAR-10/100
    "optimizer":       "SGD",
    "momentum":         0.9,
    "weight_decay":     5e-4,
    "nesterov":         True,
    "lr_scheduler":    "cosine_with_warmup",
    "warmup_epochs":    5,
    "min_lr":           1e-5,
    "augmentation":    ["random_crop", "random_horizontal_flip"],
}

# Retain-only retrain (oracle baseline)  Table 4 lines 2314-2319
RETRAIN_RESNET = {
    "cifar10":      {"epochs": 200, "batch_size": 128, "lr_init": 1e-2,
                     "momentum": 0.9, "weight_decay": 5e-4},
    "cifar100":     {"epochs": 200, "batch_size": 128, "lr_init": 1e-2,
                     "momentum": 0.9, "weight_decay": 5e-4},
    "tinyimagenet": {"epochs": 150, "batch_size": 256, "lr_init": 5e-2,
                     "momentum": 0.9, "weight_decay": 5e-4},
}


# ─────────────────────────────────────────────────────────────
# Unlearning method hyperparameters  (paper Table 4)
# ─────────────────────────────────────────────────────────────

# Epoch budgets per method  (Table 4, ResNet)
UNLEARN_EPOCHS: Dict[str, int] = {
    "random_label":                        3,   # Table 4 line 2330
    "salun":                               3,   # Table 4 line 2339
    "grad_ascent_descent":                 3,   # Table 4 line 2348  (NegGrad+)
    "scrub":                               3,   # Table 4 line 2357
    "tarun":                               3,   # Table 4 line 2366  (UNSIR)
    "SVD":                                 1,   # Table 4: training-free
    # CMF variants
    "random_label_CMF_RemoveFC":           4,   # Table 4 line 2390
    "salun_CMF_RemoveFC":                  4,   # Table 4 line 2399
    "grad_ascent_descent_CMF_RemoveFC":    3,   # Table 4 line 2408
    "scrub_CMF_RemoveFC":                  3,   # Table 4 line 2417
    "tarun_CMF_RemoveFC":                  3,   # Table 4 line 2426
}

# Learning rates  (Table 4 verbatim — CIFAR-10, CIFAR-100, Tiny-ImageNet)
# Table 4 does not split single vs multi; one LR per dataset per method.
# We use the same value for both scenarios.
UNLEARN_LR: Dict[str, Dict[str, Dict[str, float]]] = {
    # ── Random Label ──  Table 4: C10=1×10⁻⁴, C100=3×10⁻³, Tiny=5×10⁻⁴
    "random_label": {
        "cifar10":      {"single": 1e-4,  "multi": 1e-4},   # Table 4 line 2330
        "cifar100":     {"single": 3e-3,  "multi": 3e-3},   # Table 4 line 2333
        "tinyimagenet": {"single": 5e-4,  "multi": 5e-4},   # Table 4 line 2336
    },
    # ── SalUn ──  Table 4: C10=1×10⁻⁴, C100=1×10⁻³, Tiny=5×10⁻⁴
    "salun": {
        "cifar10":      {"single": 1e-4,  "multi": 1e-4},   # Table 4 line 2339
        "cifar100":     {"single": 1e-3,  "multi": 1e-3},   # Table 4 line 2342
        "tinyimagenet": {"single": 5e-4,  "multi": 5e-4},   # Table 4 line 2345
    },
    # ── NegGrad+ ──  Table 4: C10=1×10⁻⁴, C100=5×10⁻³, Tiny=5×10⁻⁴
    "grad_ascent_descent": {
        "cifar10":      {"single": 1e-4,  "multi": 1e-4},   # Table 4 line 2348
        "cifar100":     {"single": 5e-3,  "multi": 5e-3},   # Table 4 line 2351
        "tinyimagenet": {"single": 5e-4,  "multi": 5e-4},   # Table 4 line 2354
    },
    # ── SCRUB ──  Table 4: C10=1×10⁻⁴, C100=1×10⁻³, Tiny=5×10⁻³; batch=64
    "scrub": {
        "cifar10":      {"single": 1e-4,  "multi": 1e-4},   # Table 4 line 2357
        "cifar100":     {"single": 1e-3,  "multi": 1e-3},   # Table 4 line 2360
        "tinyimagenet": {"single": 5e-3,  "multi": 5e-3},   # Table 4 line 2363
    },
    # ── UNSIR / tarun ──  Table 4: C10=5×10⁻⁵, C100=3×10⁻⁵, Tiny=2×10⁻⁵; FULL-MODEL
    "tarun": {
        "cifar10":      {"single": 5e-5,  "multi": 5e-5},   # Table 4 line 2366
        "cifar100":     {"single": 3e-5,  "multi": 3e-5},   # Table 4 line 2369
        "tinyimagenet": {"single": 2e-5,  "multi": 2e-5},   # Table 4 line 2372
    },
    # ── SVD (training-free; LR not used) ──
    "SVD": {
        "cifar10":      {"single": 0.0,   "multi": 0.0},
        "cifar100":     {"single": 0.0,   "multi": 0.0},
        "tinyimagenet": {"single": 0.0,   "multi": 0.0},
    },
    # ── CMF variants ── (Table 4; no single/multi distinction)
    # Random Label + CMF:  C10=2×10⁻³, C100=2×10⁻³, Tiny=1×10⁻²
    "random_label_CMF_RemoveFC": {
        "cifar10":      {"single": 2e-3,  "multi": 2e-3},   # Table 4 line 2390
        "cifar100":     {"single": 2e-3,  "multi": 2e-3},   # Table 4 line 2393
        "tinyimagenet": {"single": 1e-2,  "multi": 1e-2},   # Table 4 line 2396
    },
    # SalUn + CMF:  C10=2×10⁻³, C100=2×10⁻³, Tiny=1×10⁻²
    "salun_CMF_RemoveFC": {
        "cifar10":      {"single": 2e-3,  "multi": 2e-3},   # Table 4 line 2399
        "cifar100":     {"single": 2e-3,  "multi": 2e-3},   # Table 4 line 2402
        "tinyimagenet": {"single": 1e-2,  "multi": 1e-2},   # Table 4 line 2405
    },
    # NegGrad+ + CMF:  C10=1×10⁻⁴, C100=1×10⁻⁴, Tiny=3×10⁻⁵
    "grad_ascent_descent_CMF_RemoveFC": {
        "cifar10":      {"single": 1e-4,  "multi": 1e-4},   # Table 4 line 2408
        "cifar100":     {"single": 1e-4,  "multi": 1e-4},   # Table 4 line 2411
        "tinyimagenet": {"single": 3e-5,  "multi": 3e-5},   # Table 4 line 2414
    },
    # SCRUB + CMF:  C10=5×10⁻³, C100=5×10⁻³, Tiny=1×10⁻³; batch=64
    "scrub_CMF_RemoveFC": {
        "cifar10":      {"single": 5e-3,  "multi": 5e-3},   # Table 4 line 2417
        "cifar100":     {"single": 5e-3,  "multi": 5e-3},   # Table 4 line 2420
        "tinyimagenet": {"single": 1e-3,  "multi": 1e-3},   # Table 4 line 2423
    },
    # UNSIR + CMF:  C10=5×10⁻⁵, C100=5×10⁻⁵, Tiny=2×10⁻⁵
    "tarun_CMF_RemoveFC": {
        "cifar10":      {"single": 5e-5,  "multi": 5e-5},   # Table 4 line 2426
        "cifar100":     {"single": 5e-5,  "multi": 5e-5},   # Table 4 line 2429
        "tinyimagenet": {"single": 2e-5,  "multi": 2e-5},   # Table 4 line 2432
    },
}

# Batch sizes  (Table 4: SCRUB variants use 64, all others 128)
UNLEARN_BATCH: Dict[str, int] = {
    "scrub":              64,   # Table 4 line 2357
    "scrub_CMF_RemoveFC": 64,   # Table 4 line 2417
}
UNLEARN_BATCH_DEFAULT: int = 128

# Method-specific extra hyperparameters  (Table 4 verbatim)
UNLEARN_EXTRA: Dict[str, dict] = {
    "salun": {
        "salun_threshold": 0.5,          # Table 4 line 2339
    },
    "salun_CMF_RemoveFC": {
        "salun_threshold": 0.5,          # Table 4 line 2399
    },
    "grad_ascent_descent": {
        "grad_norm_clip": 1.0,           # Table 4 line 2348
    },
    "grad_ascent_descent_CMF_RemoveFC": {
        "grad_norm_clip": 1.0,           # Table 4 line 2408
    },
    "SVD": {
        # Table 4 lines 2375-2388: CIFAR-10 α_r=1000, α_f=30, samples=900
        "SVD_alpha_r":     1000,         # Table 4 line 2375 (prev: 100 — corrected)
        "SVD_alpha_f":     30,           # Table 4 line 2375 (prev: 3 — corrected)
        "SVD_samples":     900,          # Table 4 line 2375
        "SVD_max_patches": 10_000,       # standard default
        "cifar100": {
            "SVD_alpha_r": 1000,         # Table 4 line 2380
            "SVD_alpha_f": 30,           # Table 4 line 2380
            "SVD_samples": 990,          # Table 4 line 2380
        },
        "tinyimagenet": {
            "SVD_alpha_r": 30,           # Table 4 line 2385
            "SVD_alpha_f": 10,           # Table 4 line 2385
            "SVD_samples": 999,          # Table 4 line 2385
        },
    },
    "tarun": {
        # Table 4 line 2366: "3 epochs impair/repair training"; LR=5×10⁻⁵ (full-model).
        # impair_lr = same as repair/main lr (Table 4 lists one LR for the cycle).
        # NOTE: previously 1e-4 (shell script value) — corrected to Table 4.
        "tarun_impair_lr":         5e-5,  # Table 4 line 2366
        "tarun_samples_per_class": 1_000, # standard default
        "cifar100":     {"tarun_impair_lr": 3e-5},   # Table 4 line 2369
        "tinyimagenet": {"tarun_impair_lr": 2e-5},   # Table 4 line 2372
    },
    "tarun_CMF_RemoveFC": {
        # Table 4 line 2426: lr=5×10⁻⁵; impair_lr = same as main lr
        "tarun_impair_lr":         5e-5,  # Table 4 line 2426
        "tarun_samples_per_class": 1_000, # standard default
        "cifar100":     {"tarun_impair_lr": 5e-5},   # Table 4 line 2429
        "tinyimagenet": {"tarun_impair_lr": 2e-5},   # Table 4 line 2432
    },
    "scrub": {
        "scrub_del_bsz":   64,            # Table 4 line 2357
        "scrub_sgda_bsz":  64,            # Table 4 line 2357
        "scrub_msteps":    2,             # Table 4 line 2357
    },
    "scrub_CMF_RemoveFC": {
        "scrub_del_bsz":   64,            # Table 4 line 2417
        "scrub_sgda_bsz":  64,            # Table 4 line 2417
        "scrub_msteps":    2,             # Table 4 line 2417
    },
}


# ─────────────────────────────────────────────────────────────
# CMF classifier settings  (paper §4.3)
# ─────────────────────────────────────────────────────────────
CMF = {
    "CMFClassifier": True,
    "CMF_momentum":  0.9,      # EMA momentum for class-mean update
    "remove_FC":     True,     # replace FC head with CMF head
    # A.1 fix: mean_source="train" is paper-faithful default
    # mean_source="retain" is explicit ablation only
    "mean_source":   "train",
}


# ─────────────────────────────────────────────────────────────
# Evaluation settings  (paper §3)
# ─────────────────────────────────────────────────────────────
EVAL = {
    "output_metrics":  ["forget_accuracy", "retain_accuracy"],
    "feature_metrics": ["linear_probe", "NCC"],
    "nc_metrics":      ["NC3", "NCC"],
    "lp_epochs":       50,
    "lp_lr":           1e-2,
    "test_batch_size": 256,
    # Convention: forget acc compared against oracle's own forget acc (NOT 0%)
    # because under random-mix the oracle itself does not reach 0%.
    "forget_baseline": "oracle_forget_acc",
}


# ─────────────────────────────────────────────────────────────
# Checkpoint / result paths
# ─────────────────────────────────────────────────────────────
# Notebooks (Kaggle): /kaggle/working/checkpoints/
# Local reference:    ./checkpoints/  (clean after archiving stale shell runs)
CKPT_ROOT_KAGGLE = "/kaggle/working/checkpoints"
CKPT_ROOT_LOCAL  = "./checkpoints"

# Stale checkpoints (shell-derived hparams) archived here:
CKPT_ARCHIVE     = "./archive_stale_shell_hparams/checkpoints_shell_derived"

# Split configs (match paper_hparams.py)
SPLIT_RATIOS = [30, 10]  # % of each class to forget
SEEDS        = [0, 1, 2]

# Hyp-source tag embedded in every checkpoint
HPARAM_SOURCE_CURRENT = "table4"
HPARAM_SOURCE_STALE   = "shell_deprecated"


# ─────────────────────────────────────────────────────────────
# Convenience helpers
# ─────────────────────────────────────────────────────────────

def get_lr(method: str, dataset: str, forget_classes: list) -> float:
    """Return the Table-4-specified learning rate for a given setup."""
    scenario = "single" if len(forget_classes) == 1 else "multi"
    try:
        return UNLEARN_LR[method][dataset][scenario]
    except KeyError:
        raise KeyError(
            f"No LR entry for method='{method}', dataset='{dataset}', "
            f"scenario='{scenario}'. Check UNLEARN_LR in config.py."
        )


def get_forget_sets(dataset: str, scenario: str = "single") -> List[List[int]]:
    """Return the list of forget-class sets used in the paper."""
    return UNLEARN_SCENARIOS[dataset][scenario]


def get_dataset_info(dataset: str) -> dict:
    """Return dataset metadata (num_classes, arch, etc.)."""
    return DATASETS[dataset]


def get_batch(method: str) -> int:
    """Return batch size (64 for SCRUB variants, 128 for others)."""
    return UNLEARN_BATCH.get(method, UNLEARN_BATCH_DEFAULT)
