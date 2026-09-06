"""
paper_hparams.py — Single auditable source for ALL hyperparameters used in the paper.

Paper: "An Illusion of Unlearning? Assessing Machine Unlearning Through Internal
        Representations" (Gao, Unal, Rangamani, Zhu — AISTATS 2026, arXiv:2604.08271v1)

Official GitHub: https://github.com/ycgao1/CMF_Unlearning

SOURCE PRIORITY (strict, no tie-breaking)
=========================================
  1. Table 4 (Appendix D.1, PDF lines 2307–2434) — primary source for every
     method's LR, epochs, batch size, and method-specific flags (SVD α_r/α_f,
     UNSIR LR, SCRUB msteps, etc.).  Values are transcribed verbatim.
  2. §A.4 text — ONLY for parameters Table 4 does not cover.
     Exception: Table 4 DOES list the original CIFAR-10 training LR (0.01),
     so Table 4 takes precedence over §A.4's 5×10⁻² for that parameter.
  3. Standard ResNet-18/CIFAR defaults — ONLY when neither Table 4 nor §A.4
     specify a value; always explicitly labeled "standard default".

Shell scripts (script/run/) are NOT a reference source.  The three ResNet
shell scripts (run_resnet_unlearning.sh, run_resnet_18_CMF_unlearning.sh,
run_resnet_retrain.sh) have been deleted from the working copy.

PART A FIX REPORT
=================
A.1 — CMF update rule (mean_source default):
      Default mean_source = "train": both global mean μ AND all per-class means m_c
      are computed from the FULL dataset (retain + forget combined), matching Algorithm 1.
      mean_source = "retain" is only an explicit ablation, never the default.
      DISCLOSED DEVIATION: recompute_cmf() L2-normalizes features before averaging
      (paper Algorithm 1 uses raw z_θ(x)). This is intentional and documented; not a bug.

A.2 — cmf_static (Algorithm 2): per-epoch recompute_cmf → freeze W → update encoder only.
      No per-round shortcuts. Epoch budgets from Table 4:
        Random Label + CMF: 4 epochs   (Table 4 line 2390)
        SalUn + CMF:        4 epochs   (Table 4 line 2399)
        NegGrad+ + CMF:     3 epochs   (Table 4 line 2408)
        SCRUB + CMF:        3 epochs   (Table 4 line 2417)
        UNSIR + CMF:        3 epochs   (Table 4 line 2426)

A.3 — All hyperparameters sourced exclusively from Table 4. No shell-script values.

A.4 — NCC/NC3 weight extraction: evaluation/nc.py get_classifier_weights() uses
      model.CMFweights.weight for CMF models (tier-2 fallback), verified correct.

A.5 — CMF geometry consistency: _features_for_nc() applies ẑ = normalize(normalize(f)−μ)
      via model._preprocess_feats_for_cmf() for CMF models; raw features for non-CMF.

A.6 — Evaluation protocol: ONE split, generated once in NB1, loaded by all downstream.
      Paper uses whole-class unlearning. Our notebooks use stratified random-mix splits
      (documented intentional deviation, labeled as alternative experiment).

A.7 — Runtime bugs: CUDA events guarded with `if device.type == 'cuda':` in unlearn/naive.py.
      "grad_descent" key now maps to unlearn_grad_descent_only (not unlearn_naive).

A.8 — Notebook provenance: all notebooks clone from official ycgao1/CMF_Unlearning at main.

TABLE 4 VERBATIM — CIFAR-10 ResNet-18 (PDF lines 2307–2434):
  Original:              epochs=300, batch=128, lr=0.01,  mom=0.9, cosine LR, WD=5×10⁻⁴
  Retain-only Retrain:   epochs=200, batch=128, lr=0.01,  mom=0.9, WD=5×10⁻⁴, val-ratio=0.1
  Retain-only FT:        epochs=3,   batch=128, lr=1×10⁻³, mom=0.9
  Random Label:          epochs=3,   batch=128, lr=1×10⁻⁴, mom=0.9
  SalUn:                 epochs=3,   batch=128, lr=1×10⁻⁴, threshold=0.5
  NegGrad+:              epochs=3,   batch=128, lr=1×10⁻⁴, grad-clip=1.0
  SCRUB:                 epochs=3,   batch=64,  lr=1×10⁻⁴, sgda-bsz=64, msteps=2
  UNSIR:                 epochs=3,   batch=128, lr=5×10⁻⁵, "3 epochs impair/repair training"
  SVD (TF):              batch=900,  lr=–,      α_r=1000, α_f=30
  Random Label + CMF:    epochs=4,   batch=128, lr=2×10⁻³
  SalUn + CMF:           epochs=4,   batch=128, lr=2×10⁻³, threshold=0.5
  NegGrad+ + CMF:        epochs=3,   batch=128, lr=1×10⁻⁴, grad-clip=1.0
  SCRUB + CMF:           epochs=3,   batch=64,  lr=5×10⁻³, sgda-bsz=64, msteps=2
  UNSIR + CMF:           epochs=3,   batch=128, lr=5×10⁻⁵, "3 epochs impair/repair training"

UNSIR MODE NOTE:
  Table 4 lists LR=5×10⁻⁵ under "3 epochs impair/repair training" — this is the
  FULL-MODEL UNSIR (encoder + classifier updated). Table 2 (main text) confirms this:
  it explicitly separates "Full Model" vs "Classifier only" UNSIR rows, and the main
  results (Table 1) use full-model UNSIR (the default, Table 4). We use full-model mode.
  The classifier-only variant (LR=2e-3) was a shell-only experiment, NOT Table 4.
"""
from __future__ import annotations
from typing import Dict, List


# ─────────────────────────────────────────────────────────────
# Official repo info
# ─────────────────────────────────────────────────────────────
OFFICIAL_REPO_URL    = "https://github.com/ycgao1/CMF_Unlearning"
OFFICIAL_REPO_BRANCH = "main"
OFFICIAL_REPO_COMMIT = "main"   # pin to SHA after first run


# ─────────────────────────────────────────────────────────────
# A.1  CMF defaults
# ─────────────────────────────────────────────────────────────
CMF_MEAN_SOURCE_DEFAULT = "train"
CMF_DISCLOSED_DEVIATION = (
    "recompute_cmf() L2-normalizes features before per-class averaging. "
    "Paper Algorithm 1 uses raw z_θ(x). This is intentional and documented; not a bug."
)


# ─────────────────────────────────────────────────────────────
# A.4  Original Model Training
#      Table 4 (line 2308): CIFAR-10 ResNet-18 lr=0.01
#      Table 4 (line 2312): Tiny-ImageNet ResNet-50 lr=0.05
#      §A.4 text supplements: batch=128, epochs≤300, early-stop patience=50,
#        SGD momentum=0.9, WD=5×10⁻⁴, cosine LR with 5-epoch warmup, min_lr=1×10⁻⁵
# ─────────────────────────────────────────────────────────────
PRETRAIN_RESNET_CIFAR = {
    "batch_size":          128,          # §A.4 text
    "epochs":              300,          # Table 4 line 2308
    "early_stop_patience":  50,          # §A.4 text
    "optimizer":           "SGD",
    "momentum":            0.9,          # Table 4 line 2308
    "weight_decay":        5e-4,         # Table 4 line 2308 (WD=5×10⁻⁴)
    "nesterov":            True,         # §A.4 text
    "lr_init":             1e-2,         # Table 4 line 2308: lr=0.01 for CIFAR-10 ResNet-18
    "warmup_epochs":        5,           # §A.4 text: 5-epoch linear warmup
    "lr_scheduler":        "cosine_with_warmup",   # Table 4: "cosine LR"
    "min_lr":              1e-5,         # §A.4 text
    "augmentation":        ["random_crop", "random_horizontal_flip"],  # §A.4 text
}

PRETRAIN_RESNET_TINY = {
    **PRETRAIN_RESNET_CIFAR,
    "arch":    "resnet50",
    "lr_init": 5e-2,   # Table 4 line 2312: 0.05 for Tiny-ImageNet ResNet-50
}

PRETRAIN_VIT = {
    "arch":            "vit_s_16",
    "pretrained":       True,            # Table 5: "pretrained backbone"
    "input_size":       224,             # §A.4 text: "resized to 224×224"
    "batch_size":       128,             # Table 5 line 2447
    "epochs":           10,              # Table 5 line 2447
    "lr_init":          3e-4,            # Table 5 line 2447: lr=3×10⁻⁴ for CIFAR-10/100
    "optimizer":       "SGD",
    "momentum":         0.9,
    "weight_decay":     5e-4,
    "nesterov":         True,
    "lr_scheduler":    "cosine_with_warmup",
    "warmup_epochs":    5,
    "min_lr":           1e-5,
    "augmentation":    ["random_crop", "random_horizontal_flip"],
}

# ─────────────────────────────────────────────────────────────
# Retrain-on-Retain baseline  (Table 4)
# CIFAR-10/100: epochs=200, lr=0.01   (lines 2314–2318)
# Tiny-ImageNet: epochs=150, batch=256, lr=0.05  (line 2319)
# ─────────────────────────────────────────────────────────────
ORACLE_RETRAIN = {
    "cifar10":      {"epochs": 200, "batch_size": 128, "lr_init": 1e-2, "momentum": 0.9, "weight_decay": 5e-4},
    "cifar100":     {"epochs": 200, "batch_size": 128, "lr_init": 1e-2, "momentum": 0.9, "weight_decay": 5e-4},
    "tinyimagenet": {"epochs": 150, "batch_size": 256, "lr_init": 5e-2, "momentum": 0.9, "weight_decay": 5e-4},
}


# ─────────────────────────────────────────────────────────────
# Checkpoint naming convention
# {method}_{variant}_{mean_source}_seed{seed}.pt
# ─────────────────────────────────────────────────────────────
CKPT_DIR_DEFAULT = "/kaggle/working/checkpoints"


# ─────────────────────────────────────────────────────────────
# Unlearning epoch budgets  (Table 4, ResNet)
# ─────────────────────────────────────────────────────────────
UNLEARN_EPOCHS: Dict[str, int] = {
    "random_label":                        3,   # Table 4 line 2330
    "salun":                               3,   # Table 4 line 2339
    "grad_ascent_descent":                 3,   # Table 4 line 2348  (NegGrad+)
    "scrub":                               3,   # Table 4 line 2357
    "tarun":                               3,   # Table 4 line 2366  (UNSIR)
    "SVD":                                 1,   # Table 4: training-free (1 pass)
    "random_label_CMF_RemoveFC":           4,   # Table 4 line 2390
    "salun_CMF_RemoveFC":                  4,   # Table 4 line 2399
    "grad_ascent_descent_CMF_RemoveFC":    3,   # Table 4 line 2408
    "scrub_CMF_RemoveFC":                  3,   # Table 4 line 2417
    "tarun_CMF_RemoveFC":                  3,   # Table 4 line 2426
}

# CMF epoch budgets per base method (for cmf_static NB4)
CMF_EPOCHS_BY_METHOD: Dict[str, int] = {
    "random_label":        4,   # Table 4 line 2390
    "salun":               4,   # Table 4 line 2399
    "grad_ascent_descent": 3,   # Table 4 line 2408
    "scrub":               3,   # Table 4 line 2417
    "tarun":               3,   # Table 4 line 2426
}
CMF_EPOCHS_DEFAULT: int = 4   # maximum across methods

# Post-hoc W calibration (NB4 4b)
POSTHOC_K_VALUES: List[int] = [2, 5, 10]
POSTHOC_PHASE2_DATA: List[str] = ["retain_only", "retain_plus_forget"]

# Budget-shared two-stage (NB5)
BUDGET_SHARED_K_VALUES: List[int] = [1]

# ─────────────────────────────────────────────────────────────
# Learning rates  (Table 4 ONLY — verbatim transcription)
# Table 4 does not distinguish single-class vs multi-class LR.
# We use the same value for both scenarios (the paper lists one
# LR per dataset per method).
# ─────────────────────────────────────────────────────────────
UNLEARN_LR: Dict[str, Dict[str, Dict[str, float]]] = {
    # ── Random Label ── (Table 4: CIFAR-10=1×10⁻⁴, CIFAR-100=3×10⁻³, Tiny=5×10⁻⁴)
    "random_label": {
        "cifar10":      {"single": 1e-4,  "multi": 1e-4},   # Table 4 line 2330
        "cifar100":     {"single": 3e-3,  "multi": 3e-3},   # Table 4 line 2333
        "tinyimagenet": {"single": 5e-4,  "multi": 5e-4},   # Table 4 line 2336
    },
    # ── SalUn ── (Table 4: CIFAR-10=1×10⁻⁴, CIFAR-100=1×10⁻³, Tiny=5×10⁻⁴)
    "salun": {
        "cifar10":      {"single": 1e-4,  "multi": 1e-4},   # Table 4 line 2339
        "cifar100":     {"single": 1e-3,  "multi": 1e-3},   # Table 4 line 2342
        "tinyimagenet": {"single": 5e-4,  "multi": 5e-4},   # Table 4 line 2345
    },
    # ── NegGrad+ ── (Table 4: CIFAR-10=1×10⁻⁴, CIFAR-100=5×10⁻³, Tiny=5×10⁻⁴)
    "grad_ascent_descent": {
        "cifar10":      {"single": 1e-4,  "multi": 1e-4},   # Table 4 line 2348
        "cifar100":     {"single": 5e-3,  "multi": 5e-3},   # Table 4 line 2351
        "tinyimagenet": {"single": 5e-4,  "multi": 5e-4},   # Table 4 line 2354
    },
    # ── SCRUB ── (Table 4: CIFAR-10=1×10⁻⁴, CIFAR-100=1×10⁻³, Tiny=5×10⁻³; batch=64)
    "scrub": {
        "cifar10":      {"single": 1e-4,  "multi": 1e-4},   # Table 4 line 2357
        "cifar100":     {"single": 1e-3,  "multi": 1e-3},   # Table 4 line 2360
        "tinyimagenet": {"single": 5e-3,  "multi": 5e-3},   # Table 4 line 2363
    },
    # ── UNSIR / tarun ── (Table 4: CIFAR-10=5×10⁻⁵, CIFAR-100=3×10⁻⁵, Tiny=2×10⁻⁵)
    # FULL-MODEL mode (Table 4 / Table 1 main results). impair_lr = same as repair lr per §A.6.
    "tarun": {
        "cifar10":      {"single": 5e-5,  "multi": 5e-5},   # Table 4 line 2366
        "cifar100":     {"single": 3e-5,  "multi": 3e-5},   # Table 4 line 2369
        "tinyimagenet": {"single": 2e-5,  "multi": 2e-5},   # Table 4 line 2372
    },
    # ── SVD (training-free, lr unused) ──
    "SVD": {
        "cifar10":      {"single": 0.0,  "multi": 0.0},
        "cifar100":     {"single": 0.0,  "multi": 0.0},
        "tinyimagenet": {"single": 0.0,  "multi": 0.0},
    },
    # ── CMF variants ── (Table 4; no single/multi distinction in the table)
    # Random Label + CMF: CIFAR-10=2×10⁻³, CIFAR-100=2×10⁻³, Tiny=1×10⁻²
    "random_label_CMF_RemoveFC": {
        "cifar10":      {"single": 2e-3,  "multi": 2e-3},   # Table 4 line 2390
        "cifar100":     {"single": 2e-3,  "multi": 2e-3},   # Table 4 line 2393
        "tinyimagenet": {"single": 1e-2,  "multi": 1e-2},   # Table 4 line 2396
    },
    # SalUn + CMF: CIFAR-10=2×10⁻³, CIFAR-100=2×10⁻³, Tiny=1×10⁻²
    "salun_CMF_RemoveFC": {
        "cifar10":      {"single": 2e-3,  "multi": 2e-3},   # Table 4 line 2399
        "cifar100":     {"single": 2e-3,  "multi": 2e-3},   # Table 4 line 2402
        "tinyimagenet": {"single": 1e-2,  "multi": 1e-2},   # Table 4 line 2405
    },
    # NegGrad+ + CMF: CIFAR-10=1×10⁻⁴, CIFAR-100=1×10⁻⁴, Tiny=3×10⁻⁵
    "grad_ascent_descent_CMF_RemoveFC": {
        "cifar10":      {"single": 1e-4,  "multi": 1e-4},   # Table 4 line 2408
        "cifar100":     {"single": 1e-4,  "multi": 1e-4},   # Table 4 line 2411
        "tinyimagenet": {"single": 3e-5,  "multi": 3e-5},   # Table 4 line 2414
    },
    # SCRUB + CMF: CIFAR-10=5×10⁻³, CIFAR-100=5×10⁻³, Tiny=1×10⁻³; batch=64
    "scrub_CMF_RemoveFC": {
        "cifar10":      {"single": 5e-3,  "multi": 5e-3},   # Table 4 line 2417
        "cifar100":     {"single": 5e-3,  "multi": 5e-3},   # Table 4 line 2420
        "tinyimagenet": {"single": 1e-3,  "multi": 1e-3},   # Table 4 line 2423
    },
    # UNSIR + CMF: CIFAR-10=5×10⁻⁵, CIFAR-100=5×10⁻⁵, Tiny=2×10⁻⁵
    "tarun_CMF_RemoveFC": {
        "cifar10":      {"single": 5e-5,  "multi": 5e-5},   # Table 4 line 2426
        "cifar100":     {"single": 5e-5,  "multi": 5e-5},   # Table 4 line 2429
        "tinyimagenet": {"single": 2e-5,  "multi": 2e-5},   # Table 4 line 2432
    },
}

# Batch sizes — SCRUB uses 64, all others 128  (Table 4)
UNLEARN_BATCH: Dict[str, int] = {
    "scrub":              64,   # Table 4 line 2357
    "scrub_CMF_RemoveFC": 64,   # Table 4 line 2417
}
UNLEARN_BATCH_DEFAULT: int = 128

# Method-specific extras  (Table 4 verbatim)
UNLEARN_EXTRA: Dict[str, dict] = {
    "salun":                    {"salun_threshold": 0.5},    # Table 4 line 2339
    "salun_CMF_RemoveFC":       {"salun_threshold": 0.5},    # Table 4 line 2399
    "grad_ascent_descent":      {"grad_norm_clip": 1.0},     # Table 4 line 2348
    "grad_ascent_descent_CMF_RemoveFC": {"grad_norm_clip": 1.0},  # Table 4 line 2408
    "SVD": {
        # Table 4 lines 2375–2388: CIFAR-10 α_r=1000, α_f=30, samples=900
        "SVD_alpha_r":     1000,   # Table 4 line 2375
        "SVD_alpha_f":     30,     # Table 4 line 2375
        "SVD_samples":     900,    # Table 4 line 2375 (batch=900)
        "SVD_max_patches": 10_000, # standard default; not specified in Table 4
        "cifar100": {
            "SVD_alpha_r": 1000,   # Table 4 line 2380
            "SVD_alpha_f": 30,     # Table 4 line 2380
            "SVD_samples": 990,    # Table 4 line 2380 (batch=990)
        },
        "tinyimagenet": {
            "SVD_alpha_r": 30,     # Table 4 line 2385
            "SVD_alpha_f": 10,     # Table 4 line 2385
            "SVD_samples": 999,    # Table 4 line 2385 (batch=999)
        },
    },
    "tarun": {
        # Table 4 line 2366: "3 epochs impair/repair training"; LR=5×10⁻⁵ (full-model).
        # impair_lr = same as repair/main lr (Table 4 does not distinguish them; §A.6
        # describes one LR for the full impair-repair cycle).
        "tarun_impair_lr":         5e-5,   # Table 4 line 2366: same as main lr
        "tarun_samples_per_class": 1_000,  # standard default (not in Table 4)
        "cifar100":     {"tarun_impair_lr": 3e-5},   # Table 4 line 2369
        "tinyimagenet": {"tarun_impair_lr": 2e-5},   # Table 4 line 2372
    },
    "tarun_CMF_RemoveFC": {
        # Table 4 line 2426: lr=5×10⁻⁵; impair_lr = same as main lr
        "tarun_impair_lr":         5e-5,   # Table 4 line 2426
        "tarun_samples_per_class": 1_000,  # standard default
        "cifar100":     {"tarun_impair_lr": 5e-5},   # Table 4 line 2429
        "tinyimagenet": {"tarun_impair_lr": 2e-5},   # Table 4 line 2432
    },
    "scrub": {
        "scrub_del_bsz":   64,   # Table 4 line 2357 (del-bsz same as sgda-bsz)
        "scrub_sgda_bsz":  64,   # Table 4 line 2357 (sgda-bsz=64)
        "scrub_msteps":    2,    # Table 4 line 2357 (msteps=2)
    },
    "scrub_CMF_RemoveFC": {
        "scrub_del_bsz":   64,   # Table 4 line 2417
        "scrub_sgda_bsz":  64,   # Table 4 line 2417
        "scrub_msteps":    2,    # Table 4 line 2417
    },
}


# ─────────────────────────────────────────────────────────────
# VALUE CHANGE LOG — shell-deprecated → Table-4
# Records every value that changed when we dropped shell scripts.
# Checkpoints produced under shell-derived values carry
#   hparam_source="shell_deprecated" in their metadata.
# ─────────────────────────────────────────────────────────────
SHELL_TO_TABLE4_CHANGES: Dict[str, dict] = {
    "pretrain_lr_cifar10": {
        "old_shell_derived": "5e-2 (§A.4 text, no pre-train shell)",
        "new_table4":        1e-2,   # Table 4 line 2308: 0.01
        "source_citation":   "Table 4, line 2308",
    },
    "random_label_lr_cifar10": {
        "old_shell_derived": 1e-2,
        "new_table4":        1e-4,   # Table 4 line 2330
        "source_citation":   "Table 4, line 2330",
    },
    "salun_lr_cifar10": {
        "old_shell_derived": 1e-2,
        "new_table4":        1e-4,   # Table 4 line 2339
        "source_citation":   "Table 4, line 2339",
    },
    "neggrad_lr_cifar10": {
        "old_shell_derived": 1e-3,
        "new_table4":        1e-4,   # Table 4 line 2348
        "source_citation":   "Table 4, line 2348",
    },
    "unsir_lr_cifar10": {
        "old_shell_derived": "2e-3 (classifier-only mode)",
        "new_table4":        5e-5,   # Table 4 line 2366 (full-model mode)
        "source_citation":   "Table 4, line 2366",
        "mode_change":       "classifier-only → full-model",
    },
    "unsir_impair_lr_cifar10": {
        "old_shell_derived": 1e-4,
        "new_table4":        5e-5,   # same as main lr per Table 4
        "source_citation":   "Table 4, line 2366",
    },
    "svd_alpha_r_cifar10": {
        "old_shell_derived": 100,
        "new_table4":        1000,   # Table 4 line 2375
        "source_citation":   "Table 4, line 2375",
    },
    "svd_alpha_f_cifar10": {
        "old_shell_derived": 3,
        "new_table4":        30,     # Table 4 line 2375
        "source_citation":   "Table 4, line 2375",
    },
    "random_label_cmf_lr_cifar10": {
        "old_shell_derived": "1e-4 (single-class shell value)",
        "new_table4":        2e-3,   # Table 4 line 2390 (no single/multi split)
        "source_citation":   "Table 4, line 2390",
    },
    "salun_cmf_lr_cifar10": {
        "old_shell_derived": "2e-4 (single-class shell value)",
        "new_table4":        2e-3,   # Table 4 line 2399
        "source_citation":   "Table 4, line 2399",
    },
    "scrub_cifar100_multi_lr": {
        "old_shell_derived": "3e-4 (shell multi value)",
        "new_table4":        1e-3,   # Table 4 line 2360 (one value per dataset)
        "source_citation":   "Table 4, line 2360",
    },
    "neggrad_cifar100_lr": {
        "old_shell_derived": "5e-5 (single) / 1e-4 (multi) from shell",
        "new_table4":        5e-3,   # Table 4 line 2351
        "source_citation":   "Table 4, line 2351",
    },
}

# Stale checkpoint tag — embed this in checkpoint metadata for any run
# produced under the old shell-derived values so they are never silently
# mixed with Table-4-derived results.
HPARAM_SOURCE_CURRENT  = "table4"
HPARAM_SOURCE_STALE    = "shell_deprecated"


# ─────────────────────────────────────────────────────────────
# Evaluation settings
# ─────────────────────────────────────────────────────────────
EVAL = {
    "output_metrics":  ["forget_accuracy", "retain_accuracy"],
    "feature_metrics": ["linear_probe", "NCC"],
    "nc_metrics":      ["NC3", "NCC"],
    "lp_epochs":       50,
    "lp_lr":           1e-2,
    "test_batch_size": 256,
    "forget_baseline": "oracle_forget_acc",
}

# Split configs
SPLIT_RATIOS: List[int] = [30, 10]
SEEDS: List[int] = [0, 1, 2]


# ─────────────────────────────────────────────────────────────
# Convenience helpers
# ─────────────────────────────────────────────────────────────
def get_lr(method: str, dataset: str, forget_classes: list) -> float:
    """Return Table 4 LR for (method, dataset, scenario)."""
    scenario = "single" if len(forget_classes) == 1 else "multi"
    try:
        return UNLEARN_LR[method][dataset][scenario]
    except KeyError:
        raise KeyError(
            f"No LR entry for method='{method}', dataset='{dataset}', "
            f"scenario='{scenario}'. Check UNLEARN_LR in paper_hparams.py."
        )


def get_cmf_epochs(base_method: str) -> int:
    """Return Table 4 epoch count for CMF variant of base_method."""
    return CMF_EPOCHS_BY_METHOD.get(base_method, CMF_EPOCHS_DEFAULT)


def get_batch(method: str) -> int:
    """Return batch size for method (64 for SCRUB variants, 128 for others)."""
    return UNLEARN_BATCH.get(method, UNLEARN_BATCH_DEFAULT)


def ckpt_name(method: str, variant: str, mean_source: str, seed: int,
              ratio: int = None, suffix: str = "", testmode: bool = False,
              hparam_source: str = HPARAM_SOURCE_CURRENT) -> str:
    """Return standard checkpoint filename.

    Pattern: {method}_{variant}_{mean_source}_seed{seed}[_ratio{ratio}][_{hparam_source}][{suffix}].pt
    hparam_source='shell_deprecated' is appended for stale checkpoints.
    """
    parts = [p for p in [method, variant, mean_source] if p]
    name  = "_".join(parts)
    if ratio is not None:
        name += f"_ratio{ratio}"
    name += f"_seed{seed}"
    if hparam_source == HPARAM_SOURCE_STALE:
        name += "_shell_deprecated"
    if suffix:
        name += f"_{suffix}"
    if testmode:
        name += "_testmode"
    return name + ".pt"
