# CMF μ-Source Ablation Experiment

**Location:** `experiments/cmf_loader_ablation/`  
**Does NOT touch:** the original `unlearn/`, `main.py`, or any saved checkpoints.

---

## Hypothesis

The global mean **μ** in `recompute_cmf()` is currently computed over the full
training set (forget + retain combined). When the forget set is small (1/9 ratio)
this contamination is negligible.  Under a larger **3/7 split** (30 % forgotten),
forget-class features shift μ away from the retain-class centroid, potentially
distorting every W_c (all classes depend on the shared μ) and hurting retain
accuracy and NC geometry.

**This experiment tests whether switching to a retain-only μ** fixes that distortion.

---

## Files

| File | Purpose |
|---|---|
| `ablation_model.py` | `AblationModelModule` — extends `ModelModule` with `recompute_cmf(loader_full, loader_retain, device, mean_source)` |
| `ablation_unlearn.py` | Self-contained copies of NegGrad+, Random-label, SalUn, SCRUB with `mean_source` threaded through every call site |
| `ablation_partition.py` | `make_37_partition()` — 3/7 split helper; `log_partition_stats()` — prints per-class counts |
| `run_ablation.py` | Top-level runner: builds the full 2 × N_method × N_seed experiment matrix and saves incremental CSV |
| `aggregate_results.py` | Post-run aggregator: produces `results_aggregated.csv`, `results_table.md`, `findings_summary.txt` |

---

## Partition Design

**Default forget classes: `[0, 1, 2]`** (CIFAR-10).

| Class ID | Class name | Set |
|---|---|---|
| 0 | airplane | **Forget** |
| 1 | automobile | **Forget** |
| 2 | bird | **Forget** |
| 3 | cat | Retain |
| 4 | deer | Retain |
| 5 | dog | Retain |
| 6 | frog | Retain |
| 7 | horse | Retain |
| 8 | ship | Retain |
| 9 | truck | Retain |

CIFAR-10 is balanced (5 000 samples/class, 50 000 total):
- **Forget set:** 15 000 samples (30 %)
- **Retain set:** 35 000 samples (70 %)
- **Ratio:** 3:7

---

## `mean_source` parameter

| Value | μ computed from | Behavior |
|---|---|---|
| `"train"` | Full training loader (forget + retain) | **Legacy / current** — μ contaminated by forget features |
| `"retain"` | Retain-only loader | **Ablation** — μ excludes forget-class samples |

Per-class means **m_c** for ALL classes (including forget classes) are always computed
from their own samples using the full training loader — so W_c for forget classes still
exists in both modes. Only the **centering reference μ** changes source.

> Note: The L2-normalize-before-averaging step (`z_i = normalize(f_i)`) is preserved
> exactly as in production. This deviates from the original CMF paper's Algorithm 1
> (which averages raw features), but is the established behavior of this codebase.

---

## Experiment Matrix

For the 3/7 split:

```
methods × {train, retain} × seeds
─────────────────────────────────────────────────────────
scrub         × train × [0,1,2]
scrub         × retain × [0,1,2]
random_label  × train × [0,1,2]
random_label  × retain × [0,1,2]
naive         × train × [0,1,2]     (NegGrad+ = ascent+descent)
naive         × retain × [0,1,2]
salun         × train × [0,1,2]
salun         × retain × [0,1,2]
original      × (single eval, no training)
retrain       × train × [0,1,2]     (retain-only retraining baseline)
```

---

## Metrics

For every run (reported as mean ± std across seeds):

| Metric | Description |
|---|---|
| `output_retain_acc` | CMF-head accuracy on retain test samples |
| `output_forget_acc` | CMF-head accuracy on forget test samples |
| `probe_retain_acc` | Linear-probe accuracy (frozen encoder) on retain test samples |
| `probe_forget_acc` | Linear-probe accuracy on forget test samples |
| `ncc_retain_acc` | NCC accuracy on retain test samples |
| `ncc_forget_acc` | NCC accuracy on forget test samples |
| `mean_mu_drift_avg` | Mean ‖μ_e − μ_{e-1}‖₂ across epochs |
| `mean_mu_drift_max` | Max  ‖μ_e − μ_{e-1}‖₂ across epochs |

---

## How to Run

### 1. Pre-requisite: pre-trained checkpoint

You need a ResNet18 pre-trained on full CIFAR-10 saved at:

```
./checkpoints/pre_train/cifar10_resnet18.pt
```

This is the same checkpoint used by the existing 1/9 benchmarks.

### 2. Full experiment (3 seeds, priority methods)

```bash
python experiments/cmf_loader_ablation/run_ablation.py \
    --arch resnet18 \
    --dataset cifar10 \
    --num_classes 10 \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-3 \
    --methods scrub random_label naive \
    --mean_sources train retain \
    --seeds 0 1 2 \
    --forget_classes 0 1 2 \
    --checkpoint_path ./checkpoints/pre_train/cifar10_resnet18.pt \
    --output_dir ./experiments/cmf_loader_ablation/results \
    --lp_every 1 \
    --ncc_every 1 \
    --num_forget_samples 15000 \
    --num_retain_samples 35000
```

### 3. Add SalUn (optional, after priority methods)

```bash
python experiments/cmf_loader_ablation/run_ablation.py \
    --methods salun \
    --mean_sources train retain \
    --seeds 0 1 2 \
    --checkpoint_path ./checkpoints/pre_train/cifar10_resnet18.pt \
    --output_dir ./experiments/cmf_loader_ablation/results \
    ...
```

### 4. Add baselines

```bash
python experiments/cmf_loader_ablation/run_ablation.py \
    --methods original retrain \
    --seeds 0 1 2 \
    --checkpoint_path ./checkpoints/pre_train/cifar10_resnet18.pt \
    --output_dir ./experiments/cmf_loader_ablation/results \
    ...
```

### 5. Aggregate results

```bash
python experiments/cmf_loader_ablation/aggregate_results.py \
    --input_csv  ./experiments/cmf_loader_ablation/results/results_summary.csv \
    --output_dir ./experiments/cmf_loader_ablation/results
```

Produces:
- `results_aggregated.csv` — mean ± std per (method, mean_source)
- `results_table.md` — Markdown summary table
- `findings_summary.txt` — per-method written analysis

### 6. Smoke-test (1 epoch, 1 seed, no GPU needed)

```bash
python experiments/cmf_loader_ablation/run_ablation.py \
    --arch resnet18 --dataset cifar10 --num_classes 10 \
    --epochs 1 --batch_size 128 \
    --methods scrub \
    --mean_sources train retain \
    --seeds 0 \
    --forget_classes 0 1 2 \
    --checkpoint_path ./checkpoints/pre_train/cifar10_resnet18.pt \
    --output_dir ./experiments/cmf_loader_ablation/results \
    --lp_every 0 --ncc_every 0 \
    --dry_run
```

---

## Output Files

After `run_ablation.py`:

```
results/
├── results_summary.csv    # per-seed rows
└── results_full.json      # per-seed rows + full history + partition stats
```

After `aggregate_results.py`:

```
results/
├── results_aggregated.csv  # mean ± std per (method, mean_source)
├── results_table.md        # Markdown table (copy-paste ready)
└── findings_summary.txt    # written analysis
```

---

## Design Constraints Satisfied

- ✅ Does **not** modify `unlearn/`, `main.py`, `utils.py`, or any existing checkpoint.
- ✅ Reuses existing evaluation harness (`evaluation/linear_prob.py`, `evaluation/nc_cmf.py`).
- ✅ Same model architecture (ResNet18), optimizer (SGD+Nesterov), and hyperparameters as the 1/9 benchmarks.
- ✅ Only variables changed: (a) forget/retain class partition, (b) `mean_source` flag for μ computation.
- ✅ L2-normalize-before-averaging step preserved exactly (deviation from original CMF paper noted in code comments).
- ✅ Per-class means m_c for forget classes are always computed from forget-class samples — only μ (the centering reference) changes source.
- ✅ Results saved incrementally — partial results survive crashes.
- ✅ 3 seeds, mean ± std reporting, matching existing benchmark variance style.
