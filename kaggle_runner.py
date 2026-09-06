"""
kaggle_runner.py — Execute all 5 CMF-Unlearning notebooks sequentially on Kaggle.

Usage (inside a Kaggle notebook cell or terminal):
    python kaggle_runner.py [--test-mode] [--start NB] [--nb1-only]

Flags:
    --test-mode    Run every notebook in TEST_MODE=True (fast smoke test, ~5 min total)
    --start N      Start from notebook N (1–5). Skips NB1..N-1 if checkpoints exist.
    --nb1-only     Run only NB1 (pre-train + splits), then stop.

Prerequisites:
    pip install nbconvert jupyter nbformat

What this script does:
  1. Clears /kaggle/working/checkpoints/ (the active checkpoint path used by all notebooks)
     so no stale shell-derived checkpoint can trigger the "skip if exists" logic.
     * NOTE: local ./checkpoints/ was already archived to
       archive_stale_shell_hparams/checkpoints_shell_derived/ before this run.
     * This script only clears the KAGGLE path (which lives in /kaggle/working/).
  2. Runs each notebook via nbconvert, injecting the correct TEST_MODE value.
  3. Logs wall-clock time per notebook.
  4. Re-runs _validate_table4.py before NB1 to confirm 72/72.

Order:
    NB1 → NB2 → NB3 → NB4 → NB5

CMF epoch config (Table 4, unchanged — K_SHARED=[1] per paper_hparams.py):
    SCRUB+CMF: 3 total epochs → k_shared=1 → 2 Stage-1 + 1 Stage-2 epochs
    This is consistent with the Table-4 epoch count; no change from prior session.
"""

import argparse
import os
import subprocess
import sys
import time


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

REPO_DIR    = "/kaggle/working/CMF_Unlearning"
CKPT_ROOT   = "/kaggle/working/checkpoints"
NB_DIR      = os.path.join(REPO_DIR, "notebooks")

NOTEBOOKS = [
    (1, os.path.join(NB_DIR, "01_train_full.ipynb"),            "NB1: Train Θ_o + Splits"),
    (2, os.path.join(NB_DIR, "02_retrain_oracle.ipynb"),        "NB2: Oracle Retrain"),
    (3, os.path.join(NB_DIR, "03_unlearn_no_cmf.ipynb"),        "NB3: Non-CMF Baselines"),
    (4, os.path.join(NB_DIR, "04_unlearn_cmf.ipynb"),           "NB4: CMF Static + Post-Hoc W"),
    (5, os.path.join(NB_DIR, "05_unlearn_cmf_budget_shared.ipynb"), "NB5: Budget-Shared Two-Stage"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_cmd(cmd, description="", check=True):
    print(f"\n[runner] {description or cmd}")
    result = subprocess.run(cmd, shell=True, text=True)
    if check and result.returncode != 0:
        print(f"[runner] ERROR: command failed (rc={result.returncode})")
        sys.exit(result.returncode)
    return result.returncode


def clear_kaggle_checkpoints():
    """Remove all checkpoint files from /kaggle/working/checkpoints/ so that
    no old result can trigger a notebook's 'skip if exists' guard."""
    if not os.path.isdir(CKPT_ROOT):
        print(f"[runner] {CKPT_ROOT} does not exist — nothing to clear.")
        return
    # Walk and delete .pt / .pkl / .csv files only (preserve directory structure)
    n_deleted = 0
    for root, dirs, files in os.walk(CKPT_ROOT):
        for fname in files:
            if fname.endswith((".pt", ".pkl", ".csv", ".json")):
                fpath = os.path.join(root, fname)
                os.remove(fpath)
                n_deleted += 1
    print(f"[runner] Cleared {n_deleted} stale checkpoint/result files from {CKPT_ROOT}")


def validate_hparams():
    """Re-run _validate_table4.py. Must pass 72/72 before any notebook runs."""
    validate_script = os.path.join(REPO_DIR, "_validate_table4.py")
    if not os.path.exists(validate_script):
        validate_script = "_validate_table4.py"  # try local fallback
    if not os.path.exists(validate_script):
        print("[runner] WARNING: _validate_table4.py not found, skipping validation.")
        return
    rc = run_cmd(f"python {validate_script}", "Validating Table-4 hyperparameters (72/72 check)")
    if rc != 0:
        print("[runner] FATAL: _validate_table4.py failed — aborting.")
        sys.exit(1)
    print("[runner] ✓ 72/72 hyperparameter checks passed.")


def inject_test_mode(nb_path: str, test_mode: bool) -> str:
    """Write a copy of the notebook with TEST_MODE hard-coded, return copy path."""
    import json
    with open(nb_path) as f:
        nb = json.load(f)

    mode_str = "True" if test_mode else "False"
    modified = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        new_src = []
        for line in src:
            # Replace any `TEST_MODE = True/False` assignment
            if line.startswith("TEST_MODE") and "=" in line and "print" not in line:
                line = f"TEST_MODE = {mode_str}  # injected by kaggle_runner.py\n"
                modified = True
            new_src.append(line)
        cell["source"] = new_src

    suffix = "_testmode" if test_mode else "_fullrun"
    out_path = nb_path.replace(".ipynb", f"{suffix}_run.ipynb")
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)

    if modified:
        print(f"[runner]   Injected TEST_MODE={mode_str} into {os.path.basename(nb_path)}")
    return out_path


def run_notebook(nb_path: str, nb_label: str) -> float:
    """Execute a notebook via nbconvert. Returns wall-clock seconds."""
    out_path = nb_path.replace(".ipynb", "_executed.ipynb")
    cmd = (
        f"jupyter nbconvert --to notebook --execute "
        f"--ExecutePreprocessor.timeout=7200 "
        f"--output {out_path!r} {nb_path!r}"
    )
    t0 = time.time()
    rc = run_cmd(cmd, f"Executing {nb_label}", check=False)
    elapsed = time.time() - t0
    if rc != 0:
        print(f"[runner] ⚠  {nb_label} finished with errors (rc={rc}). "
              f"Wall: {elapsed/60:.1f} min.")
    else:
        print(f"[runner] ✓  {nb_label} completed. Wall: {elapsed/60:.1f} min.")
    return elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run CMF-Unlearning notebooks sequentially.")
    parser.add_argument("--test-mode", action="store_true",
                        help="Inject TEST_MODE=True into every notebook (fast smoke test).")
    parser.add_argument("--start", type=int, default=1, metavar="N",
                        help="Start from notebook N (1–5). Default: 1.")
    parser.add_argument("--nb1-only", action="store_true",
                        help="Run only NB1, then stop.")
    args = parser.parse_args()

    print("=" * 70)
    print("CMF Unlearning — Full Pipeline Runner")
    print(f"  TEST_MODE : {args.test_mode}")
    print(f"  start_nb  : {args.start}")
    print(f"  nb1_only  : {args.nb1_only}")
    print("=" * 70)

    # Step 0: validate hyperparameters
    validate_hparams()

    # Step 1: clear stale Kaggle checkpoints
    print("\n[runner] Step 1: Clearing stale Kaggle checkpoints...")
    clear_kaggle_checkpoints()

    # Step 2: run notebooks
    total_wall = 0.0
    results = []

    for nb_num, nb_path, nb_label in NOTEBOOKS:
        if nb_num < args.start:
            print(f"[runner] Skipping {nb_label} (--start={args.start})")
            continue

        print(f"\n{'='*70}")
        print(f"[runner] Starting {nb_label}")
        print(f"{'='*70}")

        if not os.path.exists(nb_path):
            print(f"[runner] ERROR: notebook not found: {nb_path}")
            sys.exit(1)

        # Inject TEST_MODE
        run_path = inject_test_mode(nb_path, args.test_mode)

        # Execute
        elapsed = run_notebook(run_path, nb_label)
        total_wall += elapsed
        results.append((nb_label, elapsed))

        if args.nb1_only and nb_num == 1:
            print("[runner] --nb1-only: stopping after NB1.")
            break

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — Wall-clock summary")
    print("=" * 70)
    for label, secs in results:
        print(f"  {label:<45s}  {secs/60:.1f} min")
    print(f"  {'TOTAL':<45s}  {total_wall/60:.1f} min")
    print()
    print("Checkpoints saved to:", CKPT_ROOT)
    print("Results CSVs saved to:", CKPT_ROOT)
    print()
    print("Next steps:")
    print("  1. Publish /kaggle/working/checkpoints/ as a Kaggle dataset.")
    print("  2. Mount it in subsequent notebooks as CKPT_DATASET_DIR.")
    print("  3. Compare NB4 (4a vs 4b) and NB5 (stage1end vs final) tables.")


if __name__ == "__main__":
    main()
