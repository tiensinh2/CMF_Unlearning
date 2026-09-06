"""
_smoke_test_notebooks.py — Local pre-flight check for all 5 notebooks.

Runs WITHOUT executing cells (no GPU, no data download).
Checks:
  1. Every notebook is valid JSON / valid nbformat.
  2. Every code cell parses as valid Python.
  3. No cell references a variable that hasn't been assigned in a prior cell
     of the same notebook (basic NameError detection for known patterns).
  4. No cell uses the stale SVD_alpha_r=100 / SVD_alpha_f=3 values.
  5. No cell references undefined CMF_EPOCHS (without _BY_METHOD suffix).
  6. No cell references undefined TOTAL_EPOCHS (without assignment).
  7. _validate_table4.py passes 72/72.

Usage:
    python _smoke_test_notebooks.py
"""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

NOTEBOOKS = [
    "notebooks/01_train_full.ipynb",
    "notebooks/02_retrain_oracle.ipynb",
    "notebooks/03_unlearn_no_cmf.ipynb",
    "notebooks/04_unlearn_cmf.ipynb",
    "notebooks/05_unlearn_cmf_budget_shared.ipynb",
]

STALE_PATTERNS = [
    # (regex pattern, description, allow_in_comment)
    (r"SVD_alpha_r\s*=\s*100[^0]",    "stale SVD_alpha_r=100",         False),
    (r"SVD_alpha_f\s*=\s*3[^0]",      "stale SVD_alpha_f=3 (not 30)",  False),
    (r"tarun_impair_lr\s*=\s*1e-4",   "stale tarun_impair_lr=1e-4",    False),
    (r"PRETRAIN_LR\s*=\s*0\.05",      "stale PRETRAIN_LR=0.05",        False),
    (r"PRETRAIN_LR\s*=\s*5e-2",       "stale PRETRAIN_LR=5e-2",        False),
    (r"RETRAIN_EPOCHS\s*=\s*50",      "stale RETRAIN_EPOCHS=50",       False),
]

# Variables that must be defined before use; simple pattern check
UNDEFINED_PATTERNS = [
    # (undefined ref pattern, correct alternative, notebook_filter)
    (r"\bCMF_EPOCHS\b(?!_BY_METHOD)", "CMF_EPOCHS without _BY_METHOD suffix",
     ["04_unlearn_cmf.ipynb", "05_unlearn_cmf_budget_shared.ipynb"]),
]

errors = []
warnings = []

# ─────────────────────────────────────────────────────────────────────────────
# Step 0: validate table4
# ─────────────────────────────────────────────────────────────────────────────
print("Step 0: Running _validate_table4.py...")
result = subprocess.run([sys.executable, "_validate_table4.py"],
                        capture_output=True, text=True)
if result.returncode != 0:
    errors.append("_validate_table4.py FAILED:\n" + result.stdout + result.stderr)
    print("  FAIL:", result.stdout.strip())
else:
    print("  PASS:", result.stdout.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: check each notebook
# ─────────────────────────────────────────────────────────────────────────────
for nb_path in NOTEBOOKS:
    nb_name = os.path.basename(nb_path)
    print(f"\nChecking {nb_name}...")

    # 1a. File exists
    if not os.path.exists(nb_path):
        errors.append(f"{nb_name}: FILE NOT FOUND at {nb_path}")
        print("  FAIL: file not found")
        continue

    # 1b. Valid JSON
    try:
        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"{nb_name}: INVALID JSON — {e}")
        print(f"  FAIL: invalid JSON — {e}")
        continue

    # 1c. Has cells
    cells = nb.get("cells", [])
    if not cells:
        errors.append(f"{nb_name}: no cells found")
        continue

    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    print(f"  {len(cells)} cells ({len(code_cells)} code cells)")

    # 1d. Collect all source text
    all_code_lines = []   # list of (cell_idx, line_no_in_cell, line_text)
    for ci, cell in enumerate(code_cells):
        src = cell.get("source", [])
        if isinstance(src, list):
            src_text = "".join(src)
        else:
            src_text = src
        for li, line in enumerate(src_text.splitlines()):
            all_code_lines.append((ci, li, line))

    # 1e. Parse each cell as Python
    parse_fails = 0
    for ci, cell in enumerate(code_cells):
        src = cell.get("source", [])
        src_text = "".join(src) if isinstance(src, list) else src
        try:
            ast.parse(src_text)
        except SyntaxError as e:
            parse_fails += 1
            errors.append(f"{nb_name} cell {ci}: SyntaxError — {e}")
    if parse_fails == 0:
        print(f"  OK: all {len(code_cells)} code cells parse as valid Python")
    else:
        print(f"  FAIL: {parse_fails} cells have SyntaxErrors")

    # 1f. Check for stale patterns
    stale_found = 0
    for ci, li, line in all_code_lines:
        line_stripped = line.strip()
        is_comment = line_stripped.startswith("#")
        for pattern, desc, allow_comment in STALE_PATTERNS:
            if allow_comment and is_comment:
                continue
            if not is_comment and re.search(pattern, line):
                errors.append(
                    f"{nb_name} cell {ci} line {li}: STALE VALUE — {desc!r}: {line.strip()!r}"
                )
                stale_found += 1
    if stale_found == 0:
        print(f"  OK: no stale shell-derived hyperparameter values")
    else:
        print(f"  FAIL: {stale_found} stale hyperparameter values found")

    # 1g. Check for known undefined-variable patterns
    undef_found = 0
    for pat, desc, nb_filter in UNDEFINED_PATTERNS:
        if nb_filter and nb_name not in nb_filter:
            continue
        for ci, li, line in all_code_lines:
            line_stripped = line.strip()
            if line_stripped.startswith("#"):
                continue
            if re.search(pat, line):
                # Distinguish assignment vs usage
                # Assignment: starts with the variable name followed by =
                if re.match(r"^\s*CMF_EPOCHS\s*=", line) or re.match(r"^\s*TOTAL_EPOCHS\s*=", line):
                    continue  # this is a definition, fine
                errors.append(
                    f"{nb_name} cell {ci} line {li}: UNDEFINED REF — {desc!r}: {line.strip()!r}"
                )
                undef_found += 1
    if undef_found == 0 and any(nb_name in (nb_filter or []) for _, _, nb_filter in UNDEFINED_PATTERNS):
        print(f"  OK: no undefined variable references detected")

    # 1h. Check TOTAL_EPOCHS is not referenced (NB5 specific)
    if nb_name == "05_unlearn_cmf_budget_shared.ipynb":
        for ci, li, line in all_code_lines:
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("print"):
                continue
            if "TOTAL_EPOCHS" in line and "=" not in line:
                # Reference to TOTAL_EPOCHS in non-assignment, non-comment, non-print line
                errors.append(
                    f"{nb_name} cell {ci} line {li}: possible TOTAL_EPOCHS reference "
                    f"outside assignment/comment: {stripped!r}"
                )

    # 1i. Check that CKPT_ROOT points to Kaggle path (not local ./checkpoints/)
    for ci, cell in enumerate(code_cells):
        src = cell.get("source", [])
        src_text = "".join(src) if isinstance(src, list) else src
        for match in re.finditer(r"CKPT_ROOT\s*=\s*['\"]([^'\"]+)['\"]", src_text):
            path = match.group(1)
            if path.startswith("./checkpoints") or path == "checkpoints":
                warnings.append(
                    f"{nb_name} cell {ci}: CKPT_ROOT={path!r} — "
                    f"should be /kaggle/working/checkpoints/... on Kaggle"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: check config.py consistency
# ─────────────────────────────────────────────────────────────────────────────
print("\nChecking config.py ...")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", "config.py")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    # Pretrain LR must be 1e-2
    lr = cfg.TRAIN_RESNET["lr_init"]
    if abs(lr - 1e-2) > 1e-12:
        errors.append(f"config.py: TRAIN_RESNET lr_init={lr!r} — expected 1e-2 (Table 4)")
    else:
        print(f"  OK: TRAIN_RESNET lr_init={lr} (Table 4: 1e-2)")

    # Retrain epochs
    re_ep = cfg.RETRAIN_RESNET.get("cifar10", {}).get("epochs", -1)
    if re_ep != 200:
        errors.append(f"config.py: RETRAIN_RESNET cifar10 epochs={re_ep} — expected 200 (Table 4)")
    else:
        print(f"  OK: RETRAIN_RESNET cifar10 epochs={re_ep} (Table 4: 200)")

    # SVD alpha_r
    svd_ar = cfg.UNLEARN_EXTRA["SVD"]["SVD_alpha_r"]
    if svd_ar != 1000:
        errors.append(f"config.py: SVD_alpha_r={svd_ar} — expected 1000 (Table 4)")
    else:
        print(f"  OK: SVD_alpha_r={svd_ar} (Table 4: 1000)")

    # UNSIR impair lr
    tarun_lr = cfg.UNLEARN_EXTRA["tarun"]["tarun_impair_lr"]
    if abs(tarun_lr - 5e-5) > 1e-12:
        errors.append(f"config.py: tarun_impair_lr={tarun_lr!r} — expected 5e-5 (Table 4)")
    else:
        print(f"  OK: tarun_impair_lr={tarun_lr} (Table 4: 5e-5)")

except Exception as e:
    errors.append(f"config.py: import/check failed — {e}")
    print(f"  FAIL: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: check archive exists (stale checkpoints archived)
# ─────────────────────────────────────────────────────────────────────────────
print("\nChecking stale checkpoint archive ...")
archive_path = "archive_stale_shell_hparams/checkpoints_shell_derived"
if not os.path.isdir(archive_path):
    warnings.append(
        f"archive_stale_shell_hparams/checkpoints_shell_derived/ not found — "
        f"have stale checkpoints been archived?"
    )
    print("  WARNING: stale archive not found")
else:
    n_archived = sum(len(files) for _, _, files in os.walk(archive_path))
    print(f"  OK: {n_archived} files in stale archive ({archive_path})")

# Check local ./checkpoints/ is clean (no .pt files)
print("Checking local ./checkpoints/ is clean ...")
local_pts = []
if os.path.isdir("checkpoints"):
    for root, dirs, files in os.walk("checkpoints"):
        for f in files:
            if f.endswith(".pt"):
                local_pts.append(os.path.join(root, f))
if local_pts:
    warnings.append(
        f"{len(local_pts)} .pt files found in ./checkpoints/ — "
        f"these should have been archived: {local_pts[:3]}"
    )
    print(f"  WARNING: {len(local_pts)} .pt files in ./checkpoints/")
else:
    print("  OK: ./checkpoints/ has no .pt files (clean)")


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
if errors:
    print(f"SMOKE TEST FAILED — {len(errors)} error(s):")
    for i, e in enumerate(errors, 1):
        print(f"  [{i}] {e}")
else:
    print("SMOKE TEST PASSED — 0 errors")

if warnings:
    print(f"\n{len(warnings)} warning(s):")
    for i, w in enumerate(warnings, 1):
        print(f"  [W{i}] {w}")

print("=" * 70)
sys.exit(1 if errors else 0)
