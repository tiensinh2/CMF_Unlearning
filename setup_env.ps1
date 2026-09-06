# ============================================================
# setup_env.ps1  –  CMF_Unlearning environment bootstrap
#
# Paper spec: Python 3.11, PyTorch 2.5.1, CUDA 12.1
#
# Usage (PowerShell):
#   .\setup_env.ps1               # CPU-only (default)
#   .\setup_env.ps1 -Cuda         # CUDA 12.1 GPU wheels
#   .\setup_env.ps1 -EnvName myenv
#
# Prerequisites: Python 3.11 must be on PATH (py -3.11 launcher
# or `python` resolving to 3.11).  Install from python.org if
# needed.
# ============================================================

param(
    [string]$EnvName = ".venv",
    [switch]$Cuda
)

$ErrorActionPreference = "Stop"

# ── 1. Locate Python 3.11 ────────────────────────────────────
Write-Host "==> Locating Python 3.11 ..." -ForegroundColor Cyan
$py = $null
foreach ($candidate in @("py -3.11", "python3.11", "python")) {
    try {
        $ver = & ($candidate.Split(" ")[0]) ($candidate.Split(" ")[1..99]) -c `
            "import sys; print(sys.version_info[:2])" 2>$null
        if ($ver -like "*(3, 11)*") { $py = $candidate; break }
    } catch {}
}
if (-not $py) {
    Write-Error ("Python 3.11 not found. Install from https://www.python.org/downloads/ " +
                 "and re-run this script.")
}
Write-Host "   Found: $py" -ForegroundColor Green

# ── 2. Create virtual environment ────────────────────────────
if (-not (Test-Path $EnvName)) {
    Write-Host "==> Creating venv '$EnvName' ..." -ForegroundColor Cyan
    & ($py.Split(" ")[0]) ($py.Split(" ")[1..99]) -m venv $EnvName
} else {
    Write-Host "==> venv '$EnvName' already exists, skipping creation." -ForegroundColor Yellow
}

$pip = Join-Path $EnvName "Scripts\pip.exe"
$python = Join-Path $EnvName "Scripts\python.exe"

# ── 3. Upgrade pip / setuptools ──────────────────────────────
Write-Host "==> Upgrading pip, setuptools, wheel ..." -ForegroundColor Cyan
& $pip install --upgrade pip setuptools wheel

# ── 4. Install PyTorch 2.5.1 ─────────────────────────────────
Write-Host "==> Installing PyTorch 2.5.1 ..." -ForegroundColor Cyan
if ($Cuda) {
    # CUDA 12.1 wheels from the official PyTorch index
    & $pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 `
        --index-url https://download.pytorch.org/whl/cu121
} else {
    # CPU-only (no CUDA)
    & $pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 `
        --index-url https://download.pytorch.org/whl/cpu
}

# ── 5. Install remaining dependencies ────────────────────────
Write-Host "==> Installing remaining requirements ..." -ForegroundColor Cyan
# Install everything except the torch lines already handled above
& $pip install `
    timm>=0.9.12 `
    "numpy>=1.24,<3.0" `
    "scipy>=1.11" `
    "scikit-learn>=1.3" `
    "pandas>=1.5" `
    "matplotlib>=3.7" `
    "seaborn>=0.12" `
    "tqdm>=4.65" `
    "pytorch-lightning>=2.0" `
    "lightning>=2.0" `
    "lightning-utilities>=0.9" `
    "torchmetrics>=1.0"

# ── 6. Smoke-test ─────────────────────────────────────────────
Write-Host "==> Running smoke test ..." -ForegroundColor Cyan
& $python -c @"
import sys
print('Python:', sys.version)
import torch, torchvision, timm, numpy, sklearn, pandas
print('torch       :', torch.__version__)
print('torchvision :', torchvision.__version__)
print('timm        :', timm.__version__)
print('numpy       :', numpy.__version__)
print('scikit-learn:', sklearn.__version__)
print('pandas      :', pandas.__version__)
print('CUDA available:', torch.cuda.is_available())
print('All imports OK.')
"@

Write-Host ""
Write-Host "==> Environment ready.  Activate with:" -ForegroundColor Green
Write-Host "      .$EnvName\Scripts\Activate.ps1" -ForegroundColor White
