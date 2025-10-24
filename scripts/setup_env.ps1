<#
PowerShell environment setup for the RLHF project.

Usage (interactive):
  1. Open PowerShell as Administrator (recommended for conda/use of symlinks).
  2. Run: .\scripts\setup_env.ps1

This script supports three modes:
  1 - GPU with CUDA 11.7 (recommended when you have CUDA 11.x)
  2 - GPU with CUDA 12.1 (if your drivers/toolkit support 12.x)
  3 - CPU-only

The script will create a conda environment named `rlhf` with Python 3.10,
install a matching PyTorch wheel (2.1.x), pin NumPy to <2.0 (to avoid ABI
issues with older compiled extensions), and install common RLHF tooling:
transformers, accelerate, trl, datasets, scikit-learn, and evaluation libs.

Notes:
- If you prefer a non-conda flow, use the comments below to run pip commands
  inside your chosen virtualenv.
- If you have a custom CUDA version, choose the CPU path and install a
  wheel appropriate for your system manually.
#>

param()

function Run-Command([string]$cmd) {
    Write-Host "==> $cmd"
    & pwsh -Command $cmd
}

Write-Host "RLHF environment setup script"
Write-Host "Choose installation type:`n 1) GPU (CUDA 11.7)`n 2) GPU (CUDA 12.1)`n 3) CPU-only"
$choice = Read-Host "Enter 1, 2 or 3"

if ($choice -notin @('1','2','3')) {
    Write-Host "Invalid choice. Exiting."; exit 1
}

$envName = 'rlhf'
Write-Host "Creating conda environment '$envName' with Python 3.10..."
conda create -n $envName python=3.10 -y

Write-Host "Activating environment..."
conda activate $envName

Write-Host "Upgrading pip and installing base packages..."
python -m pip install --upgrade pip setuptools wheel

# Pin numpy to 1.25.x (avoid NumPy 2.0 ABI breakage with some wheels)
Write-Host "Installing pinned NumPy and common dependencies..."
python -m pip install "numpy<2.0"

if ($choice -eq '1') {
    Write-Host "Installing PyTorch 2.1 with CUDA 11.7 via conda (channel: pytorch)..."
    conda install -y -c pytorch pytorch=2.1 torchvision torchaudio cudatoolkit=11.7
} elseif ($choice -eq '2') {
    Write-Host "Installing PyTorch 2.1 with CUDA 12.1 via conda (channel: pytorch)..."
    conda install -y -c pytorch pytorch=2.1 torchvision torchaudio cudatoolkit=12.1
} else {
    Write-Host "Installing CPU-only PyTorch via pip (torch==2.1.*+cpu)..."
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.1.*+cpu torchvision --extra-index-url https://download.pytorch.org/whl/cpu
}

Write-Host "Installing Transformers, Accelerate, TRL, Datasets, and evaluation libs..."
# Use requirements-rlhf.txt if present, otherwise install common packages
if (Test-Path "requirements-rlhf.txt") {
    python -m pip install -r requirements-rlhf.txt
} else {
    python -m pip install transformers accelerate trl datasets scikit-learn nltk sacrebleu bert-score rouge-score sentencepiece tokenizers evaluate wandb
}

Write-Host "Post-install: downloading tokenizer caches and verifying torch..."
python - <<'PY'
import sys
try:
    import torch
    print('torch', torch.__version__, 'cuda:', torch.version.cuda, 'cuda_available:', torch.cuda.is_available())
except Exception as e:
    print('Torch import failed:', e)
    sys.exit(1)

print('Installing NLTK punkt data (if missing)...')
import nltk
nltk.download('punkt')
PY

Write-Host "Environment setup complete. To activate the environment later run: conda activate rlhf"
Write-Host "Recommended next steps: run scripts/prepare_pairs.py (already done), then run scripts/train_reward_model_hf_prod.py --config configs/reward_train.yaml"

exit 0
