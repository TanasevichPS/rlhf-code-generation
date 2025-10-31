# Fix environment dependencies for Modern RLHF
# This script fixes PyTorch and NumPy version issues

Write-Host "Fixing environment dependencies..." -ForegroundColor Cyan

# Step 1: Downgrade NumPy to < 2.0 to avoid ABI issues
Write-Host "`nStep 1: Installing NumPy < 2.0..." -ForegroundColor Yellow
python -m pip install "numpy<2.0" --force-reinstall --no-cache-dir
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: NumPy installation had issues, but continuing..." -ForegroundColor Yellow
}

# Step 2: Upgrade PyTorch to >= 2.1
Write-Host "`nStep 2: Upgrading PyTorch to >= 2.1..." -ForegroundColor Yellow
# Check if CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>$null
if ($?) {
    $cuda_available = python -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2>$null
    if ($cuda_available -eq "1") {
        Write-Host "Installing PyTorch 2.1+ with CUDA support..." -ForegroundColor Green
        python -m pip install torch>=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
    } else {
        Write-Host "Installing PyTorch 2.1+ CPU version..." -ForegroundColor Green
        python -m pip install torch>=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --upgrade
    }
} else {
    # Try CUDA version first, fallback to CPU
    Write-Host "Installing PyTorch 2.1+ with CUDA 11.8 support..." -ForegroundColor Green
    python -m pip install torch>=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CUDA installation failed, trying CPU version..." -ForegroundColor Yellow
        python -m pip install torch>=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --upgrade
    }
}

# Step 3: Reinstall transformers to ensure it detects PyTorch correctly
Write-Host "`nStep 3: Reinstalling transformers..." -ForegroundColor Yellow
python -m pip install transformers>=4.21.0 --upgrade --force-reinstall

# Step 4: Verify installation
Write-Host "`nStep 4: Verifying installation..." -ForegroundColor Yellow
$verifyScript = @"
import sys
try:
    import torch
    import transformers
    import numpy as np
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    
    # Check PyTorch version
    torch_version = torch.__version__.split('+')[0]
    major, minor = map(int, torch_version.split('.')[:2])
    if major > 2 or (major == 2 and minor >= 1):
        print("✅ PyTorch version is >= 2.1")
    else:
        print(f"❌ PyTorch version {torch.__version__} is < 2.1")
        sys.exit(1)
    
    # Check NumPy version
    np_version = np.__version__.split('.')
    if int(np_version[0]) < 2:
        print("✅ NumPy version is < 2.0")
    else:
        print(f"❌ NumPy version {np.__version__} is >= 2.0")
        sys.exit(1)
    
    # Test transformers import
    from transformers import AutoModel
    print("✅ Transformers can import AutoModel successfully")
    
    # Test CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ️  CUDA is not available (using CPU)")
    
    print("\n✅ All checks passed! Environment is ready.")
except Exception as e:
    print(f"❌ Error during verification: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@
python -c $verifyScript

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Environment fixed successfully!" -ForegroundColor Green
    Write-Host "You can now run: python run_modern_rlhf.py" -ForegroundColor Green
} else {
    Write-Host "`n❌ Environment fix had issues. Please check the errors above." -ForegroundColor Red
    exit 1
}

