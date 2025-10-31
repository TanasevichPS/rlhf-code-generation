# GPU Verification Script
"""Verify that all models will use GPU."""

import torch
import sys

print("="*70)
print("GPU VERIFICATION")
print("="*70)

# Check CUDA availability
if not torch.cuda.is_available():
    print("ERROR: CUDA GPU is not available!")
    print("Please ensure:")
    print("  1. NVIDIA GPU drivers are installed")
    print("  2. CUDA toolkit is installed")
    print("  3. PyTorch with CUDA support is installed")
    sys.exit(1)

print("[OK] CUDA is available")
print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
print(f"[OK] CUDA Version: {torch.version.cuda}")
print(f"[OK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"[OK] PyTorch Version: {torch.__version__}")

# Test tensor creation on GPU
try:
    test_tensor = torch.randn(10, 10).cuda()
    print(f"[OK] Test tensor created on GPU: {test_tensor.device}")
    del test_tensor
    torch.cuda.empty_cache()
except Exception as e:
    print(f"ERROR: Failed to create tensor on GPU: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("GPU VERIFICATION PASSED - Ready for training!")
print("="*70)

