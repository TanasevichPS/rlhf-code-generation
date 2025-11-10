#!/usr/bin/env python3
"""
Quick diagnostic script for Modern RLHF.
Run this before training to check for common issues.
"""

import sys
import os
import importlib.util

def check_python_version():
    """Check Python version."""
    print("🐍 Python Version Check")
    print(f"   Python: {sys.version}")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Python version OK")
        return True
    else:
        print("   ❌ Python 3.8+ required")
        return False

def check_imports():
    """Check that required modules can be imported."""
    print("\n📦 Import Checks")

    required_modules = [
        'torch',
        'transformers',
        'numpy',
        'pandas',
        'tqdm'
    ]

    failed = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} - MISSING")
            failed.append(module)

    if failed:
        print(f"\n   Install missing modules: pip install {' '.join(failed)}")
        return False
    return True

def check_cuda():
    """Check CUDA availability."""
    print("\n🖥️  CUDA Check")

    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.get_device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3

            print("   ✅ CUDA available")
            print(f"   Devices: {device_count}")
            print(f"   Current: {device_name}")
            print(f"   Memory: {memory:.1f} GB")
            return True
        else:
            print("   ❌ CUDA not available - will use CPU (slow)")
            return False
    except Exception as e:
        print(f"   ❌ CUDA check failed: {e}")
        return False

def check_modern_rlhf():
    """Check Modern RLHF modules."""
    print("\n🤖 Modern RLHF Module Check")

    modules = [
        'modern_rlhf.config',
        'modern_rlhf.reward_model',
        'modern_rlhf.trainer',
        'modern_rlhf.metrics',
        'modern_rlhf.pipeline'
    ]

    failed = []
    for module in modules:
        try:
            spec = importlib.util.find_spec(module)
            if spec is None:
                raise ImportError(f"Module {module} not found")
            print(f"   ✅ {module}")
        except Exception as e:
            print(f"   ❌ {module} - {e}")
            failed.append(module)

    return len(failed) == 0

def check_data():
    """Check data availability."""
    print("\n📁 Data Check")

    data_paths = [
        'conala-corpus',
        'datasets_for_training',
        'datasets_for_eval'
    ]

    found = False
    for path in data_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                contents = os.listdir(path)
                print(f"   ✅ {path}/ ({len(contents)} items)")
                found = True
            else:
                print(f"   ✅ {path} (file)")
                found = True

    if not found:
        print("   ❌ No data directories found")
        print("   Create directories or download CoNaLa dataset")
        return False

    return True

def check_config():
    """Check configuration."""
    print("\n⚙️  Configuration Check")

    try:
        from modern_rlhf.config import get_research_config

        config = get_research_config()
        print("   ✅ Config created successfully")
        print(f"   Model: {config.model.base_model_name}")
        print(f"   Target BERTScore: {config.evaluation.target_bertscore}")
        print(f"   Target CodeBLEU: {config.evaluation.target_codebleu}")

        # Check target metrics are reasonable
        if config.evaluation.target_bertscore > 0.5:
            print("   ✅ Target metrics look reasonable")
            return True
        else:
            print("   ⚠️  Target metrics seem low - may need adjustment")
            return True

    except Exception as e:
        print(f"   ❌ Config check failed: {e}")
        return False

def main():
    """Run all diagnostic checks."""
    print("=" * 70)
    print("🔍 MODERN RLHF QUICK DIAGNOSTICS")
    print("=" * 70)
    print("This script checks for common issues before training.")
    print("=" * 70)

    checks = [
        check_python_version,
        check_imports,
        check_cuda,
        check_modern_rlhf,
        check_data,
        check_config,
    ]

    results = []
    for check in checks:
        try:
            results.append(check())
        except Exception as e:
            print(f"   ❌ {check.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 70)
    print("📊 DIAGNOSTIC RESULTS")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    check_names = [
        "Python Version",
        "Required Imports",
        "CUDA Availability",
        "Modern RLHF Modules",
        "Data Availability",
        "Configuration"
    ]

    for i, (name, result) in enumerate(zip(check_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1:2d}. {name}: {status}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All checks passed! Ready to train.")
        print("Run: python run_modern_rlhf.py --config research")
    elif passed >= total - 1:  # Allow 1 failure
        print("\n⚠️  Most checks passed. Training may work with limitations.")
        print("Run: python run_modern_rlhf.py --config fast")
    else:
        print("\n❌ Multiple issues found. Fix them before training.")

    print("=" * 70)

if __name__ == "__main__":
    main()
