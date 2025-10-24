#!/usr/bin/env python3
"""
Minimal Installation Script
===========================

Install only the essential packages for the RLHF system to work.
"""

import subprocess
import sys

def install_package(package, force_reinstall=False):
    """Install a package using pip."""
    try:
        cmd = [sys.executable, "-m", "pip", "install", package]
        if force_reinstall:
            cmd.append("--force-reinstall")
        subprocess.check_call(cmd)
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install minimal required packages."""
    print("🔧 Installing minimal dependencies for RLHF system...")
    print("=" * 60)
    
    # Essential packages that should work
    packages = [
        "numpy>=2.0.0",      # Compatible NumPy version (fixes lighteval conflict)
        "evaluate",           # For BERTScore, BLEU, ROUGE
        "codebleu",          # For CodeBLEU
        "pandas",            # For data processing
        "tqdm",              # For progress bars
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package in packages:
        print(f"📦 Installing {package}...")
        if install_package(package):
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"📊 Installation Results: {success_count}/{total_count} packages installed")
    
    if success_count == total_count:
        print("🎉 All packages installed successfully!")
        print("\n📝 Next steps:")
        print("1. Run: python test_basic.py")
        print("2. Run: python run_simplified_rlhf.py")
        return True
    else:
        print("⚠️  Some packages failed to install.")
        print("The system will still work with the Simple DPO trainer.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
