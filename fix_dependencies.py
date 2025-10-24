#!/usr/bin/env python3
"""
Fix Dependencies Script
======================

Script to resolve dependency conflicts and install required packages.
"""

import subprocess
import sys

def run_command(cmd):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Command successful: {cmd}")
            return True
        else:
            print(f"❌ Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception running command {cmd}: {e}")
        return False

def main():
    """Fix dependency conflicts."""
    print("🔧 Fixing dependency conflicts...")
    print("=" * 60)
    
    # Step 1: Upgrade pip
    print("📦 Step 1: Upgrading pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    print()
    
    # Step 2: Install compatible NumPy version
    print("📦 Step 2: Installing compatible NumPy...")
    run_command(f"{sys.executable} -m pip install 'numpy>=2.0.0' --force-reinstall")
    print()
    
    # Step 3: Install essential packages
    print("📦 Step 3: Installing essential packages...")
    packages = [
        "evaluate",
        "codebleu", 
        "pandas",
        "tqdm"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        run_command(f"{sys.executable} -m pip install {package}")
        print()
    
    # Step 4: Check for conflicts
    print("📦 Step 4: Checking for remaining conflicts...")
    run_command(f"{sys.executable} -m pip check")
    print()
    
    print("=" * 60)
    print("🎉 Dependency fixing completed!")
    print("\n📝 Next steps:")
    print("1. Run: python test_basic.py")
    print("2. Run: python run_simplified_rlhf.py")

if __name__ == "__main__":
    main()
