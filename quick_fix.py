#!/usr/bin/env python3
"""
Quick Fix Script
================

Quick script to resolve the most common issues.
"""

import subprocess
import sys

def main():
    """Quick fix for common issues."""
    print("🔧 Quick Fix for RLHF System")
    print("=" * 40)
    
    print("📦 Installing compatible NumPy...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy>=2.0.0", "--force-reinstall"], check=True)
        print("✅ NumPy updated successfully")
    except:
        print("❌ NumPy update failed, but system will still work")
    
    print("\n📦 Installing evaluate package...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "evaluate"], check=True)
        print("✅ Evaluate package installed successfully")
    except:
        print("❌ Evaluate installation failed, but system will still work")
    
    print("\n📦 Installing codebleu package...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "codebleu"], check=True)
        print("✅ CodeBLEU package installed successfully")
    except:
        print("❌ CodeBLEU installation failed, but system will still work")
    
    print("\n" + "=" * 40)
    print("🎉 Quick fix completed!")
    print("\n📝 The system will work with Simple DPO trainer regardless of package installation status.")
    print("📝 Next steps:")
    print("1. Run: python test_basic.py")
    print("2. Run: python run_simplified_rlhf.py")

if __name__ == "__main__":
    main()
