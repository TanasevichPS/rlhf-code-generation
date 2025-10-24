#!/usr/bin/env python3
"""
Basic Test for RLHF System
==========================

Simple test that should work with minimal dependencies.
"""

import sys
import os
from pathlib import Path

# Add rlhf_code_project to path
sys.path.insert(0, str(Path(__file__).parent / "rlhf_code_project"))

def test_basic_imports():
    """Test basic imports."""
    print("🧪 Testing basic imports...")
    
    try:
        # Test config
        from config import RLHFConfig, get_fast_config
        print("✅ Config imports successful")
        
        # Test data
        from data import PreferenceDataset
        print("✅ Data imports successful")
        
        # Test training
        from training import SimpleDPOTrainer
        print("✅ Training imports successful")
        
        # Test evaluation
        from evaluation import MetricCalculator
        print("✅ Evaluation imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("🧪 Testing basic functionality...")
    
    try:
        from config import get_fast_config
        from data import PreferenceDataset
        from training import SimpleDPOTrainer
        from evaluation import MetricCalculator
        
        # Test config
        config = get_fast_config()
        print(f"✅ Config created: method={config.method}")
        
        # Test dataset
        dataset = PreferenceDataset("nonexistent.csv", max_samples=3)
        print(f"✅ Dataset created: {len(dataset)} samples")
        
        # Test trainer
        trainer = SimpleDPOTrainer(config)
        print("✅ Trainer created")
        
        # Test metrics
        calculator = MetricCalculator()
        print("✅ Metrics calculator created")
        
        return True
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Run basic tests."""
    print("🧪 Basic RLHF System Test")
    print("=" * 40)
    
    tests = [test_basic_imports, test_basic_functionality]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Basic system is working!")
        return True
    else:
        print("⚠️  Some tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
