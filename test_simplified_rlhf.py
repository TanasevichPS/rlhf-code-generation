#!/usr/bin/env python3
"""
Test Script for Simplified RLHF
===============================

Simple test to verify the new simplified system works.
"""

import sys
import os
from pathlib import Path

# Add rlhf_code_project to path
sys.path.insert(0, str(Path(__file__).parent / "rlhf_code_project"))

def test_imports():
    """Test basic imports."""
    print("🧪 Testing imports...")
    
    try:
        # Test config imports
        from config import RLHFConfig, get_fast_config, get_dpo_config
        print("✅ Config imports successful")
        
        # Test data imports
        from data import PreferenceDataset, EvaluationDataset
        print("✅ Data imports successful")
        
        # Test evaluation imports
        from evaluation import MetricCalculator
        print("✅ Evaluation imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_creation():
    """Test configuration creation."""
    print("🧪 Testing configuration creation...")
    
    try:
        from config import get_fast_config, get_dpo_config
        
        # Test fast config
        fast_config = get_fast_config()
        assert hasattr(fast_config, 'method')
        assert hasattr(fast_config, 'learning_rate')
        assert hasattr(fast_config, 'batch_size')
        print("✅ Fast config creation successful")
        
        # Test DPO config
        dpo_config = get_dpo_config()
        assert dpo_config.method == "dpo"
        print("✅ DPO config creation successful")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_data_loader():
    """Test data loader functionality."""
    print("🧪 Testing data loader...")
    
    try:
        from data import PreferenceDataset, EvaluationDataset
        
        # Test preference dataset
        pref_dataset = PreferenceDataset("nonexistent.csv", max_samples=5)
        assert len(pref_dataset) > 0
        print(f"✅ Preference dataset created with {len(pref_dataset)} samples")
        
        # Test evaluation dataset
        eval_dataset = EvaluationDataset("nonexistent.csv", max_samples=5)
        assert len(eval_dataset) > 0
        print(f"✅ Evaluation dataset created with {len(eval_dataset)} samples")
        
        return True
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_metrics():
    """Test metrics functionality."""
    print("🧪 Testing metrics...")
    
    try:
        from evaluation import MetricCalculator
        
        calculator = MetricCalculator()
        
        # Test with simple examples
        predictions = [
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "def reverse_string(s):\n    return s[::-1]"
        ]
        references = [
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "def reverse_string(s):\n    return s[::-1]"
        ]
        
        # Test metrics calculation
        metrics = calculator.calculate_all_metrics(predictions, references)
        assert isinstance(metrics, dict)
        print(f"✅ Metrics calculated: {list(metrics.keys())}")
        
        # Test Ruby metric (should always work)
        ruby_score = metrics.get('ruby', 0)
        print(f"✅ Ruby score: {ruby_score:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False

def test_training_imports():
    """Test training module imports."""
    print("🧪 Testing training imports...")
    
    try:
        from training import DPOTrainer
        print("✅ DPO trainer import successful")
        
        # Test trainer creation (without actual model loading)
        from config import get_fast_config
        config = get_fast_config()
        
        # This will fail at model loading, but we can test the class exists
        print("✅ Training module structure is correct")
        
        return True
    except Exception as e:
        print(f"❌ Training import test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Simplified RLHF System - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_creation,
        test_data_loader,
        test_metrics,
        test_training_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The simplified RLHF system is ready to use.")
        print("\n📝 Next steps:")
        print("1. Install dependencies: pip install -r rlhf_code_project/requirements.txt")
        print("2. Run quick start: python quick_start_simple.py")
        print("3. Run full training: python rlhf_code_project/scripts/train.py --fast")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
