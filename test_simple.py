#!/usr/bin/env python3
"""
Simple Test for Modern RLHF Framework
=====================================

Basic test to verify the framework works with minimal dependencies.
"""

import sys
import os
from pathlib import Path

# Add modern_rlhf to path
sys.path.insert(0, str(Path(__file__).parent / "modern_rlhf"))

def test_basic_imports():
    """Test basic imports without heavy dependencies."""
    print("🧪 Testing basic imports...")
    
    try:
        # Test config imports
        from modern_rlhf.config import ModernRLHFConfig, get_research_config
        print("✅ Config imports successful!")
        
        # Test data loader imports
        from modern_rlhf.data_loader import ModernDataLoader
        print("✅ Data loader imports successful!")
        
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_config_creation():
    """Test configuration creation."""
    print("🧪 Testing configuration creation...")
    
    try:
        from modern_rlhf.config import get_research_config
        
        config = get_research_config()
        
        # Check basic properties
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'evaluation')
        assert hasattr(config, 'data')
        
        print("✅ Configuration creation successful!")
        return True
    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        return False

def test_data_loader():
    """Test data loader functionality."""
    print("🧪 Testing data loader...")
    
    try:
        from modern_rlhf.config import get_research_config
        from modern_rlhf.data_loader import ModernDataLoader
        
        config = get_research_config()
        data_loader = ModernDataLoader(config)
        
        # Test synthetic data generation
        synthetic_data = data_loader._generate_synthetic_data()
        assert len(synthetic_data) > 0
        
        print("✅ Data loader test successful!")
        return True
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_metrics_basic():
    """Test basic metrics functionality."""
    print("🧪 Testing basic metrics...")
    
    try:
        from modern_rlhf.metrics import ModernMetricsEvaluator
        
        evaluator = ModernMetricsEvaluator()
        
        # Test with simple examples
        predictions = ["def test(): return 1", "def hello(): return 'world'"]
        references = ["def test(): return 1", "def hello(): return 'world'"]
        
        # Test BLEU (should work without external dependencies)
        bleu_result = evaluator.compute_bleu(predictions, references)
        assert bleu_result.metric_name == "bleu"
        
        # Test Ruby metric (custom implementation)
        ruby_result = evaluator.compute_ruby(predictions, references)
        assert ruby_result.metric_name == "ruby"
        
        print("✅ Basic metrics test successful!")
        return True
    except Exception as e:
        print(f"❌ Basic metrics test failed: {e}")
        return False

def main():
    """Run basic tests."""
    print("🧪 Modern RLHF Framework - Simple Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_config_creation,
        test_data_loader,
        test_metrics_basic
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
        print("🎉 All basic tests passed! The framework is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
