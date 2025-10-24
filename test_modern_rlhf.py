#!/usr/bin/env python3
"""
Test Script for Modern RLHF Framework
=====================================

Simple test to verify the framework works correctly.
"""

import sys
import os
from pathlib import Path

# Add modern_rlhf to path
sys.path.insert(0, str(Path(__file__).parent / "modern_rlhf"))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from modern_rlhf import ModernRLHFPipeline, ModernRLHFConfig
        from modern_rlhf.config import get_research_config, get_production_config, get_fast_config
        from modern_rlhf.metrics import ModernMetricsEvaluator
        from modern_rlhf.reward_model import ModernRewardModel
        from modern_rlhf.trainer import ModernRLHFTrainer
        from modern_rlhf.data_loader import ModernDataLoader
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration creation."""
    print("🧪 Testing configuration...")
    
    try:
        from modern_rlhf.config import get_research_config
        
        config = get_research_config()
        
        # Check basic properties
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'evaluation')
        assert hasattr(config, 'data')
        
        # Check model config
        assert config.model.base_model_name is not None
        assert config.model.reward_model_name is not None
        
        # Check training config
        assert config.training.learning_rate > 0
        assert config.training.batch_size > 0
        
        # Check evaluation config
        assert config.evaluation.target_bertscore > 0
        assert config.evaluation.target_codebleu > 0
        
        print("✅ Configuration test passed!")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_metrics():
    """Test metrics evaluation."""
    print("🧪 Testing metrics...")
    
    try:
        from modern_rlhf.metrics import ModernMetricsEvaluator
        
        evaluator = ModernMetricsEvaluator()
        
        # Test with simple examples
        predictions = [
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "def reverse_string(s):\n    return s[::-1]"
        ]
        references = [
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "def reverse_string(s):\n    return s[::-1]"
        ]
        
        # Test individual metrics
        bertscore_result = evaluator.compute_bertscore(predictions, references)
        assert bertscore_result.metric_name == "bertscore"
        
        codebleu_result = evaluator.compute_codebleu(predictions, references)
        assert codebleu_result.metric_name == "codebleu"
        
        bleu_result = evaluator.compute_bleu(predictions, references)
        assert bleu_result.metric_name == "bleu"
        
        rouge_result = evaluator.compute_rouge(predictions, references)
        assert rouge_result.metric_name == "rouge"
        
        ruby_result = evaluator.compute_ruby(predictions, references)
        assert ruby_result.metric_name == "ruby"
        
        print("✅ Metrics test passed!")
        return True
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False

def test_data_loader():
    """Test data loader."""
    print("🧪 Testing data loader...")
    
    try:
        from modern_rlhf.config import get_research_config
        from modern_rlhf.data_loader import ModernDataLoader
        
        config = get_research_config()
        data_loader = ModernDataLoader(config)
        
        # Test synthetic data generation
        synthetic_data = data_loader._generate_synthetic_data()
        assert len(synthetic_data) > 0
        
        # Test data filtering
        filtered_data = data_loader._filter_samples(synthetic_data)
        assert len(filtered_data) <= len(synthetic_data)
        
        print("✅ Data loader test passed!")
        return True
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_pipeline_creation():
    """Test pipeline creation."""
    print("🧪 Testing pipeline creation...")
    
    try:
        from modern_rlhf import ModernRLHFPipeline
        from modern_rlhf.config import get_fast_config
        
        config = get_fast_config()
        
        # Create pipeline (this should not fail)
        pipeline = ModernRLHFPipeline(config)
        
        assert pipeline.config is not None
        assert pipeline.data_loader is not None
        assert pipeline.metrics_evaluator is not None
        
        print("✅ Pipeline creation test passed!")
        return True
    except Exception as e:
        print(f"❌ Pipeline creation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Modern RLHF Framework - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_metrics,
        test_data_loader,
        test_pipeline_creation
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
        print("🎉 All tests passed! The framework is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
