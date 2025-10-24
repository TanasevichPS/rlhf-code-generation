#!/usr/bin/env python3
"""
Simple Test for RLHF Code Project
=================================

Test script that works with minimal dependencies.
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
        
        # Test training imports
        from training import SimpleDPOTrainer, DPO_AVAILABLE
        print("✅ Training imports successful")
        print(f"   Full DPO available: {DPO_AVAILABLE}")
        
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

def test_simple_trainer():
    """Test simple trainer functionality."""
    print("🧪 Testing simple trainer...")
    
    try:
        from training import SimpleDPOTrainer
        from config import get_fast_config
        
        config = get_fast_config()
        trainer = SimpleDPOTrainer(config)
        
        # Test mock training step
        mock_batch = {
            'prompts': ['Write a function to calculate factorial'],
            'chosen_responses': ['def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)'],
            'rejected_responses': ['def factorial(n):\n    return 1']
        }
        
        stats = trainer.train_step(mock_batch)
        assert 'loss' in stats
        print(f"✅ Training step successful: loss = {stats['loss']:.4f}")
        
        # Test response generation
        responses = trainer.generate_responses(['Write a function to reverse a string'])
        assert len(responses) == 1
        print(f"✅ Response generation successful: {responses[0][:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Simple trainer test failed: {e}")
        return False

def test_full_pipeline():
    """Test full pipeline with simple trainer."""
    print("🧪 Testing full pipeline...")
    
    try:
        from config import get_fast_config
        from data import PreferenceDataset
        from training import SimpleDPOTrainer
        from evaluation import MetricCalculator
        
        # Create config
        config = get_fast_config()
        config.num_epochs = 1  # Just one epoch for testing
        config.batch_size = 2
        
        # Create dataset
        dataset = PreferenceDataset("nonexistent.csv", max_samples=4)
        
        # Create trainer
        trainer = SimpleDPOTrainer(config)
        
        # Create mock data loader
        class MockDataLoader:
            def __init__(self, dataset):
                self.dataset = dataset
                self.data = [dataset[i] for i in range(len(dataset))]
            
            def __iter__(self):
                # Yield batches
                batch_size = 2
                for i in range(0, len(self.data), batch_size):
                    batch_data = self.data[i:i+batch_size]
                    yield {
                        'prompts': [item['prompt'] for item in batch_data],
                        'chosen_responses': [item['chosen_response'] for item in batch_data],
                        'rejected_responses': [item['rejected_response'] for item in batch_data]
                    }
        
        # Mock training
        mock_loader = MockDataLoader(dataset)
        training_results = trainer.train(mock_loader)
        assert 'training_stats' in training_results
        print("✅ Training completed successfully")
        
        # Test evaluation
        calculator = MetricCalculator()
        predictions = trainer.generate_responses(['Write a function to add two numbers'])
        references = ['def add(a, b):\n    return a + b']
        
        metrics = calculator.calculate_all_metrics(predictions, references)
        print(f"✅ Evaluation completed: {list(metrics.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Simple RLHF System - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_creation,
        test_data_loader,
        test_metrics,
        test_simple_trainer,
        test_full_pipeline
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
        print("🎉 All tests passed! The simple RLHF system is ready to use.")
        print("\n📝 Next steps:")
        print("1. Run quick start: python quick_start_simple.py")
        print("2. Run full training: python rlhf_code_project/scripts/train.py --fast")
        print("3. Install full dependencies for production use")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
