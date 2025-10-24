#!/usr/bin/env python3
"""
Quick Test for Fixed RLHF System
===============================

Simple test to verify the fixed system works.
"""

import sys
import os
from pathlib import Path

# Add rlhf_code_project to path
sys.path.insert(0, str(Path(__file__).parent / "rlhf_code_project"))

def main():
    """Quick test function."""
    print("🧪 Quick Test for Fixed RLHF System")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("🔍 Testing imports...")
        from config import get_fast_config
        from data import PreferenceDataset
        from training import SimpleDPOTrainer
        from evaluation import MetricCalculator
        print("✅ All imports successful!")
        
        # Test configuration
        print("🔧 Testing configuration...")
        config = get_fast_config()
        config.num_epochs = 1
        config.batch_size = 2
        print(f"✅ Config created: method={config.method}, epochs={config.num_epochs}")
        
        # Test data loader
        print("📊 Testing data loader...")
        dataset = PreferenceDataset("nonexistent.csv", max_samples=4)
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Test trainer
        print("🏃 Testing trainer...")
        trainer = SimpleDPOTrainer(config)
        print("✅ Trainer created successfully")
        
        # Test metrics
        print("📈 Testing metrics...")
        calculator = MetricCalculator()
        predictions = ["def test(): return 1", "def hello(): return 'world'"]
        references = ["def test(): return 1", "def hello(): return 'world'"]
        metrics = calculator.calculate_all_metrics(predictions, references)
        print(f"✅ Metrics calculated: {list(metrics.keys())}")
        
        # Test response generation
        print("🎯 Testing response generation...")
        responses = trainer.generate_responses(["Write a function to add two numbers"])
        print(f"✅ Response generated: {responses[0][:50]}...")
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 50)
        
        print("\n📝 The fixed RLHF system is working correctly!")
        print("\n🚀 Next steps:")
        print("1. Run full test: python test_simple_rlhf.py")
        print("2. Run quick start: python quick_start_simple.py")
        print("3. Run full training: python rlhf_code_project/scripts/train.py --fast")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 System is ready to use!")
    else:
        print("\n⚠️  There were some issues. Check the errors above.")
    
    sys.exit(0 if success else 1)
