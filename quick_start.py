#!/usr/bin/env python3
"""
Quick Start Script for Modern RLHF
==================================

Minimal script to demonstrate the framework with basic functionality.
"""

import sys
import os
from pathlib import Path

# Add modern_rlhf to path
sys.path.insert(0, str(Path(__file__).parent / "modern_rlhf"))

def main():
    """Quick start demonstration."""
    print("🚀 Modern RLHF Framework - Quick Start")
    print("=" * 50)
    
    try:
        # Import basic components
        from config import ModernRLHFConfig, get_research_config
        from data_loader import ModernDataLoader
        from metrics import ModernMetricsEvaluator
        
        print("✅ All imports successful!")
        
        # Create configuration
        print("🔧 Creating configuration...")
        config = get_research_config()
        
        # Adjust for quick demo
        config.data.output_path = "./modern_outputs"
        config.training.ppo_epochs = 2
        config.training.total_steps = 100
        config.evaluation.eval_samples = 10
        
        print(f"📁 Output directory: {config.data.output_path}")
        print(f"🎯 Target BERTScore: {config.evaluation.target_bertscore}")
        print(f"🎯 Target CodeBLEU: {config.evaluation.target_codebleu}")
        
        # Create output directory
        os.makedirs(config.data.output_path, exist_ok=True)
        
        # Test data loader
        print("📊 Testing data loader...")
        data_loader = ModernDataLoader(config)
        synthetic_data = data_loader._generate_synthetic_data()
        print(f"✅ Generated {len(synthetic_data)} synthetic samples")
        
        # Test metrics
        print("📈 Testing metrics...")
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
        
        # Test BLEU metric
        bleu_result = evaluator.compute_bleu(predictions, references)
        print(f"✅ BLEU score: {bleu_result.score:.3f}")
        
        # Test Ruby metric
        ruby_result = evaluator.compute_ruby(predictions, references)
        print(f"✅ Ruby score: {ruby_result.score:.3f}")
        
        # Save configuration
        config_path = os.path.join(config.data.output_path, 'config.json')
        config.save(config_path)
        print(f"💾 Configuration saved to: {config_path}")
        
        print("\n" + "=" * 50)
        print("🎉 Quick Start Demo Completed Successfully!")
        print("=" * 50)
        
        print("\n📊 Results Summary:")
        print(f"  BLEU Score: {bleu_result.score:.3f}")
        print(f"  Ruby Score: {ruby_result.score:.3f}")
        print(f"  Synthetic Samples: {len(synthetic_data)}")
        
        print("\n📝 Next Steps:")
        print("1. Install full dependencies: pip install -r modern_rlhf/requirements.txt")
        print("2. Run full pipeline: python modern_rlhf/main.py --mode fast")
        print("3. Check results in: ./modern_outputs/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Framework is working correctly!")
    else:
        print("\n⚠️  There were some issues. Check the errors above.")
    
    sys.exit(0 if success else 1)
