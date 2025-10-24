#!/usr/bin/env python3
"""
Quick Start Script for Simplified RLHF
======================================

Simple script to run the new simplified RLHF system.
"""

import sys
import os
from pathlib import Path

# Add rlhf_code_project to path
sys.path.insert(0, str(Path(__file__).parent / "rlhf_code_project"))

def main():
    """Quick start function."""
    print("🚀 Simplified RLHF Code Project - Quick Start")
    print("=" * 60)
    
    try:
        # Import modules
        from config import get_fast_config
        from scripts.train import main as train_main
        
        print("✅ All imports successful!")
        
        # Create fast configuration
        print("🔧 Creating configuration...")
        config = get_fast_config()
        
        # Adjust paths to use existing data
        config.train_data_path = "./datasets_for_training"
        config.eval_data_path = "./datasets_for_eval"
        config.output_dir = "./rlhf_outputs"
        
        # Set experiment name
        config.experiment_name = "simplified_rlhf_experiment"
        
        print(f"📁 Training data: {config.train_data_path}")
        print(f"📁 Evaluation data: {config.eval_data_path}")
        print(f"📁 Output directory: {config.output_dir}")
        print(f"🎯 Method: {config.method}")
        print(f"🎯 Target BERTScore: {config.target_bertscore}")
        print(f"🎯 Target CodeBLEU: {config.target_codebleu}")
        print()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Run training
        print("🏃 Starting training...")
        results = train_main(config)
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Print results
        if 'evaluation_results' in results:
            eval_results = results['evaluation_results']
            
            print("\n📊 EVALUATION RESULTS:")
            print("-" * 30)
            
            metrics = eval_results.get('metrics', {})
            for metric, value in metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")
            
            print(f"\n🎯 TARGET ACHIEVEMENT:")
            print("-" * 30)
            
            targets_met = eval_results.get('targets_met', {})
            for metric, met in targets_met.items():
                status = "✅" if met else "❌"
                target_value = getattr(config, f'target_{metric}', 0)
                print(f"  {status} {metric.upper()}: {metrics.get(metric, 0):.4f} / {target_value:.4f}")
            
            summary = eval_results.get('summary', {})
            print(f"\n📈 OVERALL SUMMARY:")
            print("-" * 30)
            print(f"  Targets Met: {summary.get('targets_met_count', 0)}/{summary.get('targets_total', 0)}")
            print(f"  All Targets Met: {'✅' if summary.get('all_targets_met', False) else '❌'}")
        
        print(f"\n📁 RESULTS SAVED TO: {config.output_dir}")
        print("=" * 60)
        
        print("\n📝 Next Steps:")
        print("1. Check results in ./rlhf_outputs/")
        print("2. Run full training: python rlhf_code_project/scripts/train.py --method dpo --epochs 10")
        print("3. Customize configuration for your research needs")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Simplified RLHF system is working correctly!")
    else:
        print("\n⚠️  There were some issues. Check the errors above.")
    
    sys.exit(0 if success else 1)
