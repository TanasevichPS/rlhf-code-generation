#!/usr/bin/env python3
"""
Quick Start Script for Modern RLHF
==================================

Simple script to run the modern RLHF framework with your existing data.
"""

import sys
import os
from pathlib import Path

# Add modern_rlhf to path
sys.path.insert(0, str(Path(__file__).parent / "modern_rlhf"))

# Ensure stdout/stderr use UTF-8 where possible to avoid console encoding errors
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    # Older Python / environments may not support reconfigure; ignore
    pass

from modern_rlhf import ModernRLHFPipeline, get_research_config
from modern_rlhf.config import ModernRLHFConfig

def main():
    """Quick start function."""
    print("Modern RLHF Framework - Quick Start")
    print("=" * 50)
    
    # Create configuration
    config = get_research_config()
    
    # Adjust paths to use existing data
    config.data.train_data_path = r"C:\Users\Полина\Desktop\Работа\huawei\rlhf\conala-corpus\conala-train.json"
    config.data.eval_data_path = r"C:\Users\Полина\Desktop\Работа\huawei\rlhf\conala-corpus\conala-test.json"
    config.data.human_feedback_path = "./evaluation_results_server"
    config.data.output_path = "./modern_outputs"
    config.data.min_prompt_length = 0
    config.data.min_response_length = 0
    # Force local CoNaLa corpus (preferred over Hub)
    config.data.conala_local_path = r"C:\Users\Полина\Desktop\Работа\huawei\rlhf\conala-corpus"

    config.data.train_data_path = "./datasets_for_training"
    config.data.eval_data_path = "./datasets_for_eval"
    config.data.human_feedback_path = "./evaluation_results_server"
    config.data.output_path = "./modern_outputs"
    
    # Set experiment name
    config.experiment_name = "modern_rlhf_experiment"
    # Adjust training parameters for better convergence
    config.training.ppo_epochs = 10
    config.training.total_steps = 2000
    config.evaluation.eval_samples = 100
    config.training.learning_rate = 1e-5
    # Adjust training parameters for quick testing
    config.training.ppo_epochs = 3
    config.training.total_steps = 500
    config.evaluation.eval_samples = 50
    # Set target metrics
    config.evaluation.target_bertscore = 0.7
    config.evaluation.target_codebleu = 0.6
    config.evaluation.target_bleu = 0.4
    config.evaluation.target_rouge = 0.5
    config.evaluation.target_ruby = 0.3
    config.data.conala_local_path = r"C:\Users\Полина\Desktop\Работа\huawei\rlhf\conala-corpus"

    print(f"Training data: {config.data.train_data_path}")
    print(f"Evaluation data: {config.data.eval_data_path}")
    print(f"Human feedback: {config.data.human_feedback_path}")
    print(f"Output directory: {config.data.output_path}")
    if getattr(config.data, 'conala_local_path', None):
        print(f"CoNaLa local corpus: {config.data.conala_local_path}")

    print(f"Target BERTScore: {config.evaluation.target_bertscore}")
    print(f"Target CodeBLEU: {config.evaluation.target_codebleu}")
    print(f"Target BLEU: {config.evaluation.target_bleu}")
    print(f"Target ROUGE: {config.evaluation.target_rouge}")
    print(f"Target Ruby: {config.evaluation.target_ruby}")
    print()
    
    # Create output directory
    os.makedirs(config.data.output_path, exist_ok=True)
    
    # Create pipeline
    print("Initializing Modern RLHF Pipeline...")
    pipeline = ModernRLHFPipeline(config)

    # Run pipeline (let exceptions bubble up so we see full trace during debugging)
    print("Starting training pipeline...")
    results = pipeline.run_full_pipeline()

    # Create visualizations
    print("Creating visualizations...")
    pipeline.visualize_results()

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    if results.success:
        print("Pipeline completed successfully!")
        print(f"Total time: {results.total_time:.2f} seconds")
        print(f"Training time: {results.training_time:.2f} seconds")

        print("\nFinal Metrics:")
        for metric, value in results.final_metrics.items():
            print(f"  {metric}: {value}")

        print("\nEvaluation Metrics:")
        for metric, value in results.evaluation_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")

        # Check targets
        if 'targets_met' in results.evaluation_metrics:
            targets_met = results.evaluation_metrics['targets_met']
            met_count = sum(targets_met.values())
            total_count = len(targets_met)
            print(f"\nTargets Met: {met_count}/{total_count}")

            if met_count == total_count:
                print("All targets achieved!")
            else:
                print("Some targets not met:")
                for metric, met in targets_met.items():
                    status = "✅" if met else "❌"
                    print(f"  {status} {metric}")

        print(f"\nResults saved to: {config.data.output_path}")
    else:
        print("Pipeline failed!")
        print(f"Error: {results.error_message}")

if __name__ == "__main__":
    main()
