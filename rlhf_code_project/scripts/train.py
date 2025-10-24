"""
Main Training Script for RLHF Code Project
==========================================

Simple, clean training script for DPO/PPO training.
"""

import torch
from torch.utils.data import DataLoader
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any

# Import our modules
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RLHFConfig, get_dpo_config, get_fast_config
from training import DPOTrainer, SimpleDPOTrainer, DPO_AVAILABLE
from data import PreferenceDataset, EvaluationDataset
from evaluation import MetricCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(config: RLHFConfig = None):
    """Main training function."""
    if config is None:
        config = get_fast_config()  # Default to fast config for quick testing
    
    logger.info("Starting RLHF training...")
    logger.info(f"Method: {config.method}")
    logger.info(f"Model: {config.policy_model_name}")
    logger.info(f"Device: {config.device}")
    
    try:
        # 1. Load training data
        logger.info("Loading training data...")
        train_dataset = PreferenceDataset(
            data_path=os.path.join(config.train_data_path, "pairwise_prefs.csv"),
            max_samples=100  # Limit for quick testing
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            collate_fn=lambda x: {
                'prompts': [item['prompt'] for item in x],
                'chosen_responses': [item['chosen_response'] for item in x],
                'rejected_responses': [item['rejected_response'] for item in x]
            }
        )
        
        # 2. Initialize trainer
        logger.info("Initializing trainer...")
        if config.method == "dpo":
            if DPO_AVAILABLE:
                try:
                    trainer = DPOTrainer(config)
                    logger.info("Using full DPO trainer")
                except Exception as e:
                    logger.warning(f"Full DPO trainer failed: {e}. Using simple trainer.")
                    trainer = SimpleDPOTrainer(config)
            else:
                logger.info("Using simple DPO trainer (full trainer not available)")
                trainer = SimpleDPOTrainer(config)
        else:
            raise ValueError(f"Method {config.method} not implemented yet")
        
        # 3. Train the model
        logger.info("Starting training...")
        training_results = trainer.train(train_loader)
        
        # 4. Save the model
        model_save_path = os.path.join(config.output_dir, "trained_model")
        trainer.save_model(model_save_path)
        
        # 5. Quick evaluation
        logger.info("Running evaluation...")
        eval_results = evaluate_model(trainer, config)
        
        # 6. Save results
        results = {
            'config': config.__dict__,
            'training_results': training_results,
            'evaluation_results': eval_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add epoch metrics if available
        if 'epoch_metrics' in training_results:
            results['epoch_metrics'] = training_results['epoch_metrics']
        
        results_path = os.path.join(config.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # 7. Print summary
        print_summary(eval_results, config, training_results)
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def evaluate_model(trainer: DPOTrainer, config: RLHFConfig) -> Dict[str, Any]:
    """Quick evaluation of the trained model."""
    logger.info("Evaluating model...")
    
    # Load evaluation data
    eval_dataset = EvaluationDataset(
        data_path=os.path.join(config.eval_data_path, "T2C-CONALA-CODEGEN-FINETUNED-SO.csv"),
        max_samples=config.eval_samples
    )
    
    # Generate responses
    prompts = [sample['prompt'] for sample in eval_dataset]
    references = [sample['reference'] for sample in eval_dataset]
    
    logger.info(f"Generating responses for {len(prompts)} prompts...")
    generated_responses = trainer.generate_responses(prompts, max_new_tokens=256)
    
    # Calculate metrics
    metric_calculator = MetricCalculator()
    metrics = metric_calculator.calculate_all_metrics(generated_responses, references)
    
    # Check against targets
    targets = {
        'bertscore': config.target_bertscore,
        'codebleu': config.target_codebleu,
        'bleu': config.target_bleu,
        'rouge': config.target_rouge
    }
    
    target_results = metric_calculator.evaluate_against_targets(metrics, targets)
    summary = metric_calculator.get_summary(metrics, targets)
    
    return {
        'metrics': metrics,
        'targets_met': target_results,
        'summary': summary,
        'generated_responses': generated_responses[:5],  # Save first 5 for inspection
        'references': references[:5]
    }


def print_summary(eval_results: Dict[str, Any], config: RLHFConfig, training_results: Dict[str, Any] = None):
    """Print training and evaluation summary."""
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Print epoch metrics if available
    if training_results and 'epoch_metrics' in training_results:
        print(f"\n📈 METRICS BY EPOCH:")
        print("-" * 50)
        
        epoch_metrics = training_results['epoch_metrics']
        for i, metrics in enumerate(epoch_metrics):
            print(f"  Epoch {i+1:2d}: ", end="")
            for metric, value in metrics.items():
                print(f"{metric.upper()}={value:.3f} ", end="")
            print()
        
        # Show improvement
        if len(epoch_metrics) > 1:
            print(f"\n📊 IMPROVEMENT:")
            print("-" * 30)
            first_epoch = epoch_metrics[0]
            last_epoch = epoch_metrics[-1]
            for metric in ['bertscore', 'codebleu', 'bleu', 'rouge', 'ruby']:
                if metric in first_epoch and metric in last_epoch:
                    improvement = last_epoch[metric] - first_epoch[metric]
                    print(f"  {metric.upper()}: {first_epoch[metric]:.3f} → {last_epoch[metric]:.3f} ({improvement:+.3f})")
    
    print(f"\n📊 FINAL EVALUATION RESULTS:")
    print("-" * 30)
    
    metrics = eval_results['metrics']
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print(f"\n🎯 TARGET ACHIEVEMENT:")
    print("-" * 30)
    
    targets_met = eval_results['targets_met']
    for metric, met in targets_met.items():
        status = "✅" if met else "❌"
        target_value = getattr(config, f'target_{metric}', 0)
        print(f"  {status} {metric.upper()}: {metrics.get(metric, 0):.4f} / {target_value:.4f}")
    
    summary = eval_results['summary']
    print(f"\n📈 OVERALL SUMMARY:")
    print("-" * 30)
    print(f"  Targets Met: {summary['targets_met_count']}/{summary['targets_total']}")
    print(f"  All Targets Met: {'✅' if summary['all_targets_met'] else '❌'}")
    
    print(f"\n📁 RESULTS SAVED TO: {config.output_dir}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RLHF model")
    parser.add_argument("--method", choices=["dpo", "ppo"], default="dpo", help="Training method")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--fast", action="store_true", help="Use fast config for quick testing")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--device", type=str, help="Device to use")
    
    args = parser.parse_args()
    
    # Create config
    if args.fast:
        config = get_fast_config()
    else:
        config = get_dpo_config()
    
    # Override with command line arguments
    if args.method:
        config.method = args.method
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.device:
        config.device = args.device
    
    # Run training
    main(config)
