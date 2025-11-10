"""
Fix Training Script for Modern RLHF
===================================

This script fixes the critical issues that caused metrics to drop from 0.8 to 0.01:
1. Restores do_sample=True for PPO exploration
2. Uses local CoNaLa dataset
3. Generates proper synthetic human feedback
4. Sets realistic target metrics
5. Validates GPU usage

Run this to start corrected training.
"""

import os
import sys
import logging
import torch
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fixed_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def validate_environment():
    """Validate that the environment is set up correctly."""
    logger.info("="*60)
    logger.info("VALIDATING ENVIRONMENT")
    logger.info("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("[ERROR] CUDA is not available! GPU training is required.")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"[OK] GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Check CoNaLa dataset
    conala_path = Path("./conala-corpus")
    if not conala_path.exists():
        logger.error(f"[ERROR] CoNaLa corpus not found at {conala_path}")
        logger.error("   Please ensure conala-corpus directory exists with train/test files")
        sys.exit(1)
    
    train_file = conala_path / "conala-train.json"
    test_file = conala_path / "conala-test.json"
    
    if not train_file.exists():
        logger.error(f"[ERROR] Training file not found: {train_file}")
        sys.exit(1)
    if not test_file.exists():
        logger.error(f"[ERROR] Test file not found: {test_file}")
        sys.exit(1)
    
    logger.info(f"[OK] CoNaLa dataset found:")
    logger.info(f"  - Train: {train_file}")
    logger.info(f"  - Test: {test_file}")
    
    # Check output directories
    output_dirs = ["./modern_outputs", "./evaluation_results_server"]
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"[OK] Output directory ready: {dir_path}")
    
    logger.info("="*60)
    logger.info("ENVIRONMENT VALIDATION PASSED [OK]")
    logger.info("="*60)
    return True

def generate_synthetic_feedback():
    """Generate synthetic human feedback based on CoNaLa data."""
    logger.info("\n" + "="*60)
    logger.info("GENERATING SYNTHETIC HUMAN FEEDBACK")
    logger.info("="*60)
    
    from modern_rlhf import ModernRLHFConfig
    from modern_rlhf.data_loader import ModernDataLoader
    
    config = ModernRLHFConfig()
    data_loader = ModernDataLoader(config)
    
    # Generate 500 synthetic feedback entries based on CoNaLa
    target_size = 500
    logger.info(f"Generating {target_size} synthetic human feedback entries...")
    
    feedback_items = data_loader.generate_synthetic_human_feedback(
        n=target_size,
        output_dir=config.data.human_feedback_path
    )
    
    logger.info(f"[OK] Generated {len(feedback_items)} feedback entries")
    logger.info(f"  Saved to: {config.data.human_feedback_path}")
    logger.info("="*60)
    
    return feedback_items

def print_config_summary(config):
    """Print a summary of the training configuration."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info("="*60)
    
    logger.info("\n[Generation Settings]")
    logger.info(f"  - do_sample: {config.generation.do_sample} (CRITICAL: must be True for PPO)")
    logger.info(f"  - temperature: {config.generation.temperature}")
    logger.info(f"  - top_p: {config.generation.top_p}")
    logger.info(f"  - max_new_tokens: {config.generation.max_new_tokens}")
    
    logger.info("\n[Target Metrics (Realistic)]")
    logger.info(f"  - BERTScore: {config.evaluation.target_bertscore}")
    logger.info(f"  - CodeBLEU: {config.evaluation.target_codebleu}")
    logger.info(f"  - BLEU: {config.evaluation.target_bleu}")
    logger.info(f"  - ROUGE: {config.evaluation.target_rouge}")
    logger.info(f"  - RUBY: {config.evaluation.target_ruby}")
    
    logger.info("\n[Training Parameters]")
    logger.info(f"  - Learning rate: {config.training.learning_rate}")
    logger.info(f"  - Batch size: {config.training.batch_size}")
    logger.info(f"  - PPO epochs: {config.training.ppo_epochs}")
    logger.info(f"  - Total steps: {config.training.total_steps}")
    
    logger.info("\n[Data Configuration]")
    logger.info(f"  - CoNaLa path: {config.data.conala_local_path}")
    logger.info(f"  - Max train samples: {config.data.max_train_samples}")
    logger.info(f"  - Max eval samples: {config.data.max_eval_samples}")
    logger.info(f"  - Human feedback: {config.data.human_feedback_path}")
    
    logger.info("\n[Hardware]")
    logger.info(f"  - Device: {config.hardware.device}")
    logger.info(f"  - Mixed precision: {config.hardware.mixed_precision}")
    logger.info(f"  - Gradient checkpointing: {config.hardware.gradient_checkpointing}")
    
    logger.info("="*60)

def run_training():
    """Run the corrected training pipeline."""
    logger.info("\n" + "="*60)
    logger.info("STARTING CORRECTED RLHF TRAINING")
    logger.info("="*60)
    
    from modern_rlhf import ModernRLHFPipeline, ModernRLHFConfig
    
    # Create configuration
    config = ModernRLHFConfig()
    
    # CRITICAL: Ensure do_sample is True
    if not config.generation.do_sample:
        logger.warning("[WARNING] do_sample was False, forcing to True for PPO training")
        config.generation.do_sample = True
    
    # Print configuration summary
    print_config_summary(config)
    
    # Create pipeline
    logger.info("\nInitializing RLHF pipeline...")
    pipeline = ModernRLHFPipeline(config)
    
    # Run training
    logger.info("\nStarting training pipeline...")
    logger.info("This will:")
    logger.info("  1. Load CoNaLa training data")
    logger.info("  2. Generate/load human feedback")
    logger.info("  3. Train reward model")
    logger.info("  4. Train policy model with PPO")
    logger.info("  5. Evaluate on CoNaLa test set")
    logger.info("")
    
    results = pipeline.run_full_pipeline()
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)
    
    logger.info("\n[Final Results]")
    # Handle PipelineResults object
    if hasattr(results, 'final_metrics'):
        final_metrics = results.final_metrics
    elif isinstance(results, dict):
        final_metrics = results
    else:
        final_metrics = {}
    
    for metric_name, value in final_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  - {metric_name}: {value:.4f}")
        else:
            logger.info(f"  - {metric_name}: {value}")
    
    # Check if targets were met
    logger.info("\n[Target Achievement]")
    targets = {
        'bertscore': config.evaluation.target_bertscore,
        'codebleu': config.evaluation.target_codebleu,
        'bleu': config.evaluation.target_bleu,
        'rouge': config.evaluation.target_rouge,
        'ruby': config.evaluation.target_ruby
    }
    
    for metric, target in targets.items():
        actual = final_metrics.get(metric, 0.0)
        status = "[OK]" if actual >= target else "[FAIL]"
        logger.info(f"  {status} {metric}: {actual:.4f} (target: {target:.4f})")
    
    logger.info("\n" + "="*60)
    logger.info(f"Results saved to: {config.data.output_path}")
    logger.info("="*60)
    
    return results

def main():
    """Main entry point."""
    try:
        # Step 1: Validate environment
        validate_environment()
        
        # Step 2: Generate synthetic feedback
        generate_synthetic_feedback()
        
        # Step 3: Run training
        results = run_training()
        
        logger.info("\n[SUCCESS] ALL STEPS COMPLETED SUCCESSFULLY")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n[WARNING] Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n[ERROR] {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())

