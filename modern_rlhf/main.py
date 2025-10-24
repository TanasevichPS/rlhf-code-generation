#!/usr/bin/env python3
"""
Modern RLHF Main Script
=======================

Main entry point for the Modern RLHF framework.
Supports different modes: research, production, fast prototype.
"""

import argparse
import sys
import os
import logging
from pathlib import Path
import json
from datetime import datetime

# Add the parent directory to the path to import modern_rlhf
sys.path.insert(0, str(Path(__file__).parent.parent))

from modern_rlhf import (
    ModernRLHFPipeline,
    ModernRLHFConfig,
    get_research_config,
    get_production_config,
    get_fast_config
)
from modern_rlhf.pipeline import run_research_experiment, run_production_training, run_fast_prototype

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_custom_config(args) -> ModernRLHFConfig:
    """Create a custom configuration based on command line arguments."""
    # Start with base config
    if args.mode == 'research':
        config = get_research_config()
    elif args.mode == 'production':
        config = get_production_config()
    elif args.mode == 'fast':
        config = get_fast_config()
    else:
        config = ModernRLHFConfig()
    
    # Override with command line arguments
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    if args.batch_size:
        config.training.batch_size = args.batch_size
    
    if args.epochs:
        config.training.ppo_epochs = args.epochs
    
    if args.steps:
        config.training.total_steps = args.steps
    
    if args.device:
        config.hardware.device = args.device
    
    if args.output_dir:
        config.data.output_path = args.output_dir
    
    if args.model_name:
        config.model.base_model_name = args.model_name
    
    if args.reward_model_name:
        config.model.reward_model_name = args.reward_model_name
    
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    # Set run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = f"{config.experiment_name}_{timestamp}"
    
    return config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Modern RLHF Framework for Code Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run research experiment
  python main.py --mode research --epochs 10 --steps 2000
  
  # Run production training
  python main.py --mode production --device cuda --batch-size 8
  
  # Run fast prototype
  python main.py --mode fast --epochs 2 --steps 500
  
  # Custom configuration
  python main.py --learning-rate 1e-5 --batch-size 4 --model-name microsoft/CodeGPT-small-py
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['research', 'production', 'fast', 'custom'],
        default='research',
        help='Training mode (default: research)'
    )
    
    # Training parameters
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate for training'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--steps',
        type=int,
        help='Total number of training steps'
    )
    
    # Model parameters
    parser.add_argument(
        '--model-name',
        type=str,
        help='Base model name (e.g., microsoft/CodeGPT-small-py)'
    )
    parser.add_argument(
        '--reward-model-name',
        type=str,
        help='Reward model name (e.g., microsoft/codebert-base)'
    )
    
    # Hardware parameters
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use for training (default: auto)'
    )
    
    # Data parameters
    parser.add_argument(
        '--train-data-path',
        type=str,
        help='Path to training data directory'
    )
    parser.add_argument(
        '--eval-data-path',
        type=str,
        help='Path to evaluation data directory'
    )
    parser.add_argument(
        '--human-feedback-path',
        type=str,
        help='Path to human feedback data'
    )
    
    # Output parameters
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Name of the experiment'
    )
    
    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (JSON)'
    )
    
    # Logging parameters
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only run evaluation (skip training)'
    )
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        help='Path to model checkpoint for evaluation'
    )
    
    # Target metrics
    parser.add_argument(
        '--target-bertscore',
        type=float,
        default=0.7,
        help='Target BERTScore (default: 0.7)'
    )
    parser.add_argument(
        '--target-codebleu',
        type=float,
        default=0.6,
        help='Target CodeBLEU (default: 0.6)'
    )
    parser.add_argument(
        '--target-bleu',
        type=float,
        default=0.4,
        help='Target BLEU (default: 0.4)'
    )
    parser.add_argument(
        '--target-rouge',
        type=float,
        default=0.5,
        help='Target ROUGE (default: 0.5)'
    )
    parser.add_argument(
        '--target-ruby',
        type=float,
        default=0.3,
        help='Target Ruby (default: 0.3)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.debug)
    
    try:
        # Load configuration
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            config = ModernRLHFConfig.load(args.config)
        else:
            logger.info("Creating configuration from command line arguments")
            config = create_custom_config(args)
        
        # Override target metrics if specified
        config.evaluation.target_bertscore = args.target_bertscore
        config.evaluation.target_codebleu = args.target_codebleu
        config.evaluation.target_bleu = args.target_bleu
        config.evaluation.target_rouge = args.target_rouge
        config.evaluation.target_ruby = args.target_ruby
        
        # Override data paths if specified
        if args.train_data_path:
            config.data.train_data_path = args.train_data_path
        if args.eval_data_path:
            config.data.eval_data_path = args.eval_data_path
        if args.human_feedback_path:
            config.data.human_feedback_path = args.human_feedback_path
        
        # Set device
        if args.device == 'auto':
            import torch
            config.hardware.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            config.hardware.device = args.device
        
        # Create output directory
        os.makedirs(config.data.output_path, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(config.data.output_path, 'config.json')
        config.save(config_path)
        logger.info(f"Configuration saved to {config_path}")
        
        # Print configuration summary
        logger.info("Configuration Summary:")
        logger.info(f"  Mode: {args.mode}")
        logger.info(f"  Model: {config.model.base_model_name}")
        logger.info(f"  Reward Model: {config.model.reward_model_name}")
        logger.info(f"  Device: {config.hardware.device}")
        logger.info(f"  Learning Rate: {config.training.learning_rate}")
        logger.info(f"  Batch Size: {config.training.batch_size}")
        logger.info(f"  Epochs: {config.training.ppo_epochs}")
        logger.info(f"  Steps: {config.training.total_steps}")
        logger.info(f"  Output Directory: {config.data.output_path}")
        
        # Run pipeline
        if args.eval_only:
            logger.info("Running evaluation only...")
            # TODO: Implement evaluation-only mode
            logger.warning("Evaluation-only mode not yet implemented")
        else:
            logger.info("Starting full RLHF pipeline...")
            
            # Create pipeline
            pipeline = ModernRLHFPipeline(config)
            
            # Run pipeline
            results = pipeline.run_full_pipeline()
            
            # Create visualizations
            pipeline.visualize_results()
            
            # Print results
            logger.info("Pipeline Results:")
            logger.info(f"  Success: {results.success}")
            logger.info(f"  Total Time: {results.total_time:.2f} seconds")
            logger.info(f"  Training Time: {results.training_time:.2f} seconds")
            
            if results.success:
                logger.info("  Final Metrics:")
                for metric, value in results.final_metrics.items():
                    logger.info(f"    {metric}: {value}")
                
                logger.info("  Evaluation Metrics:")
                for metric, value in results.evaluation_metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"    {metric}: {value:.4f}")
                
                # Check if targets were met
                if 'targets_met' in results.evaluation_metrics:
                    targets_met = results.evaluation_metrics['targets_met']
                    met_count = sum(targets_met.values())
                    total_count = len(targets_met)
                    logger.info(f"  Targets Met: {met_count}/{total_count}")
                    
                    if met_count == total_count:
                        logger.info("  🎉 All targets achieved!")
                    else:
                        logger.info("  ⚠️  Some targets not met")
            else:
                logger.error(f"  Error: {results.error_message}")
        
        logger.info("Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
