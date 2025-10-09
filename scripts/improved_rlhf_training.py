#!/usr/bin/env python3
"""Improved RLHF training script for meaningful text generation."""

import logging
import random
import numpy as np
import torch
from datetime import datetime
import os
import sys
from typing import List, Dict, Any, Optional
import re

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    # Try to import with datasets
    from src.data.dataset_utils import CodeDatasetLoader
    print("Using datasets version")
except ImportError as e:
    print(f"datasets import failed: {e}")
    print("Using standalone dataset implementation")
    # Fallback to our standalone implementation
    from src.data.dataset_utils import CodeDatasetLoader

from src.config import CodeRLHFConfig
from src.data.dataset_utils import CodeDatasetLoader
from src.models.model_loader import ModelLoader, CodeRewardModel
from src.train.ppo_trainer import CodeRLHFTrainer


def setup_logging(output_dir: str) -> None:
    """Set up logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def main() -> None:
    """Main training function for code generation."""
    # Configuration for code generation
    config = CodeRLHFConfig()
    
    # Setup
    setup_logging(config.output_dir)
    set_seed(config.seed)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Code Generation RLHF Training")
    logger.info(f"Configuration: {config}")
    
    try:
        # Load code dataset
        logger.info("Loading code dataset...")
        dataset_loader = CodeDatasetLoader(config)
        train_dataset = dataset_loader.load_dataset()
        
        # Load models
        logger.info("Loading models...")
        model_loader = ModelLoader(config)
        tokenizer, policy_model, ref_model = model_loader.load_models()
        
        # Initialize code reward model
        reward_model = CodeRewardModel(config)
        
        # Initialize code trainer
        trainer = CodeRLHFTrainer(config, tokenizer, policy_model, ref_model, reward_model)
        
        # Training loop
        logger.info("Starting code training...")
        for epoch in range(config.ppo_epochs):
            epoch_stats = trainer.train_epoch(train_dataset, epoch)
            logger.info(f"Epoch {epoch} statistics: {epoch_stats}")
            
            # Early stopping based on code quality
            if epoch_stats.get('syntax_score', 0) > 0.8 and epoch_stats.get('mean_reward', 0) > 0.6:
                logger.info("Good code quality achieved, stopping early")
                break
        
        # Save final results
        trainer.save_final_results()
        trainer.evaluate_code_quality()
        
        logger.info("Code RLHF training completed successfully!")
        
    except Exception as e:
        logger.error(f"Code training failed: {e}")
        raise


if __name__ == "__main__":
    main()