#!/usr/bin/env python3
"""Main RLHF training script with improved reward model."""

import logging
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import CodeRLHFConfig
from src.data.dataset_loader import CodeDatasetLoader
from src.models.model_loader import ModelLoader
from src.models.reward_model import ImprovedCodeRewardModel
from src.train.ppo_trainer import CodeRLHFTrainer 

def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "training.log"))
        ]
    )

def main():
    config = CodeRLHFConfig()
    setup_logging(config.output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Improved RLHF Training Pipeline")
    
    try:
        # 1. Load trained reward model
        logger.info("Loading trained reward model...")
        reward_model = ImprovedCodeRewardModel(config.reward_model_name)
        
        reward_model_path = os.path.join(config.output_dir, "trained_reward_model.pt")
        if os.path.exists(reward_model_path):
            state_dict = torch.load(reward_model_path, map_location=config.device)
            # Удаляем проблемные ключи
            keys_to_remove = [k for k in state_dict.keys() if 'position_ids' in k]
            for k in keys_to_remove:
                del state_dict[k]
            reward_model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded reward model, ignored keys: {keys_to_remove}")
            logger.info("Loaded trained reward model")
        else:
            logger.warning("No trained reward model found, using untrained model")
        
        reward_model.eval()
        
        # 2. Load dataset
        logger.info("Loading code dataset...")
        dataset_loader = CodeDatasetLoader(config)
        train_dataset = dataset_loader.load_dataset()
        logger.info(f"Loaded dataset with {len(train_dataset)} examples")
        
        # 3. Load policy model
        logger.info("Loading policy model...")
        model_loader = ModelLoader(config)
        tokenizer, policy_model, ref_model = model_loader.load_models()
        
        # 4. Initialize trainer
        logger.info("Initializing RLHF trainer...")
        trainer = CodeRLHFTrainer(config, tokenizer, policy_model, ref_model, reward_model)
        
        # 5. Training loop
        logger.info("Starting RLHF training...")
        best_reward = -float('inf')
        
        for epoch in range(config.ppo_epochs):
            epoch_stats = trainer.train_epoch(train_dataset, epoch)
            
            current_reward = epoch_stats.get('mean_reward', 0)
            logger.info(f"Epoch {epoch} completed:")
            logger.info(f"  Mean Reward: {current_reward:.4f}")
            logger.info(f"  Syntax Score: {epoch_stats.get('syntax_score', 0):.4f}")
            
            # Save best model
            if current_reward > best_reward:
                best_reward = current_reward
                trainer.save_final_results()
                logger.info(f"New best model saved with reward: {best_reward:.4f}")
            
            # Early stopping
            if current_reward > 0.7:  # Good quality threshold
                logger.info("High quality achieved, stopping early")
                break
        
        logger.info(" RLHF training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()