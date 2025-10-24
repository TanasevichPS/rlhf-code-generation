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
import importlib.util
eval_utils = None
try:
    # load scripts/eval_utils.py as a module when running as a script
    spec = importlib.util.spec_from_file_location("scripts.eval_utils", os.path.join(os.path.dirname(__file__), "eval_utils.py"))
    eval_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_utils)
except Exception:
    eval_utils = None
import subprocess
import sys

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
        # 1. Load or train a reward model
        logger.info("Preparing reward model...")
        reward_model = ImprovedCodeRewardModel(config.reward_model_name)
        # Prefer HF-style folder if present, then fall back to single-file state_dict
        hf_dir = os.path.join(config.output_dir, "reward_model_hf")
        reward_model_path = os.path.join(config.output_dir, "trained_reward_model.pt")

        if os.path.exists(hf_dir):
            logger.info(f"Found HF-style reward model directory: {hf_dir}. Initializing from directory and loading improved state if present.")
            try:
                # Initialize model to pick up tokenizer/base model from hf_dir
                reward_model = ImprovedCodeRewardModel(hf_dir)
                # Look for an improved state dict inside the HF dir and load it (non-strict)
                improved_state = os.path.join(hf_dir, "improved_state_dict.pt")
                if os.path.exists(improved_state):
                    state_dict = torch.load(improved_state, map_location=config.device)
                elif os.path.exists(reward_model_path):
                    state_dict = torch.load(reward_model_path, map_location=config.device)
                else:
                    state_dict = None

                if state_dict is not None:
                    keys_to_remove = [k for k in state_dict.keys() if 'position_ids' in k]
                    for k in keys_to_remove:
                        del state_dict[k]
                    try:
                        reward_model.load_state_dict(state_dict, strict=False)
                        logger.info(f"Loaded improved reward model from HF dir, ignored keys: {keys_to_remove}")
                    except RuntimeError as re:
                        logger.warning(f"Could not fully load improved state into reward model (shape mismatch): {re}. Continuing with partially initialized model.")
            except Exception as e:
                logger.warning(f"Failed to initialize reward model from HF dir {hf_dir}: {e}. Falling back to default init.")
        elif os.path.exists(reward_model_path):
            state_dict = torch.load(reward_model_path, map_location=config.device)
            keys_to_remove = [k for k in state_dict.keys() if 'position_ids' in k]
            for k in keys_to_remove:
                del state_dict[k]
            try:
                reward_model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded reward model, ignored keys: {keys_to_remove}")
            except RuntimeError as re:
                # Shape mismatches may occur when switching to a different underlying pretrained model
                logger.warning(f"Could not load saved reward model due to shape mismatch: {re}. Skipping loading and continuing with uninitialized reward model.")
        else:
            # If no reward model - build training TSV from human prefs and run the quick trainer stub
            logger.warning("No trained reward model found. Building preferences TSV and training a quick reward model (stub)...")
            prefs_folder = config.human_eval_path if hasattr(config, 'human_eval_path') else os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluation_results_server')
            script_path = os.path.join(os.path.dirname(__file__), 'pref_convert_and_reward_train.py')
            out_tsv = os.path.join(config.output_dir, 'training_data', 'prefs.tsv')
            # include --pairwise to get a pairwise reward trainer by default
            cmd = [sys.executable, script_path, '--prefs-folder', prefs_folder, '--out-tsv', out_tsv, '--reward-output', os.path.join(config.output_dir, 'reward_model'), '--pairwise', '--pairwise-epochs', '1']
            logger.info('Running reward model train stub: ' + ' '.join(cmd))
            try:
                subprocess.run(cmd, check=True)
                logger.info('Reward model stub training completed.')
                # Try to load the weights from the stub if available
                stub_model_dir = os.path.join(config.output_dir, 'reward_model')
                if os.path.exists(stub_model_dir):
                    try:
                        reward_model = ImprovedCodeRewardModel(stub_model_dir)
                        logger.info('Loaded reward model from stub directory.')
                    except Exception:
                        logger.warning('Could not initialize ImprovedCodeRewardModel from stub dir — will continue with untrained model')
            except subprocess.CalledProcessError as e:
                logger.error(f'Reward model stub failed: {e}. Continuing with untrained reward model')

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

            # Early stopping (configurable target, default 0.8)
            stop_target = getattr(config, 'early_stop_reward', 0.8)
            if current_reward >= stop_target:
                logger.info(f"Early stopping: reached target mean reward {stop_target}")
                break
        
        logger.info(" RLHF training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()