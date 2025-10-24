#!/usr/bin/env python3
"""Quick PPO stability test: run a single short epoch with conservative generation/gym settings
to observe KL behavior and batch ratio warnings. This uses the same trainer wrapper but overrides
config values for a short run.
"""
import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import CodeRLHFConfig
from src.data.dataset_loader import CodeDatasetLoader
from src.models.model_loader import ModelLoader
from src.models.reward_model import ImprovedCodeRewardModel
from src.train.ppo_trainer import CodeRLHFTrainer

def main():
    cfg = CodeRLHFConfig()
    # Conservative overrides
    cfg.ppo_epochs = 1
    cfg.batch_size = 1
    cfg.mini_batch_size = 1
    cfg.learning_rate = 2e-5
    cfg.max_prompt_length = 128
    cfg.max_response_length = 64
    cfg.device = 'cuda' if 'cuda' in cfg.device else cfg.device

    # Lower temperature and disable sampling to reduce KL surprises
    cfg.temperature = 0.3
    cfg.do_sample = False
    cfg.top_p = 0.95

    # Logging
    logging.basicConfig(level=logging.INFO)

    reward_model = ImprovedCodeRewardModel(cfg.reward_model_name)
    dataset_loader = CodeDatasetLoader(cfg)
    dataset = dataset_loader.load_dataset()
    tokenizer, policy_model, ref_model = ModelLoader(cfg).load_models()

    trainer = CodeRLHFTrainer(cfg, tokenizer, policy_model, ref_model, reward_model)

    # Run one epoch and print summary
    stats = trainer.train_epoch(dataset, epoch=0)
    print('Stability test stats:', stats)

if __name__ == '__main__':
    main()
