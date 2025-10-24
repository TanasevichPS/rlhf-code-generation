"""
Training Configuration for RLHF Code Project
===========================================

Simple, clean configuration management for RLHF training.
"""

from dataclasses import dataclass
from typing import Optional, List
import torch


@dataclass
class RLHFConfig:
    """Main configuration class for RLHF training."""
    
    # Model settings
    policy_model_name: str = "microsoft/CodeGPT-small-py"
    reward_model_name: str = "microsoft/codebert-base"
    
    # Training method
    method: str = "dpo"  # "ppo" or "dpo"
    
    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    max_length: int = 512
    num_epochs: int = 3
    warmup_steps: int = 100
    
    # DPO specific parameters
    beta: float = 0.1
    reference_free: bool = False
    
    # PPO specific parameters (if using PPO)
    ppo_epochs: int = 4
    ppo_clip_ratio: float = 0.2
    value_loss_coef: float = 0.1
    entropy_coef: float = 0.01
    
    # Human feedback integration
    use_human_feedback: bool = True
    human_feedback_dim: int = 64
    human_feedback_weight: float = 0.3
    
    # Data settings
    train_data_path: str = "./datasets_for_training"
    eval_data_path: str = "./datasets_for_eval"
    human_feedback_path: str = "./evaluation_results_server"
    output_dir: str = "./rlhf_outputs"
    
    # Evaluation settings
    eval_batch_size: int = 8
    eval_samples: int = 100
    
    # Target metrics (your research goals)
    target_bertscore: float = 0.7
    target_codebleu: float = 0.6
    target_bleu: float = 0.4
    target_rouge: float = 0.5
    
    # Hardware settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Logging and saving
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Post-initialization setup."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("Warning: CUDA not available, falling back to CPU")


# Predefined configurations for different use cases
def get_dpo_config() -> RLHFConfig:
    """Get configuration optimized for DPO training."""
    config = RLHFConfig()
    config.method = "dpo"
    config.beta = 0.1
    config.learning_rate = 1e-5
    config.batch_size = 8
    config.num_epochs = 5
    return config


def get_ppo_config() -> RLHFConfig:
    """Get configuration optimized for PPO training."""
    config = RLHFConfig()
    config.method = "ppo"
    config.learning_rate = 5e-6
    config.batch_size = 4
    config.num_epochs = 10
    config.ppo_epochs = 4
    return config


def get_fast_config() -> RLHFConfig:
    """Get configuration for fast prototyping."""
    config = RLHFConfig()
    config.method = "dpo"
    config.num_epochs = 2
    config.batch_size = 2
    config.eval_samples = 20
    config.save_steps = 100
    return config


def get_research_config() -> RLHFConfig:
    """Get configuration optimized for research experiments."""
    config = RLHFConfig()
    config.method = "dpo"
    config.beta = 0.1
    config.learning_rate = 1e-5
    config.batch_size = 6
    config.num_epochs = 8
    config.eval_samples = 200
    config.target_bertscore = 0.8
    config.target_codebleu = 0.7
    return config
