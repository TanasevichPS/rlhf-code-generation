from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class CodeRLHFConfig:
    """Configuration for improved code generation RLHF."""
    
    # Model settings
    model_name: str = "microsoft/CodeGPT-small-py"
    reward_model_name: str = "microsoft/codebert-base"
    
    # Dataset settings
    dataset_path: str = "./datasets_for_eval"
    human_eval_path: str = "./evaluation_results_server"
    
    # Training settings
    learning_rate: float = 5e-6
    batch_size: int = 2
    ppo_epochs: int = 10
    reward_training_epochs: int = 5
    mini_batch_size: int = 1
    
    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.2
    
    # Code-specific settings
    max_prompt_length: int = 256
    max_response_length: int = 512
    min_code_length: int = 20
    
    # Hardware settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging and saving
    output_dir: str = "./outputs"
    save_steps: int = 500
    logging_steps: int = 100
    
    # Reproducibility
    seed: int = 42