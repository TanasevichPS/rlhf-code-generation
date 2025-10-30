"""
Modern RLHF Configuration
========================

Configuration management for the modern RLHF framework with support for
state-of-the-art methods and comprehensive evaluation metrics.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch
import os


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    
    # Base model settings
    base_model_name: str = "microsoft/CodeGPT-small-py"
    reward_model_name: str = "microsoft/codebert-base"
    
    # Model sizes for different components
    policy_model_size: str = "small"  # small, medium, large
    reward_model_size: str = "base"   # base, large
    
    # Model loading settings
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = True
    torch_dtype: str = "float16"  # float16, float32, bfloat16
    
    # Model architecture settings
    max_position_embeddings: int = 1024
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12


@dataclass
class TrainingConfig:
    """Configuration for training settings."""
    
    # Basic training parameters
    learning_rate: float = 5e-6
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # PPO specific settings
    ppo_epochs: int = 4
    ppo_clip_ratio: float = 0.2
    ppo_value_loss_coef: float = 0.1
    ppo_entropy_coef: float = 0.01
    ppo_kl_penalty: float = 0.02
    
    # DPO specific settings (alternative to PPO)
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"  # sigmoid, hinge, ipo
    
    # Training schedule
    warmup_steps: int = 100
    total_steps: int = 1000
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    
    # Code-specific generation
    max_prompt_length: int = 512
    max_response_length: int = 512
    min_code_length: int = 10
    
    # Generation strategies
    num_beams: int = 1
    num_return_sequences: int = 1
    early_stopping: bool = True


@dataclass
class RewardConfig:
    """Configuration for reward modeling."""
    
    # Reward model training
    reward_learning_rate: float = 2e-5
    reward_batch_size: int = 8
    reward_epochs: int = 3
    
    # Human feedback integration
    human_feedback_weight: float = 0.3
    use_human_logits: bool = True
    human_logits_layer: str = "last"  # last, second_last, custom
    
    # Reward components
    syntax_reward_weight: float = 0.2
    execution_reward_weight: float = 0.3
    semantic_reward_weight: float = 0.3
    human_preference_weight: float = 0.2
    
    # Reward normalization
    reward_normalization: bool = True
    reward_clipping: bool = True
    reward_clip_value: float = 5.0


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    
    # Target metrics (thresholds for success)
    target_bertscore: float = 0.7
    target_codebleu: float = 0.6
    target_bleu: float = 0.4
    target_rouge: float = 0.5
    target_ruby: float = 0.3  # Custom metric for code quality
    
    # Evaluation settings
    eval_batch_size: int = 8
    eval_samples: int = 100
    eval_datasets: List[str] = field(default_factory=lambda: [
        "T2C-CONALA-CODEGEN-FINETUNED-SO.csv",
        "T2C-CONALA-CODEGEN-VANILLA.csv",
        "T2C-CONALA-CODEGEN2B-FINETUNED-CONALA-IMPORTS.csv"
    ])
    
    # Metric computation
    use_cached_embeddings: bool = True
    cache_embeddings: bool = True
    embedding_model: str = "microsoft/codebert-base"


@dataclass
class DataConfig:
    """Configuration for data handling."""
    
    # Data paths
    train_data_path: str = "./datasets_for_training"
    eval_data_path: str = "./datasets_for_eval"
    human_feedback_path: str = "./evaluation_results_server"
    output_path: str = "./modern_outputs"
<<<<<<< HEAD
<<<<<<< HEAD
    # Optional local CoNaLa corpus root (if provided, prefer local files)
    conala_local_path: Optional[str] = None
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
    
    # Data processing
    max_train_samples: int = 10000
    max_eval_samples: int = 1000
    train_test_split: float = 0.9
    
    # Data augmentation
    use_data_augmentation: bool = True
    augmentation_ratio: float = 0.1
    
    # Data filtering
    min_prompt_length: int = 10
    max_prompt_length: int = 512
    min_response_length: int = 5
    max_response_length: int = 512


@dataclass
class HardwareConfig:
    """Configuration for hardware settings."""
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Memory optimization
    max_memory_usage: float = 0.9  # Fraction of GPU memory to use
    offload_to_cpu: bool = False
    use_deepspeed: bool = False
    
    # Distributed training
    local_rank: int = -1
    world_size: int = 1
    ddp_backend: str = "nccl"


@dataclass
class ModernRLHFConfig:
    """Main configuration class for Modern RLHF framework."""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    # Global settings
    seed: int = 42
    debug: bool = False
    verbose: bool = True
    
    # Experiment tracking
    experiment_name: str = "modern_rlhf_experiment"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Create output directory
        os.makedirs(self.data.output_path, exist_ok=True)
        
        # Set device
        if self.hardware.device == "cuda" and not torch.cuda.is_available():
            self.hardware.device = "cpu"
            print("Warning: CUDA not available, falling back to CPU")
        
<<<<<<< HEAD
<<<<<<< HEAD
        # Ensure dtype is compatible with device (float32 on CPU)
        if self.hardware.device == "cpu" and getattr(self.model, "torch_dtype", "float16") != "float32":
            self.model.torch_dtype = "float32"
        
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
        # Set run name if not provided
        if self.run_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{self.experiment_name}_{timestamp}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "generation": self.generation.__dict__,
            "reward": self.reward.__dict__,
            "evaluation": self.evaluation.__dict__,
            "data": self.data.__dict__,
            "hardware": self.hardware.__dict__,
            "seed": self.seed,
            "debug": self.debug,
            "verbose": self.verbose,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "tags": self.tags
        }
    
    def save(self, path: str):
        """Save configuration to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ModernRLHFConfig':
        """Load configuration from file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct the configuration
        config = cls()
        config.model = ModelConfig(**config_dict["model"])
        config.training = TrainingConfig(**config_dict["training"])
        config.generation = GenerationConfig(**config_dict["generation"])
        config.reward = RewardConfig(**config_dict["reward"])
        config.evaluation = EvaluationConfig(**config_dict["evaluation"])
        config.data = DataConfig(**config_dict["data"])
        config.hardware = HardwareConfig(**config_dict["hardware"])
        config.seed = config_dict["seed"]
        config.debug = config_dict["debug"]
        config.verbose = config_dict["verbose"]
        config.experiment_name = config_dict["experiment_name"]
        config.run_name = config_dict["run_name"]
        config.tags = config_dict["tags"]
        
        return config


# Predefined configurations for common use cases
def get_research_config() -> ModernRLHFConfig:
    """Get configuration optimized for research experiments."""
    config = ModernRLHFConfig()
    config.training.total_steps = 2000
    config.training.learning_rate = 3e-6
    config.evaluation.eval_samples = 200
    config.tags = ["research", "experimental"]
    return config


def get_production_config() -> ModernRLHFConfig:
    """Get configuration optimized for production deployment."""
    config = ModernRLHFConfig()
    config.training.total_steps = 5000
    config.training.learning_rate = 1e-6
    config.evaluation.eval_samples = 500
    config.tags = ["production", "stable"]
    return config


def get_fast_config() -> ModernRLHFConfig:
    """Get configuration optimized for fast experimentation."""
    config = ModernRLHFConfig()
    config.training.total_steps = 500
    config.training.learning_rate = 1e-5
    config.evaluation.eval_samples = 50
    config.tags = ["fast", "prototype"]
    return config
