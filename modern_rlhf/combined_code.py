

# ------------------------------------------------------------
# FILE: .\config.py
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# FILE: .\data_loader.py
# ------------------------------------------------------------

"""
Modern Data Loader for RLHF
===========================

A comprehensive data loader that handles:
- Training data preparation
- Evaluation data loading
- Human feedback integration
- Data preprocessing and augmentation
"""

import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import random
from pathlib import Path

from .config import ModernRLHFConfig, DataConfig

logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    """Container for a single data sample."""
    prompt: str
    response: str
    reference: Optional[str] = None
    rating: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ModernDataLoader:
    """Modern data loader for RLHF training."""
    
    def __init__(self, config: ModernRLHFConfig):
        self.config = config
        self.data_config = config.data
        
        # Set random seed for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        logger.info(f"Initialized ModernDataLoader with config: {self.data_config}")
    
    def load_training_data(self) -> List[DataSample]:
        """Load training data from various sources."""
        logger.info("Loading training data...")
        
        all_samples = []
        
        # Load from different sources
        sources = [
            self._load_sft_data,
            self._load_preference_data,
            self._load_synthetic_data
        ]
        
        for source_func in sources:
            try:
                samples = source_func()
                all_samples.extend(samples)
                logger.info(f"Loaded {len(samples)} samples from {source_func.__name__}")
            except Exception as e:
                logger.warning(f"Failed to load from {source_func.__name__}: {e}")
        
        # Filter and clean data
        filtered_samples = self._filter_samples(all_samples)
        
        # Limit samples if specified
        if self.data_config.max_train_samples > 0:
            filtered_samples = filtered_samples[:self.data_config.max_train_samples]
        
        logger.info(f"Total training samples loaded: {len(filtered_samples)}")
        
        return filtered_samples
    
    def load_evaluation_data(self) -> List[DataSample]:
        """Load evaluation data."""
        logger.info("Loading evaluation data...")
        
        all_samples = []
        
        # Load from evaluation datasets
        eval_path = Path(self.data_config.eval_data_path)
        
        if eval_path.exists():
            for dataset_file in self.data_config.evaluation.eval_datasets:
                try:
                    samples = self._load_evaluation_dataset(eval_path / dataset_file)
                    all_samples.extend(samples)
                    logger.info(f"Loaded {len(samples)} samples from {dataset_file}")
                except Exception as e:
                    logger.warning(f"Failed to load {dataset_file}: {e}")
        
        # Filter and clean data
        filtered_samples = self._filter_samples(all_samples)
        
        # Limit samples if specified
        if self.data_config.max_eval_samples > 0:
            filtered_samples = filtered_samples[:self.data_config.max_eval_samples]
        
        logger.info(f"Total evaluation samples loaded: {len(filtered_samples)}")
        
        return filtered_samples
    
    def load_human_feedback(self) -> Optional[str]:
        """Load human feedback data."""
        logger.info("Loading human feedback data...")
        
        feedback_path = Path(self.data_config.human_feedback_path)
        
        if feedback_path.exists():
            # Look for JSON files with human feedback
            json_files = list(feedback_path.glob("*.json"))
            
            if json_files:
                # Use the most recent file
                latest_file = max(json_files, key=os.path.getmtime)
                logger.info(f"Found human feedback file: {latest_file}")
                return str(latest_file)
            else:
                logger.warning("No JSON files found in human feedback directory")
        else:
            logger.warning(f"Human feedback directory not found: {feedback_path}")
        
        return None
    
    def _load_sft_data(self) -> List[DataSample]:
        """Load supervised fine-tuning data."""
        samples = []
        
        sft_path = Path(self.data_config.train_data_path) / "sft_dataset.csv"
        
        if sft_path.exists():
            df = pd.read_csv(sft_path)
            
            # Find appropriate columns
            prompt_col = self._find_column(df, ['prompt', 'instruction', 'question', 'input'])
            response_col = self._find_column(df, ['response', 'answer', 'output', 'completion'])
            
            if prompt_col and response_col:
                for _, row in df.iterrows():
                    sample = DataSample(
                        prompt=str(row[prompt_col]),
                        response=str(row[response_col]),
                        metadata={'source': 'sft', 'row_id': row.name}
                    )
                    samples.append(sample)
        
        return samples
    
    def _load_preference_data(self) -> List[DataSample]:
        """Load preference data."""
        samples = []
        
        pref_path = Path(self.data_config.train_data_path) / "pairwise_prefs.csv"
        
        if pref_path.exists():
            df = pd.read_csv(pref_path)
            
            # Find appropriate columns
            prompt_col = self._find_column(df, ['prompt', 'instruction', 'question', 'input'])
            chosen_col = self._find_column(df, ['chosen', 'preferred', 'better'])
            rejected_col = self._find_column(df, ['rejected', 'not_preferred', 'worse'])
            rating_col = self._find_column(df, ['rating', 'score', 'preference'])
            
            if prompt_col and chosen_col and rejected_col:
                for _, row in df.iterrows():
                    # Create sample for chosen response
                    chosen_sample = DataSample(
                        prompt=str(row[prompt_col]),
                        response=str(row[chosen_col]),
                        rating=float(row[rating_col]) if rating_col else 1.0,
                        metadata={'source': 'preference', 'type': 'chosen', 'row_id': row.name}
                    )
                    samples.append(chosen_sample)
                    
                    # Create sample for rejected response
                    rejected_sample = DataSample(
                        prompt=str(row[prompt_col]),
                        response=str(row[rejected_col]),
                        rating=0.0,
                        metadata={'source': 'preference', 'type': 'rejected', 'row_id': row.name}
                    )
                    samples.append(rejected_sample)
        
        return samples
    
    def _load_synthetic_data(self) -> List[DataSample]:
        """Load synthetic data or generate if needed."""
        samples = []
        
        # Check for existing synthetic data
        synthetic_path = Path(self.data_config.train_data_path) / "synthetic_data.csv"
        
        if synthetic_path.exists():
            df = pd.read_csv(synthetic_path)
            
            prompt_col = self._find_column(df, ['prompt', 'instruction', 'question', 'input'])
            response_col = self._find_column(df, ['response', 'answer', 'output', 'completion'])
            
            if prompt_col and response_col:
                for _, row in df.iterrows():
                    sample = DataSample(
                        prompt=str(row[prompt_col]),
                        response=str(row[response_col]),
                        metadata={'source': 'synthetic', 'row_id': row.name}
                    )
                    samples.append(sample)
        else:
            # Generate some basic synthetic data if none exists
            samples = self._generate_synthetic_data()
        
        return samples
    
    def _load_evaluation_dataset(self, dataset_path: Path) -> List[DataSample]:
        """Load a specific evaluation dataset."""
        samples = []
        
        if dataset_path.suffix == '.csv':
            df = pd.read_csv(dataset_path)
            
            # Find appropriate columns
            prompt_col = self._find_column(df, ['prompt', 'instruction', 'question', 'input', 'text'])
            response_col = self._find_column(df, ['response', 'answer', 'output', 'completion', 'code'])
            reference_col = self._find_column(df, ['reference', 'ground_truth', 'expected'])
            
            if prompt_col:
                for _, row in df.iterrows():
                    sample = DataSample(
                        prompt=str(row[prompt_col]),
                        response=str(row[response_col]) if response_col else "",
                        reference=str(row[reference_col]) if reference_col else None,
                        metadata={'source': 'evaluation', 'dataset': dataset_path.name, 'row_id': row.name}
                    )
                    samples.append(sample)
        
        return samples
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find a column with one of the possible names."""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def _filter_samples(self, samples: List[DataSample]) -> List[DataSample]:
        """Filter and clean samples based on criteria."""
        filtered_samples = []
        
        for sample in samples:
            # Check length constraints
            if len(sample.prompt) < self.data_config.min_prompt_length:
                continue
            if len(sample.prompt) > self.data_config.max_prompt_length:
                continue
            if len(sample.response) < self.data_config.min_response_length:
                continue
            if len(sample.response) > self.data_config.max_response_length:
                continue
            
            # Check for empty or invalid content
            if not sample.prompt.strip() or not sample.response.strip():
                continue
            
            # Check for code-like content (basic heuristic)
            if self._is_code_like(sample.prompt) or self._is_code_like(sample.response):
                filtered_samples.append(sample)
        
        logger.info(f"Filtered {len(samples)} samples to {len(filtered_samples)} valid samples")
        
        return filtered_samples
    
    def _is_code_like(self, text: str) -> bool:
        """Check if text looks like code."""
        # Simple heuristics for code detection
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ',
            'return ', 'print(', 'function', 'var ', 'let ', 'const ',
            '{', '}', '(', ')', ';', '=', '==', '!='
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in code_indicators)
    
    def _generate_synthetic_data(self) -> List[DataSample]:
        """Generate basic synthetic data for training."""
        samples = []
        
        # Basic code generation prompts
        basic_prompts = [
            "Write a function to calculate the factorial of a number",
            "Create a function that reverses a string",
            "Write a function to check if a number is prime",
            "Create a function that finds the maximum element in a list",
            "Write a function to sort a list of numbers",
            "Create a function that counts the frequency of each character in a string",
            "Write a function to find the greatest common divisor of two numbers",
            "Create a function that checks if a string is a palindrome",
            "Write a function to generate the Fibonacci sequence",
            "Create a function that removes duplicates from a list"
        ]
        
        # Basic responses (these would be improved with actual code generation)
        basic_responses = [
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "def reverse_string(s):\n    return s[::-1]",
            "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
            "def find_max(lst):\n    return max(lst)",
            "def sort_list(lst):\n    return sorted(lst)",
            "def count_chars(s):\n    return {char: s.count(char) for char in set(s)}",
            "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
            "def is_palindrome(s):\n    return s == s[::-1]",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "def remove_duplicates(lst):\n    return list(set(lst))"
        ]
        
        for prompt, response in zip(basic_prompts, basic_responses):
            sample = DataSample(
                prompt=prompt,
                response=response,
                metadata={'source': 'synthetic', 'generated': True}
            )
            samples.append(sample)
        
        logger.info(f"Generated {len(samples)} synthetic samples")
        
        return samples
    
    def augment_data(self, samples: List[DataSample]) -> List[DataSample]:
        """Augment training data if enabled."""
        if not self.data_config.use_data_augmentation:
            return samples
        
        logger.info("Augmenting training data...")
        
        augmented_samples = samples.copy()
        augmentation_count = int(len(samples) * self.data_config.augmentation_ratio)
        
        # Select random samples for augmentation
        indices_to_augment = random.sample(range(len(samples)), augmentation_count)
        
        for idx in indices_to_augment:
            original_sample = samples[idx]
            
            # Simple augmentation: add variations to prompts
            augmented_prompt = self._augment_prompt(original_sample.prompt)
            
            augmented_sample = DataSample(
                prompt=augmented_prompt,
                response=original_sample.response,
                reference=original_sample.reference,
                rating=original_sample.rating,
                metadata={**(original_sample.metadata or {}), 'augmented': True}
            )
            
            augmented_samples.append(augmented_sample)
        
        logger.info(f"Augmented {augmentation_count} samples, total: {len(augmented_samples)}")
        
        return augmented_samples
    
    def _augment_prompt(self, prompt: str) -> str:
        """Augment a single prompt."""
        # Simple augmentation strategies
        augmentations = [
            lambda p: f"Please {p.lower()}",
            lambda p: f"Can you {p.lower()}?",
            lambda p: f"I need help with: {p}",
            lambda p: f"Write code to {p.lower()}",
            lambda p: f"Create a solution for: {p}"
        ]
        
        # Randomly select an augmentation
        augmentation = random.choice(augmentations)
        return augmentation(prompt)
    
    def create_train_test_split(self, samples: List[DataSample]) -> Tuple[List[DataSample], List[DataSample]]:
        """Create train-test split."""
        random.shuffle(samples)
        
        split_idx = int(len(samples) * self.data_config.train_test_split)
        
        train_samples = samples[:split_idx]
        test_samples = samples[split_idx:]
        
        logger.info(f"Created train-test split: {len(train_samples)} train, {len(test_samples)} test")
        
        return train_samples, test_samples
    
    def save_samples(self, samples: List[DataSample], filepath: str):
        """Save samples to a file."""
        data = []
        
        for sample in samples:
            data.append({
                'prompt': sample.prompt,
                'response': sample.response,
                'reference': sample.reference,
                'rating': sample.rating,
                'metadata': sample.metadata
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Saved {len(samples)} samples to {filepath}")
    
    def load_samples(self, filepath: str) -> List[DataSample]:
        """Load samples from a file."""
        df = pd.read_csv(filepath)
        
        samples = []
        for _, row in df.iterrows():
            sample = DataSample(
                prompt=str(row['prompt']),
                response=str(row['response']),
                reference=str(row['reference']) if 'reference' in row and pd.notna(row['reference']) else None,
                rating=float(row['rating']) if 'rating' in row and pd.notna(row['rating']) else None,
                metadata=json.loads(row['metadata']) if 'metadata' in row and pd.notna(row['metadata']) else None
            )
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples from {filepath}")
        
        return samples


# ------------------------------------------------------------
# FILE: .\main.py
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# FILE: .\metrics.py
# ------------------------------------------------------------

"""
Modern Evaluation Metrics for Code Generation
============================================

Comprehensive evaluation metrics for code generation tasks including:
- BERTScore for semantic similarity
- CodeBLEU for code-specific evaluation
- BLEU for n-gram overlap
- ROUGE for summarization metrics
- Custom Ruby metric for code quality
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass
import re
import ast
import subprocess
import tempfile
import os

# Import evaluation libraries
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logging.warning("BERTScore not available. Install with: pip install bert-score")

try:
    from codebleu import calc_codebleu
    CODEBLEU_AVAILABLE = True
except ImportError:
    CODEBLEU_AVAILABLE = False
    logging.warning("CodeBLEU not available. Install with: pip install codebleu")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("ROUGE not available. Install with: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    logging.warning("BLEU not available. Install with: pip install nltk")

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Container for metric evaluation results."""
    metric_name: str
    score: float
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class CodeQualityAnalyzer:
    """Analyzer for code quality metrics."""
    
    def __init__(self):
        self.smoothing = SmoothingFunction().method1 if BLEU_AVAILABLE else None
    
    def analyze_syntax(self, code: str) -> Dict[str, Any]:
        """Analyze syntax correctness of code."""
        try:
            # Try to parse the code
            ast.parse(code)
            return {
                "syntax_correct": True,
                "syntax_error": None,
                "syntax_score": 1.0
            }
        except SyntaxError as e:
            return {
                "syntax_correct": False,
                "syntax_error": str(e),
                "syntax_score": 0.0
            }
        except Exception as e:
            return {
                "syntax_correct": False,
                "syntax_error": f"Parse error: {str(e)}",
                "syntax_score": 0.0
            }
    
    def analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity metrics."""
        try:
            tree = ast.parse(code)
            
            # Count different constructs
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            loops = len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))])
            conditionals = len([node for node in ast.walk(tree) if isinstance(node, ast.If)])
            
            # Calculate complexity score (simplified)
            complexity_score = min(1.0, max(0.0, 1.0 - (functions + classes + loops + conditionals) / 20.0))
            
            return {
                "functions": functions,
                "classes": classes,
                "loops": loops,
                "conditionals": conditionals,
                "complexity_score": complexity_score
            }
        except Exception as e:
            return {
                "functions": 0,
                "classes": 0,
                "loops": 0,
                "conditionals": 0,
                "complexity_score": 0.0,
                "error": str(e)
            }
    
    def analyze_style(self, code: str) -> Dict[str, Any]:
        """Analyze code style metrics."""
        lines = code.split('\n')
        
        # Basic style metrics
        avg_line_length = np.mean([len(line) for line in lines if line.strip()])
        long_lines = sum(1 for line in lines if len(line) > 80)
        empty_lines = sum(1 for line in lines if not line.strip())
        
        # Style score (simplified)
        style_score = 1.0
        if avg_line_length > 100:
            style_score -= 0.2
        if long_lines / len(lines) > 0.1:
            style_score -= 0.2
        if empty_lines / len(lines) > 0.3:
            style_score -= 0.1
        
        return {
            "avg_line_length": avg_line_length,
            "long_lines": long_lines,
            "empty_lines": empty_lines,
            "style_score": max(0.0, style_score)
        }


class ModernMetricsEvaluator:
    """Modern metrics evaluator for code generation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.code_analyzer = CodeQualityAnalyzer()
        self.rouge_scorer = None
        
        # Initialize ROUGE scorer if available
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def compute_bertscore(self, predictions: List[str], references: List[str]) -> MetricResult:
        """Compute BERTScore for semantic similarity."""
        if not BERTSCORE_AVAILABLE:
            return MetricResult(
                metric_name="bertscore",
                score=0.0,
                error="BERTScore not available"
            )
        
        try:
            P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
            
            # Return F1 score (harmonic mean of precision and recall)
            score = float(F1.mean())
            
            return MetricResult(
                metric_name="bertscore",
                score=score,
                details={
                    "precision": float(P.mean()),
                    "recall": float(R.mean()),
                    "f1": score
                }
            )
        except Exception as e:
            return MetricResult(
                metric_name="bertscore",
                score=0.0,
                error=str(e)
            )
    
    def compute_codebleu(self, predictions: List[str], references: List[str]) -> MetricResult:
        """Compute CodeBLEU for code-specific evaluation."""
        if not CODEBLEU_AVAILABLE:
            return MetricResult(
                metric_name="codebleu",
                score=0.0,
                error="CodeBLEU not available"
            )
        
        try:
            # CodeBLEU expects specific format
            results = []
            for pred, ref in zip(predictions, references):
                try:
                    # Ensure we have valid strings
                    if not pred or not ref:
                        results.append(0.0)
                        continue
                    
                    # CodeBLEU expects references as list of strings
                    score = calc_codebleu(
                        [ref], pred, lang="python", weights=[0.25, 0.25, 0.25, 0.25]
                    )
                    results.append(score)
                except Exception as e:
                    logger.warning(f"CodeBLEU computation failed for sample: {e}")
                    results.append(0.0)
            
            score = np.mean(results) if results else 0.0
            
            return MetricResult(
                metric_name="codebleu",
                score=score,
                details={"individual_scores": results}
            )
        except Exception as e:
            return MetricResult(
                metric_name="codebleu",
                score=0.0,
                error=str(e)
            )
    
    def compute_bleu(self, predictions: List[str], references: List[str]) -> MetricResult:
        """Compute BLEU score for n-gram overlap."""
        if not BLEU_AVAILABLE:
            return MetricResult(
                metric_name="bleu",
                score=0.0,
                error="BLEU not available"
            )
        
        try:
            results = []
            for pred, ref in zip(predictions, references):
                # Tokenize (simple whitespace tokenization)
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                
                if len(pred_tokens) == 0:
                    results.append(0.0)
                    continue
                
                # Compute BLEU score
                score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.code_analyzer.smoothing)
                results.append(score)
            
            score = np.mean(results)
            
            return MetricResult(
                metric_name="bleu",
                score=score,
                details={"individual_scores": results}
            )
        except Exception as e:
            return MetricResult(
                metric_name="bleu",
                score=0.0,
                error=str(e)
            )
    
    def compute_rouge(self, predictions: List[str], references: List[str]) -> MetricResult:
        """Compute ROUGE scores for summarization metrics."""
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            return MetricResult(
                metric_name="rouge",
                score=0.0,
                error="ROUGE not available"
            )
        
        try:
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for pred, ref in zip(predictions, references):
                scores = self.rouge_scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            # Return average ROUGE-L score
            score = np.mean(rougeL_scores)
            
            return MetricResult(
                metric_name="rouge",
                score=score,
                details={
                    "rouge1": np.mean(rouge1_scores),
                    "rouge2": np.mean(rouge2_scores),
                    "rougeL": score
                }
            )
        except Exception as e:
            return MetricResult(
                metric_name="rouge",
                score=0.0,
                error=str(e)
            )
    
    def compute_ruby(self, predictions: List[str], references: List[str]) -> MetricResult:
        """Compute custom Ruby metric for code quality."""
        try:
            results = []
            
            for pred, ref in zip(predictions, references):
                # Analyze syntax
                syntax_analysis = self.code_analyzer.analyze_syntax(pred)
                syntax_score = syntax_analysis["syntax_score"]
                
                # Analyze complexity
                complexity_analysis = self.code_analyzer.analyze_complexity(pred)
                complexity_score = complexity_analysis["complexity_score"]
                
                # Analyze style
                style_analysis = self.code_analyzer.analyze_style(pred)
                style_score = style_analysis["style_score"]
                
                # Simple execution test (if possible)
                execution_score = self._test_execution(pred)
                
                # Combined Ruby score
                ruby_score = (
                    syntax_score * 0.4 +
                    complexity_score * 0.2 +
                    style_score * 0.2 +
                    execution_score * 0.2
                )
                
                results.append(ruby_score)
            
            score = np.mean(results)
            
            return MetricResult(
                metric_name="ruby",
                score=score,
                details={
                    "syntax_scores": [self.code_analyzer.analyze_syntax(p)["syntax_score"] for p in predictions],
                    "complexity_scores": [self.code_analyzer.analyze_complexity(p)["complexity_score"] for p in predictions],
                    "style_scores": [self.code_analyzer.analyze_style(p)["style_score"] for p in predictions],
                    "execution_scores": [self._test_execution(p) for p in predictions]
                }
            )
        except Exception as e:
            return MetricResult(
                metric_name="ruby",
                score=0.0,
                error=str(e)
            )
    
    def _test_execution(self, code: str) -> float:
        """Test if code can be executed (simplified version)."""
        try:
            # Create a safe execution environment
            safe_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round,
                    'int': int,
                    'float': float,
                    'str': str,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'bool': bool,
                    'type': type,
                    'isinstance': isinstance,
                    'hasattr': hasattr,
                    'getattr': getattr,
                    'setattr': setattr,
                }
            }
            
            # Try to compile and execute
            compiled = compile(code, '<string>', 'exec')
            exec(compiled, safe_globals)
            return 1.0
            
        except Exception:
            return 0.0
    
    def compute_all_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, MetricResult]:
        """Compute all available metrics."""
        metrics = {}
        
        # Compute each metric
        metrics["bertscore"] = self.compute_bertscore(predictions, references)
        metrics["codebleu"] = self.compute_codebleu(predictions, references)
        metrics["bleu"] = self.compute_bleu(predictions, references)
        metrics["rouge"] = self.compute_rouge(predictions, references)
        metrics["ruby"] = self.compute_ruby(predictions, references)
        
        return metrics
    
    def evaluate_against_targets(self, metrics: Dict[str, MetricResult], targets: Dict[str, float]) -> Dict[str, bool]:
        """Evaluate if metrics meet target thresholds."""
        results = {}
        
        for metric_name, target in targets.items():
            if metric_name in metrics:
                results[metric_name] = metrics[metric_name].score >= target
            else:
                results[metric_name] = False
        
        return results
    
    def get_summary(self, metrics: Dict[str, MetricResult]) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        summary = {
            "scores": {},
            "errors": {},
            "overall_success": True
        }
        
        for metric_name, result in metrics.items():
            summary["scores"][metric_name] = result.score
            if result.error:
                summary["errors"][metric_name] = result.error
                summary["overall_success"] = False
        
        return summary


# Utility functions for batch evaluation
def evaluate_batch(
    predictions: List[str],
    references: List[str],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Evaluate a batch of predictions against references."""
    evaluator = ModernMetricsEvaluator(config)
    metrics = evaluator.compute_all_metrics(predictions, references)
    summary = evaluator.get_summary(metrics)
    
    return {
        "metrics": metrics,
        "summary": summary
    }


def evaluate_single(
    prediction: str,
    reference: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Evaluate a single prediction against a reference."""
    return evaluate_batch([prediction], [reference], config)


# ------------------------------------------------------------
# FILE: .\pipeline.py
# ------------------------------------------------------------

"""
Modern RLHF Pipeline
===================

A complete, modern RLHF pipeline for code generation with:
- Data loading and preprocessing
- Reward model training
- PPO/DPO training
- Comprehensive evaluation
- Results visualization
"""

import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import os
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

from .config import ModernRLHFConfig, get_research_config, get_production_config, get_fast_config
from .reward_model import ModernRewardModel, RewardModelTrainer
from .trainer import ModernRLHFTrainer
from .metrics import ModernMetricsEvaluator
from .data_loader import ModernDataLoader

logger = logging.getLogger(__name__)


@dataclass
class PipelineResults:
    """Container for pipeline results."""
    config: ModernRLHFConfig
    reward_model_metrics: Dict[str, float]
    training_metrics: Dict[str, float]
    evaluation_metrics: Dict[str, float]
    final_metrics: Dict[str, float]
    training_time: float
    total_time: float
    success: bool
    error_message: Optional[str] = None


class ModernRLHFPipeline:
    """Main RLHF pipeline class."""
    
    def __init__(self, config: Optional[ModernRLHFConfig] = None):
        self.config = config or get_research_config()
        self.device = torch.device(self.config.hardware.device)
        
        # Initialize components
        self.data_loader = ModernDataLoader(self.config)
        self.metrics_evaluator = ModernMetricsEvaluator()
        
        # Training components (initialized later)
        self.reward_model = None
        self.reward_trainer = None
        self.rlhf_trainer = None
        
        # Results
        self.results = None
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Initialized Modern RLHF Pipeline with config: {self.config.experiment_name}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.config.debug else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.config.data.output_path, 'pipeline.log'))
            ]
        )
    
    def load_data(self) -> Tuple[Any, Any, Any]:
        """Load training and evaluation data."""
        logger.info("Loading data...")
        
        # Load training data
        train_data = self.data_loader.load_training_data()
        
        # Load evaluation data
        eval_data = self.data_loader.load_evaluation_data()
        
        # Load human feedback data
        human_feedback = self.data_loader.load_human_feedback()
        
        logger.info(f"Loaded {len(train_data)} training samples, {len(eval_data)} eval samples")
        
        return train_data, eval_data, human_feedback
    
    def prepare_reward_model(self, train_data: Any, human_feedback: Any) -> ModernRewardModel:
        """Prepare and train the reward model."""
        logger.info("Preparing reward model...")
        
        # Initialize reward model
        self.reward_model = ModernRewardModel(
            self.config.reward,
            self.config.model.reward_model_name
        )
        
        # Load human feedback if available
        if human_feedback:
            self.reward_model.load_human_feedback(human_feedback)
        
        # Initialize reward trainer
        self.reward_trainer = RewardModelTrainer(self.reward_model, self.config.reward)
        
        # Train reward model if needed
        if self.config.reward.reward_epochs > 0:
            logger.info("Training reward model...")
            self._train_reward_model(train_data)
        
        return self.reward_model
    
    def _train_reward_model(self, train_data: Any):
        """Train the reward model."""
        # Convert data to training format
        train_batches = self._prepare_reward_training_batches(train_data)
        
        # Training loop
        for epoch in range(self.config.reward.reward_epochs):
            epoch_metrics = []
            
            for batch in tqdm(train_batches, desc=f"Reward Training Epoch {epoch}"):
                metrics = self.reward_trainer.train_step(batch)
                epoch_metrics.append(metrics)
            
            # Average metrics
            avg_metrics = {}
            for key in epoch_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
            
            logger.info(f"Reward Model Epoch {epoch}: {avg_metrics}")
        
        # Save reward model
        reward_model_path = os.path.join(self.config.data.output_path, "reward_model")
        self.reward_model.save_model(reward_model_path)
        logger.info(f"Reward model saved to {reward_model_path}")
    
    def _prepare_reward_training_batches(self, train_data: Any) -> List[Dict[str, Any]]:
        """Prepare batches for reward model training."""
        batches = []
        batch_size = self.config.reward.reward_batch_size
        
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i + batch_size]
            
            batch = {
                'prompts': [item['prompt'] for item in batch_data],
                'responses': [item['response'] for item in batch_data],
                'human_ratings': [item.get('rating', None) for item in batch_data]
            }
            
            batches.append(batch)
        
        return batches
    
    def prepare_rlhf_trainer(self) -> ModernRLHFTrainer:
        """Prepare the RLHF trainer."""
        logger.info("Preparing RLHF trainer...")
        
        if self.reward_model is None:
            raise ValueError("Reward model must be prepared before RLHF trainer")
        
        # Initialize RLHF trainer
        self.rlhf_trainer = ModernRLHFTrainer(self.config, self.reward_model)
        
        return self.rlhf_trainer
    
    def train_rlhf(self, train_data: Any, eval_data: Any) -> Dict[str, float]:
        """Train the RLHF model."""
        logger.info("Starting RLHF training...")
        
        if self.rlhf_trainer is None:
            raise ValueError("RLHF trainer must be prepared before training")
        
        # Prepare data loaders
        train_dataloader = self._prepare_rlhf_dataloader(train_data, is_training=True)
        eval_dataloader = self._prepare_rlhf_dataloader(eval_data, is_training=False)
        
        # Train
        training_metrics = self.rlhf_trainer.train(train_dataloader, eval_dataloader)
        
        logger.info(f"RLHF training completed. Final metrics: {training_metrics}")
        
        return training_metrics
    
    def _prepare_rlhf_dataloader(self, data: Any, is_training: bool = True) -> List[Dict[str, Any]]:
        """Prepare data loader for RLHF training."""
        dataloader = []
        batch_size = self.config.training.batch_size
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            
            if is_training:
                # For training, we need prompt-response pairs
                batch = {
                    'prompts': [item['prompt'] for item in batch_data],
                    'responses': [item.get('response', '') for item in batch_data]
                }
            else:
                # For evaluation, we need prompts and references
                batch = {
                    'prompts': [item['prompt'] for item in batch_data],
                    'references': [item.get('reference', '') for item in batch_data]
                }
            
            dataloader.append(batch)
        
        return dataloader
    
    def evaluate_model(self, eval_data: Any) -> Dict[str, float]:
        """Evaluate the trained model."""
        logger.info("Evaluating model...")
        
        if self.rlhf_trainer is None:
            raise ValueError("RLHF trainer must be prepared before evaluation")
        
        # Generate responses
        all_prompts = [item['prompt'] for item in eval_data]
        all_references = [item.get('reference', '') for item in eval_data]
        
        # Generate responses in batches
        all_responses = []
        batch_size = self.config.evaluation.eval_batch_size
        
        for i in tqdm(range(0, len(all_prompts), batch_size), desc="Generating responses"):
            batch_prompts = all_prompts[i:i + batch_size]
            
            # Generate responses
            generation_output = self.rlhf_trainer.trainer.generate_responses(batch_prompts)
            batch_responses = generation_output['response_texts']
            
            all_responses.extend(batch_responses)
        
        # Compute metrics
        metrics_results = self.metrics_evaluator.compute_all_metrics(all_responses, all_references)
        
        # Convert to simple dict
        evaluation_metrics = {}
        for metric_name, result in metrics_results.items():
            evaluation_metrics[metric_name] = result.score
        
        # Check against targets
        targets = {
            'bertscore': self.config.evaluation.target_bertscore,
            'codebleu': self.config.evaluation.target_codebleu,
            'bleu': self.config.evaluation.target_bleu,
            'rouge': self.config.evaluation.target_rouge,
            'ruby': self.config.evaluation.target_ruby
        }
        
        target_results = self.metrics_evaluator.evaluate_against_targets(metrics_results, targets)
        evaluation_metrics['targets_met'] = target_results
        
        logger.info(f"Evaluation completed. Metrics: {evaluation_metrics}")
        
        return evaluation_metrics
    
    def run_full_pipeline(self) -> PipelineResults:
        """Run the complete RLHF pipeline."""
        start_time = time.time()
        
        try:
            logger.info("Starting full RLHF pipeline...")
            
            # Step 1: Load data
            train_data, eval_data, human_feedback = self.load_data()
            
            # Step 2: Prepare reward model
            reward_model_start = time.time()
            self.prepare_reward_model(train_data, human_feedback)
            reward_model_time = time.time() - reward_model_start
            
            # Step 3: Prepare RLHF trainer
            self.prepare_rlhf_trainer()
            
            # Step 4: Train RLHF model
            training_start = time.time()
            training_metrics = self.train_rlhf(train_data, eval_data)
            training_time = time.time() - training_start
            
            # Step 5: Evaluate model
            evaluation_start = time.time()
            evaluation_metrics = self.evaluate_model(eval_data)
            evaluation_time = time.time() - evaluation_start
            
            # Step 6: Compute final metrics
            final_metrics = self._compute_final_metrics(evaluation_metrics)
            
            # Create results
            total_time = time.time() - start_time
            
            self.results = PipelineResults(
                config=self.config,
                reward_model_metrics={'training_time': reward_model_time},
                training_metrics=training_metrics,
                evaluation_metrics=evaluation_metrics,
                final_metrics=final_metrics,
                training_time=training_time,
                total_time=total_time,
                success=True
            )
            
            # Save results
            self._save_results()
            
            logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            
            self.results = PipelineResults(
                config=self.config,
                reward_model_metrics={},
                training_metrics={},
                evaluation_metrics={},
                final_metrics={},
                training_time=0.0,
                total_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
            
            return self.results
    
    def _compute_final_metrics(self, evaluation_metrics: Dict[str, float]) -> Dict[str, float]:
        """Compute final success metrics."""
        final_metrics = {}
        
        # Check if targets are met
        targets_met = evaluation_metrics.get('targets_met', {})
        final_metrics['all_targets_met'] = all(targets_met.values())
        final_metrics['targets_met_count'] = sum(targets_met.values())
        final_metrics['targets_total'] = len(targets_met)
        
        # Overall success score
        if 'targets_met' in evaluation_metrics:
            success_score = sum(targets_met.values()) / len(targets_met)
            final_metrics['success_score'] = success_score
        else:
            final_metrics['success_score'] = 0.0
        
        return final_metrics
    
    def _save_results(self):
        """Save pipeline results."""
        if self.results is None:
            return
        
        # Save results to JSON
        results_path = os.path.join(self.config.data.output_path, 'pipeline_results.json')
        
        results_dict = {
            'config': self.results.config.to_dict(),
            'reward_model_metrics': self.results.reward_model_metrics,
            'training_metrics': self.results.training_metrics,
            'evaluation_metrics': self.results.evaluation_metrics,
            'final_metrics': self.results.final_metrics,
            'training_time': self.results.training_time,
            'total_time': self.results.total_time,
            'success': self.results.success,
            'error_message': self.results.error_message,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save configuration
        config_path = os.path.join(self.config.data.output_path, 'config.json')
        self.config.save(config_path)
        
        logger.info(f"Results saved to {results_path}")
    
    def visualize_results(self):
        """Create visualizations of the results."""
        if self.results is None:
            logger.warning("No results to visualize")
            return
        
        # Create output directory for plots
        plots_dir = os.path.join(self.config.data.output_path, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot 1: Evaluation metrics
        self._plot_evaluation_metrics(plots_dir)
        
        # Plot 2: Training progress
        self._plot_training_progress(plots_dir)
        
        # Plot 3: Target achievement
        self._plot_target_achievement(plots_dir)
        
        logger.info(f"Visualizations saved to {plots_dir}")
    
    def _plot_evaluation_metrics(self, plots_dir: str):
        """Plot evaluation metrics."""
        metrics = self.results.evaluation_metrics
        
        # Filter out non-numeric metrics
        numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and k != 'targets_met'}
        
        if not numeric_metrics:
            return
        
        plt.figure(figsize=(10, 6))
        metric_names = list(numeric_metrics.keys())
        metric_values = list(numeric_metrics.values())
        
        bars = plt.bar(metric_names, metric_values, color='skyblue', alpha=0.7)
        
        # Add target lines
        targets = {
            'bertscore': self.config.evaluation.target_bertscore,
            'codebleu': self.config.evaluation.target_codebleu,
            'bleu': self.config.evaluation.target_bleu,
            'rouge': self.config.evaluation.target_rouge,
            'ruby': self.config.evaluation.target_ruby
        }
        
        for i, (metric_name, target) in enumerate(targets.items()):
            if metric_name in numeric_metrics:
                plt.axhline(y=target, color='red', linestyle='--', alpha=0.7, label=f'{metric_name} target' if i == 0 else "")
        
        plt.title('Evaluation Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, 'evaluation_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_progress(self, plots_dir: str):
        """Plot training progress."""
        # This would require training history data
        # For now, create a simple placeholder
        plt.figure(figsize=(10, 6))
        plt.title('Training Progress (Placeholder)')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.text(0.5, 0.5, 'Training progress visualization\nwould be implemented here', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.savefig(os.path.join(plots_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_target_achievement(self, plots_dir: str):
        """Plot target achievement."""
        if 'targets_met' not in self.results.evaluation_metrics:
            return
        
        targets_met = self.results.evaluation_metrics['targets_met']
        
        plt.figure(figsize=(8, 6))
        metric_names = list(targets_met.keys())
        achieved = [1 if targets_met[name] else 0 for name in metric_names]
        
        colors = ['green' if a else 'red' for a in achieved]
        bars = plt.bar(metric_names, achieved, color=colors, alpha=0.7)
        
        plt.title('Target Achievement')
        plt.ylabel('Achieved (1) / Not Achieved (0)')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.2)
        
        # Add text labels
        for bar, achieved in zip(bars, achieved):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    '✓' if achieved else '✗', ha='center', va='bottom', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'target_achievement.png'), dpi=300, bbox_inches='tight')
        plt.close()


# Convenience functions for different use cases
def run_research_experiment() -> PipelineResults:
    """Run a research experiment with optimized settings."""
    config = get_research_config()
    pipeline = ModernRLHFPipeline(config)
    results = pipeline.run_full_pipeline()
    pipeline.visualize_results()
    return results


def run_production_training() -> PipelineResults:
    """Run production training with stable settings."""
    config = get_production_config()
    pipeline = ModernRLHFPipeline(config)
    results = pipeline.run_full_pipeline()
    pipeline.visualize_results()
    return results


def run_fast_prototype() -> PipelineResults:
    """Run a fast prototype for quick testing."""
    config = get_fast_config()
    pipeline = ModernRLHFPipeline(config)
    results = pipeline.run_full_pipeline()
    pipeline.visualize_results()
    return results


if __name__ == "__main__":
    # Example usage
    results = run_research_experiment()
    print(f"Pipeline completed with success: {results.success}")
    print(f"Final metrics: {results.final_metrics}")


# ------------------------------------------------------------
# FILE: .\reward_model.py
# ------------------------------------------------------------

"""
Modern Reward Model with Human Feedback Integration
=================================================

A state-of-the-art reward model that combines multiple signals:
- Syntax correctness
- Execution success
- Semantic similarity
- Human preference feedback
- Code quality metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    PreTrainedModel, PreTrainedTokenizer
)
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import numpy as np
from dataclasses import dataclass
import json
import os
import ast
import subprocess
import tempfile

from .metrics import ModernMetricsEvaluator, CodeQualityAnalyzer
from .config import RewardConfig

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Container for different reward components."""
    syntax_reward: float = 0.0
    execution_reward: float = 0.0
    semantic_reward: float = 0.0
    human_preference_reward: float = 0.0
    quality_reward: float = 0.0
    total_reward: float = 0.0


class HumanFeedbackIntegrator:
    """Integrates human feedback into reward computation."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.human_logits_cache = {}
        self.feedback_weights = {}
        
    def load_human_feedback(self, feedback_path: str):
        """Load human feedback data from file."""
        try:
            if os.path.exists(feedback_path):
                with open(feedback_path, 'r') as f:
                    feedback_data = json.load(f)
                
                # Process feedback data
                for item in feedback_data:
                    prompt = item.get('prompt', '')
                    response = item.get('response', '')
                    rating = item.get('rating', 0.0)
                    logits = item.get('logits', None)
                    
                    # Store human logits if available
                    if logits and self.config.use_human_logits:
                        key = f"{prompt[:50]}_{response[:50]}"
                        self.human_logits_cache[key] = {
                            'logits': logits,
                            'rating': rating
                        }
                
                logger.info(f"Loaded {len(self.human_logits_cache)} human feedback entries")
                
        except Exception as e:
            logger.warning(f"Failed to load human feedback: {e}")
    
    def get_human_logits(self, prompt: str, response: str) -> Optional[torch.Tensor]:
        """Get human logits for a prompt-response pair."""
        key = f"{prompt[:50]}_{response[:50]}"
        
        if key in self.human_logits_cache:
            logits_data = self.human_logits_cache[key]['logits']
            if isinstance(logits_data, list):
                return torch.tensor(logits_data, dtype=torch.float32)
            elif isinstance(logits_data, dict):
                # Handle different logits formats
                if 'last_layer' in logits_data:
                    return torch.tensor(logits_data['last_layer'], dtype=torch.float32)
                elif 'logits' in logits_data:
                    return torch.tensor(logits_data['logits'], dtype=torch.float32)
        
        return None
    
    def compute_human_preference_reward(self, prompt: str, response: str) -> float:
        """Compute reward based on human preferences."""
        key = f"{prompt[:50]}_{response[:50]}"
        
        if key in self.human_logits_cache:
            rating = self.human_logits_cache[key]['rating']
            # Normalize rating to [0, 1] range
            return max(0.0, min(1.0, rating / 5.0))  # Assuming 5-point scale
        
        return 0.5  # Neutral reward if no human feedback available


class SyntaxChecker:
    """Advanced syntax checking for code."""
    
    def __init__(self):
        self.supported_languages = ['python', 'javascript', 'java', 'cpp']
    
    def check_syntax(self, code: str, language: str = 'python') -> Tuple[bool, float, str]:
        """Check syntax correctness of code."""
        if language == 'python':
            return self._check_python_syntax(code)
        else:
            # For other languages, use basic parsing
            return self._check_generic_syntax(code)
    
    def _check_python_syntax(self, code: str) -> Tuple[bool, float, str]:
        """Check Python syntax."""
        try:
            ast.parse(code)
            return True, 1.0, ""
        except SyntaxError as e:
            return False, 0.0, str(e)
        except Exception as e:
            return False, 0.0, f"Parse error: {str(e)}"
    
    def _check_generic_syntax(self, code: str) -> Tuple[bool, float, str]:
        """Generic syntax checking."""
        # Basic checks
        if not code.strip():
            return False, 0.0, "Empty code"
        
        # Check for balanced brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return False, 0.0, "Unbalanced brackets"
                if brackets[stack.pop()] != char:
                    return False, 0.0, "Unbalanced brackets"
        
        if stack:
            return False, 0.0, "Unbalanced brackets"
        
        return True, 0.8, ""  # Partial credit for basic structure


class ExecutionTester:
    """Test code execution in a safe environment."""
    
    def __init__(self):
        self.timeout = 5  # seconds
        self.max_memory = 100 * 1024 * 1024  # 100MB
    
    def test_execution(self, code: str, language: str = 'python') -> Tuple[bool, float, str]:
        """Test if code can be executed successfully."""
        if language == 'python':
            return self._test_python_execution(code)
        else:
            return False, 0.0, f"Execution testing not supported for {language}"
    
    def _test_python_execution(self, code: str) -> Tuple[bool, float, str]:
        """Test Python code execution."""
        try:
            # Create a safe execution environment
            safe_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round,
                    'int': int,
                    'float': float,
                    'str': str,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'bool': bool,
                    'type': type,
                    'isinstance': isinstance,
                    'hasattr': hasattr,
                    'getattr': getattr,
                    'setattr': setattr,
                }
            }
            
            # Try to execute
            exec(code, safe_globals)
            return True, 1.0, ""
            
        except Exception as e:
            return False, 0.0, str(e)


class ModernRewardModel(nn.Module):
    """Modern reward model with multiple signal integration."""
    
    def __init__(self, config: RewardConfig, model_name: str = "microsoft/codebert-base"):
        super().__init__()
        self.config = config
        self.model_name = model_name
        
        # Load base model
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Reward head
        hidden_size = self.base_model.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Component-specific heads
        self.syntax_head = nn.Linear(hidden_size, 1)
        self.execution_head = nn.Linear(hidden_size, 1)
        self.semantic_head = nn.Linear(hidden_size, 1)
        self.quality_head = nn.Linear(hidden_size, 1)
        
        # Human feedback integration
        self.human_feedback_integrator = HumanFeedbackIntegrator(config)
        
        # Utility components
        self.syntax_checker = SyntaxChecker()
        self.execution_tester = ExecutionTester()
        self.metrics_evaluator = ModernMetricsEvaluator()
        self.code_analyzer = CodeQualityAnalyzer()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, prompts: List[str], responses: List[str]) -> Dict[str, torch.Tensor]:
        """Forward pass through the reward model."""
        # Tokenize inputs
        inputs = self._tokenize_pairs(prompts, responses)
        
        # Get base model outputs
        with torch.no_grad():
            outputs = self.base_model(**inputs)
        
        # Get pooled representation
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        # Compute different reward components
        rewards = {}
        rewards['total'] = self.reward_head(pooled_output)
        rewards['syntax'] = self.syntax_head(pooled_output)
        rewards['execution'] = self.execution_head(pooled_output)
        rewards['semantic'] = self.semantic_head(pooled_output)
        rewards['quality'] = self.quality_head(pooled_output)
        
        return rewards
    
    def _tokenize_pairs(self, prompts: List[str], responses: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize prompt-response pairs."""
        # Combine prompts and responses
        texts = [f"{prompt} <SEP> {response}" for prompt, response in zip(prompts, responses)]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return inputs
    
    def compute_reward_components(self, prompts: List[str], responses: List[str]) -> List[RewardComponents]:
        """Compute detailed reward components for each prompt-response pair."""
        components_list = []
        
        for prompt, response in zip(prompts, responses):
            components = RewardComponents()
            
            # Syntax reward
            syntax_correct, syntax_score, syntax_error = self.syntax_checker.check_syntax(response)
            components.syntax_reward = syntax_score
            
            # Execution reward
            exec_success, exec_score, exec_error = self.execution_tester.test_execution(response)
            components.execution_reward = exec_score
            
            # Semantic reward (using BERTScore)
            try:
                semantic_result = self.metrics_evaluator.compute_bertscore([response], [prompt])
                components.semantic_reward = semantic_result.score
            except Exception as e:
                logger.warning(f"Semantic reward computation failed: {e}")
                components.semantic_reward = 0.0
            
            # Human preference reward
            components.human_preference_reward = self.human_feedback_integrator.compute_human_preference_reward(
                prompt, response
            )
            
            # Quality reward
            try:
                quality_analysis = self.code_analyzer.analyze_complexity(response)
                style_analysis = self.code_analyzer.analyze_style(response)
                components.quality_reward = (
                    quality_analysis['complexity_score'] * 0.6 +
                    style_analysis['style_score'] * 0.4
                )
            except Exception as e:
                logger.warning(f"Quality reward computation failed: {e}")
                components.quality_reward = 0.0
            
            # Compute total reward
            components.total_reward = (
                components.syntax_reward * self.config.syntax_reward_weight +
                components.execution_reward * self.config.execution_reward_weight +
                components.semantic_reward * self.config.semantic_reward_weight +
                components.human_preference_reward * self.config.human_preference_weight +
                components.quality_reward * 0.1  # Small weight for quality
            )
            
            components_list.append(components)
        
        return components_list
    
    def compute_reward(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute final reward scores."""
        # Get neural network predictions
        neural_rewards = self.forward(prompts, responses)
        
        # Get component-based rewards
        component_rewards = self.compute_reward_components(prompts, responses)
        
        # Combine neural and component rewards
        final_rewards = []
        for i, (neural_reward, component_reward) in enumerate(zip(neural_rewards['total'], component_rewards)):
            # Weighted combination
            combined_reward = (
                neural_reward.item() * 0.7 +  # Neural network prediction
                component_reward.total_reward * 0.3  # Component-based reward
            )
            
            # Apply normalization and clipping
            if self.config.reward_normalization:
                combined_reward = torch.sigmoid(torch.tensor(combined_reward))
            
            if self.config.reward_clipping:
                combined_reward = torch.clamp(
                    torch.tensor(combined_reward),
                    -self.config.reward_clip_value,
                    self.config.reward_clip_value
                )
            
            final_rewards.append(combined_reward.item())
        
        return torch.tensor(final_rewards, dtype=torch.float32)
    
    def load_human_feedback(self, feedback_path: str):
        """Load human feedback data."""
        self.human_feedback_integrator.load_human_feedback(feedback_path)
    
    def save_model(self, save_path: str):
        """Save the reward model."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(save_path, "reward_model.pt"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        config_dict = {
            "model_name": self.model_name,
            "reward_config": self.config.__dict__
        }
        with open(os.path.join(save_path, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Reward model saved to {save_path}")
    
    @classmethod
    def load_model(cls, load_path: str, config: RewardConfig):
        """Load a saved reward model."""
        # Load config
        with open(os.path.join(load_path, "config.json"), 'r') as f:
            config_dict = json.load(f)
        
        # Create model
        model = cls(config, config_dict["model_name"])
        
        # Load state dict
        state_dict = torch.load(os.path.join(load_path, "reward_model.pt"))
        model.load_state_dict(state_dict)
        
        # Load tokenizer
        model.tokenizer = AutoTokenizer.from_pretrained(load_path)
        
        logger.info(f"Reward model loaded from {load_path}")
        return model


class RewardModelTrainer:
    """Trainer for the reward model."""
    
    def __init__(self, model: ModernRewardModel, config: RewardConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.reward_learning_rate,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.reward_epochs
        )
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        prompts = batch['prompts']
        responses = batch['responses']
        human_ratings = batch.get('human_ratings', None)
        
        # Compute rewards
        predicted_rewards = self.model.compute_reward(prompts, responses)
        
        # Compute loss
        if human_ratings is not None:
            # Use human ratings as targets
            target_rewards = torch.tensor(human_ratings, dtype=torch.float32)
            loss = F.mse_loss(predicted_rewards, target_rewards)
        else:
            # Use component-based rewards as targets
            component_rewards = self.model.compute_reward_components(prompts, responses)
            target_rewards = torch.tensor([c.total_reward for c in component_rewards], dtype=torch.float32)
            loss = F.mse_loss(predicted_rewards, target_rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'predicted_reward_mean': predicted_rewards.mean().item(),
            'predicted_reward_std': predicted_rewards.std().item()
        }
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = []
        
        for batch in dataloader:
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
        
        return avg_metrics


# ------------------------------------------------------------
# FILE: .\trainer.py
# ------------------------------------------------------------

"""
Modern RLHF Trainer with PPO and DPO Support
===========================================

A state-of-the-art trainer that supports both PPO and DPO (Direct Preference Optimization)
for code generation tasks with comprehensive evaluation and monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    PreTrainedModel, PreTrainedTokenizer
)
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import numpy as np
from dataclasses import dataclass
import json
import os
import time
from tqdm import tqdm
import wandb
from collections import defaultdict

from .config import ModernRLHFConfig, TrainingConfig
from .reward_model import ModernRewardModel
from .metrics import ModernMetricsEvaluator

logger = logging.getLogger(__name__)


@dataclass
class TrainingStep:
    """Container for training step results."""
    step: int
    loss: float
    reward: float
    kl_divergence: float
    entropy: float
    learning_rate: float
    metrics: Dict[str, float]


class PPOTrainer:
    """Modern PPO trainer for RLHF."""
    
    def __init__(self, config: ModernRLHFConfig, reward_model: ModernRewardModel):
        self.config = config
        self.reward_model = reward_model
        self.device = torch.device(config.hardware.device)
        
        # Load models
        self.policy_model = self._load_policy_model()
        self.reference_model = self._load_reference_model()
        self.tokenizer = self._load_tokenizer()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=0.01
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.total_steps
        )
        
        # Metrics evaluator
        self.metrics_evaluator = ModernMetricsEvaluator()
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_reward = -float('inf')
        self.training_history = []
        
        # Initialize wandb if available
        if config.verbose and not config.debug:
            try:
                wandb.init(
                    project=config.experiment_name,
                    name=config.run_name,
                    config=config.to_dict(),
                    tags=config.tags
                )
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
    
    def _load_policy_model(self) -> PreTrainedModel:
        """Load the policy model."""
        model = AutoModel.from_pretrained(
            self.config.model.base_model_name,
            torch_dtype=getattr(torch, self.config.model.torch_dtype),
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        if self.config.hardware.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model.to(self.device)
    
    def _load_reference_model(self) -> PreTrainedModel:
        """Load the reference model (frozen)."""
        model = AutoModel.from_pretrained(
            self.config.model.base_model_name,
            torch_dtype=getattr(torch, self.config.model.torch_dtype),
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        # Freeze reference model
        for param in model.parameters():
            param.requires_grad = False
        
        return model.to(self.device)
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_name,
            use_fast=self.config.model.use_fast_tokenizer,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def generate_responses(self, prompts: List[str]) -> Dict[str, Any]:
        """Generate responses for given prompts."""
        self.policy_model.eval()
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.generation.max_prompt_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate responses
        with torch.no_grad():
            outputs = self.policy_model.generate(
                **inputs,
                max_new_tokens=self.config.generation.max_new_tokens,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                top_k=self.config.generation.top_k,
                do_sample=self.config.generation.do_sample,
                repetition_penalty=self.config.generation.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode responses
        response_texts = []
        for i, output in enumerate(outputs):
            # Remove prompt from output
            prompt_length = inputs['input_ids'][i].shape[0]
            response_tokens = output[prompt_length:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            response_texts.append(response_text)
        
        return {
            "response_texts": response_texts,
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask']
        }
    
    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute rewards for prompt-response pairs."""
        return self.reward_model.compute_reward(prompts, responses)
    
    def compute_kl_divergence(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute KL divergence between policy and reference models."""
        # Tokenize responses
        inputs = self.tokenizer(
            responses,
            padding=True,
            truncation=True,
            max_length=self.config.generation.max_response_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Get logits from both models
        with torch.no_grad():
            policy_logits = self.policy_model(**inputs).logits
            reference_logits = self.reference_model(**inputs).logits
        
        # Compute KL divergence
        policy_probs = F.softmax(policy_logits, dim=-1)
        reference_probs = F.softmax(reference_logits, dim=-1)
        
        kl_div = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            reference_probs,
            reduction='none'
        ).sum(dim=-1)
        
        return kl_div.mean(dim=1)
    
    def compute_entropy(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute entropy of policy model outputs."""
        # Tokenize responses
        inputs = self.tokenizer(
            responses,
            padding=True,
            truncation=True,
            max_length=self.config.generation.max_response_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Get logits
        with torch.no_grad():
            logits = self.policy_model(**inputs).logits
        
        # Compute entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        
        return entropy.mean(dim=1)
    
    def ppo_step(self, batch: Dict[str, Any]) -> TrainingStep:
        """Single PPO training step."""
        self.policy_model.train()
        
        prompts = batch['prompts']
        responses = batch['responses']
        old_log_probs = batch.get('old_log_probs', None)
        
        # Compute rewards
        rewards = self.compute_rewards(prompts, responses)
        
        # Compute KL divergence
        kl_div = self.compute_kl_divergence(prompts, responses)
        
        # Compute entropy
        entropy = self.compute_entropy(prompts, responses)
        
        # Compute advantages (simplified)
        advantages = rewards - rewards.mean()
        
        # Compute policy loss
        if old_log_probs is not None:
            # Compute new log probabilities
            inputs = self.tokenizer(
                responses,
                padding=True,
                truncation=True,
                max_length=self.config.generation.max_response_length,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.policy_model(**inputs)
            new_log_probs = F.log_softmax(outputs.logits, dim=-1)
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute clipped loss
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.training.ppo_clip_ratio,
                1 + self.config.training.ppo_clip_ratio
            )
            
            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
        else:
            # Simplified policy loss
            policy_loss = -rewards.mean()
        
        # Compute value loss (simplified)
        value_loss = F.mse_loss(rewards, rewards.mean().expand_as(rewards))
        
        # Compute entropy loss
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (
            policy_loss +
            self.config.training.ppo_value_loss_coef * value_loss +
            self.config.training.ppo_entropy_coef * entropy_loss +
            self.config.training.ppo_kl_penalty * kl_div.mean()
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.training.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        
        # Create training step
        step = TrainingStep(
            step=self.step,
            loss=total_loss.item(),
            reward=rewards.mean().item(),
            kl_divergence=kl_div.mean().item(),
            entropy=entropy.mean().item(),
            learning_rate=self.optimizer.param_groups[0]['lr'],
            metrics={
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'kl_penalty': kl_div.mean().item()
            }
        )
        
        self.step += 1
        return step
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = defaultdict(list)
        
        for batch in tqdm(dataloader, desc=f"Epoch {self.epoch}"):
            step = self.ppo_step(batch)
            
            # Collect metrics
            epoch_metrics['loss'].append(step.loss)
            epoch_metrics['reward'].append(step.reward)
            epoch_metrics['kl_divergence'].append(step.kl_divergence)
            epoch_metrics['entropy'].append(step.entropy)
            epoch_metrics['learning_rate'].append(step.learning_rate)
            
            # Log to wandb
            if hasattr(self, 'wandb') and self.wandb:
                wandb.log({
                    'step': step.step,
                    'loss': step.loss,
                    'reward': step.reward,
                    'kl_divergence': step.kl_divergence,
                    'entropy': step.entropy,
                    'learning_rate': step.learning_rate,
                    **step.metrics
                })
            
            # Save checkpoint
            if step.step % self.config.training.save_steps == 0:
                self.save_checkpoint()
        
        # Average metrics
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            avg_metrics[key] = np.mean(values)
        
        self.epoch += 1
        return avg_metrics
    
    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """Evaluate the model."""
        self.policy_model.eval()
        
        all_prompts = []
        all_responses = []
        all_rewards = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                prompts = batch['prompts']
                
                # Generate responses
                generation_output = self.generate_responses(prompts)
                responses = generation_output['response_texts']
                
                # Compute rewards
                rewards = self.compute_rewards(prompts, responses)
                
                all_prompts.extend(prompts)
                all_responses.extend(responses)
                all_rewards.extend(rewards.tolist())
        
        # Compute evaluation metrics
        eval_metrics = {}
        eval_metrics['avg_reward'] = np.mean(all_rewards)
        eval_metrics['reward_std'] = np.std(all_rewards)
        
        # Compute other metrics if references are available
        if 'references' in batch:
            references = batch['references']
            metrics_results = self.metrics_evaluator.compute_all_metrics(all_responses, references)
            
            for metric_name, result in metrics_results.items():
                eval_metrics[f'eval_{metric_name}'] = result.score
        
        return eval_metrics
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.data.output_path, f"checkpoint-{self.step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.policy_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            'step': self.step,
            'epoch': self.epoch,
            'best_reward': self.best_reward,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        
        with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w') as f:
            json.dump(training_state, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint."""
        # Load model
        self.policy_model = AutoModel.from_pretrained(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        
        # Load training state
        with open(os.path.join(checkpoint_dir, 'training_state.json'), 'r') as f:
            training_state = json.load(f)
        
        self.step = training_state['step']
        self.epoch = training_state['epoch']
        self.best_reward = training_state['best_reward']
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
        self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")


class DPOTrainer:
    """Direct Preference Optimization trainer."""
    
    def __init__(self, config: ModernRLHFConfig, reward_model: ModernRewardModel):
        self.config = config
        self.reward_model = reward_model
        self.device = torch.device(config.hardware.device)
        
        # Load models
        self.policy_model = self._load_policy_model()
        self.reference_model = self._load_reference_model()
        self.tokenizer = self._load_tokenizer()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=0.01
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.training_history = []
    
    def _load_policy_model(self) -> PreTrainedModel:
        """Load the policy model."""
        model = AutoModel.from_pretrained(
            self.config.model.base_model_name,
            torch_dtype=getattr(torch, self.config.model.torch_dtype),
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        if self.config.hardware.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model.to(self.device)
    
    def _load_reference_model(self) -> PreTrainedModel:
        """Load the reference model (frozen)."""
        model = AutoModel.from_pretrained(
            self.config.model.base_model_name,
            torch_dtype=getattr(torch, self.config.model.torch_dtype),
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        # Freeze reference model
        for param in model.parameters():
            param.requires_grad = False
        
        return model.to(self.device)
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_name,
            use_fast=self.config.model.use_fast_tokenizer,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def dpo_step(self, batch: Dict[str, Any]) -> TrainingStep:
        """Single DPO training step."""
        self.policy_model.train()
        
        prompts = batch['prompts']
        chosen_responses = batch['chosen_responses']
        rejected_responses = batch['rejected_responses']
        
        # Compute log probabilities for chosen responses
        chosen_log_probs = self._compute_log_probs(prompts, chosen_responses)
        
        # Compute log probabilities for rejected responses
        rejected_log_probs = self._compute_log_probs(prompts, rejected_responses)
        
        # Compute reference log probabilities
        with torch.no_grad():
            chosen_ref_log_probs = self._compute_log_probs(prompts, chosen_responses, use_reference=True)
            rejected_ref_log_probs = self._compute_log_probs(prompts, rejected_responses, use_reference=True)
        
        # Compute DPO loss
        pi_logratios = chosen_log_probs - rejected_log_probs
        ref_logratios = chosen_ref_log_probs - rejected_ref_log_probs
        
        logits = pi_logratios - ref_logratios
        
        if self.config.training.dpo_loss_type == "sigmoid":
            losses = -F.logsigmoid(self.config.training.dpo_beta * logits)
        elif self.config.training.dpo_loss_type == "hinge":
            losses = torch.relu(1 - self.config.training.dpo_beta * logits)
        else:
            raise ValueError(f"Unknown DPO loss type: {self.config.training.dpo_loss_type}")
        
        loss = losses.mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.training.max_grad_norm)
        self.optimizer.step()
        
        # Create training step
        step = TrainingStep(
            step=self.step,
            loss=loss.item(),
            reward=0.0,  # DPO doesn't use explicit rewards
            kl_divergence=0.0,
            entropy=0.0,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            metrics={
                'dpo_loss': loss.item(),
                'chosen_log_prob': chosen_log_probs.mean().item(),
                'rejected_log_prob': rejected_log_probs.mean().item(),
                'log_ratio': logits.mean().item()
            }
        )
        
        self.step += 1
        return step
    
    def _compute_log_probs(self, prompts: List[str], responses: List[str], use_reference: bool = False) -> torch.Tensor:
        """Compute log probabilities for prompt-response pairs."""
        model = self.reference_model if use_reference else self.policy_model
        
        # Tokenize
        inputs = self.tokenizer(
            [f"{prompt} {response}" for prompt, response in zip(prompts, responses)],
            padding=True,
            truncation=True,
            max_length=self.config.generation.max_prompt_length + self.config.generation.max_response_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Compute logits
        with torch.no_grad() if use_reference else torch.enable_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract log probabilities for response tokens
        response_log_probs = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            
            # Get log probabilities for response tokens
            response_start = len(prompt_tokens)
            response_end = response_start + len(response_tokens)
            
            if response_end <= logits.shape[1]:
                response_log_prob = log_probs[i, response_start:response_end, response_tokens].sum()
                response_log_probs.append(response_log_prob)
            else:
                response_log_probs.append(torch.tensor(0.0))
        
        return torch.stack(response_log_probs)
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = defaultdict(list)
        
        for batch in tqdm(dataloader, desc=f"DPO Epoch {self.epoch}"):
            step = self.dpo_step(batch)
            
            # Collect metrics
            epoch_metrics['loss'].append(step.loss)
            epoch_metrics['learning_rate'].append(step.learning_rate)
            
            # Log metrics
            for key, value in step.metrics.items():
                epoch_metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            avg_metrics[key] = np.mean(values)
        
        self.epoch += 1
        return avg_metrics


class ModernRLHFTrainer:
    """Main trainer that supports both PPO and DPO."""
    
    def __init__(self, config: ModernRLHFConfig, reward_model: ModernRewardModel):
        self.config = config
        self.reward_model = reward_model
        
        # Choose trainer based on config
        if hasattr(config.training, 'use_dpo') and config.training.use_dpo:
            self.trainer = DPOTrainer(config, reward_model)
        else:
            self.trainer = PPOTrainer(config, reward_model)
        
        logger.info(f"Initialized {type(self.trainer).__name__} trainer")
    
    def train(self, train_dataloader, eval_dataloader=None) -> Dict[str, Any]:
        """Main training loop."""
        logger.info("Starting training...")
        
        best_metrics = {}
        patience_counter = 0
        
        for epoch in range(self.config.training.ppo_epochs):
            # Training
            train_metrics = self.trainer.train_epoch(train_dataloader)
            
            # Evaluation
            if eval_dataloader is not None:
                eval_metrics = self.trainer.evaluate(eval_dataloader)
                
                # Check for improvement
                if eval_metrics.get('avg_reward', 0) > best_metrics.get('avg_reward', -float('inf')):
                    best_metrics = eval_metrics
                    patience_counter = 0
                    self.trainer.save_checkpoint()
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.config.training.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
            
            # Log metrics
            logger.info(f"Epoch {epoch}: {train_metrics}")
            if eval_dataloader is not None:
                logger.info(f"Eval metrics: {eval_metrics}")
        
        return best_metrics


# ------------------------------------------------------------
# FILE: .\__init__.py
# ------------------------------------------------------------

"""
Modern RLHF Framework for Code Generation
=========================================

A clean, modern implementation of RLHF (Reinforcement Learning from Human Feedback)
specifically designed for code generation tasks with state-of-the-art methods.

Key Features:
- Direct Preference Optimization (DPO) support
- Modern reward modeling with human feedback integration
- Comprehensive evaluation metrics (BERTScore, CodeBLEU, BLEU, ROUGE)
- Efficient training pipeline with GPU optimization
- Clean, modular architecture

Author: Research Team
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Research Team"

# Import main classes
from .config import ModernRLHFConfig, get_research_config, get_production_config, get_fast_config
from .pipeline import ModernRLHFPipeline
from .metrics import ModernMetricsEvaluator
from .reward_model import ModernRewardModel
from .trainer import ModernRLHFTrainer
from .data_loader import ModernDataLoader

# Make main classes available at package level
__all__ = [
    'ModernRLHFConfig',
    'get_research_config',
    'get_production_config', 
    'get_fast_config',
    'ModernRLHFPipeline',
    'ModernMetricsEvaluator',
    'ModernRewardModel',
    'ModernRLHFTrainer',
    'ModernDataLoader'
]
