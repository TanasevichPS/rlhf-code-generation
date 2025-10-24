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
