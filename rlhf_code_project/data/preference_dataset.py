"""
Preference Dataset for RLHF Training
===================================

Simple dataset for handling preference data for DPO training.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    """Dataset for preference-based training."""
    
    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        """
        Initialize preference dataset.
        
        Args:
            data_path: Path to preference data (CSV file)
            max_samples: Maximum number of samples to load
        """
        self.data_path = data_path
        self.max_samples = max_samples
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} preference samples from {data_path}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load preference data from file."""
        if not os.path.exists(self.data_path):
            logger.warning(f"Data file not found: {self.data_path}. Creating synthetic data.")
            return self._create_synthetic_data()
        
        try:
            # Load CSV data
            df = pd.read_csv(self.data_path)
            
            # Find appropriate columns
            prompt_col = self._find_column(df, ['prompt', 'instruction', 'question', 'input'])
            chosen_col = self._find_column(df, ['chosen', 'preferred', 'better', 'response'])
            rejected_col = self._find_column(df, ['rejected', 'not_preferred', 'worse'])
            
            if not prompt_col or not chosen_col:
                logger.warning("Required columns not found. Creating synthetic data.")
                return self._create_synthetic_data()
            
            # Convert to list of dictionaries
            data = []
            for _, row in df.iterrows():
                sample = {
                    'prompt': str(row[prompt_col]),
                    'chosen_response': str(row[chosen_col])
                }
                
                if rejected_col and rejected_col in df.columns:
                    sample['rejected_response'] = str(row[rejected_col])
                else:
                    # Generate a simple rejected response
                    sample['rejected_response'] = self._generate_rejected_response(sample['chosen_response'])
                
                data.append(sample)
            
            # Limit samples if specified
            if self.max_samples:
                data = data[:self.max_samples]
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load data from {self.data_path}: {e}. Creating synthetic data.")
            return self._create_synthetic_data()
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find a column with one of the possible names."""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def _create_synthetic_data(self) -> List[Dict[str, Any]]:
        """Create synthetic preference data for testing."""
        synthetic_data = [
            {
                'prompt': 'Write a function to calculate factorial',
                'chosen_response': 'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)',
                'rejected_response': 'def factorial(n):\n    return 1'  # Incomplete implementation
            },
            {
                'prompt': 'Write a function to reverse a string',
                'chosen_response': 'def reverse_string(s):\n    return s[::-1]',
                'rejected_response': 'def reverse_string(s):\n    return s'  # Wrong implementation
            },
            {
                'prompt': 'Write a function to check if a number is prime',
                'chosen_response': 'def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True',
                'rejected_response': 'def is_prime(n):\n    return True'  # Always returns True
            },
            {
                'prompt': 'Write a function to find the maximum element in a list',
                'chosen_response': 'def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)',
                'rejected_response': 'def find_max(lst):\n    return lst[0]'  # Only returns first element
            },
            {
                'prompt': 'Write a function to sort a list of numbers',
                'chosen_response': 'def sort_list(lst):\n    return sorted(lst)',
                'rejected_response': 'def sort_list(lst):\n    return lst'  # No sorting
            },
            {
                'prompt': 'Write a function to count the frequency of each character in a string',
                'chosen_response': 'def count_chars(s):\n    return {char: s.count(char) for char in set(s)}',
                'rejected_response': 'def count_chars(s):\n    return {}'  # Empty dictionary
            },
            {
                'prompt': 'Write a function to find the greatest common divisor of two numbers',
                'chosen_response': 'def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a',
                'rejected_response': 'def gcd(a, b):\n    return 1'  # Always returns 1
            },
            {
                'prompt': 'Write a function to check if a string is a palindrome',
                'chosen_response': 'def is_palindrome(s):\n    return s == s[::-1]',
                'rejected_response': 'def is_palindrome(s):\n    return True'  # Always returns True
            },
            {
                'prompt': 'Write a function to generate the Fibonacci sequence',
                'chosen_response': 'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)',
                'rejected_response': 'def fibonacci(n):\n    return 0'  # Always returns 0
            },
            {
                'prompt': 'Write a function to remove duplicates from a list',
                'chosen_response': 'def remove_duplicates(lst):\n    return list(set(lst))',
                'rejected_response': 'def remove_duplicates(lst):\n    return lst'  # No deduplication
            }
        ]
        
        # Limit samples if specified
        if self.max_samples:
            synthetic_data = synthetic_data[:self.max_samples]
        
        logger.info(f"Created {len(synthetic_data)} synthetic preference samples")
        return synthetic_data
    
    def _generate_rejected_response(self, chosen_response: str) -> str:
        """Generate a simple rejected response."""
        # Simple strategy: return a truncated or modified version
        lines = chosen_response.split('\n')
        if len(lines) > 1:
            # Return only the first line (incomplete)
            return lines[0]
        else:
            # Return a simple placeholder
            return "def placeholder():\n    pass"
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """Get a sample by index."""
        return self.data[idx]
    
    def get_batch(self, indices: List[int]) -> Dict[str, List[str]]:
        """Get a batch of samples."""
        batch = {
            'prompts': [],
            'chosen_responses': [],
            'rejected_responses': []
        }
        
        for idx in indices:
            sample = self.data[idx]
            batch['prompts'].append(sample['prompt'])
            batch['chosen_responses'].append(sample['chosen_response'])
            batch['rejected_responses'].append(sample['rejected_response'])
        
        return batch


class EvaluationDataset(Dataset):
    """Dataset for evaluation."""
    
    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        """
        Initialize evaluation dataset.
        
        Args:
            data_path: Path to evaluation data (CSV file)
            max_samples: Maximum number of samples to load
        """
        self.data_path = data_path
        self.max_samples = max_samples
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} evaluation samples from {data_path}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load evaluation data from file."""
        if not os.path.exists(self.data_path):
            logger.warning(f"Data file not found: {self.data_path}. Creating synthetic data.")
            return self._create_synthetic_data()
        
        try:
            # Load CSV data
            df = pd.read_csv(self.data_path)
            
            # Find appropriate columns
            prompt_col = self._find_column(df, ['prompt', 'instruction', 'question', 'input'])
            reference_col = self._find_column(df, ['reference', 'ground_truth', 'expected', 'response'])
            
            if not prompt_col:
                logger.warning("Required columns not found. Creating synthetic data.")
                return self._create_synthetic_data()
            
            # Convert to list of dictionaries
            data = []
            for _, row in df.iterrows():
                sample = {
                    'prompt': str(row[prompt_col])
                }
                
                if reference_col and reference_col in df.columns:
                    sample['reference'] = str(row[reference_col])
                else:
                    # Generate a simple reference
                    sample['reference'] = self._generate_reference(sample['prompt'])
                
                data.append(sample)
            
            # Limit samples if specified
            if self.max_samples:
                data = data[:self.max_samples]
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load data from {self.data_path}: {e}. Creating synthetic data.")
            return self._create_synthetic_data()
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find a column with one of the possible names."""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def _create_synthetic_data(self) -> List[Dict[str, Any]]:
        """Create synthetic evaluation data."""
        synthetic_data = [
            {
                'prompt': 'Write a function to calculate the sum of two numbers',
                'reference': 'def add(a, b):\n    return a + b'
            },
            {
                'prompt': 'Write a function to multiply two numbers',
                'reference': 'def multiply(a, b):\n    return a * b'
            },
            {
                'prompt': 'Write a function to check if a number is even',
                'reference': 'def is_even(n):\n    return n % 2 == 0'
            },
            {
                'prompt': 'Write a function to get the length of a string',
                'reference': 'def get_length(s):\n    return len(s)'
            },
            {
                'prompt': 'Write a function to convert a string to uppercase',
                'reference': 'def to_uppercase(s):\n    return s.upper()'
            }
        ]
        
        # Limit samples if specified
        if self.max_samples:
            synthetic_data = synthetic_data[:self.max_samples]
        
        logger.info(f"Created {len(synthetic_data)} synthetic evaluation samples")
        return synthetic_data
    
    def _generate_reference(self, prompt: str) -> str:
        """Generate a simple reference for a prompt."""
        # Simple strategy: return a basic implementation
        return "def solution():\n    pass"
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """Get a sample by index."""
        return self.data[idx]
