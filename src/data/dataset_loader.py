from typing import List, Dict, Any
import pandas as pd
import os
from glob import glob
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)

class CodeDatasetLoader:
    """Improved dataset loader for code generation tasks."""
    
    def __init__(self, config):
        self.config = config
    
    def load_dataset(self) -> Dataset:
        """Load dataset from CSV files."""
        dataset_path = self.config.dataset_path
        csv_files = glob(os.path.join(dataset_path, "*.csv"))
        
        if not csv_files:
            logger.warning("No CSV files found, using synthetic dataset")
            return self._load_synthetic_dataset()
        
        all_prompts = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"Loaded dataset: {os.path.basename(csv_file)} with {len(df)} rows")
                
                # Find prompt column
                prompt_column = None
                for col in ['Question', 'Prompt', 'prompt', 'instruction', 'input', 'text']:
                    if col in df.columns:
                        prompt_column = col
                        break
                
                if prompt_column is None:
                    prompt_column = df.columns[0]
                
                prompts = df[prompt_column].dropna().astype(str).tolist()
                all_prompts.extend(prompts)
                
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                continue
        
        if not all_prompts:
            raise ValueError("No valid prompts found in any CSV files")
        
        dataset = Dataset.from_dict({"prompt": all_prompts})
        logger.info(f"Successfully loaded {len(all_prompts)} prompts")
        
        return dataset
    
    def _load_synthetic_dataset(self) -> Dataset:
        """Create synthetic dataset as fallback."""
        synthetic_prompts = [
            "Write a Python function to calculate factorial",
            "Create a function to reverse a string in Python",
            "Write code to read a CSV file and print its contents",
            "Create a Python class for a simple calculator",
            "Write a function to check if a number is prime",
            "Create code to download a file from URL using requests",
            "Write a Python script to parse JSON data",
            "Create a function to sort a list of dictionaries by key"
        ]
        
        return Dataset.from_dict({"prompt": synthetic_prompts})