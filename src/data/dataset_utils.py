from typing import Tuple, List, Dict, Any
import sys
from datasets import Dataset, load_dataset
import logging
import re
import pandas as pd
import os
from glob import glob

logger = logging.getLogger(__name__)

class CodeDatasetLoader:
    """Improved dataset loader for code generation tasks."""
    
    def __init__(self, config) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate dataset configuration."""
        required_fields = ['dataset_name', 'max_prompt_length']
        for field in required_fields:
            if not hasattr(self.config, field):
                raise ValueError(f"Config missing required field: {field}")

    def _load_custom_eval_dataset(self) -> Dataset:
        """Load custom evaluation dataset from CSV files."""
        try:
            dataset_path = self.config.dataset_path
            csv_files = glob(os.path.join(dataset_path, "*.csv"))
            
            if not csv_files:
                logger.warning("No CSV files found, using synthetic dataset")
                return self._load_synthetic_code_dataset()
            
            all_prompts = []
            all_codes = []
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    logger.info(f"Loaded dataset: {os.path.basename(csv_file)} with {len(df)} rows")
                    
                    # Extract prompts from various possible columns
                    prompt_column = None
                    for col in ['Question', 'Prompt', 'prompt', 'instruction', 'input']:
                        if col in df.columns:
                            prompt_column = col
                            break
                    
                    if prompt_column is None:
                        logger.warning(f"No prompt column found in {csv_file}, using first column")
                        prompt_column = df.columns[0]
                    
                    prompts = df[prompt_column].dropna().astype(str).tolist()
                    all_prompts.extend(prompts)
                    
                    # Use empty strings for code as we're generating it
                    all_codes.extend([""] * len(prompts))
                    
                except Exception as e:
                    logger.error(f"Error loading {csv_file}: {e}")
                    continue
            
            if not all_prompts:
                raise ValueError("No valid prompts found in any CSV files")
                
            dataset = Dataset.from_dict({
                "prompt": all_prompts,
                "code": all_codes
            })
            
            logger.info(f"Successfully loaded {len(all_prompts)} prompts from {len(csv_files)} files")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load custom eval dataset: {e}")
            return self._load_synthetic_code_dataset()

    def _load_synthetic_code_dataset(self) -> Dataset:
        """Create synthetic code generation prompts optimized for the dataset style."""
        synthetic_prompts = [
            "Write Python code to send a signal to the current process",
            "How to decode a hex string to UTF-8 in Python?",
            "Remove None values from a dictionary in Python",
            "Capture output of system commands using subprocess",
            "Find intersection between two pandas Series",
            "Send HTTP headers to a client",
            "Format datetime string to extract date only",
            "Split multi-line string into separate lines",
            "Concatenate list elements with a colon",
            "Get first object from Django model queryset",
            "Calculate sum of 2D numpy array rows",
            "Run Python script with arguments using subprocess",
            "Parse time string with milliseconds",
            "Convert string with commas to float",
            "Set Python path in script",
            "Split string using regex pattern",
            "Open file in append mode",
            "Download file from URL and save locally"
        ]
        
        return Dataset.from_dict({
            "prompt": synthetic_prompts,
            "code": [""] * len(synthetic_prompts)
        })

    def load_dataset(self) -> Dataset:
        """Main method to load and prepare the dataset."""
        logger.info(f"Loading code dataset: {self.config.dataset_name}")
        
        try:
            if self.config.dataset_name == "code_search_net":
                dataset = self._load_code_search_net()
            elif self.config.dataset_name == "synthetic_code":
                dataset = self._load_synthetic_code_dataset()
            elif self.config.dataset_name == "custom_code":
                dataset = self._load_custom_eval_dataset()
            else:
                dataset = self._load_custom_dataset()
            
            return self._format_code_dataset(dataset)
        except Exception as e:
            logger.error(f"Failed to load code dataset: {e}")
            raise

    def _format_code_dataset(self, dataset: Dataset) -> Dataset:
        """Format dataset for code generation training."""
        def format_code_prompts(batch: Dict) -> Dict:
            """Format batch of code prompts."""
            prompts = []
            for prompt in batch["prompt"]:
                prompt = str(prompt).strip()
                
                # Clean and standardize prompts
                if prompt.startswith('"') and prompt.endswith('"'):
                    prompt = prompt[1:-1]
                
                # Ensure prompt is properly formatted
                if not prompt.endswith((".", "?", "!")):
                    prompt += "."
                
                # Add Python context if missing
                if not any(keyword in prompt.lower() for keyword in 
                          ["python", "code", "function", "def ", "import"]):
                    prompt = "Write Python code to " + prompt.lower()
                
                prompts.append(prompt)
            
            return {"prompt": prompts}
        
        return dataset.map(format_code_prompts, batched=True)