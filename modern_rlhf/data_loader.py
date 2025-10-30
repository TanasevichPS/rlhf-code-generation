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
from datasets import load_dataset  # Added import for Hugging Face datasets
import sys
from contextlib import contextmanager
import tempfile
from typing import cast
try:
    from huggingface_hub import hf_hub_download, list_repo_files
    _HF_HUB_AVAILABLE = True
except Exception:
    _HF_HUB_AVAILABLE = False

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

    @contextmanager
    def _no_local_dataset_scripts(self):
        """Temporarily remove project paths from sys.path to avoid picking local conala.py."""
        project_root = Path(__file__).resolve().parents[2]  # repo root
        removed = []
        original_sys_path = list(sys.path)
        for p in list(sys.path):
            try:
                pr = Path(p).resolve()
                if project_root in pr.parents or pr == project_root or p in ("", "."):
                    sys.path.remove(p)
                    removed.append(p)
            except Exception:
                # Non-pathy entries, ignore
                if p in ("", "."):
                    try:
                        sys.path.remove(p)
                        removed.append(p)
                    except Exception:
                        pass
        try:
            yield
        finally:
            # Restore original sys.path order
            sys.path[:] = original_sys_path

    @contextmanager
    def _temp_cwd(self):
        """Temporarily change working directory to a safe temp folder."""
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                yield
            finally:
                os.chdir(old_cwd)

    def _load_conala_split(self, split: str):
        """Load CoNaLa curated split directly from Hugging Face Hub without dataset scripts.

        Strategy:
        1) Try discovering curated parquet files via huggingface_hub and load with pandas.
        2) If that fails, try datasets parquet builder with explicit URLs.
        3) Finally, try repo id APIs.
        """
        # 0) Prefer local corpus if provided
        local = self._load_conala_local(split)
        if local is not None:
            return local
        # 1) Discover and load curated parquet(s) via Hub API
        if _HF_HUB_AVAILABLE:
            try:
                files = list_repo_files(repo_id="neulab/conala", repo_type="dataset")
                # Prefer curated parquet paths containing the split name
                candidate_paths = [
                    f for f in files
                    if f.lower().endswith('.parquet') and (
                        ('/curated/' in f.replace('\\', '/')) or ('curated' in f)
                    ) and (f"/{split}" in f.replace('\\', '/') or f"{split}-" in f or f"{split}.parquet" in f)
                ]
                # If nothing found under curated, fall back to any parquet with split in name
                if not candidate_paths:
                    candidate_paths = [
                        f for f in files
                        if f.lower().endswith('.parquet') and (f"/{split}" in f.replace('\\', '/') or f"{split}-" in f)
                    ]
                if candidate_paths:
                    dfs = []
                    for rel_path in candidate_paths:
                        try:
                            local_path = hf_hub_download(repo_id="neulab/conala", filename=rel_path, repo_type="dataset")
                            dfs.append(pd.read_parquet(local_path))
                        except Exception as e_dl:
                            logger.warning(f"Failed to download/read parquet {rel_path}: {e_dl}")
                    if dfs:
                        df = pd.concat(dfs, ignore_index=True)
                        return df.to_dict(orient='records')
            except Exception as e:
                logger.warning(f"hf_hub listing/parquet load failed: {e}")

        # 2) Datasets parquet builder with explicit URL(s)
        with self._no_local_dataset_scripts(), self._temp_cwd():
            try:
                url = f"https://huggingface.co/datasets/neulab/conala/resolve/main/curated/{split}-00000-of-00001.parquet"
                ds = load_dataset("parquet", data_files={split: [url]}, split=split)
                return ds
            except Exception as e1:
                logger.warning(f"datasets parquet builder failed: {e1}")
                try:
                    # Прямой доступ к подконфигурации на HF
                    return load_dataset("hf://datasets/neulab/conala/curated", split=split)
                except Exception as e2:
                    logger.warning(f"Direct curated load failed: {e2}")
                    try:
                        # Резерв: стандартный ID
                        return load_dataset("neulab/conala", "curated", split=split)
                    except Exception as e3:
                        logger.error(f"All loading methods failed for split '{split}': {e3}")
                        raise

    def _load_conala_local(self, split: str) -> Optional[List[Dict[str, Any]]]:
        """Load CoNaLa curated split from a local corpus directory if available.

        Supports common file names and formats in the official corpus:
        - conala-train.json / conala-test.json (JSON array or JSONL)
        - curated_train.json / curated_test.json
        - train.json / test.json
        """
        root = getattr(self.data_config, 'conala_local_path', None)
        if not root:
            return None
        corpus_dir = Path(root)
        if not corpus_dir.exists():
            logger.warning(f"Conala local path not found: {corpus_dir}")
            return None

        candidates = [
            f"conala-{split}.json",
            f"conala_{split}.json",
            f"curated_{split}.json",
            f"{split}.json",
            f"conala-{split}.jsonl",
            f"conala_{split}.jsonl",
            f"curated_{split}.jsonl",
            f"{split}.jsonl",
            # nested under 'conala-corpus' subdir if user points to parent
            f"conala-corpus/conala-{split}.json",
            f"conala-corpus/conala_{split}.json",
        ]

        file_path = None
        for name in candidates:
            p = corpus_dir / name
            if p.exists():
                file_path = p
                break

        if file_path is None:
            # Try to locate any json/jsonl mentioning split in name
            for p in corpus_dir.rglob("*.json*"):
                if split in p.name.lower() and ("train" in p.name.lower() or "test" in p.name.lower()):
                    file_path = p
                    break

        if file_path is None:
            logger.warning(f"No local CoNaLa file found for split '{split}' in {corpus_dir}")
            return None

        logger.info(f"Loading local CoNaLa {split} from: {file_path}")

        records: List[Dict[str, Any]] = []
        try:
            if file_path.suffix == '.jsonl' or file_path.suffixes[-1] == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        records.append(json.loads(line))
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                    if isinstance(obj, dict) and split in obj:
                        records = obj[split]
                    elif isinstance(obj, list):
                        records = obj
                    else:
                        # Some dumps store under keys like 'data'
                        for key in ['data', 'examples', 'items']:
                            if isinstance(obj, dict) and key in obj and isinstance(obj[key], list):
                                records = obj[key]
                                break
            # Normalize to expected fields
            normalized = []
            for item in records:
                prompt = item.get('rewritten_intent') or item.get('intent') or item.get('question') or ""
                response = item.get('snippet') or item.get('code') or item.get('answer') or ""
                qid = item.get('question_id') or item.get('id')
                normalized.append({
                    'prompt': prompt,
                    'response': response,
                    'reference': response,
                    'rating': None,
                    'metadata': {'source': f'conala_{split}_local', 'question_id': qid}
                })
            return normalized
        except Exception as e:
            logger.error(f"Failed to load local CoNaLa {split} from {file_path}: {e}")
            return None

    def load_training_data(self) -> List[Dict[str, Any]]:
        """Load training data from Hugging Face CoNaLa dataset."""
        logger.info("Loading training data from Hugging Face: neulab/conala (train split)...")
        
        # Load curated dataset directly from HF (avoid local conala.py)
        dataset = self._load_conala_split('train')
        
        samples = []
        for item in dataset:
            # Robust field access for local/HF variants
            if isinstance(item, dict):
                prompt = item.get('rewritten_intent') or item.get('intent') or item.get('question') or ""
                response = item.get('snippet') or item.get('code') or item.get('answer') or ""
                qid = item.get('question_id') or item.get('id')
            else:
                # datasets arrow row
                try:
                    prompt = item['rewritten_intent'] if item['rewritten_intent'] else item['intent']
                except Exception:
                    prompt = item.get('intent', "")  # type: ignore[attr-defined]
                response = item.get('snippet', "")  # type: ignore[attr-defined]
                qid = item.get('question_id', None)  # type: ignore[attr-defined]

            samples.append({
                'prompt': str(prompt),
                'response': str(response),  # snippet is the code to generate
                'reference': str(response),  # Use snippet as reference for supervised fine-tuning
                'rating': None,
                'metadata': {'source': 'conala_train', 'question_id': qid}
            })
        
        # Filter and clean data
        filtered_samples = self._filter_samples(samples, allow_empty_response=False)

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
        
        logger.info(f"Total training samples loaded from CoNaLa: {len(filtered_samples)}")
        
        return filtered_samples
    
    def load_evaluation_data(self) -> List[Dict[str, Any]]:
        """Load evaluation data from Hugging Face CoNaLa dataset."""
        logger.info("Loading evaluation data from Hugging Face: neulab/conala (test split)...")
        
        # Load curated dataset directly from HF (avoid local conala.py)
        dataset = self._load_conala_split('test')
        
        samples = []
        for item in dataset:
            if isinstance(item, dict):
                prompt = item.get('rewritten_intent') or item.get('intent') or item.get('question') or ""
                snippet = item.get('snippet') or item.get('code') or item.get('answer') or ""
                qid = item.get('question_id') or item.get('id')
            else:
                try:
                    prompt = item['rewritten_intent'] if item['rewritten_intent'] else item['intent']
                except Exception:
                    prompt = item.get('intent', "")  # type: ignore[attr-defined]
                snippet = item.get('snippet', "")  # type: ignore[attr-defined]
                qid = item.get('question_id', None)  # type: ignore[attr-defined]

            samples.append({
                'prompt': str(prompt),
                'response': "",  # model will generate; keep empty to avoid leakage
                'reference': str(snippet),  # snippet is the gold code
                'rating': None,
                'metadata': {'source': 'conala_test', 'question_id': qid}
            })
        
        # Filter and clean data (allow empty responses for eval)
        filtered_samples = self._filter_samples(samples, allow_empty_response=True)

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

        logger.info(f"Total evaluation samples loaded from CoNaLa: {len(filtered_samples)}")

        logger.info(f"Total evaluation samples loaded: {len(filtered_samples)}")
        logger.info(f"Total evaluation samples loaded: {len(filtered_samples)}")
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

    def _filter_samples(self, samples: List[Dict[str, Any]], allow_empty_response: bool = False) -> List[Dict[str, Any]]:
        """Filter and clean samples based on criteria.
        If allow_empty_response is True, do not enforce min response length and allow empty responses (for eval).
        """

    def _filter_samples(self, samples: List[DataSample]) -> List[DataSample]:
        """Filter and clean samples based on criteria."""

    def _filter_samples(self, samples: List[DataSample]) -> List[DataSample]:
        """Filter and clean samples based on criteria."""


    def _filter_samples(self, samples: List[DataSample]) -> List[DataSample]:
        """Filter and clean samples based on criteria."""
        filtered_samples = []
        
        for sample in samples:
            if len(sample['prompt']) < self.data_config.min_prompt_length:
                continue
            if len(sample['prompt']) > self.data_config.max_prompt_length:
                continue
            if not allow_empty_response:
                if len(sample['response']) < self.data_config.min_response_length:
                    continue
            if len(sample['response']) > self.data_config.max_response_length:
                continue
            
            # Check for empty or invalid content
            if not sample['prompt'].strip():
                continue
            if not allow_empty_response and not sample['response'].strip():
                continue
            
            # Check for code-like content (basic heuristic)
            if allow_empty_response:
                # For evaluation, accept as long as prompt/reference exist
                filtered_samples.append(sample)
            else:
                if self._is_code_like(sample['prompt']) or self._is_code_like(sample['response']):
                    filtered_samples.append(sample)

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
