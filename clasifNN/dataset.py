"""
Dataset functionality for ClassifMLP
"""

from pathlib import Path
from typing import Dict, List, Any, Iterator
import torch
from torch.utils.data import Dataset


class FeedbackClassificationDataset(Dataset):
    """Dataset for code quality classification feedback."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            'question': sample.get('question', ''),
            'answer': sample.get('answer', ''),
            'label': float(sample.get('label', 0)),
            'metadata': sample.get('metadata', {})
        }


def iter_feedback_samples(feedback_dir: Path) -> Iterator[Dict[str, Any]]:
    """Iterate over feedback samples from directory.

    This is a simplified implementation for demonstration.
    In practice, this would load real feedback data.
    """
    # Generate dummy samples for demonstration
    questions = [
        "How to sort a list in Python?",
        "How to read a file in Python?",
        "How to calculate factorial recursively?",
        "How to handle exceptions in Python?",
        "How to use list comprehensions?",
        "How to work with dictionaries?",
        "How to write functions in Python?",
        "How to use classes in Python?",
        "How to handle file I/O?",
        "How to use loops in Python?"
    ]

    answers = [
        "sorted_list = sorted(my_list)",
        "with open('file.txt', 'r') as f: content = f.read()",
        "def factorial(n): return n * factorial(n-1) if n > 1 else 1",
        "try: risky_code() except Exception as e: handle_error(e)",
        "squares = [x**2 for x in range(10)]",
        "my_dict = {'key': 'value'}; value = my_dict.get('key')",
        "def greet(name): return f'Hello {name}'",
        "class Calculator: def add(self, a, b): return a + b",
        "with open('data.txt', 'w') as f: f.write('content')",
        "for i in range(5): print(i)"
    ]

    labels = [1, 1, 1, 0, 1, 1, 1, 1, 1, 0]  # Mix of good/bad examples

    for i, (q, a, label) in enumerate(zip(questions, answers, labels)):
        yield {
            'question': q,
            'answer': a,
            'label': label,
            'metadata': {
                'sample_id': i,
                'csv_path': str(feedback_dir / 'dummy.csv'),
                'question_id': str(i)
            }
        }


def iter_multihead_samples(feedback_dir: Path) -> Iterator[Dict[str, Any]]:
    """Iterate over multi-head feedback samples.

    Simplified version for demonstration.
    """
    # Generate samples with multiple labels (consistent, correct, useful)
    base_samples = list(iter_feedback_samples(feedback_dir))

    for sample in base_samples:
        # Convert single label to multi-head labels
        base_label = sample['label']
        sample['labels'] = {
            'consistent': base_label,
            'correct': base_label if base_label > 0.5 else 0.0,
            'useful': base_label
        }
        yield sample
