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


def load_real_datasets(eval_dir: Path, feedback_dir: Path) -> List[Dict[str, Any]]:
    """Load real datasets from CSV files and human feedback JSON files."""
    import csv
    import json
    import glob
    import os

    print(f"Loading real datasets from eval_dir: {eval_dir}, feedback_dir: {feedback_dir}")

    samples = []

    # Load human feedback data
    feedback_data = {}
    json_pattern = str(feedback_dir / "*.json")
    json_files = glob.glob(json_pattern)
    print(f"Found {len(json_files)} JSON files in feedback_dir")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Process each question in the feedback
            for question_data in data.get('questions_df', []):
                question_id = question_data.get('ID')
                answer = question_data.get('Answer', '').strip()

                if question_id and answer:
                    key = f"{question_id}_{hash(answer) % 1000000}"  # Create unique key

                    # Get ratings (convert from -2/+2 scale to 0-1 scale)
                    consistent = (data.get('consistent_R', 0) + 2) / 4  # -2->0, +2->1
                    correct = (data.get('correct_R', 0) + 2) / 4
                    useful = (data.get('useful_R', 0) + 2) / 4

                    feedback_data[key] = {
                        'consistent': consistent,
                        'correct': correct,
                        'useful': useful
                    }
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    print(f"Loaded feedback for {len(feedback_data)} samples")

    # Load CSV data and match with feedback
    csv_pattern = str(eval_dir / "*.csv")
    csv_files = glob.glob(csv_pattern)
    print(f"Found {len(csv_files)} CSV files in eval_dir")

    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    question = row.get('intent') or row.get('Question', '').strip()
                    answer = row.get('snippet') or row.get('Answer', '').strip()
                    question_id = row.get('question_id') or row.get('ID')

                    if not question or not answer:
                        continue

                    # Create key to match with feedback
                    key = f"{question_id}_{hash(answer) % 1000000}"

                    if key in feedback_data:
                        feedback = feedback_data[key]

                        # Create sample with multi-head labels
                        sample = {
                            'question': question,
                            'answer': answer,
                            'labels': {
                                'consistent': feedback['consistent'],
                                'correct': feedback['correct'],
                                'useful': feedback['useful']
                            },
                            'metadata': {
                                'source_file': os.path.basename(csv_file),
                                'question_id': question_id,
                                'has_feedback': True
                            }
                        }
                        samples.append(sample)
                    else:
                        # No feedback available - use heuristic scoring
                        sample = {
                            'question': question,
                            'answer': answer,
                            'labels': {
                                'consistent': 0.5,  # Neutral
                                'correct': 0.5,     # Neutral
                                'useful': 0.5       # Neutral
                            },
                            'metadata': {
                                'source_file': os.path.basename(csv_file),
                                'question_id': question_id,
                                'has_feedback': False
                            }
                        }
                        samples.append(sample)

        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

    print(f"Loaded {len(samples)} samples from CSV files")
    return samples


def iter_feedback_samples(feedback_dir: Path) -> Iterator[Dict[str, Any]]:
    """Iterate over feedback samples from directory.

    Now uses real datasets with human feedback!
    """
    print("DEBUG: iter_feedback_samples function called!")
    print(f"DEBUG: feedback_dir parameter: {feedback_dir}")

    # Try to load real datasets first
    # feedback_dir is something like "evaluation_results_server"
    # We need to go up one level to find datasets_for_eval
    current_dir = Path.cwd()
    eval_dir = current_dir / "clasifNN" / "datasets_for_eval"
    feedback_dir_full = current_dir / "clasifNN" / str(feedback_dir).replace("clasifNN/", "")

    print(f"Looking for datasets in: {eval_dir}")
    print(f"Looking for feedback in: {feedback_dir_full}")
    print(f"Current dir: {current_dir}")
    print(f"eval_dir exists: {eval_dir.exists()}")
    print(f"feedback_dir_full exists: {feedback_dir_full.exists()}")

    if eval_dir.exists() and feedback_dir_full.exists():
        print("Both directories exist, attempting to load real datasets...")
        try:
            real_samples = load_real_datasets(eval_dir, feedback_dir_full)
            if real_samples:
                print(f"Using real datasets with {len(real_samples)} samples!")
                for sample in real_samples:
                    yield sample
                return
        except Exception as e:
            print(f"Error loading real datasets: {e}. Falling back to enhanced dummy data.")

    # Fallback to enhanced dummy data if real data loading fails
    print("Using enhanced dummy dataset...")
    # Good quality code examples
    good_questions = [
        "How to sort a list in Python?",
        "How to read a file in Python?",
        "How to calculate factorial recursively?",
        "How to use list comprehensions?",
        "How to work with dictionaries?",
        "How to write functions in Python?",
        "How to use classes in Python?",
        "How to handle file I/O properly?",
        "How to use loops in Python?",
        "How to format strings in Python?",
        "How to work with dates in Python?",
        "How to handle JSON data?",
        "How to create a simple web server?",
        "How to use regular expressions?",
        "How to work with NumPy arrays?",
        "How to read CSV files?",
        "How to create unit tests?",
        "How to use context managers?",
        "How to work with command line arguments?",
        "How to implement a binary search?"
    ]

    good_answers = [
        "sorted_list = sorted(my_list)",
        "with open('file.txt', 'r') as f: content = f.read()",
        "def factorial(n): return n * factorial(n-1) if n > 1 else 1",
        "squares = [x**2 for x in range(10)]",
        "my_dict = {'key': 'value'}; value = my_dict.get('key', 'default')",
        "def greet(name: str) -> str: return f'Hello {name}'",
        "class Calculator: def __init__(self): self.value = 0\n    def add(self, x): self.value += x",
        "with open('data.txt', 'w') as f: f.write('content')",
        "for i in range(5): print(f'Number: {i}')",
        "name = 'Alice'; message = f'Hello, {name}!'",
        "from datetime import datetime; now = datetime.now()",
        "import json; data = json.loads('{\"key\": \"value\"}')",
        "from http.server import HTTPServer, BaseHTTPRequestHandler; # Simple server setup",
        "import re; result = re.findall(r'\d+', 'abc123def456')",
        "import numpy as np; arr = np.array([1, 2, 3, 4, 5])",
        "import csv; with open('data.csv') as f: reader = csv.reader(f)",
        "import unittest; class TestMath(unittest.TestCase): pass",
        "with open('file.txt') as f: content = f.read().strip()",
        "import argparse; parser = argparse.ArgumentParser()",
        "def binary_search(arr, target): left, right = 0, len(arr)-1\n    while left <= right: mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1"
    ]

    # Poor quality code examples
    bad_questions = [
        "How to handle exceptions in Python?",
        "How to sort a list?",
        "How to read files?",
        "How to write functions?",
        "How to handle errors?",
        "How to work with data?",
        "How to create classes?",
        "How to use loops?",
        "How to format output?",
        "How to manage memory?",
        "How to handle network requests?",
        "How to process text?",
        "How to validate input?",
        "How to handle databases?",
        "How to implement algorithms?",
        "How to debug code?",
        "How to optimize performance?",
        "How to write documentation?",
        "How to handle concurrency?",
        "How to manage dependencies?"
    ]

    bad_answers = [
        "try: risky_code() except: pass  # Bare except is bad practice",
        "my_list.sort()  # Modifies original list unexpectedly",
        "f = open('file.txt'); content = f.read(); f.close()  # No context manager",
        "def func(): return 42  # No type hints or docstring",
        "if error: print('Error!')  # Poor error handling",
        "data = 'some data'  # No validation",
        "class BadClass: pass  # Empty class with no methods",
        "i = 0; while i < 10: print(i); i += 1  # Manual loop control",
        "print('Result: ' + str(result))  # Old-style string formatting",
        "big_list = [i for i in range(1000000)]  # Memory inefficient",
        "import requests; r = requests.get('url')  # No error handling",
        "text = 'hello'; processed = text.upper()  # Minimal processing",
        "user_input = input(); process(user_input)  # No input validation",
        "conn = sqlite3.connect('db.db'); cursor = conn.cursor()  # No cleanup",
        "def sort(arr): return sorted(arr)  # Unnecessarily wraps built-in",
        "print(variable)  # Debugging left in production code",
        "result = slow_function()  # No performance considerations",
        "# TODO: Add documentation  # Missing docstring",
        "import threading; t = threading.Thread(target=func); t.start()  # No join",
        "from some_lib import *  # Pollutes namespace"
    ]

    # Combine good and bad examples
    all_questions = good_questions + bad_questions
    all_answers = good_answers + bad_answers
    all_labels = [1] * len(good_answers) + [0] * len(bad_answers)  # 1 for good, 0 for bad

    # Add some medium quality examples to make it more realistic
    medium_questions = [
        "How to calculate sum of list?",
        "How to reverse a string?",
        "How to check if number is prime?",
        "How to merge dictionaries?",
        "How to remove duplicates from list?"
    ]

    medium_answers = [
        "total = 0; for num in numbers: total += num; return total  # Works but not Pythonic",
        "reversed_str = ''; for char in string: reversed_str = char + reversed_str  # Inefficient",
        "def is_prime(n): if n < 2: return False\n    for i in range(2, int(n**0.5)+1): \n        if n % i == 0: return False\n    return True  # Correct but could be optimized",
        "dict1.update(dict2); return dict1  # Modifies original dict",
        "seen = set(); result = []; for item in lst:\n    if item not in seen:\n        seen.add(item); result.append(item)  # Verbose but correct"
    ]

    all_questions.extend(medium_questions)
    all_answers.extend(medium_answers)
    all_labels.extend([0.5] * len(medium_answers))  # Medium quality

    # Shuffle to avoid patterns
    import random
    combined = list(zip(all_questions, all_answers, all_labels))
    random.shuffle(combined)
    all_questions, all_answers, all_labels = zip(*combined)

    for i, (q, a, label) in enumerate(zip(all_questions, all_answers, all_labels)):
        yield {
            'question': q,
            'answer': a,
            'label': label,
            'metadata': {
                'sample_id': i,
                'csv_path': str(feedback_dir / 'enhanced_dataset.csv'),
                'question_id': str(i),
                'quality_score': label
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
