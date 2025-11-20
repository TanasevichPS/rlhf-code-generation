"""
ClassifMLP - Neural Code Quality Classification Pipeline

Core MLP classifier with anti-overfitting techniques for code quality assessment.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

try:
    from .model import build_feature_vector, _ensure_tensor
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from clasifNN.model import build_feature_vector, _ensure_tensor


# =============================================================================
# ENHANCED MLP CLASSIFIER WITH FEATURES
# =============================================================================

class EnhancedClassifierWithFeatures(nn.Module):
    """Enhanced MLP classifier with code features and anti-overfitting techniques."""

    def __init__(self, embedding_dim: int, code_feature_dim: int = 74, hidden_dim: int = 512, dropout: float = 0.4):
        super().__init__()
        input_dim = embedding_dim * 4 + code_feature_dim  # embeddings + code features
        hidden_dim = max(hidden_dim, embedding_dim)
        mid_dim = max(hidden_dim // 2, embedding_dim // 2)

        # Deep MLP with regularization
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        logits = self.net(features)
        return logits.squeeze(-1)


# =============================================================================
# DUMMY EMBEDDING ENCODER
# =============================================================================

class DummyEmbeddingEncoder:
    """Deterministic embedding encoder for reproducible demos."""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self._cache: Dict[str, torch.Tensor] = {}

    def encode(self, texts):
        """Return deterministic embeddings derived from text hashes."""
        texts_list = list(texts) if not isinstance(texts, torch.Tensor) else [texts]

        if not texts_list:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)

        embeddings = []
        for text in texts_list:
            normalized = self._normalize_text(text)
            embeddings.append(self._get_embedding(normalized))

        return torch.stack(embeddings, dim=0)

    def _normalize_text(self, text: str) -> str:
        if text is None:
            return ""
        return " ".join(str(text).split()).lower()

    def _get_embedding(self, normalized_text: str) -> torch.Tensor:
        cached = self._cache.get(normalized_text)
        if cached is not None:
            return cached.clone()

        tensor = self._build_embedding(normalized_text)
        self._cache[normalized_text] = tensor
        return tensor.clone()

    def _build_embedding(self, normalized_text: str) -> torch.Tensor:
        import hashlib
        digest = hashlib.sha256(normalized_text.encode("utf-8")).digest()[:8]
        seed = int.from_bytes(digest, "little", signed=False)
        rng = np.random.default_rng(seed)
        values = rng.standard_normal(self.embedding_dim, dtype=np.float32)
        return torch.from_numpy(values)


# =============================================================================
# CODE FEATURE EXTRACTION
# =============================================================================

def extract_code_features(code: str) -> Dict[str, float]:
    """Extract 74-dimensional code features from code snippet."""
    features = {}

    # Basic text metrics
    features['code_length'] = len(code)
    features['num_lines'] = len(code.split('\n'))
    features['avg_line_length'] = len(code) / max(1, features['num_lines'])

    # Python keywords (25 features)
    python_keywords = {
        'import', 'from', 'def', 'class', 'if', 'for', 'while', 'try', 'except',
        'return', 'yield', 'lambda', 'and', 'or', 'not', 'in', 'is', 'None',
        'True', 'False', 'with', 'as', 'pass', 'break', 'continue', 'raise',
        'assert', 'global', 'nonlocal', 'del', 'await', 'async'
    }

    for keyword in python_keywords:
        count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', code))
        features[f'keyword_{keyword}'] = count

    features['total_keywords'] = sum(features[f'keyword_{k}'] for k in python_keywords)

    # Common modules (13 features)
    common_modules = {
        'os', 'sys', 're', 'json', 'math', 'datetime', 'collections', 'itertools',
        'numpy', 'pandas', 'torch', 'tensorflow', 'sklearn', 'matplotlib', 'PIL'
    }

    imported_modules = set()
    for module in common_modules:
        features[f'imports_{module}'] = 1.0 if module in code else 0.0

    # Syntax validity
    features['syntax_valid'] = 1.0 if check_syntax_validity(code) else 0.0

    # AST features (simplified - 20 features)
    ast_features = extract_basic_ast_features(code)
    features.update(ast_features)

    # Structural features (10 features)
    structural_features = extract_basic_structural_features(code)
    features.update(structural_features)

    return features


def check_syntax_validity(code: str) -> bool:
    """Check if Python code is syntactically valid."""
    try:
        compile(code, '<string>', 'exec')
        return True
    except (SyntaxError, IndentationError, TypeError):
        return False


def extract_basic_ast_features(code: str) -> Dict[str, float]:
    """Extract basic AST features."""
    features = {
        'ast_num_functions': 0, 'ast_num_classes': 0, 'ast_num_loops': 0,
        'ast_num_conditionals': 0, 'ast_max_nesting': 0, 'ast_num_assignments': 0,
        'ast_num_calls': 0, 'ast_num_returns': 0
    }

    try:
        import ast
        tree = ast.parse(code)

        class ASTVisitor(ast.NodeVisitor):
            def __init__(self):
                self.features = features.copy()
                self.nesting_level = 0
                self.max_nesting = 0

            def visit_FunctionDef(self, node):
                self.features['ast_num_functions'] += 1

            def visit_ClassDef(self, node):
                self.features['ast_num_classes'] += 1

            def visit_For(self, node):
                self.features['ast_num_loops'] += 1

            def visit_While(self, node):
                self.features['ast_num_loops'] += 1

            def visit_If(self, node):
                self.features['ast_num_conditionals'] += 1

            def visit_Assign(self, node):
                self.features['ast_num_assignments'] += 1

            def visit_Call(self, node):
                self.features['ast_num_calls'] += 1

            def visit_Return(self, node):
                self.features['ast_num_returns'] += 1

        visitor = ASTVisitor()
        visitor.visit(tree)
        features.update(visitor.features)

    except SyntaxError:
        pass

    return features


def extract_basic_structural_features(code: str) -> Dict[str, float]:
    """Extract basic structural features."""
    features = {}

    # Cyclomatic complexity (simplified)
    predicates = len(re.findall(r'\b(if|while|for|and|or|not)\b', code))
    features['cyclomatic_complexity'] = predicates + 1

    # Indentation statistics
    lines = code.split('\n')
    indent_levels = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            indent = len(line) - len(line.lstrip())
            indent_levels.append(indent)

    if indent_levels:
        features['max_indent'] = max(indent_levels) / 4
        features['avg_indent'] = sum(indent_levels) / len(indent_levels) / 4
        features['indent_variance'] = np.var(indent_levels) if len(indent_levels) > 1 else 0
    else:
        features['max_indent'] = 0
        features['avg_indent'] = 0
        features['indent_variance'] = 0

    # Comment ratio
    comment_lines = len([line for line in lines if line.strip().startswith('#')])
    code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
    features['comment_ratio'] = comment_lines / max(1, code_lines)

    # String and numeric literals
    import re
    string_literals = len(re.findall(r'["\'].*?["\']', code))
    numeric_literals = len(re.findall(r'\b\d+\.?\d*\b', code))
    features['num_string_literals'] = string_literals
    features['num_numeric_literals'] = numeric_literals

    return features


def build_feature_vector_with_code_features(
    question_emb: torch.Tensor,
    answer_emb: torch.Tensor,
    answer_code: str = None
) -> torch.Tensor:
    """Build combined features with code analysis."""
    base_features = build_feature_vector(question_emb, answer_emb)

    if answer_code is None:
        return base_features

    code_features = extract_code_features(answer_code)
    code_feature_values = list(code_features.values())
    code_features_tensor = torch.tensor(code_feature_values, dtype=torch.float32, device=base_features.device)

    combined_features = torch.cat([base_features, code_features_tensor.unsqueeze(0)], dim=-1)
    return combined_features


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class FeedbackClassificationDataset(Dataset):
    """Dataset for code quality classification."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'question': sample['question'],
            'answer': sample['answer'],
            'label': float(sample.get('label', 0)),
            'metadata': sample.get('metadata', {})
        }


def iter_feedback_samples(feedback_dir: Path) -> List[Dict[str, Any]]:
    """Iterate over feedback samples (dummy implementation)."""
    # Generate dummy samples for demonstration
    samples = []
    questions = [
        "How to sort a list in Python?",
        "How to read a file in Python?",
        "How to calculate factorial?",
        "How to handle exceptions?",
        "How to use list comprehensions?"
    ]

    answers = [
        "sorted_list = sorted(my_list)",
        "with open('file.txt') as f: content = f.read()",
        "def factorial(n): return n * factorial(n-1) if n > 0 else 1",
        "try: risky_operation() except Exception as e: print(e)",
        "squares = [x**2 for x in range(10)]"
    ]

    for i, (q, a) in enumerate(zip(questions, answers)):
        samples.append({
            'question': q,
            'answer': a,
            'label': 1 if i % 2 == 0 else 0,  # Alternate good/bad
            'metadata': {'sample_id': i}
        })

    return samples


class IntegratedTrainingPipeline:
    """Integrated training pipeline with anti-overfitting techniques."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        self.embedding_encoder = DummyEmbeddingEncoder()
        self.code_feature_dim = 74
        self.total_input_dim = 768 * 4 + self.code_feature_dim

        self.classifier = EnhancedClassifierWithFeatures(
            embedding_dim=768,
            code_feature_dim=self.code_feature_dim,
            hidden_dim=config.get('hidden_dim', 512),
            dropout=config.get('dropout', 0.4)
        )

        # Adjust input dimension if needed
        self.classifier.net[0] = nn.Linear(self.total_input_dim, self.classifier.net[0].out_features)
        self.classifier.to(self.device)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 20) -> Dict[str, Any]:
        """Train the MLP classifier with anti-overfitting techniques."""
        print("=== ClassifMLP Training ===")
        print(f"Device: {self.device}")
        print(f"Model: EnhancedClassifierWithFeatures")
        print(f"Input dim: {self.total_input_dim}, Hidden dim: {self.config.get('hidden_dim', 512)}")
        print(f"Dropout: {self.config.get('dropout', 0.4)}, Batch size: {self.config.get('batch_size', 16)}")

        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.config.get('learning_rate', 1e-5),
            weight_decay=self.config.get('weight_decay', 1e-3)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('patience', 5)
        history = []

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")

            # Training
            self.classifier.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for batch in tqdm(train_loader, desc="Training"):
                questions = batch['question']
                answers = batch['answer']
                labels = batch['label'].to(self.device)

                question_emb = self.embedding_encoder.encode(questions)
                answer_emb = self.embedding_encoder.encode(answers)

                extended_features = []
                for q, a in zip(questions, answers):
                    features = build_feature_vector_with_code_features(
                        question_emb[len(extended_features):len(extended_features)+1],
                        answer_emb[len(extended_features):len(extended_features)+1],
                        a
                    )
                    extended_features.append(features[0])

                extended_features = torch.stack(extended_features).to(self.device)

                optimizer.zero_grad()
                logits = self.classifier(extended_features)
                loss = criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * len(questions)
                predictions = (torch.sigmoid(logits) > 0.5).long()
                total_correct += (predictions == labels.long()).sum().item()
                total_samples += len(questions)

            avg_train_loss = total_loss / total_samples
            train_acc = total_correct / total_samples

            # Validation
            val_loss, val_acc = self.validate(val_loader, criterion)
            scheduler.step(val_loss)

            print(".4f"
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model("best_model.pt")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            })

        return {'history': history, 'best_val_loss': best_val_loss}

    def validate(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        """Validate the model."""
        self.classifier.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                questions = batch['question']
                answers = batch['answer']
                labels = batch['label'].to(self.device)

                question_emb = self.embedding_encoder.encode(questions)
                answer_emb = self.embedding_encoder.encode(answers)

                extended_features = []
                for q, a in zip(questions, answers):
                    features = build_feature_vector_with_code_features(
                        question_emb[len(extended_features):len(extended_features)+1],
                        answer_emb[len(extended_features):len(extended_features)+1],
                        a
                    )
                    extended_features.append(features[0])

                extended_features = torch.stack(extended_features).to(self.device)

                logits = self.classifier(extended_features)
                loss = criterion(logits, labels)

                total_loss += loss.item() * len(questions)
                predictions = (torch.sigmoid(logits) > 0.5).long()
                total_correct += (predictions == labels.long()).sum().item()
                total_samples += len(questions)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'classifier': self.classifier.state_dict(),
            'config': self.config
        }, path)
        print(f"Model saved: {path}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def collate_fn(batch):
    """Collate function for DataLoader."""
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    metadata = [item['metadata'] for item in batch]
    return {
        'question': questions,
        'answer': answers,
        'label': labels,
        'metadata': metadata
    }


def train_integrated_system(args: argparse.Namespace) -> None:
    """Main training function for ClassifMLP."""
    print("=== ClassifMLP Training ===")

    config = {
        'device': args.device,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': getattr(args, 'patience', 5),
    }

    feedback_dir = Path(args.feedback_dir)
    samples = iter_feedback_samples(feedback_dir)
    print(f"Loaded {len(samples)} samples")

    dataset = FeedbackClassificationDataset(samples)
    val_size = max(int(len(dataset) * getattr(args, 'val_ratio', 0.2)), 1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    pipeline = IntegratedTrainingPipeline(config)
    training_results = pipeline.train(train_loader, val_loader, num_epochs=args.epochs)

    # Save history
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "training_history.json"
    with history_path.open('w') as f:
        json.dump(training_results['history'], f, indent=2)

    print(f"Training completed! Results saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ClassifMLP - Neural Code Quality Classifier")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feedback-dir", type=str, default="evaluation_results_server")
    parser.add_argument("--output-dir", type=str, default="clasifNN/results")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--patience", type=int, default=5)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_integrated_system(args)
