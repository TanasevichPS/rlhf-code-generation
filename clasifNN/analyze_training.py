#!/usr/bin/env python3
"""
Simple Training Analysis Script for ClassifMLP

Analyzes training results and displays key metrics.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def analyze_training_history(history_path: str):
    """Analyze training history and display results."""
    print("=== ClassifMLP Training Analysis ===\n")

    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)

    print(f"Total epochs trained: {len(history)}")
    print(f"Early stopping: {'Yes' if len(history) < 20 else 'No'}")
    print()

    # Extract metrics
    epochs = [entry['epoch'] for entry in history]
    train_losses = [entry['train_loss'] for entry in history]
    val_losses = [entry['val_loss'] for entry in history]
    train_accs = [entry.get('train_acc', 0) for entry in history]
    val_accs = [entry.get('val_acc', 0) for entry in history]

    # Check for code quality metrics
    has_code_metrics = 'val_bertscore' in history[0]
    if has_code_metrics:
        bert_scores = [entry.get('val_bertscore', 0) for entry in history]
        code_bleu_scores = [entry.get('val_codebleu', 0) for entry in history]
        bleu_scores = [entry.get('val_bleu', 0) for entry in history]
        rouge_scores = [entry.get('val_rouge', 0) for entry in history]
        ruby_scores = [entry.get('val_ruby', 0) for entry in history]

    # Display summary
    print("Training Summary:")
    print(".4f")
    print(".4f")
    print(".3f")
    print(".3f")
    print()

    # Check for stability (no metric degradation)
    if len(val_losses) > 1:
        loss_trend = "Improving" if val_losses[-1] < val_losses[0] else "Degrading"
        print(f"Loss trend: {loss_trend}")

    if has_code_metrics and len(bert_scores) > 1:
        bert_trend = "Improving" if bert_scores[-1] >= bert_scores[0] else "Degrading"
        code_bleu_trend = "Improving" if code_bleu_scores[-1] >= code_bleu_scores[0] else "Degrading"
        print(f"BERTScore trend: {bert_trend}")
        print(f"CodeBLEU trend: {code_bleu_trend}")
        print()

        if bert_scores[-1] >= bert_scores[0] and code_bleu_scores[-1] >= code_bleu_scores[0]:
            print("SUCCESS: Anti-overfitting measures working!")
            print("   Code quality metrics are stable/improving")
        else:
            print("WARNING: Metrics still degrading - may need adjustment")
    else:
        print("INFO: Basic training only (no code quality metrics)")

    print()
    print("Detailed Results:")

    # Display table
    print(f"{'Epoch':<5} {'Train Loss':<10} {'Val Loss':<10} {'Train Acc':<10} {'Val Acc':<10}")
    print("-" * 55)

    for entry in history:
        epoch = entry['epoch']
        t_loss = entry['train_loss']
        v_loss = entry['val_loss']
        t_acc = entry.get('train_acc', 0)
        v_acc = entry.get('val_acc', 0)
        print("4.1f")

    if has_code_metrics:
        print()
        print("Code Quality Metrics:")
        print(f"{'Epoch':<5} {'BERTScore':<10} {'CodeBLEU':<10} {'BLEU':<10} {'ROUGE':<10} {'RUBY':<10}")
        print("-" * 60)

        for entry in history:
            epoch = entry['epoch']
            bert = entry.get('val_bertscore', 0)
            cbleu = entry.get('val_codebleu', 0)
            bleu = entry.get('val_bleu', 0)
            rouge = entry.get('val_rouge', 0)
            ruby = entry.get('val_ruby', 0)
            print("4.1f")

    # Simple plot if matplotlib available
    try:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_losses, 'b-', label='Train', marker='o')
        plt.plot(epochs, val_losses, 'r-', label='Validation', marker='s')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(epochs, train_accs, 'b-', label='Train', marker='o')
        plt.plot(epochs, val_accs, 'r-', label='Validation', marker='s')
        plt.title('Training & Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if has_code_metrics:
            plt.subplot(1, 3, 3)
            plt.plot(epochs, bert_scores, 'g-', label='BERTScore', marker='^')
            plt.plot(epochs, code_bleu_scores, 'm-', label='CodeBLEU', marker='v')
            plt.title('Code Quality Metrics')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_analysis.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved as 'training_analysis.png'")
    except ImportError:
        print("\nMatplotlib not available for plotting")

    print("\n=== Analysis Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Analyze ClassifMLP training results")
    parser.add_argument("--history-path", type=str, required=True,
                       help="Path to training history JSON file")

    args = parser.parse_args()

    if not Path(args.history_path).exists():
        print(f"❌ Error: History file not found: {args.history_path}")
        return

    analyze_training_history(args.history_path)


if __name__ == "__main__":
    main()
