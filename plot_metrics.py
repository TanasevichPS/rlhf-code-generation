#!/usr/bin/env python3
"""
Plot Metrics Script
===================

Script to visualize training metrics by epoch.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_results(results_path: str = "./rlhf_outputs/training_results.json"):
    """Load training results from JSON file."""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def plot_metrics_by_epoch(results: dict, save_path: str = "./rlhf_outputs/metrics_by_epoch.png"):
    """Plot metrics by epoch."""
    if 'epoch_metrics' not in results:
        print("No epoch metrics found in results")
        return
    
    epoch_metrics = results['epoch_metrics']
    if not epoch_metrics:
        print("No epoch metrics data available")
        return
    
    # Extract data
    epochs = list(range(1, len(epoch_metrics) + 1))
    metrics_names = ['bertscore', 'codebleu', 'bleu', 'rouge', 'ruby']
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Metrics by Epoch', fontsize=16)
    
    # Plot each metric
    for i, metric in enumerate(metrics_names):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        values = [epoch_metrics[epoch-1].get(metric, 0) for epoch in epochs]
        
        ax.plot(epochs, values, 'o-', linewidth=2, markersize=6)
        ax.set_title(f'{metric.upper()}', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add target line if available
        target_value = results.get('config', {}).get(f'target_{metric}', None)
        if target_value:
            ax.axhline(y=target_value, color='red', linestyle='--', alpha=0.7, label=f'Target: {target_value}')
            ax.legend()
    
    # Remove empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics plot saved to: {save_path}")
    
    # Also create a combined plot
    plt.figure(figsize=(12, 8))
    
    for metric in metrics_names:
        values = [epoch_metrics[epoch-1].get(metric, 0) for epoch in epochs]
        plt.plot(epochs, values, 'o-', linewidth=2, markersize=6, label=metric.upper())
    
    plt.title('All Metrics by Epoch', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    combined_save_path = "./rlhf_outputs/all_metrics_by_epoch.png"
    plt.savefig(combined_save_path, dpi=300, bbox_inches='tight')
    print(f"Combined metrics plot saved to: {combined_save_path}")
    
    plt.show()

def print_metrics_summary(results: dict):
    """Print a summary of metrics."""
    if 'epoch_metrics' not in results:
        print("No epoch metrics found")
        return
    
    epoch_metrics = results['epoch_metrics']
    if not epoch_metrics:
        print("No epoch metrics data available")
        return
    
    print("\n📊 METRICS SUMMARY:")
    print("=" * 50)
    
    metrics_names = ['bertscore', 'codebleu', 'bleu', 'rouge', 'ruby']
    
    for metric in metrics_names:
        values = [epoch_metrics[epoch-1].get(metric, 0) for epoch in range(1, len(epoch_metrics) + 1)]
        
        if values:
            initial = values[0]
            final = values[-1]
            improvement = final - initial
            best = max(values)
            
            print(f"\n{metric.upper()}:")
            print(f"  Initial: {initial:.4f}")
            print(f"  Final:   {final:.4f}")
            print(f"  Best:    {best:.4f}")
            print(f"  Improvement: {improvement:+.4f}")
            
            # Check if target was met
            target = results.get('config', {}).get(f'target_{metric}', None)
            if target:
                target_met = final >= target
                status = "✅" if target_met else "❌"
                print(f"  Target:  {target:.4f} {status}")

def main():
    """Main function."""
    print("📊 RLHF Training Metrics Visualization")
    print("=" * 50)
    
    # Load results
    results = load_training_results()
    if not results:
        return
    
    # Print summary
    print_metrics_summary(results)
    
    # Plot metrics
    try:
        plot_metrics_by_epoch(results)
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error plotting metrics: {e}")

if __name__ == "__main__":
    main()
