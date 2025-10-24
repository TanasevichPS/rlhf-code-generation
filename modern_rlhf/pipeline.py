"""
Modern RLHF Pipeline
===================

A complete, modern RLHF pipeline for code generation with:
- Data loading and preprocessing
- Reward model training
- PPO/DPO training
- Comprehensive evaluation
- Results visualization
"""

import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import os
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

from .config import ModernRLHFConfig, get_research_config, get_production_config, get_fast_config
from .reward_model import ModernRewardModel, RewardModelTrainer
from .trainer import ModernRLHFTrainer
from .metrics import ModernMetricsEvaluator
from .data_loader import ModernDataLoader

logger = logging.getLogger(__name__)


@dataclass
class PipelineResults:
    """Container for pipeline results."""
    config: ModernRLHFConfig
    reward_model_metrics: Dict[str, float]
    training_metrics: Dict[str, float]
    evaluation_metrics: Dict[str, float]
    final_metrics: Dict[str, float]
    training_time: float
    total_time: float
    success: bool
    error_message: Optional[str] = None


class ModernRLHFPipeline:
    """Main RLHF pipeline class."""
    
    def __init__(self, config: Optional[ModernRLHFConfig] = None):
        self.config = config or get_research_config()
        self.device = torch.device(self.config.hardware.device)
        
        # Initialize components
        self.data_loader = ModernDataLoader(self.config)
        self.metrics_evaluator = ModernMetricsEvaluator()
        
        # Training components (initialized later)
        self.reward_model = None
        self.reward_trainer = None
        self.rlhf_trainer = None
        
        # Results
        self.results = None
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Initialized Modern RLHF Pipeline with config: {self.config.experiment_name}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.config.debug else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.config.data.output_path, 'pipeline.log'))
            ]
        )
    
    def load_data(self) -> Tuple[Any, Any, Any]:
        """Load training and evaluation data."""
        logger.info("Loading data...")
        
        # Load training data
        train_data = self.data_loader.load_training_data()
        
        # Load evaluation data
        eval_data = self.data_loader.load_evaluation_data()
        
        # Load human feedback data
        human_feedback = self.data_loader.load_human_feedback()
        
        logger.info(f"Loaded {len(train_data)} training samples, {len(eval_data)} eval samples")
        
        return train_data, eval_data, human_feedback
    
    def prepare_reward_model(self, train_data: Any, human_feedback: Any) -> ModernRewardModel:
        """Prepare and train the reward model."""
        logger.info("Preparing reward model...")
        
        # Initialize reward model
        self.reward_model = ModernRewardModel(
            self.config.reward,
            self.config.model.reward_model_name
        )
        
        # Load human feedback if available
        if human_feedback:
            self.reward_model.load_human_feedback(human_feedback)
        
        # Initialize reward trainer
        self.reward_trainer = RewardModelTrainer(self.reward_model, self.config.reward)
        
        # Train reward model if needed
        if self.config.reward.reward_epochs > 0:
            logger.info("Training reward model...")
            self._train_reward_model(train_data)
        
        return self.reward_model
    
    def _train_reward_model(self, train_data: Any):
        """Train the reward model."""
        # Convert data to training format
        train_batches = self._prepare_reward_training_batches(train_data)
        
        # Training loop
        for epoch in range(self.config.reward.reward_epochs):
            epoch_metrics = []
            
            for batch in tqdm(train_batches, desc=f"Reward Training Epoch {epoch}"):
                metrics = self.reward_trainer.train_step(batch)
                epoch_metrics.append(metrics)
            
            # Average metrics
            avg_metrics = {}
            for key in epoch_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
            
            logger.info(f"Reward Model Epoch {epoch}: {avg_metrics}")
        
        # Save reward model
        reward_model_path = os.path.join(self.config.data.output_path, "reward_model")
        self.reward_model.save_model(reward_model_path)
        logger.info(f"Reward model saved to {reward_model_path}")
    
    def _prepare_reward_training_batches(self, train_data: Any) -> List[Dict[str, Any]]:
        """Prepare batches for reward model training."""
        batches = []
        batch_size = self.config.reward.reward_batch_size
        
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i + batch_size]
            
            batch = {
                'prompts': [item['prompt'] for item in batch_data],
                'responses': [item['response'] for item in batch_data],
                'human_ratings': [item.get('rating', None) for item in batch_data]
            }
            
            batches.append(batch)
        
        return batches
    
    def prepare_rlhf_trainer(self) -> ModernRLHFTrainer:
        """Prepare the RLHF trainer."""
        logger.info("Preparing RLHF trainer...")
        
        if self.reward_model is None:
            raise ValueError("Reward model must be prepared before RLHF trainer")
        
        # Initialize RLHF trainer
        self.rlhf_trainer = ModernRLHFTrainer(self.config, self.reward_model)
        
        return self.rlhf_trainer
    
    def train_rlhf(self, train_data: Any, eval_data: Any) -> Dict[str, float]:
        """Train the RLHF model."""
        logger.info("Starting RLHF training...")
        
        if self.rlhf_trainer is None:
            raise ValueError("RLHF trainer must be prepared before training")
        
        # Prepare data loaders
        train_dataloader = self._prepare_rlhf_dataloader(train_data, is_training=True)
        eval_dataloader = self._prepare_rlhf_dataloader(eval_data, is_training=False)
        
        # Train
        training_metrics = self.rlhf_trainer.train(train_dataloader, eval_dataloader)
        
        logger.info(f"RLHF training completed. Final metrics: {training_metrics}")
        
        return training_metrics
    
    def _prepare_rlhf_dataloader(self, data: Any, is_training: bool = True) -> List[Dict[str, Any]]:
        """Prepare data loader for RLHF training."""
        dataloader = []
        batch_size = self.config.training.batch_size
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            
            if is_training:
                # For training, we need prompt-response pairs
                batch = {
                    'prompts': [item['prompt'] for item in batch_data],
                    'responses': [item.get('response', '') for item in batch_data]
                }
            else:
                # For evaluation, we need prompts and references
                batch = {
                    'prompts': [item['prompt'] for item in batch_data],
                    'references': [item.get('reference', '') for item in batch_data]
                }
            
            dataloader.append(batch)
        
        return dataloader
    
    def evaluate_model(self, eval_data: Any) -> Dict[str, float]:
        """Evaluate the trained model."""
        logger.info("Evaluating model...")
        
        if self.rlhf_trainer is None:
            raise ValueError("RLHF trainer must be prepared before evaluation")
        
        # Generate responses
        all_prompts = [item['prompt'] for item in eval_data]
        all_references = [item.get('reference', '') for item in eval_data]
        
        # Generate responses in batches
        all_responses = []
        batch_size = self.config.evaluation.eval_batch_size
        
        for i in tqdm(range(0, len(all_prompts), batch_size), desc="Generating responses"):
            batch_prompts = all_prompts[i:i + batch_size]
            
            # Generate responses
            generation_output = self.rlhf_trainer.trainer.generate_responses(batch_prompts)
            batch_responses = generation_output['response_texts']
            
            all_responses.extend(batch_responses)
        
        # Compute metrics
        metrics_results = self.metrics_evaluator.compute_all_metrics(all_responses, all_references)
        
        # Convert to simple dict
        evaluation_metrics = {}
        for metric_name, result in metrics_results.items():
            evaluation_metrics[metric_name] = result.score
        
        # Check against targets
        targets = {
            'bertscore': self.config.evaluation.target_bertscore,
            'codebleu': self.config.evaluation.target_codebleu,
            'bleu': self.config.evaluation.target_bleu,
            'rouge': self.config.evaluation.target_rouge,
            'ruby': self.config.evaluation.target_ruby
        }
        
        target_results = self.metrics_evaluator.evaluate_against_targets(metrics_results, targets)
        evaluation_metrics['targets_met'] = target_results
        
        logger.info(f"Evaluation completed. Metrics: {evaluation_metrics}")
        
        return evaluation_metrics
    
    def run_full_pipeline(self) -> PipelineResults:
        """Run the complete RLHF pipeline."""
        start_time = time.time()
        
        try:
            logger.info("Starting full RLHF pipeline...")
            
            # Step 1: Load data
            train_data, eval_data, human_feedback = self.load_data()
            
            # Step 2: Prepare reward model
            reward_model_start = time.time()
            self.prepare_reward_model(train_data, human_feedback)
            reward_model_time = time.time() - reward_model_start
            
            # Step 3: Prepare RLHF trainer
            self.prepare_rlhf_trainer()
            
            # Step 4: Train RLHF model
            training_start = time.time()
            training_metrics = self.train_rlhf(train_data, eval_data)
            training_time = time.time() - training_start
            
            # Step 5: Evaluate model
            evaluation_start = time.time()
            evaluation_metrics = self.evaluate_model(eval_data)
            evaluation_time = time.time() - evaluation_start
            
            # Step 6: Compute final metrics
            final_metrics = self._compute_final_metrics(evaluation_metrics)
            
            # Create results
            total_time = time.time() - start_time
            
            self.results = PipelineResults(
                config=self.config,
                reward_model_metrics={'training_time': reward_model_time},
                training_metrics=training_metrics,
                evaluation_metrics=evaluation_metrics,
                final_metrics=final_metrics,
                training_time=training_time,
                total_time=total_time,
                success=True
            )
            
            # Save results
            self._save_results()
            
            logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            
            self.results = PipelineResults(
                config=self.config,
                reward_model_metrics={},
                training_metrics={},
                evaluation_metrics={},
                final_metrics={},
                training_time=0.0,
                total_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
            
            return self.results
    
    def _compute_final_metrics(self, evaluation_metrics: Dict[str, float]) -> Dict[str, float]:
        """Compute final success metrics."""
        final_metrics = {}
        
        # Check if targets are met
        targets_met = evaluation_metrics.get('targets_met', {})
        final_metrics['all_targets_met'] = all(targets_met.values())
        final_metrics['targets_met_count'] = sum(targets_met.values())
        final_metrics['targets_total'] = len(targets_met)
        
        # Overall success score
        if 'targets_met' in evaluation_metrics:
            success_score = sum(targets_met.values()) / len(targets_met)
            final_metrics['success_score'] = success_score
        else:
            final_metrics['success_score'] = 0.0
        
        return final_metrics
    
    def _save_results(self):
        """Save pipeline results."""
        if self.results is None:
            return
        
        # Save results to JSON
        results_path = os.path.join(self.config.data.output_path, 'pipeline_results.json')
        
        results_dict = {
            'config': self.results.config.to_dict(),
            'reward_model_metrics': self.results.reward_model_metrics,
            'training_metrics': self.results.training_metrics,
            'evaluation_metrics': self.results.evaluation_metrics,
            'final_metrics': self.results.final_metrics,
            'training_time': self.results.training_time,
            'total_time': self.results.total_time,
            'success': self.results.success,
            'error_message': self.results.error_message,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save configuration
        config_path = os.path.join(self.config.data.output_path, 'config.json')
        self.config.save(config_path)
        
        logger.info(f"Results saved to {results_path}")
    
    def visualize_results(self):
        """Create visualizations of the results."""
        if self.results is None:
            logger.warning("No results to visualize")
            return
        
        # Create output directory for plots
        plots_dir = os.path.join(self.config.data.output_path, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot 1: Evaluation metrics
        self._plot_evaluation_metrics(plots_dir)
        
        # Plot 2: Training progress
        self._plot_training_progress(plots_dir)
        
        # Plot 3: Target achievement
        self._plot_target_achievement(plots_dir)
        
        logger.info(f"Visualizations saved to {plots_dir}")
    
    def _plot_evaluation_metrics(self, plots_dir: str):
        """Plot evaluation metrics."""
        metrics = self.results.evaluation_metrics
        
        # Filter out non-numeric metrics
        numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and k != 'targets_met'}
        
        if not numeric_metrics:
            return
        
        plt.figure(figsize=(10, 6))
        metric_names = list(numeric_metrics.keys())
        metric_values = list(numeric_metrics.values())
        
        bars = plt.bar(metric_names, metric_values, color='skyblue', alpha=0.7)
        
        # Add target lines
        targets = {
            'bertscore': self.config.evaluation.target_bertscore,
            'codebleu': self.config.evaluation.target_codebleu,
            'bleu': self.config.evaluation.target_bleu,
            'rouge': self.config.evaluation.target_rouge,
            'ruby': self.config.evaluation.target_ruby
        }
        
        for i, (metric_name, target) in enumerate(targets.items()):
            if metric_name in numeric_metrics:
                plt.axhline(y=target, color='red', linestyle='--', alpha=0.7, label=f'{metric_name} target' if i == 0 else "")
        
        plt.title('Evaluation Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, 'evaluation_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_progress(self, plots_dir: str):
        """Plot training progress."""
        # This would require training history data
        # For now, create a simple placeholder
        plt.figure(figsize=(10, 6))
        plt.title('Training Progress (Placeholder)')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.text(0.5, 0.5, 'Training progress visualization\nwould be implemented here', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.savefig(os.path.join(plots_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_target_achievement(self, plots_dir: str):
        """Plot target achievement."""
        if 'targets_met' not in self.results.evaluation_metrics:
            return
        
        targets_met = self.results.evaluation_metrics['targets_met']
        
        plt.figure(figsize=(8, 6))
        metric_names = list(targets_met.keys())
        achieved = [1 if targets_met[name] else 0 for name in metric_names]
        
        colors = ['green' if a else 'red' for a in achieved]
        bars = plt.bar(metric_names, achieved, color=colors, alpha=0.7)
        
        plt.title('Target Achievement')
        plt.ylabel('Achieved (1) / Not Achieved (0)')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.2)
        
        # Add text labels
        for bar, achieved in zip(bars, achieved):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    '✓' if achieved else '✗', ha='center', va='bottom', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'target_achievement.png'), dpi=300, bbox_inches='tight')
        plt.close()


# Convenience functions for different use cases
def run_research_experiment() -> PipelineResults:
    """Run a research experiment with optimized settings."""
    config = get_research_config()
    pipeline = ModernRLHFPipeline(config)
    results = pipeline.run_full_pipeline()
    pipeline.visualize_results()
    return results


def run_production_training() -> PipelineResults:
    """Run production training with stable settings."""
    config = get_production_config()
    pipeline = ModernRLHFPipeline(config)
    results = pipeline.run_full_pipeline()
    pipeline.visualize_results()
    return results


def run_fast_prototype() -> PipelineResults:
    """Run a fast prototype for quick testing."""
    config = get_fast_config()
    pipeline = ModernRLHFPipeline(config)
    results = pipeline.run_full_pipeline()
    pipeline.visualize_results()
    return results


if __name__ == "__main__":
    # Example usage
    results = run_research_experiment()
    print(f"Pipeline completed with success: {results.success}")
    print(f"Final metrics: {results.final_metrics}")
