#!/usr/bin/env python3
"""Compare trained model with baseline model."""

import sys
import os
import pandas as pd
import torch
import logging
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import CodeRLHFConfig
from src.models.model_loader import ModelLoader
from src.models.reward_model import ImprovedCodeRewardModel

class ModelComparator:
    def __init__(self, baseline_model_name: str, trained_model_path: str, reward_model_path: str):
        self.config = CodeRLHFConfig()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load baseline model
        self.logger.info("Loading baseline model...")
        self.config.model_name = baseline_model_name
        baseline_loader = ModelLoader(self.config)
        self.baseline_tokenizer, self.baseline_model, _ = baseline_loader.load_models()
        
        # Load trained model
        self.logger.info("Loading trained model...")
        self.config.model_name = trained_model_path
        trained_loader = ModelLoader(self.config)
        self.trained_tokenizer, self.trained_model, _ = trained_loader.load_models()
        
        # Load reward model
        # Initialize reward model from config name by default
        self.reward_model = ImprovedCodeRewardModel(self.config.reward_model_name)

        # If a reward model artifact exists, try to load it. Support directory (HF) or .pt state_dict
        try:
            if os.path.exists(reward_model_path):
                if os.path.isdir(reward_model_path):
                    try:
                        self.reward_model = ImprovedCodeRewardModel(reward_model_path)
                        self.logger.info(f"Initialized reward model from directory: {reward_model_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to init ImprovedCodeRewardModel from dir {reward_model_path}: {e}; using default model")
                else:
                    try:
                        state = torch.load(reward_model_path, map_location=self.config.device)
                        try:
                            self.reward_model.load_state_dict(state, strict=False)
                            self.logger.info(f"Loaded reward model state_dict from: {reward_model_path}")
                        except RuntimeError as e:
                            self.logger.warning(f"reward model state_dict load failed (shape/key mismatch): {e}; continuing with default model")
                    except Exception as e:
                        self.logger.warning(f"Failed to load reward model file {reward_model_path}: {e}; continuing with default model")
        except Exception as e:
            self.logger.warning(f"Unexpected error while preparing reward model: {e}; continuing with default model")

        try:
            self.reward_model.to(self.config.device)
        except Exception:
            pass

        self.reward_model.eval()
    
    def _setup_logging(self):
        class SafeStreamHandler(logging.StreamHandler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    msg = msg.encode('utf-8', 'replace').decode('utf-8')
                    stream = self.stream
                    stream.write(msg + self.terminator)
                    self.flush()
                except Exception:
                    self.handleError(record)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[SafeStreamHandler(sys.stdout)]
        )
    
    def generate_with_model(self, model, tokenizer, prompt: str) -> str:
        """Generate code with specified model."""
        model.eval()
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    def compare_models(self, prompts: List[str]) -> pd.DataFrame:
        """Compare models on multiple prompts."""
        results = []
        
        for i, prompt in enumerate(prompts):
            self.logger.info(f"Comparing models on prompt {i+1}/{len(prompts)}")
            
            # Generate with both models
            baseline_code = self.generate_with_model(self.baseline_model, self.baseline_tokenizer, prompt)
            trained_code = self.generate_with_model(self.trained_model, self.trained_tokenizer, prompt)
            
            # Evaluate both
            with torch.no_grad():
                baseline_reward = self.reward_model.compute_reward([prompt], [baseline_code])
                trained_reward = self.reward_model.compute_reward([prompt], [trained_code])
                
                baseline_quality = self.reward_model.predict_quality(prompt, baseline_code)
                trained_quality = self.reward_model.predict_quality(prompt, trained_code)
            
            result = {
                'prompt': prompt,
                'baseline_code': baseline_code,
                'trained_code': trained_code,
                'baseline_reward': baseline_reward.item(),
                'trained_reward': trained_reward.item(),
                'reward_improvement': trained_reward.item() - baseline_reward.item(),
                'baseline_overall': baseline_quality['overall'],
                'trained_overall': trained_quality['overall'],
                'overall_improvement': trained_quality['overall'] - baseline_quality['overall']
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def calculate_improvement_stats(self, comparison_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate improvement statistics."""
        stats = {
            'avg_baseline_reward': comparison_df['baseline_reward'].mean(),
            'avg_trained_reward': comparison_df['trained_reward'].mean(),
            'avg_reward_improvement': comparison_df['reward_improvement'].mean(),
            'avg_baseline_overall': comparison_df['baseline_overall'].mean(),
            'avg_trained_overall': comparison_df['trained_overall'].mean(),
            'avg_overall_improvement': comparison_df['overall_improvement'].mean(),
            'improvement_rate': (comparison_df['reward_improvement'] > 0).mean(),
            'significant_improvement_rate': (comparison_df['reward_improvement'] > 0.1).mean()
        }
        
        return stats

def main():
    # Model paths
    baseline_model = "microsoft/DialoGPT-medium"  # or "gpt2" for baseline
    trained_model_path = "./outputs/final_model"
    reward_model_path = "./outputs/trained_reward_model.pt"
    
    # Test prompts
    test_prompts = [
        "Write a Python function to calculate factorial",
        "Create a function to reverse a string",
        "Write code to read a CSV file and print its contents",
        "Create a Python class for a simple calculator",
        "Write a function to check if a number is prime",
        "Create code to download a file from URL using requests"
    ]
    
    # Compare models
    comparator = ModelComparator(baseline_model, trained_model_path, reward_model_path)
    comparison_results = comparator.compare_models(test_prompts)
    
    # Calculate stats
    stats = comparator.calculate_improvement_stats(comparison_results)
    
    # Save results
    os.makedirs("./evaluation_results", exist_ok=True)
    comparison_results.to_csv("./evaluation_results/model_comparison.csv", index=False, encoding='utf-8')
    
    # Print results
    comparator.logger.info("\n" + "="*60)
    comparator.logger.info("MODEL COMPARISON RESULTS")
    comparator.logger.info("="*60)
    
    for key, value in stats.items():
        comparator.logger.info(f"{key}: {value:.4f}")
    
    comparator.logger.info(f"\nDetailed results saved to: ./evaluation_results/model_comparison.csv")

if __name__ == "__main__":
    main()