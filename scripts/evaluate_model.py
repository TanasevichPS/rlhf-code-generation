#!/usr/bin/env python3
"""Evaluate trained model on multiple datasets."""

import sys
import os
import pandas as pd
import torch
import logging
from typing import List, Dict, Any
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import CodeRLHFConfig
from src.data.dataset_loader import CodeDatasetLoader
from src.models.model_loader import ModelLoader
from src.models.reward_model import ImprovedCodeRewardModel

class ModelEvaluator:
    def __init__(self, model_path: str, reward_model_path: str):
        self.config = CodeRLHFConfig()
        self.config.model_name = model_path  # Use trained model
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load models
        self.logger.info("Loading models...")
        self.model_loader = ModelLoader(self.config)
        self.tokenizer, self.policy_model, _ = self.model_loader.load_models()
        
        # Load reward model
        self.reward_model = ImprovedCodeRewardModel(self.config.reward_model_name)
        if os.path.exists(reward_model_path):
            self.reward_model.load_state_dict(torch.load(reward_model_path, map_location=self.config.device))
            self.logger.info("Loaded trained reward model")
        else:
            self.logger.warning("Using untrained reward model")
        
        self.reward_model.eval()
    
    def _setup_logging(self):
        """Setup logging without encoding issues."""
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
    
    def generate_code(self, prompt: str, max_length: int = 512) -> str:
        """Generate code for a single prompt."""
        self.policy_model.eval()
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
            padding=True
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.policy_model.generate(
                **inputs,
                max_length=min(len(inputs.input_ids[0]) + max_length, 1024),
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=self.config.repetition_penalty
            )
        
        # Extract generated part
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_code = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up the code
        generated_code = self._clean_generated_code(generated_code)
        
        return generated_code
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean generated code."""
        import re
        
        # Remove markdown code blocks
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        # Remove excessive whitespace but preserve indentation
        lines = []
        for line in code.split('\n'):
            line = line.rstrip()
            if line.strip():  # Keep non-empty lines
                lines.append(line)
        
        return '\n'.join(lines).strip()
    
    def evaluate_single_prompt(self, prompt: str) -> Dict[str, Any]:
        """Evaluate model on a single prompt."""
        try:
            generated_code = self.generate_code(prompt)
            
            # Calculate metrics
            with torch.no_grad():
                reward_score = self.reward_model.compute_reward([prompt], [generated_code])
                detailed_scores = self.reward_model.predict_quality(prompt, generated_code)
            
            # Basic code metrics
            syntax_score = self.reward_model._check_syntax(generated_code)
            structure_score = self.reward_model._check_structure(generated_code)
            
            return {
                'prompt': prompt,
                'generated_code': generated_code,
                'reward_score': reward_score.item() if reward_score.numel() == 1 else reward_score.mean().item(),
                'syntax_score': syntax_score,
                'structure_score': structure_score,
                'consistency_score': detailed_scores['consistency'],
                'correctness_score': detailed_scores['correctness'],
                'usefulness_score': detailed_scores['usefulness'],
                'overall_quality': detailed_scores['overall']
            }
        except Exception as e:
            self.logger.error(f"Error evaluating prompt: {e}")
            return {
                'prompt': prompt,
                'generated_code': '',
                'reward_score': 0.0,
                'syntax_score': 0.0,
                'structure_score': 0.0,
                'consistency_score': 0.0,
                'correctness_score': 0.0,
                'usefulness_score': 0.0,
                'overall_quality': 0.0,
                'error': str(e)
            }
    
    def evaluate_dataset(self, dataset_path: str, sample_size: int = None) -> Dict[str, Any]:
        """Evaluate model on a dataset."""
        self.logger.info(f"Evaluating dataset: {dataset_path}")
        
        try:
            # Load dataset
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            else:
                # Assume it's a directory with CSV files
                csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
                if not csv_files:
                    raise ValueError("No CSV files found in directory")
                
                all_data = []
                for csv_file in csv_files:
                    file_path = os.path.join(dataset_path, csv_file)
                    df = pd.read_csv(file_path)
                    all_data.append(df)
                
                df = pd.concat(all_data, ignore_index=True)
            
            # Find prompt column
            prompt_column = None
            for col in ['Question', 'Prompt', 'prompt', 'instruction', 'input', 'text']:
                if col in df.columns:
                    prompt_column = col
                    break
            
            if prompt_column is None:
                prompt_column = df.columns[0]  # Use first column as fallback
            
            prompts = df[prompt_column].dropna().astype(str).tolist()
            
            if sample_size and sample_size < len(prompts):
                prompts = prompts[:sample_size]
            
            self.logger.info(f"Evaluating {len(prompts)} prompts...")
            
            results = []
            total_scores = {
                'reward': 0.0,
                'syntax': 0.0,
                'structure': 0.0,
                'consistency': 0.0,
                'correctness': 0.0,
                'usefulness': 0.0,
                'overall': 0.0
            }
            
            for i, prompt in enumerate(prompts):
                self.logger.info(f"Processing prompt {i+1}/{len(prompts)}")
                result = self.evaluate_single_prompt(prompt)
                results.append(result)
                
                # Accumulate scores
                for key in total_scores:
                    score_key = f"{key}_score" if key != 'overall' else 'overall_quality'
                    total_scores[key] += result.get(score_key, 0.0)
            
            # Calculate averages
            avg_scores = {f"avg_{key}": total_scores[key] / len(results) for key in total_scores}
            
            evaluation_result = {
                'dataset': os.path.basename(dataset_path),
                'total_prompts': len(results),
                'results': results,
                **avg_scores
            }
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Error evaluating dataset {dataset_path}: {e}")
            return {'error': str(e)}
    
    def save_evaluation_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to file."""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # Save detailed results
        if 'results' in results:
            detailed_df = pd.DataFrame(results['results'])
            detailed_output = output_file.replace('.json', '_detailed.csv')
            detailed_df.to_csv(detailed_output, index=False, encoding='utf-8')
            self.logger.info(f"Detailed results saved to: {detailed_output}")
        
        # Save summary
        summary = {k: v for k, v in results.items() if k != 'results'}
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Summary results saved to: {output_file}")
        
        # Print summary
        self.logger.info("\n" + "="*50)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("="*50)
        for key, value in summary.items():
            if key not in ['dataset', 'total_prompts', 'results']:
                self.logger.info(f"{key}: {value:.4f}")

def main():
    """Main evaluation function."""
    # Paths to your trained models
    trained_model_path = "./outputs/final_model"  # Path to your trained model
    reward_model_path = "./outputs/trained_reward_model.pt"  # Path to trained reward model
    
    # Dataset to evaluate on
    dataset_path = r"C:\Users\Полина\Desktop\Работа\huawei\rlhf\datasets_for_eval"
    
    # Output file
    output_file = "./evaluation_results/final_evaluation.json"
    
    # Initialize evaluator
    evaluator = ModelEvaluator(trained_model_path, reward_model_path)
    
    # Evaluate
    results = evaluator.evaluate_dataset(dataset_path, sample_size=20)  # Evaluate on 20 samples
    
    # Save results
    evaluator.save_evaluation_results(results, output_file)
    
    # Also evaluate on individual example prompts
    test_prompts = [
        "Write a Python function to calculate factorial",
        "Create a function to reverse a string",
        "Write code to read a CSV file and print its contents",
        "Create a Python class for a simple calculator"
    ]
    
    self_test_results = []
    evaluator.logger.info("\n" + "="*50)
    evaluator.logger.info("SELF-TEST EVALUATION")
    evaluator.logger.info("="*50)
    
    for prompt in test_prompts:
        result = evaluator.evaluate_single_prompt(prompt)
        self_test_results.append(result)
        
        evaluator.logger.info(f"Prompt: {prompt}")
        evaluator.logger.info(f"Generated code: {result['generated_code'][:100]}...")
        evaluator.logger.info(f"Overall quality: {result['overall_quality']:.4f}")
        evaluator.logger.info("-" * 30)
    
    # Save self-test results
    self_test_output = "./evaluation_results/self_test_results.csv"
    pd.DataFrame(self_test_results).to_csv(self_test_output, index=False, encoding='utf-8')
    evaluator.logger.info(f"Self-test results saved to: {self_test_output}")

if __name__ == "__main__":
    main()