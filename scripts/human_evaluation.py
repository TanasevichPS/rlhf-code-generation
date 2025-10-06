#!/usr/bin/env python3
"""Generate samples for human evaluation."""

import sys
import os
import pandas as pd
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.evaluate_model import ModelEvaluator

def generate_human_evaluation_samples():
    """Generate samples for human evaluation."""
    
    # Initialize evaluator
    trained_model_path = "./outputs/final_model"
    reward_model_path = "./outputs/trained_reward_model.pt"
    evaluator = ModelEvaluator(trained_model_path, reward_model_path)
    
    # Diverse test prompts covering different aspects
    evaluation_prompts = [
        # Simple functions
        "Write a Python function to calculate factorial",
        "Create a function to reverse a string",
        
        # File operations
        "Write code to read a CSV file and print its contents",
        "Create a function to write data to a JSON file",
        
        # Web requests
        "Write code to make HTTP request and handle errors",
        "Create a function to download file from URL",
        
        # Data processing
        "Write a function to filter list of dictionaries",
        "Create code to process pandas DataFrame",
        
        # Classes and OOP
        "Write a Python class for a simple calculator",
        "Create a class to represent a person with name and age",
        
        # Error handling
        "Write a function with proper error handling",
        "Create code that uses try-except blocks",
        
        # Real-world scenarios
        "Write code to parse command line arguments",
        "Create a function to send email using smtplib"
    ]
    
    results = []
    
    evaluator.logger.info("Generating samples for human evaluation...")
    
    for i, prompt in enumerate(evaluation_prompts):
        evaluator.logger.info(f"Generating sample {i+1}/{len(evaluation_prompts)}")
        
        result = evaluator.evaluate_single_prompt(prompt)
        results.append({
            'prompt_id': i + 1,
            'prompt': prompt,
            'generated_code': result['generated_code'],
            'auto_quality_score': result['overall_quality'],
            'syntax_score': result['syntax_score'],
            'structure_score': result['structure_score']
        })
    
    # Save for human evaluation
    output_file = "./evaluation_results/human_evaluation_samples.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Also create a CSV version for easier review
    df = pd.DataFrame(results)
    csv_output = "./evaluation_results/human_evaluation_samples.csv"
    df.to_csv(csv_output, index=False, encoding='utf-8')
    
    evaluator.logger.info(f"Human evaluation samples saved to:")
    evaluator.logger.info(f"  JSON: {output_file}")
    evaluator.logger.info(f"  CSV: {csv_output}")
    
    # Print summary
    avg_quality = df['auto_quality_score'].mean()
    avg_syntax = df['syntax_score'].mean()
    avg_structure = df['structure_score'].mean()
    
    evaluator.logger.info(f"\nSummary for human evaluation:")
    evaluator.logger.info(f"Average quality score: {avg_quality:.4f}")
    evaluator.logger.info(f"Average syntax score: {avg_syntax:.4f}")
    evaluator.logger.info(f"Average structure score: {avg_structure:.4f}")

if __name__ == "__main__":
    generate_human_evaluation_samples()