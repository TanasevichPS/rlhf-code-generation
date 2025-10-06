#!/usr/bin/env python3
"""Train reward model on human evaluations."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import CodeRLHFConfig
from src.data.human_eval_processor import process_human_evaluations
from src.models.reward_model import ImprovedCodeRewardModel

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def train_reward_model():
    """Train the reward model on human evaluation data."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    config = CodeRLHFConfig()
    
    # Load human evaluations
    logger.info("Loading human evaluations...")
    human_evals = process_human_evaluations(config.human_eval_path)
    
    if human_evals.empty:
        logger.error("No human evaluation data found!")
        return
    
    # Initialize model
    reward_model = ImprovedCodeRewardModel(config.reward_model_name)
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-5)
    
    # Prepare data
    questions = human_evals['question'].tolist()
    answers = human_evals['answer'].tolist()
    consistency_scores = torch.tensor(human_evals['consistent_score'].values, dtype=torch.float32)
    correctness_scores = torch.tensor(human_evals['correct_score'].values, dtype=torch.float32)
    usefulness_scores = torch.tensor(human_evals['useful_score'].values, dtype=torch.float32)
    overall_scores = torch.tensor(human_evals['total_score'].values, dtype=torch.float32)
    
    # Training loop
    reward_model.train()
    logger.info("Starting reward model training...")
    
    for epoch in range(config.reward_training_epochs):
        total_loss = 0
        batch_count = 0
        
        for i in range(0, len(questions), config.batch_size):
            batch_questions = questions[i:i+config.batch_size]
            batch_answers = answers[i:i+config.batch_size]
            
            if not batch_questions:
                continue
                
            batch_consistency = consistency_scores[i:i+config.batch_size].to(reward_model.device)
            batch_correctness = correctness_scores[i:i+config.batch_size].to(reward_model.device)
            batch_usefulness = usefulness_scores[i:i+config.batch_size].to(reward_model.device)
            batch_overall = overall_scores[i:i+config.batch_size].to(reward_model.device)
            
            # Forward pass
            predictions = reward_model(batch_questions, batch_answers)
            
            # Compute losses
            loss = (F.mse_loss(predictions['consistency'].squeeze(), batch_consistency) +
                   F.mse_loss(predictions['correctness'].squeeze(), batch_correctness) +
                   F.mse_loss(predictions['usefulness'].squeeze(), batch_usefulness) +
                   F.mse_loss(predictions['overall'].squeeze(), batch_overall))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        logger.info(f"Epoch {epoch+1}/{config.reward_training_epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save trained model
    os.makedirs(config.output_dir, exist_ok=True)
    model_path = os.path.join(config.output_dir, "trained_reward_model.pt")
    torch.save(reward_model.state_dict(), model_path)
    logger.info(f"Trained reward model saved to: {model_path}")
    
    return reward_model

if __name__ == "__main__":
    train_reward_model()