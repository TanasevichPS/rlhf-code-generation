"""Lightweight reward-model trainer (CPU-safe placeholder).

This script reads `datasets_for_training/pairwise_prefs.csv` and trains a
very small logistic model using a single numeric feature: length difference
between preferred and other answers. It's a placeholder that verifies the
data pipeline and produces a JSON 'checkpoint' that downstream steps can
consume or replace with a real PyTorch/Transformers-trained reward model
once the environment is ready.

Usage:
    python scripts/train_reward_model.py

Outputs:
    outputs/reward_model_placeholder.json
"""
import csv
import json
import math
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(ROOT, "datasets_for_training", "pairwise_prefs.csv")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "reward_model_placeholder.json")


def load_pairs(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def featurize(row):
    a = row.get("preferred_answer", "") or ""
    b = row.get("other_answer", "") or ""
    # single numeric feature: length difference
    return float(len(a) - len(b))


def sigmoid(x):
    # numerically stable sigmoid
    try:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def train_simple(rows, epochs=200, lr=0.01):
    # simple logistic regression with single weight + bias
    w = 0.0
    b = 0.0
    n = 0
    for r in rows:
        if r.get("preference") in ("left", "right"):
            n += 1
    if n == 0:
        raise ValueError("No labeled pairwise rows found in dataset")

    for ep in range(epochs):
        dw = 0.0
        db = 0.0
        for r in rows:
            if r.get("preference") not in ("left", "right"):
                continue
            x = featurize(r)
            y = 1.0 if r.get("preference") == "left" else 0.0
            p = sigmoid(w * x + b)
            err = p - y
            dw += err * x
            db += err
        # gradient step (mean)
        w -= lr * (dw / n)
        b -= lr * (db / n)
    return {"weight": w, "bias": b}


def main():
    if not os.path.exists(IN_PATH):
        logging.error("Required input not found: %s", IN_PATH)
        logging.error("Run scripts/prepare_pairs.py first to generate datasets.")
        return 2

    rows = load_pairs(IN_PATH)
    if not rows:
        logging.error("No rows found in %s", IN_PATH)
        return 2

    try:
        model = train_simple(rows)
    except Exception as e:
        logging.error("Training failed: %s", e)
        return 3

    ckpt = {"meta": {"type": "placeholder_logistic", "rows": len(rows)}, "model": model}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)

    logging.info("Saved placeholder reward model to %s", OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
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
    # Primary state_dict (backwards-compatible single-file)
    model_path = os.path.join(config.output_dir, "trained_reward_model.pt")
    torch.save(reward_model.state_dict(), model_path)
    logger.info(f"Trained reward model saved to: {model_path}")

    # Also save HF-style artifacts (tokenizer + base model) into a folder so
    # other scripts can load by directory. We also write the full improved
    # state_dict into that folder for downstream loading into the
    # ImprovedCodeRewardModel (strict=False).
    try:
        hf_dir = os.path.join(config.output_dir, "reward_model_hf")
        os.makedirs(hf_dir, exist_ok=True)
        # Save base model weights (AutoModel) and tokenizer
        try:
            reward_model.bert.save_pretrained(hf_dir)
        except Exception:
            # Some AutoModel instances use different save semantics; swallow errors
            logger.warning("Could not save base AutoModel with save_pretrained; continuing")
        try:
            reward_model.tokenizer.save_pretrained(hf_dir)
        except Exception:
            logger.warning("Could not save tokenizer with save_pretrained; continuing")

        # Save the full improved state dict for strict=False loading later
        improved_state_path = os.path.join(hf_dir, "improved_state_dict.pt")
        torch.save(reward_model.state_dict(), improved_state_path)
        logger.info(f"Also saved HF-style reward model artifacts to: {hf_dir}")
    except Exception as e:
        logger.warning(f"Failed to save HF-style reward artifacts: {e}")
    
    return reward_model

if __name__ == "__main__":
    train_reward_model()