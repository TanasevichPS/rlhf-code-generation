"""
Direct Preference Optimization (DPO) Trainer
============================================

Modern alternative to PPO for RLHF training.
Based on: https://arxiv.org/abs/2305.18290
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
import logging
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class DPOTrainer:
    """
    Direct Preference Optimization trainer.
    
    DPO is a modern alternative to PPO that directly optimizes human preferences
    without explicit reward modeling.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load models
        self.policy_model = self._load_policy_model()
        self.reference_model = self._load_reference_model()
        self.tokenizer = self._load_tokenizer()
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        
        logger.info(f"Initialized DPO trainer with {config.method}")
    
    def _load_policy_model(self):
        """Load the policy model."""
        try:
            # Try to load as causal LM first
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                self.config.policy_model_name,
                torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load as CausalLM: {e}. Trying AutoModel...")
            # Fallback to AutoModel
            model = AutoModel.from_pretrained(
                self.config.policy_model_name,
                torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32,
                trust_remote_code=True
            )
        
        if hasattr(model, 'gradient_checkpointing_enable') and self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model.to(self.device)
    
    def _load_reference_model(self):
        """Load the reference model (frozen)."""
        try:
            # Try to load as causal LM first
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                self.config.policy_model_name,
                torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load reference as CausalLM: {e}. Trying AutoModel...")
            # Fallback to AutoModel
            model = AutoModel.from_pretrained(
                self.config.policy_model_name,
                torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32,
                trust_remote_code=True
            )
        
        # Freeze reference model
        for param in model.parameters():
            param.requires_grad = False
        
        return model.to(self.device)
    
    def _load_tokenizer(self):
        """Load the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.policy_model_name,
            use_fast=True,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def dpo_loss(self, policy_chosen_logps, policy_rejected_logps, 
                 reference_chosen_logps, reference_rejected_logps, beta=None):
        """
        Compute DPO loss.
        
        Args:
            policy_chosen_logps: Log probabilities of chosen responses from policy model
            policy_rejected_logps: Log probabilities of rejected responses from policy model
            reference_chosen_logps: Log probabilities of chosen responses from reference model
            reference_rejected_logps: Log probabilities of rejected responses from reference model
            beta: Temperature parameter (defaults to config.beta)
        """
        if beta is None:
            beta = self.config.beta
        
        # Compute log ratios
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps
        
        # DPO loss
        logits = policy_logratios - reference_logratios
        losses = -F.logsigmoid(beta * logits)
        
        return losses.mean()
    
    def compute_log_probs(self, model, input_ids, attention_mask, labels):
        """Compute log probabilities for given inputs and labels."""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # Get log probabilities
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract log probabilities for labels
        labels_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        
        # Mask out padding tokens
        mask = (labels != -100).float()
        masked_log_probs = labels_log_probs * mask
        
        # Sum over sequence length
        sequence_log_probs = masked_log_probs.sum(dim=-1)
        
        return sequence_log_probs
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step."""
        self.policy_model.train()
        
        prompts = batch['prompts']
        chosen_responses = batch['chosen_responses']
        rejected_responses = batch['rejected_responses']
        
        # Tokenize inputs
        chosen_inputs = self.tokenizer(
            [f"{prompt} {response}" for prompt, response in zip(prompts, chosen_responses)],
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        rejected_inputs = self.tokenizer(
            [f"{prompt} {response}" for prompt, response in zip(prompts, rejected_responses)],
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Create labels (shifted input_ids)
        chosen_labels = chosen_inputs['input_ids'].clone()
        rejected_labels = rejected_inputs['input_ids'].clone()
        
        # Mask prompt tokens in labels
        prompt_lengths = [len(self.tokenizer.encode(prompt, add_special_tokens=False)) 
                         for prompt in prompts]
        
        for i, prompt_len in enumerate(prompt_lengths):
            chosen_labels[i, :prompt_len] = -100
            rejected_labels[i, :prompt_len] = -100
        
        # Compute log probabilities
        with torch.no_grad():
            reference_chosen_logps = self.compute_log_probs(
                self.reference_model, 
                chosen_inputs['input_ids'], 
                chosen_inputs['attention_mask'], 
                chosen_labels
            )
            reference_rejected_logps = self.compute_log_probs(
                self.reference_model, 
                rejected_inputs['input_ids'], 
                rejected_inputs['attention_mask'], 
                rejected_labels
            )
        
        policy_chosen_logps = self.compute_log_probs(
            self.policy_model, 
            chosen_inputs['input_ids'], 
            chosen_inputs['attention_mask'], 
            chosen_labels
        )
        policy_rejected_logps = self.compute_log_probs(
            self.policy_model, 
            rejected_inputs['input_ids'], 
            rejected_inputs['attention_mask'], 
            rejected_labels
        )
        
        # Compute DPO loss
        loss = self.dpo_loss(
            policy_chosen_logps, 
            policy_rejected_logps,
            reference_chosen_logps, 
            reference_rejected_logps
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()
        
        self.step += 1
        
        return {
            'loss': loss.item(),
            'chosen_log_prob': policy_chosen_logps.mean().item(),
            'rejected_log_prob': policy_rejected_logps.mean().item(),
            'log_ratio': (policy_chosen_logps - policy_rejected_logps).mean().item()
        }
    
    def train(self, train_loader: DataLoader) -> Dict[str, Any]:
        """Train the model."""
        logger.info("Starting DPO training...")
        
        training_stats = []
        
        for epoch in range(self.config.num_epochs):
            epoch_stats = []
            
            for batch in tqdm(train_loader, desc=f"DPO Epoch {epoch + 1}"):
                stats = self.train_step(batch)
                epoch_stats.append(stats)
                
                # Logging
                if self.step % self.config.logging_steps == 0:
                    logger.info(f"Step {self.step}: Loss = {stats['loss']:.4f}")
            
            # Average epoch stats
            avg_stats = {}
            for key in epoch_stats[0].keys():
                avg_stats[key] = np.mean([s[key] for s in epoch_stats])
            
            training_stats.append(avg_stats)
            logger.info(f"Epoch {epoch + 1} completed: {avg_stats}")
            
            self.epoch += 1
        
        logger.info("DPO training completed!")
        
        return {
            'training_stats': training_stats,
            'final_model': self.policy_model
        }
    
    def save_model(self, save_path: str):
        """Save the trained model."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and tokenizer
        self.policy_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def generate_responses(self, prompts: List[str], max_new_tokens: int = 256) -> List[str]:
        """Generate responses for given prompts."""
        self.policy_model.eval()
        
        responses = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length - max_new_tokens
                ).to(self.device)
                
                # Generate response
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode response
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                responses.append(response)
        
        return responses
