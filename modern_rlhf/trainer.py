"""
Modern RLHF Trainer with PPO and DPO Support
===========================================

A state-of-the-art trainer that supports both PPO and DPO (Direct Preference Optimization)
for code generation tasks with comprehensive evaluation and monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    PreTrainedModel, PreTrainedTokenizer
)
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import numpy as np
from dataclasses import dataclass
import json
import os
import time
from tqdm import tqdm
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
try:
    import wandb  # Optional
    _WANDB_AVAILABLE = True
except Exception:  # broad to handle env issues
    wandb = None
    _WANDB_AVAILABLE = False
=======
import wandb
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
import wandb
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
import wandb
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
from collections import defaultdict

from .config import ModernRLHFConfig, TrainingConfig
from .reward_model import ModernRewardModel
from .metrics import ModernMetricsEvaluator

logger = logging.getLogger(__name__)


@dataclass
class TrainingStep:
    """Container for training step results."""
    step: int
    loss: float
    reward: float
    kl_divergence: float
    entropy: float
    learning_rate: float
    metrics: Dict[str, float]


class PPOTrainer:
    """Modern PPO trainer for RLHF."""
    
    def __init__(self, config: ModernRLHFConfig, reward_model: ModernRewardModel):
        self.config = config
        self.reward_model = reward_model
        self.device = torch.device(config.hardware.device)
        
        # Load models
        self.policy_model = self._load_policy_model()
        self.reference_model = self._load_reference_model()
        self.tokenizer = self._load_tokenizer()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=0.01
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.total_steps
        )
        
        # Metrics evaluator
        self.metrics_evaluator = ModernMetricsEvaluator()
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_reward = -float('inf')
        self.training_history = []
        
        # Initialize wandb if available
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        self._wandb_enabled = False
        if config.verbose and not config.debug and _WANDB_AVAILABLE:
=======
        if config.verbose and not config.debug:
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
        if config.verbose and not config.debug:
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
        if config.verbose and not config.debug:
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
            try:
                wandb.init(
                    project=config.experiment_name,
                    name=config.run_name,
                    config=config.to_dict(),
                    tags=config.tags
                )
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
                self._wandb_enabled = True
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
    
    def _load_policy_model(self) -> PreTrainedModel:
        """Load the policy model."""
        model = AutoModel.from_pretrained(
            self.config.model.base_model_name,
            torch_dtype=getattr(torch, self.config.model.torch_dtype),
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        if self.config.hardware.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model.to(self.device)
    
    def _load_reference_model(self) -> PreTrainedModel:
        """Load the reference model (frozen)."""
        model = AutoModel.from_pretrained(
            self.config.model.base_model_name,
            torch_dtype=getattr(torch, self.config.model.torch_dtype),
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        # Freeze reference model
        for param in model.parameters():
            param.requires_grad = False
        
        return model.to(self.device)
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_name,
            use_fast=self.config.model.use_fast_tokenizer,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def generate_responses(self, prompts: List[str]) -> Dict[str, Any]:
        """Generate responses for given prompts."""
        self.policy_model.eval()
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.generation.max_prompt_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate responses
        with torch.no_grad():
            outputs = self.policy_model.generate(
                **inputs,
                max_new_tokens=self.config.generation.max_new_tokens,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                top_k=self.config.generation.top_k,
                do_sample=self.config.generation.do_sample,
                repetition_penalty=self.config.generation.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode responses
        response_texts = []
        for i, output in enumerate(outputs):
            # Remove prompt from output
            prompt_length = inputs['input_ids'][i].shape[0]
            response_tokens = output[prompt_length:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            response_texts.append(response_text)
        
        return {
            "response_texts": response_texts,
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask']
        }
    
    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute rewards for prompt-response pairs."""
        return self.reward_model.compute_reward(prompts, responses)
    
    def compute_kl_divergence(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute KL divergence between policy and reference models."""
        # Tokenize responses
        inputs = self.tokenizer(
            responses,
            padding=True,
            truncation=True,
            max_length=self.config.generation.max_response_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Get logits from both models
        with torch.no_grad():
            policy_logits = self.policy_model(**inputs).logits
            reference_logits = self.reference_model(**inputs).logits
        
        # Compute KL divergence
        policy_probs = F.softmax(policy_logits, dim=-1)
        reference_probs = F.softmax(reference_logits, dim=-1)
        
        kl_div = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            reference_probs,
            reduction='none'
        ).sum(dim=-1)
        
        return kl_div.mean(dim=1)
    
    def compute_entropy(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute entropy of policy model outputs."""
        # Tokenize responses
        inputs = self.tokenizer(
            responses,
            padding=True,
            truncation=True,
            max_length=self.config.generation.max_response_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Get logits
        with torch.no_grad():
            logits = self.policy_model(**inputs).logits
        
        # Compute entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        
        return entropy.mean(dim=1)
    
    def ppo_step(self, batch: Dict[str, Any]) -> TrainingStep:
        """Single PPO training step."""
        self.policy_model.train()
        
        prompts = batch['prompts']
        responses = batch['responses']
        old_log_probs = batch.get('old_log_probs', None)
        
        # Compute rewards
        rewards = self.compute_rewards(prompts, responses)
        
        # Compute KL divergence
        kl_div = self.compute_kl_divergence(prompts, responses)
        
        # Compute entropy
        entropy = self.compute_entropy(prompts, responses)
        
        # Compute advantages (simplified)
        advantages = rewards - rewards.mean()
        
        # Compute policy loss
        if old_log_probs is not None:
            # Compute new log probabilities
            inputs = self.tokenizer(
                responses,
                padding=True,
                truncation=True,
                max_length=self.config.generation.max_response_length,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.policy_model(**inputs)
            new_log_probs = F.log_softmax(outputs.logits, dim=-1)
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute clipped loss
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.training.ppo_clip_ratio,
                1 + self.config.training.ppo_clip_ratio
            )
            
            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
        else:
            # Simplified policy loss
            policy_loss = -rewards.mean()
        
        # Compute value loss (simplified)
        value_loss = F.mse_loss(rewards, rewards.mean().expand_as(rewards))
        
        # Compute entropy loss
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (
            policy_loss +
            self.config.training.ppo_value_loss_coef * value_loss +
            self.config.training.ppo_entropy_coef * entropy_loss +
            self.config.training.ppo_kl_penalty * kl_div.mean()
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.training.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        
        # Create training step
        step = TrainingStep(
            step=self.step,
            loss=total_loss.item(),
            reward=rewards.mean().item(),
            kl_divergence=kl_div.mean().item(),
            entropy=entropy.mean().item(),
            learning_rate=self.optimizer.param_groups[0]['lr'],
            metrics={
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'kl_penalty': kl_div.mean().item()
            }
        )
        
        self.step += 1
        return step
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = defaultdict(list)
        
        for batch in tqdm(dataloader, desc=f"Epoch {self.epoch}"):
            step = self.ppo_step(batch)
            
            # Collect metrics
            epoch_metrics['loss'].append(step.loss)
            epoch_metrics['reward'].append(step.reward)
            epoch_metrics['kl_divergence'].append(step.kl_divergence)
            epoch_metrics['entropy'].append(step.entropy)
            epoch_metrics['learning_rate'].append(step.learning_rate)
            
            # Log to wandb
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
            if self._wandb_enabled:
                try:
                    wandb.log({
                        'step': step.step,
                        'loss': step.loss,
                        'reward': step.reward,
                        'kl_divergence': step.kl_divergence,
                        'entropy': step.entropy,
                        'learning_rate': step.learning_rate,
                        **step.metrics
                    })
                except Exception as e:
                    logger.warning(f"wandb.log failed: {e}")
=======
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
            if hasattr(self, 'wandb') and self.wandb:
                wandb.log({
                    'step': step.step,
                    'loss': step.loss,
                    'reward': step.reward,
                    'kl_divergence': step.kl_divergence,
                    'entropy': step.entropy,
                    'learning_rate': step.learning_rate,
                    **step.metrics
                })
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
            
            # Save checkpoint
            if step.step % self.config.training.save_steps == 0:
                self.save_checkpoint()
        
        # Average metrics
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            avg_metrics[key] = np.mean(values)
        
        self.epoch += 1
        return avg_metrics
    
    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """Evaluate the model."""
        self.policy_model.eval()
        
        all_prompts = []
        all_responses = []
        all_rewards = []
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        all_references = []
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
        
        with torch.no_grad():
            for batch in eval_dataloader:
                prompts = batch['prompts']
                
                # Generate responses
                generation_output = self.generate_responses(prompts)
                responses = generation_output['response_texts']
                
                # Compute rewards
                rewards = self.compute_rewards(prompts, responses)
                
                all_prompts.extend(prompts)
                all_responses.extend(responses)
                all_rewards.extend(rewards.tolist())
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
                if 'references' in batch:
                    all_references.extend(batch['references'])
        
        # Compute evaluation metrics
        eval_metrics = {}
        if all_rewards:
            eval_metrics['avg_reward'] = np.mean(all_rewards)
            eval_metrics['reward_std'] = np.std(all_rewards)
        else:
            eval_metrics['avg_reward'] = 0.0
            eval_metrics['reward_std'] = 0.0
        
        # Compute other metrics if references are available and aligned
        if all_references and len(all_references) == len(all_responses):
            metrics_results = self.metrics_evaluator.compute_all_metrics(all_responses, all_references)
=======
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
        
        # Compute evaluation metrics
        eval_metrics = {}
        eval_metrics['avg_reward'] = np.mean(all_rewards)
        eval_metrics['reward_std'] = np.std(all_rewards)
        
        # Compute other metrics if references are available
        if 'references' in batch:
            references = batch['references']
            metrics_results = self.metrics_evaluator.compute_all_metrics(all_responses, references)
            
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
            for metric_name, result in metrics_results.items():
                eval_metrics[f'eval_{metric_name}'] = result.score
        
        return eval_metrics
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.data.output_path, f"checkpoint-{self.step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.policy_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            'step': self.step,
            'epoch': self.epoch,
            'best_reward': self.best_reward,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        
        with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w') as f:
            json.dump(training_state, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint."""
        # Load model
        self.policy_model = AutoModel.from_pretrained(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        
        # Load training state
        with open(os.path.join(checkpoint_dir, 'training_state.json'), 'r') as f:
            training_state = json.load(f)
        
        self.step = training_state['step']
        self.epoch = training_state['epoch']
        self.best_reward = training_state['best_reward']
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
        self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")


class DPOTrainer:
    """Direct Preference Optimization trainer."""
    
    def __init__(self, config: ModernRLHFConfig, reward_model: ModernRewardModel):
        self.config = config
        self.reward_model = reward_model
        self.device = torch.device(config.hardware.device)
        
        # Load models
        self.policy_model = self._load_policy_model()
        self.reference_model = self._load_reference_model()
        self.tokenizer = self._load_tokenizer()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=0.01
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.training_history = []
    
    def _load_policy_model(self) -> PreTrainedModel:
        """Load the policy model."""
        model = AutoModel.from_pretrained(
            self.config.model.base_model_name,
            torch_dtype=getattr(torch, self.config.model.torch_dtype),
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        if self.config.hardware.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model.to(self.device)
    
    def _load_reference_model(self) -> PreTrainedModel:
        """Load the reference model (frozen)."""
        model = AutoModel.from_pretrained(
            self.config.model.base_model_name,
            torch_dtype=getattr(torch, self.config.model.torch_dtype),
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        # Freeze reference model
        for param in model.parameters():
            param.requires_grad = False
        
        return model.to(self.device)
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_name,
            use_fast=self.config.model.use_fast_tokenizer,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def dpo_step(self, batch: Dict[str, Any]) -> TrainingStep:
        """Single DPO training step."""
        self.policy_model.train()
        
        prompts = batch['prompts']
        chosen_responses = batch['chosen_responses']
        rejected_responses = batch['rejected_responses']
        
        # Compute log probabilities for chosen responses
        chosen_log_probs = self._compute_log_probs(prompts, chosen_responses)
        
        # Compute log probabilities for rejected responses
        rejected_log_probs = self._compute_log_probs(prompts, rejected_responses)
        
        # Compute reference log probabilities
        with torch.no_grad():
            chosen_ref_log_probs = self._compute_log_probs(prompts, chosen_responses, use_reference=True)
            rejected_ref_log_probs = self._compute_log_probs(prompts, rejected_responses, use_reference=True)
        
        # Compute DPO loss
        pi_logratios = chosen_log_probs - rejected_log_probs
        ref_logratios = chosen_ref_log_probs - rejected_ref_log_probs
        
        logits = pi_logratios - ref_logratios
        
        if self.config.training.dpo_loss_type == "sigmoid":
            losses = -F.logsigmoid(self.config.training.dpo_beta * logits)
        elif self.config.training.dpo_loss_type == "hinge":
            losses = torch.relu(1 - self.config.training.dpo_beta * logits)
        else:
            raise ValueError(f"Unknown DPO loss type: {self.config.training.dpo_loss_type}")
        
        loss = losses.mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.training.max_grad_norm)
        self.optimizer.step()
        
        # Create training step
        step = TrainingStep(
            step=self.step,
            loss=loss.item(),
            reward=0.0,  # DPO doesn't use explicit rewards
            kl_divergence=0.0,
            entropy=0.0,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            metrics={
                'dpo_loss': loss.item(),
                'chosen_log_prob': chosen_log_probs.mean().item(),
                'rejected_log_prob': rejected_log_probs.mean().item(),
                'log_ratio': logits.mean().item()
            }
        )
        
        self.step += 1
        return step
    
    def _compute_log_probs(self, prompts: List[str], responses: List[str], use_reference: bool = False) -> torch.Tensor:
        """Compute log probabilities for prompt-response pairs."""
        model = self.reference_model if use_reference else self.policy_model
        
        # Tokenize
        inputs = self.tokenizer(
            [f"{prompt} {response}" for prompt, response in zip(prompts, responses)],
            padding=True,
            truncation=True,
            max_length=self.config.generation.max_prompt_length + self.config.generation.max_response_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Compute logits
        with torch.no_grad() if use_reference else torch.enable_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract log probabilities for response tokens
        response_log_probs = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            
            # Get log probabilities for response tokens
            response_start = len(prompt_tokens)
            response_end = response_start + len(response_tokens)
            
            if response_end <= logits.shape[1]:
                response_log_prob = log_probs[i, response_start:response_end, response_tokens].sum()
                response_log_probs.append(response_log_prob)
            else:
                response_log_probs.append(torch.tensor(0.0))
        
        return torch.stack(response_log_probs)
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = defaultdict(list)
        
        for batch in tqdm(dataloader, desc=f"DPO Epoch {self.epoch}"):
            step = self.dpo_step(batch)
            
            # Collect metrics
            epoch_metrics['loss'].append(step.loss)
            epoch_metrics['learning_rate'].append(step.learning_rate)
            
            # Log metrics
            for key, value in step.metrics.items():
                epoch_metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            avg_metrics[key] = np.mean(values)
        
        self.epoch += 1
        return avg_metrics


class ModernRLHFTrainer:
    """Main trainer that supports both PPO and DPO."""
    
    def __init__(self, config: ModernRLHFConfig, reward_model: ModernRewardModel):
        self.config = config
        self.reward_model = reward_model
        
        # Choose trainer based on config
        if hasattr(config.training, 'use_dpo') and config.training.use_dpo:
            self.trainer = DPOTrainer(config, reward_model)
        else:
            self.trainer = PPOTrainer(config, reward_model)
        
        logger.info(f"Initialized {type(self.trainer).__name__} trainer")
    
    def train(self, train_dataloader, eval_dataloader=None) -> Dict[str, Any]:
        """Main training loop."""
        logger.info("Starting training...")
        
        best_metrics = {}
        patience_counter = 0
        
        for epoch in range(self.config.training.ppo_epochs):
            # Training
            train_metrics = self.trainer.train_epoch(train_dataloader)
            
            # Evaluation
            if eval_dataloader is not None:
                eval_metrics = self.trainer.evaluate(eval_dataloader)
                
                # Check for improvement
                if eval_metrics.get('avg_reward', 0) > best_metrics.get('avg_reward', -float('inf')):
                    best_metrics = eval_metrics
                    patience_counter = 0
                    self.trainer.save_checkpoint()
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.config.training.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
            
            # Log metrics
            logger.info(f"Epoch {epoch}: {train_metrics}")
            if eval_dataloader is not None:
                logger.info(f"Eval metrics: {eval_metrics}")
        
        return best_metrics
