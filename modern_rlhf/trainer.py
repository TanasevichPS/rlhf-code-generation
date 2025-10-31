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
    AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig,
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
try:
    import wandb  # Optional
    _WANDB_AVAILABLE = True
except Exception:  # broad to handle env issues
    wandb = None
    _WANDB_AVAILABLE = False
import wandb
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
        
        # Force GPU usage - verify CUDA is available
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is not available! Training requires GPU.")
        
        if config.hardware.device != "cuda":
            logger.warning(f"Config device is '{config.hardware.device}', but forcing GPU usage")
            config.hardware.device = "cuda"
        
        self.device = torch.device("cuda")  # Force GPU
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"PPOTrainer: Using GPU {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"PPOTrainer initialized on GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
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
        self.evaluation_history = []  # Track evaluation metrics per epoch
        
        # Initialize wandb if available
        self._wandb_enabled = False
        if config.verbose and not config.debug and _WANDB_AVAILABLE:
            try:
                wandb.init(
                    project=config.experiment_name,
                    name=config.run_name,
                    config=config.to_dict(),
                    tags=config.tags
                )
                self._wandb_enabled = True
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
    
    def _load_policy_model(self) -> PreTrainedModel:
        """Load the policy model (prefer causal LM / seq2seq variants to get logits/generate)."""
        model_name = self.config.model.base_model_name
        torch_dtype = getattr(torch, self.config.model.torch_dtype)

        # Prefer causal LM model to support `.generate()` and `.logits`
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=self.config.model.trust_remote_code
            )
        except Exception:
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=self.config.model.trust_remote_code
                )
            except Exception:
                # Fallback to base AutoModel (may not provide logits/generate)
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=self.config.model.trust_remote_code
                )

        if self.config.hardware.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        model = model.to(self.device)  # Move to GPU
        # Verify model is on GPU
        if next(model.parameters()).is_cuda:
            logger.info(f"Policy model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            raise RuntimeError(f"Policy model is not on GPU! Device: {next(model.parameters()).device}")
        
        return model
    
    def _load_reference_model(self) -> PreTrainedModel:
        """Load the reference model (frozen). Prefer matching class to policy model."""
        model_name = self.config.model.base_model_name
        torch_dtype = getattr(torch, self.config.model.torch_dtype)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=self.config.model.trust_remote_code
            )
        except Exception:
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=self.config.model.trust_remote_code
                )
            except Exception:
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=self.config.model.trust_remote_code
                )

        # Freeze reference model
        for param in model.parameters():
            param.requires_grad = False

        model = model.to(self.device)  # Move to GPU
        # Verify model is on GPU
        import torch
        if next(model.parameters()).is_cuda:
            logger.info(f"Reference model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            raise RuntimeError(f"Reference model is not on GPU! Device: {next(model.parameters()).device}")
        
        return model
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_name,
            use_fast=self.config.model.use_fast_tokenizer,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Ensure left padding for decoder-only models to avoid generation issues
        try:
            tokenizer.padding_side = 'left'
        except Exception:
            pass
        
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
            policy_outputs = self.policy_model(**inputs)
            reference_outputs = self.reference_model(**inputs)

            def _outputs_to_logits(model, outputs):
                if hasattr(outputs, 'logits') and outputs.logits is not None:
                    return outputs.logits
                # Try lm_head
                if hasattr(model, 'lm_head') and hasattr(outputs, 'last_hidden_state'):
                    return model.lm_head(outputs.last_hidden_state)
                # Try output embeddings weight multiplication
                try:
                    if hasattr(model, 'get_output_embeddings') and outputs.last_hidden_state is not None:
                        emb = model.get_output_embeddings()
                        if hasattr(emb, 'weight'):
                            return torch.matmul(outputs.last_hidden_state, emb.weight.t())
                except Exception:
                    pass
                raise AttributeError("Model outputs do not contain logits and no lm_head/output_embeddings available")

            policy_logits = _outputs_to_logits(self.policy_model, policy_outputs)
            reference_logits = _outputs_to_logits(self.reference_model, reference_outputs)
        
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
            outputs = self.policy_model(**inputs)
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                logits = outputs.logits
            elif hasattr(self.policy_model, 'lm_head') and hasattr(outputs, 'last_hidden_state'):
                logits = self.policy_model.lm_head(outputs.last_hidden_state)
            else:
                emb = self.policy_model.get_output_embeddings() if hasattr(self.policy_model, 'get_output_embeddings') else None
                if emb is not None and hasattr(outputs, 'last_hidden_state'):
                    logits = torch.matmul(outputs.last_hidden_state, emb.weight.t())
                else:
                    raise AttributeError("Unable to obtain logits from policy model outputs")
        
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
            # Robustly extract logits from outputs
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                logits_for_new = outputs.logits
            elif hasattr(self.policy_model, 'lm_head') and hasattr(outputs, 'last_hidden_state'):
                logits_for_new = self.policy_model.lm_head(outputs.last_hidden_state)
            else:
                emb = self.policy_model.get_output_embeddings() if hasattr(self.policy_model, 'get_output_embeddings') else None
                if emb is not None and hasattr(outputs, 'last_hidden_state'):
                    logits_for_new = torch.matmul(outputs.last_hidden_state, emb.weight.t())
                else:
                    raise AttributeError("Unable to extract logits for new_log_probs")

            new_log_probs = F.log_softmax(logits_for_new, dim=-1)
            
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
        
        # Create progress bar with detailed info
        total_batches = len(dataloader)
        pbar = tqdm(
            enumerate(dataloader), 
            total=total_batches,
            desc=f"Epoch {self.epoch+1}/{self.config.training.ppo_epochs}",
            unit="batch",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for batch_idx, batch in pbar:
            step = self.ppo_step(batch)
            
            # Collect metrics
            epoch_metrics['loss'].append(step.loss)
            epoch_metrics['reward'].append(step.reward)
            epoch_metrics['kl_divergence'].append(step.kl_divergence)
            epoch_metrics['entropy'].append(step.entropy)
            epoch_metrics['learning_rate'].append(step.learning_rate)
            
            # Update progress bar with current metrics
            pbar.set_postfix({
                'loss': f'{step.loss:.4f}',
                'reward': f'{step.reward:.4f}',
                'lr': f'{step.learning_rate:.2e}',
                'step': f'{step.step}/{self.config.training.total_steps}'
            })
            
            # Log to wandb
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
            
            # Save checkpoint
            if step.step % self.config.training.save_steps == 0:
                self.save_checkpoint()
                
            # Stop if reached total steps
            if step.step >= self.config.training.total_steps:
                break
        
        # Close progress bar
        pbar.close()
        
        # Average metrics
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            avg_metrics[key] = np.mean(values)
        # Record history for plotting/inspection
        try:
            # Ensure training_history exists and append epoch metrics
            if not hasattr(self, 'training_history') or self.training_history is None:
                self.training_history = []
            self.training_history.append(avg_metrics)
        except Exception:
            pass

        self.epoch += 1
        return avg_metrics
    
    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """Evaluate the model."""
        self.policy_model.eval()
        
        all_prompts = []
        all_responses = []
        all_rewards = []
        all_references = []
        
        # If no evaluation dataloader or it's empty, return default zero metrics
        if not eval_dataloader:
            return {'avg_reward': 0.0, 'reward_std': 0.0}

        with torch.no_grad():
            pbar = tqdm(
                eval_dataloader,
                desc="Evaluating",
                unit="batch",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            
            for batch in pbar:
                prompts = batch.get('prompts', [])

                # Generate responses
                generation_output = self.generate_responses(prompts)
                responses = generation_output['response_texts']

                # Compute rewards
                rewards = self.compute_rewards(prompts, responses)

                all_prompts.extend(prompts)
                all_responses.extend(responses)
                all_rewards.extend(rewards.tolist())
                if 'references' in batch:
                    all_references.extend(batch.get('references', []))
                
                # Update progress bar
                pbar.set_postfix({
                    'processed': len(all_responses),
                    'avg_reward': f'{np.mean(all_rewards):.4f}' if all_rewards else '0.0000'
                })
            
            pbar.close()
        
        # Compute evaluation metrics
        eval_metrics = {}
        if all_rewards:
            eval_metrics['avg_reward'] = float(np.mean(all_rewards))
            eval_metrics['reward_std'] = float(np.std(all_rewards))
        else:
            eval_metrics['avg_reward'] = 0.0
            eval_metrics['reward_std'] = 0.0
        
        # If references exist but lengths mismatch, align to the minimum length and warn
        if all_references and len(all_references) != len(all_responses):
            logger.warning(f"Mismatch between number of references ({len(all_references)}) and responses ({len(all_responses)}). Trimming to min length.")
            n = min(len(all_references), len(all_responses))
            all_references = all_references[:n]
            all_responses = all_responses[:n]

        if all_references and len(all_references) == len(all_responses):
            metrics_results = self.metrics_evaluator.compute_all_metrics(all_responses, all_references)

        # Compute evaluation metrics
        eval_metrics = {}
        eval_metrics['avg_reward'] = np.mean(all_rewards)
        eval_metrics['reward_std'] = np.std(all_rewards)
        
        # Compute other metrics if references are available
        if all_references and len(all_references) == len(all_responses):
            metrics_results = self.metrics_evaluator.compute_all_metrics(all_responses, all_references)
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
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                logits = outputs.logits
            elif hasattr(model, 'lm_head') and hasattr(outputs, 'last_hidden_state'):
                logits = model.lm_head(outputs.last_hidden_state)
            else:
                emb = model.get_output_embeddings() if hasattr(model, 'get_output_embeddings') else None
                if emb is not None and hasattr(outputs, 'last_hidden_state'):
                    logits = torch.matmul(outputs.last_hidden_state, emb.weight.t())
                else:
                    raise AttributeError("Unable to obtain logits from model outputs")
        
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
        import time
        start_time = time.time()
        
        logger.info("Starting training...")
        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"  Total steps: {self.config.training.total_steps}")
        print(f"  PPO epochs: {self.config.training.ppo_epochs}")
        print(f"  Batch size: {self.config.training.batch_size}")
        print(f"  Gradient accumulation: {self.config.training.gradient_accumulation_steps}")
        print(f"  Device: {self.config.hardware.device}")
        print(f"{'='*60}\n")
        
        best_metrics = {}
        patience_counter = 0
        
        # Main training loop with progress tracking
        for epoch in range(self.config.training.ppo_epochs):
            epoch_start = time.time()
            print(f"\n[Epoch {epoch+1}/{self.config.training.ppo_epochs}]")
            
            # Training
            train_metrics = self.trainer.train_epoch(train_dataloader)
            epoch_time = time.time() - epoch_start
            
            # Evaluation
            if eval_dataloader is not None:
                print(f"\n[Evaluation]")
                eval_start = time.time()
                eval_metrics = self.trainer.evaluate(eval_dataloader)
                eval_time = time.time() - eval_start
                
                # Store evaluation metrics for plotting
                epoch_eval_metrics = {
                    'epoch': epoch + 1,
                    **eval_metrics
                }
                # Store in trainer's evaluation history
                if hasattr(self.trainer, 'evaluation_history'):
                    self.trainer.evaluation_history.append(epoch_eval_metrics)
                
                # Check for improvement
                if eval_metrics.get('avg_reward', 0) > best_metrics.get('avg_reward', -float('inf')):
                    best_metrics = eval_metrics
                    patience_counter = 0
                    self.trainer.save_checkpoint()
                    print(f"  New best reward: {best_metrics.get('avg_reward', 0):.4f} - Checkpoint saved!")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.config.training.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    print(f"\n[Early Stopping] No improvement for {patience_counter} epochs")
                    break
                
                print(f"  Eval time: {eval_time:.1f}s | Reward: {eval_metrics.get('avg_reward', 0):.4f}")
                # Print evaluation metrics if available
                if any(k.startswith('eval_') for k in eval_metrics.keys()):
                    print(f"  Metrics: ", end="")
                    metric_strs = []
                    for k in ['eval_bertscore', 'eval_codebleu', 'eval_bleu', 'eval_rouge', 'eval_ruby']:
                        if k in eval_metrics:
                            metric_strs.append(f"{k.replace('eval_', '')}={eval_metrics[k]:.4f}")
                    print(", ".join(metric_strs))
            
            # Calculate progress
            total_time = time.time() - start_time
            progress = ((epoch + 1) / self.config.training.ppo_epochs) * 100
            avg_time_per_epoch = total_time / (epoch + 1)
            remaining_epochs = self.config.training.ppo_epochs - (epoch + 1)
            eta_seconds = avg_time_per_epoch * remaining_epochs
            
            # Log metrics
            print(f"\n[Epoch {epoch+1} Summary]")
            print(f"  Time: {epoch_time:.1f}s | Total: {total_time/60:.1f}min")
            print(f"  Progress: {progress:.1f}% | ETA: {eta_seconds/60:.1f}min")
            print(f"  Train metrics: loss={train_metrics.get('loss', 0):.4f}, reward={train_metrics.get('reward', 0):.4f}")
            logger.info(f"Epoch {epoch+1}: {train_metrics}")
            if eval_dataloader is not None:
                logger.info(f"Eval metrics: {eval_metrics}")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"{'='*60}\n")
        
        return best_metrics
