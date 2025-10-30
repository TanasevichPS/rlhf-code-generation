"""
Modern Reward Model with Human Feedback Integration
=================================================

A state-of-the-art reward model that combines multiple signals:
- Syntax correctness
- Execution success
- Semantic similarity
- Human preference feedback
- Code quality metrics
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
import ast
import subprocess
import tempfile

from .metrics import ModernMetricsEvaluator, CodeQualityAnalyzer
from .config import RewardConfig

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Container for different reward components."""
    syntax_reward: float = 0.0
    execution_reward: float = 0.0
    semantic_reward: float = 0.0
    human_preference_reward: float = 0.0
    quality_reward: float = 0.0
    total_reward: float = 0.0


class HumanFeedbackIntegrator:
    """Integrates human feedback into reward computation."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.human_logits_cache = {}
        self.feedback_weights = {}
        
    def load_human_feedback(self, feedback_path: str):
        """Load human feedback data from file."""
        try:
            if os.path.exists(feedback_path):
                with open(feedback_path, 'r') as f:
                    feedback_data = json.load(f)
                
<<<<<<< HEAD
<<<<<<< HEAD
                # Normalize to list of dicts
                items = []
                if isinstance(feedback_data, list):
                    items = feedback_data
                elif isinstance(feedback_data, dict):
                    for key in ['data', 'items', 'examples', 'feedback']:
                        v = feedback_data.get(key)
                        if isinstance(v, list):
                            items = v
                            break
                
                # Process feedback data
                for item in items:
                    if not isinstance(item, dict):
                        continue
=======
                # Process feedback data
                for item in feedback_data:
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
                # Process feedback data
                for item in feedback_data:
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
                    prompt = item.get('prompt', '')
                    response = item.get('response', '')
                    rating = item.get('rating', 0.0)
                    logits = item.get('logits', None)
                    
                    # Store human logits if available
                    if logits and self.config.use_human_logits:
                        key = f"{prompt[:50]}_{response[:50]}"
                        self.human_logits_cache[key] = {
                            'logits': logits,
                            'rating': rating
                        }
                
                logger.info(f"Loaded {len(self.human_logits_cache)} human feedback entries")
                
        except Exception as e:
            logger.warning(f"Failed to load human feedback: {e}")
    
    def get_human_logits(self, prompt: str, response: str) -> Optional[torch.Tensor]:
        """Get human logits for a prompt-response pair."""
        key = f"{prompt[:50]}_{response[:50]}"
        
        if key in self.human_logits_cache:
            logits_data = self.human_logits_cache[key]['logits']
            if isinstance(logits_data, list):
                return torch.tensor(logits_data, dtype=torch.float32)
            elif isinstance(logits_data, dict):
                # Handle different logits formats
                if 'last_layer' in logits_data:
                    return torch.tensor(logits_data['last_layer'], dtype=torch.float32)
                elif 'logits' in logits_data:
                    return torch.tensor(logits_data['logits'], dtype=torch.float32)
        
        return None
    
    def compute_human_preference_reward(self, prompt: str, response: str) -> float:
        """Compute reward based on human preferences."""
        key = f"{prompt[:50]}_{response[:50]}"
        
        if key in self.human_logits_cache:
            rating = self.human_logits_cache[key]['rating']
            # Normalize rating to [0, 1] range
            return max(0.0, min(1.0, rating / 5.0))  # Assuming 5-point scale
        
        return 0.5  # Neutral reward if no human feedback available


class SyntaxChecker:
    """Advanced syntax checking for code."""
    
    def __init__(self):
        self.supported_languages = ['python', 'javascript', 'java', 'cpp']
    
    def check_syntax(self, code: str, language: str = 'python') -> Tuple[bool, float, str]:
        """Check syntax correctness of code."""
        if language == 'python':
            return self._check_python_syntax(code)
        else:
            # For other languages, use basic parsing
            return self._check_generic_syntax(code)
    
    def _check_python_syntax(self, code: str) -> Tuple[bool, float, str]:
        """Check Python syntax."""
        try:
            ast.parse(code)
            return True, 1.0, ""
        except SyntaxError as e:
            return False, 0.0, str(e)
        except Exception as e:
            return False, 0.0, f"Parse error: {str(e)}"
    
    def _check_generic_syntax(self, code: str) -> Tuple[bool, float, str]:
        """Generic syntax checking."""
        # Basic checks
        if not code.strip():
            return False, 0.0, "Empty code"
        
        # Check for balanced brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return False, 0.0, "Unbalanced brackets"
                if brackets[stack.pop()] != char:
                    return False, 0.0, "Unbalanced brackets"
        
        if stack:
            return False, 0.0, "Unbalanced brackets"
        
        return True, 0.8, ""  # Partial credit for basic structure


class ExecutionTester:
    """Test code execution in a safe environment."""
    
    def __init__(self):
        self.timeout = 5  # seconds
        self.max_memory = 100 * 1024 * 1024  # 100MB
    
    def test_execution(self, code: str, language: str = 'python') -> Tuple[bool, float, str]:
        """Test if code can be executed successfully."""
        if language == 'python':
            return self._test_python_execution(code)
        else:
            return False, 0.0, f"Execution testing not supported for {language}"
    
    def _test_python_execution(self, code: str) -> Tuple[bool, float, str]:
        """Test Python code execution."""
        try:
            # Create a safe execution environment
            safe_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round,
                    'int': int,
                    'float': float,
                    'str': str,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'bool': bool,
                    'type': type,
                    'isinstance': isinstance,
                    'hasattr': hasattr,
                    'getattr': getattr,
                    'setattr': setattr,
                }
            }
            
            # Try to execute
            exec(code, safe_globals)
            return True, 1.0, ""
            
        except Exception as e:
            return False, 0.0, str(e)


class ModernRewardModel(nn.Module):
    """Modern reward model with multiple signal integration."""
    
    def __init__(self, config: RewardConfig, model_name: str = "microsoft/codebert-base"):
        super().__init__()
        self.config = config
        self.model_name = model_name
        
        # Load base model
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Reward head
        hidden_size = self.base_model.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Component-specific heads
        self.syntax_head = nn.Linear(hidden_size, 1)
        self.execution_head = nn.Linear(hidden_size, 1)
        self.semantic_head = nn.Linear(hidden_size, 1)
        self.quality_head = nn.Linear(hidden_size, 1)
        
        # Human feedback integration
        self.human_feedback_integrator = HumanFeedbackIntegrator(config)
        
        # Utility components
        self.syntax_checker = SyntaxChecker()
        self.execution_tester = ExecutionTester()
        self.metrics_evaluator = ModernMetricsEvaluator()
        self.code_analyzer = CodeQualityAnalyzer()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, prompts: List[str], responses: List[str]) -> Dict[str, torch.Tensor]:
        """Forward pass through the reward model."""
        # Tokenize inputs
        inputs = self._tokenize_pairs(prompts, responses)
        
        # Get base model outputs
        with torch.no_grad():
            outputs = self.base_model(**inputs)
        
        # Get pooled representation
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        # Compute different reward components
        rewards = {}
        rewards['total'] = self.reward_head(pooled_output)
        rewards['syntax'] = self.syntax_head(pooled_output)
        rewards['execution'] = self.execution_head(pooled_output)
        rewards['semantic'] = self.semantic_head(pooled_output)
        rewards['quality'] = self.quality_head(pooled_output)
        
        return rewards
    
    def _tokenize_pairs(self, prompts: List[str], responses: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize prompt-response pairs."""
        # Combine prompts and responses
        texts = [f"{prompt} <SEP> {response}" for prompt, response in zip(prompts, responses)]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return inputs
    
    def compute_reward_components(self, prompts: List[str], responses: List[str]) -> List[RewardComponents]:
        """Compute detailed reward components for each prompt-response pair."""
        components_list = []
        
        for prompt, response in zip(prompts, responses):
            components = RewardComponents()
            
            # Syntax reward
            syntax_correct, syntax_score, syntax_error = self.syntax_checker.check_syntax(response)
            components.syntax_reward = syntax_score
            
            # Execution reward
            exec_success, exec_score, exec_error = self.execution_tester.test_execution(response)
            components.execution_reward = exec_score
            
            # Semantic reward (using BERTScore)
            try:
                semantic_result = self.metrics_evaluator.compute_bertscore([response], [prompt])
                components.semantic_reward = semantic_result.score
            except Exception as e:
                logger.warning(f"Semantic reward computation failed: {e}")
                components.semantic_reward = 0.0
            
            # Human preference reward
            components.human_preference_reward = self.human_feedback_integrator.compute_human_preference_reward(
                prompt, response
            )
            
            # Quality reward
            try:
                quality_analysis = self.code_analyzer.analyze_complexity(response)
                style_analysis = self.code_analyzer.analyze_style(response)
                components.quality_reward = (
                    quality_analysis['complexity_score'] * 0.6 +
                    style_analysis['style_score'] * 0.4
                )
            except Exception as e:
                logger.warning(f"Quality reward computation failed: {e}")
                components.quality_reward = 0.0
            
            # Compute total reward
            components.total_reward = (
                components.syntax_reward * self.config.syntax_reward_weight +
                components.execution_reward * self.config.execution_reward_weight +
                components.semantic_reward * self.config.semantic_reward_weight +
                components.human_preference_reward * self.config.human_preference_weight +
                components.quality_reward * 0.1  # Small weight for quality
            )
            
            components_list.append(components)
        
        return components_list
    
    def compute_reward(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute final reward scores."""
        # Get neural network predictions
        neural_rewards = self.forward(prompts, responses)
        
        # Get component-based rewards
        component_rewards = self.compute_reward_components(prompts, responses)
        
        # Combine neural and component rewards
        final_rewards = []
        for i, (neural_reward, component_reward) in enumerate(zip(neural_rewards['total'], component_rewards)):
            # Weighted combination
            combined_reward = (
                neural_reward.item() * 0.7 +  # Neural network prediction
                component_reward.total_reward * 0.3  # Component-based reward
            )
            
            # Apply normalization and clipping
            if self.config.reward_normalization:
                combined_reward = torch.sigmoid(torch.tensor(combined_reward))
            
            if self.config.reward_clipping:
                combined_reward = torch.clamp(
                    torch.tensor(combined_reward),
                    -self.config.reward_clip_value,
                    self.config.reward_clip_value
                )
            
            final_rewards.append(combined_reward.item())
        
        return torch.tensor(final_rewards, dtype=torch.float32)
    
    def load_human_feedback(self, feedback_path: str):
        """Load human feedback data."""
        self.human_feedback_integrator.load_human_feedback(feedback_path)
    
    def save_model(self, save_path: str):
        """Save the reward model."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(save_path, "reward_model.pt"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        config_dict = {
            "model_name": self.model_name,
            "reward_config": self.config.__dict__
        }
        with open(os.path.join(save_path, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Reward model saved to {save_path}")
    
    @classmethod
    def load_model(cls, load_path: str, config: RewardConfig):
        """Load a saved reward model."""
        # Load config
        with open(os.path.join(load_path, "config.json"), 'r') as f:
            config_dict = json.load(f)
        
        # Create model
        model = cls(config, config_dict["model_name"])
        
        # Load state dict
        state_dict = torch.load(os.path.join(load_path, "reward_model.pt"))
        model.load_state_dict(state_dict)
        
        # Load tokenizer
        model.tokenizer = AutoTokenizer.from_pretrained(load_path)
        
        logger.info(f"Reward model loaded from {load_path}")
        return model


class RewardModelTrainer:
    """Trainer for the reward model."""
    
    def __init__(self, model: ModernRewardModel, config: RewardConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.reward_learning_rate,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.reward_epochs
        )
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        prompts = batch['prompts']
        responses = batch['responses']
        human_ratings = batch.get('human_ratings', None)
        
        # Compute rewards
        predicted_rewards = self.model.compute_reward(prompts, responses)
        
        # Compute loss
        if human_ratings is not None:
            # Use human ratings as targets
            target_rewards = torch.tensor(human_ratings, dtype=torch.float32)
            loss = F.mse_loss(predicted_rewards, target_rewards)
        else:
            # Use component-based rewards as targets
            component_rewards = self.model.compute_reward_components(prompts, responses)
            target_rewards = torch.tensor([c.total_reward for c in component_rewards], dtype=torch.float32)
            loss = F.mse_loss(predicted_rewards, target_rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'predicted_reward_mean': predicted_rewards.mean().item(),
            'predicted_reward_std': predicted_rewards.std().item()
        }
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = []
        
        for batch in dataloader:
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)
        
        # Average metrics
<<<<<<< HEAD
<<<<<<< HEAD
        if not epoch_metrics:
            return {'loss': 0.0, 'predicted_reward_mean': 0.0, 'predicted_reward_std': 0.0}
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
=======
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
        
<<<<<<< HEAD
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
=======
>>>>>>> e965bd9110c8eb4f5e1fc4df091eb3a8fa94a0f1
        return avg_metrics
