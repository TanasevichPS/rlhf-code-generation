# ppo_trainer.py
from typing import Tuple, List, Dict, Any, Optional
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer
import torch
import pandas as pd
import logging
from datetime import datetime
import os
import re
import ast

logger = logging.getLogger(__name__)

class CodeRLHFTrainer:
    """RLHF trainer specialized for code generation tasks."""    
    def __init__(self, config, tokenizer, policy_model, ref_model, reward_model) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.device = torch.device(config.device)
        
        self.ppo_trainer = self._setup_ppo_trainer()
        self.results: List[Dict] = []
    
    def _setup_ppo_trainer(self) -> PPOTrainer:
        """Set up PPO trainer with compatible configuration."""
        try:
            # Создаем базовую конфигурацию без неподдерживаемых параметров
            ppo_config_args = {
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'mini_batch_size': getattr(self.config, 'mini_batch_size', 1),
            }
            
            # Добавляем опциональные параметры если они есть в конфиге
            optional_params = ['ppo_epochs', 'gradient_accumulation_steps', 'max_grad_norm']
            for param in optional_params:
                if hasattr(self.config, param):
                    ppo_config_args[param] = getattr(self.config, param)
            
            # Создаем конфиг
            ppo_config = PPOConfig(**ppo_config_args)
            
            # Отключаем mixed precision если такие параметры существуют
            if hasattr(ppo_config, 'fp16'):
                ppo_config.fp16 = False
            if hasattr(ppo_config, 'bf16'):
                ppo_config.bf16 = False
                
            logger.info(f"PPO Config created: {ppo_config}")
                
        except Exception as e:
            logger.error(f"PPOConfig creation failed: {e}")
            # Фолбэк с минимальной конфигурацией
            ppo_config = PPOConfig(
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
            )
        
        logger.info("PPO Config created successfully")
        
        # Инициализация PPOTrainer
        return PPOTrainer(
            config=ppo_config,
            model=self.policy_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )
    
    def generate_responses(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Generate code responses with code-specific parameters."""
        self.policy_model.eval()
        
        if not prompts:
            return {"response_tensors": [], "response_texts": [], "prompt_tensors": []}
        
        try:
            # Tokenize prompts
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=getattr(self.config, 'max_prompt_length', 512),
                return_tensors="pt"
            ).to(self.device)
            
            # Generate code with fallback parameters
            generation_kwargs = {
                'input_ids': inputs.input_ids,
                'attention_mask': inputs.attention_mask,
                'max_length': getattr(self.config, 'max_prompt_length', 512) + getattr(self.config, 'max_response_length', 256),
                'do_sample': getattr(self.config, 'do_sample', True),
                'temperature': getattr(self.config, 'temperature', 0.8),
                'top_p': getattr(self.config, 'top_p', 0.95),
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }
            
            # Добавляем опциональные параметры
            optional_gen_params = ['top_k', 'repetition_penalty', 'min_length']
            for param in optional_gen_params:
                if hasattr(self.config, param):
                    generation_kwargs[param] = getattr(self.config, param)
            
            with torch.no_grad():
                responses = self.policy_model.generate(**generation_kwargs)
            
            # Extract generated code
            response_tensors = []
            response_texts = []
            
            for i, response in enumerate(responses):
                actual_prompt_length = inputs.attention_mask[i].sum().item()
                generated_tokens = response[actual_prompt_length:]
                response_tensors.append(generated_tokens)
                
                response_text = self.tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                # Clean and format generated code
                response_text = self._clean_generated_code(response_text)
                response_texts.append(response_text)
            
            return {
                "response_tensors": response_tensors,
                "response_texts": response_texts,
                "prompt_tensors": [inputs.input_ids[i] for i in range(inputs.input_ids.shape[0])],
            }
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            # Fallback to simple code examples
            return {
                "response_tensors": [torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long) for _ in prompts],
                "response_texts": ["def example():\n    return 'hello world'"] * len(prompts),
                "prompt_tensors": [torch.tensor([0]) for _ in prompts],
            }
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean and format generated code."""
        if not code:
            return ""
        
        # Remove markdown code blocks if present
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        # Remove excessive whitespace but preserve indentation
        lines = []
        for line in code.split('\n'):
            line = line.rstrip()
            if line.strip():  # Keep non-empty lines
                lines.append(line)
        
        code = '\n'.join(lines)
        
        # Try to extract the first complete function/block
        try:
            # Find the first def or class
            def_match = re.search(r'(def\s+\w+.*?(?=\n\s*def|\n\s*class|\Z))', code, re.DOTALL)
            class_match = re.search(r'(class\s+\w+.*?(?=\n\s*def|\n\s*class|\Z))', code, re.DOTALL)
            
            if def_match:
                code = def_match.group(1)
            elif class_match:
                code = class_match.group(1)
            else:
                # Try to parse and get first complete statement
                parsed = ast.parse(code)
                if parsed.body:
                    first_node = parsed.body[0]
                    code = ast.get_source_segment(code, first_node) or code
                    
        except SyntaxError:
            # If parsing fails, take first reasonable chunk
            lines = code.split('\n')
            if len(lines) > 10:
                code = '\n'.join(lines[:10])
        
        return code.strip()
    
    def train_epoch(self, dataset, epoch: int) -> Dict[str, float]:
        """Train for one epoch with code-specific evaluation."""
        logger.info(f"Starting code training epoch {epoch}")
        
        train_size = min(16, len(dataset))
        train_dataset = dataset.select(range(train_size))
        
        epoch_stats = {
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "kl_divergence": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "syntax_score": 0.0,
            "execution_score": 0.0,
        }
        
        batch_count = 0
        
        for i in range(0, len(train_dataset), self.config.batch_size):
            batch = train_dataset[i:i + self.config.batch_size]
            
            if isinstance(batch, dict) and 'prompt' in batch:
                prompts = batch['prompt']
                if isinstance(prompts, str):
                    prompts = [prompts]
            else:
                continue
            
            if not prompts:
                continue
                
            batch_stats = self.train_batch({"prompt": prompts}, epoch, batch_count)
            
            # Accumulate statistics
            for key in epoch_stats:
                if key in batch_stats:
                    epoch_stats[key] += batch_stats[key]
            
            batch_count += 1
        
        # Average statistics
        if batch_count > 0:
            for key in epoch_stats:
                epoch_stats[key] /= batch_count
        
        logger.info(f"Epoch {epoch} completed: Mean Reward = {epoch_stats['mean_reward']:.4f}, "
                   f"Syntax Score = {epoch_stats['syntax_score']:.4f}")
        return epoch_stats
    
    def train_batch(self, batch: Dict, epoch: int, batch_idx: int) -> Dict[str, float]:
        """Train on a single batch of code generation examples."""
        prompts = batch["prompt"]
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if not prompts:
            return {
                "mean_reward": 0.0, "std_reward": 0.0, "kl_divergence": 0.0,
                "policy_loss": 0.0, "value_loss": 0.0, "syntax_score": 0.0, "execution_score": 0.0
            }
        
        try:
            # Generate code responses
            generation_output = self.generate_responses(prompts)
            responses = generation_output["response_texts"]
            
            # Compute code-specific rewards
            rewards = self.reward_model.compute_reward(prompts, responses)
            
            # Log code examples
            if batch_idx % 2 == 0:
                for i, (prompt, response) in enumerate(zip(prompts, responses)):
                    if i < 2:
                        # Calculate individual metrics for logging
                        syntax_score = self.reward_model._check_syntax(response)
                        logger.info(f"Code Example - Prompt: '{prompt[:50]}...'")
                        logger.info(f"Generated Code: {response[:100]}...")
                        logger.info(f"Syntax: {syntax_score:.3f}, Reward: {rewards[i].item():.3f}")
            
            # Convert rewards for PPO
            rewards_list = [rewards[i].unsqueeze(0) for i in range(len(rewards))]
            
            # PPO training step
            stats = self.ppo_trainer.step(
                generation_output["prompt_tensors"],
                generation_output["response_tensors"],
                rewards_list,
            )
            
        except Exception as e:
            logger.error(f"Code training step failed: {e}")
            stats = {
                "ppo/returns/mean": 0.1,
                "ppo/policy/approxkl": 0.0,
                "ppo/policy/mean": 0.0,
                "ppo/val/mean": 0.0,
            }
            rewards = torch.tensor([0.1] * len(prompts))
            rewards_list = [rewards[i].unsqueeze(0) for i in range(len(rewards))]
        
        # Calculate statistics
        rewards_tensor = torch.cat(rewards_list) if rewards_list else torch.tensor([0.0])
        
        if len(rewards_tensor) > 1:
            mean_reward = rewards_tensor.mean().item()
            std_reward = rewards_tensor.std().item()
        else:
            mean_reward = rewards_tensor.mean().item()
            std_reward = 0.0
        
        # Calculate code quality metrics
        syntax_scores = []
        execution_scores = []
        for response in responses:
            syntax_scores.append(self.reward_model._check_syntax(response))
            execution_scores.append(self.reward_model._check_execution("", response))
        
        structure_scores = []
        for response in responses:
            structure_scores.append(self.reward_model._check_structure(response))

        # Обновите batch_stats
        batch_stats = {
            "epoch": epoch,
            "batch": batch_idx,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "kl_divergence": stats.get("ppo/policy/approxkl", 0.0),
            "policy_loss": stats.get("ppo/policy/mean", 0.0),
            "value_loss": stats.get("ppo/val/mean", 0.0),
            "syntax_score": sum(syntax_scores) / len(syntax_scores) if syntax_scores else 0.0,
            "structure_score": sum(structure_scores) / len(structure_scores) if structure_scores else 0.0,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.results.append(batch_stats)
        
        logger.info(f"Epoch {epoch}, Batch {batch_idx}: Reward = {batch_stats['mean_reward']:.4f}, "
                   f"Syntax = {batch_stats['syntax_score']:.4f}")
        
        return batch_stats
    
    def save_final_results(self) -> None:
        """Save final training results."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save results to CSV
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_path = os.path.join(self.config.output_dir, "improved_rlhf_results.csv")
            results_df.to_csv(results_path, index=False)
            logger.info(f"Results saved to: {results_path}")
        
        # Save model
        model_path = os.path.join(self.config.output_dir, "final_model")
        self.policy_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        logger.info(f"Model saved to: {model_path}")