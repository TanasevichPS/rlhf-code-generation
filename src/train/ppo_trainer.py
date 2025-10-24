# ppo_trainer.py
from typing import Tuple, List, Dict, Any, Optional
try:
    # Delay TRL import to runtime when PPO training is actually used.
    from transformers import AutoTokenizer
    from trl import PPOTrainer, PPOConfig
except Exception:
    # Fallback: define placeholders so module imports succeed in environments
    PPOTrainer = None
    PPOConfig = None
    from transformers import AutoTokenizer
import torch
import pandas as pd
import logging
from datetime import datetime
import os
import re
import ast
from src.metrics_tracker import MetricsTracker
import types


# Lightweight PPO stub used when `trl` is not installed or fails to import.
class PPOStub:
    def __init__(self, config=None, model=None, tokenizer=None, ref_model=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self._steps = 0

    def step(self, prompt_tensors, response_tensors, rewards_list):
        # Minimal no-op step that returns a stable stats dict
        self._steps += 1
        return {'step': self._steps, 'loss': 0.0, 'info': 'ppo_stub'}

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
        self.metrics_tracker = MetricsTracker(output_dir=config.output_dir)
        
        self.ppo_trainer = self._setup_ppo_trainer()
        self.results: List[Dict] = []
    
    def _setup_ppo_trainer(self):
        """Set up PPO trainer with compatible configuration."""
        try:
            # Base arguments we would like to pass to PPOConfig
            ppo_config_args = {
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'mini_batch_size': getattr(self.config, 'mini_batch_size', 1),
                'ppo_epochs': getattr(self.config, 'ppo_epochs', 2),
                'cliprange': 0.2,
                'cliprange_value': 0.2,
                'vf_coef': 0.5,
                'ent_coef': 0.01,
                'target_kl': 0.1,
                'init_kl_coef': 0.2,  # initial KL coeff
            }

            optional_params = ['gradient_accumulation_steps', 'max_grad_norm']
            for param in optional_params:
                if hasattr(self.config, param):
                    ppo_config_args[param] = getattr(self.config, param)

            # If TRL's PPOConfig is available, be defensive: introspect its __init__
            # signature and only pass supported keyword arguments. Also support a
            # small mapping of common names (e.g. ppo_epochs -> epochs) to keep
            # compatibility with multiple TRL versions.
            if PPOConfig is not None:
                import inspect

                sig = inspect.signature(PPOConfig.__init__)
                allowed = {p.name for p in sig.parameters.values() if p.name != 'self' and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}

                # Common mapping of our preferred names -> older/newer names in PPOConfig
                name_map = {
                    'ppo_epochs': 'epochs',
                    'learning_rate': 'lr',
                }

                # Build filtered args
                filtered_args = {}
                for k, v in ppo_config_args.items():
                    if k in allowed:
                        filtered_args[k] = v
                    else:
                        mapped = name_map.get(k)
                        if mapped and mapped in allowed:
                            filtered_args[mapped] = v

                ppo_config = PPOConfig(**filtered_args)
                # Ensure mixed precision flags disabled if present
                if hasattr(ppo_config, 'fp16'):
                    ppo_config.fp16 = False
                if hasattr(ppo_config, 'bf16'):
                    ppo_config.bf16 = False
                logger.info(f"PPO Config created: {ppo_config}")
            else:
                # Fallback SimpleNamespace config for PPOStub
                logger.warning('TRL PPOConfig not available; using PPOStub config fallback')
                ppo_config = types.SimpleNamespace(**ppo_config_args)

        except Exception as e:
            logger.error(f"PPOConfig creation failed: {e}")
            # Minimal fallback config
            if PPOConfig is not None:
                try:
                    # Try a minimal set of arguments that are commonly supported
                    ppo_config = PPOConfig(
                        learning_rate=self.config.learning_rate,
                        batch_size=self.config.batch_size,
                        mini_batch_size=1,
                    )
                except Exception:
                    ppo_config = types.SimpleNamespace(learning_rate=self.config.learning_rate, batch_size=self.config.batch_size, mini_batch_size=1)
            else:
                ppo_config = types.SimpleNamespace(learning_rate=self.config.learning_rate, batch_size=self.config.batch_size, mini_batch_size=1)

        logger.info("PPO Config created successfully")

        # Initialize trainer; use stub if TRL not available
        try:
            if PPOTrainer is not None:
                import inspect
                try:
                    # Try to call PPOTrainer using supported keyword args when
                    # available. Different TRL versions use different signatures,
                    # so we introspect __init__ and adapt.
                    init_sig = inspect.signature(PPOTrainer.__init__)
                    params = [p.name for p in init_sig.parameters.values() if p.name != 'self']

                    kwargs = {}
                    if 'config' in params:
                        kwargs['config'] = ppo_config
                    if 'model' in params:
                        kwargs['model'] = self.policy_model
                    if 'ref_model' in params:
                        kwargs['ref_model'] = self.ref_model
                    if 'tokenizer' in params:
                        kwargs['tokenizer'] = self.tokenizer

                    # Attempt keyword-based init first
                    try:
                        return PPOTrainer(**kwargs)
                    except Exception as e_kw:
                        logger.error(f"PPOTrainer keyword init failed: {e_kw}")
                        # Try positional ordering fallback
                        args = []
                        for name in ['model', 'ref_model', 'tokenizer', 'config']:
                            if name in params:
                                if name == 'model':
                                    args.append(self.policy_model)
                                elif name == 'ref_model':
                                    args.append(self.ref_model)
                                elif name == 'tokenizer':
                                    args.append(self.tokenizer)
                                elif name == 'config':
                                    args.append(ppo_config)
                        return PPOTrainer(*args)
                except Exception as e:
                    logger.error(f"PPOTrainer initialization failed with ref_model: {e}")
                    # Fall back to trying a minimal keyword set
                    try:
                        return PPOTrainer(model=self.policy_model, tokenizer=self.tokenizer)
                    except Exception as e2:
                        logger.error(f"PPOTrainer minimal init failed: {e2}")
                        raise
            else:
                # Use stub implementation
                logger.warning('PPOTrainer (trl) unavailable — using PPOStub. PPO steps will be no-ops.')
                return PPOStub(config=ppo_config, model=self.policy_model, tokenizer=self.tokenizer, ref_model=self.ref_model)
        except Exception as e:
            logger.error(f"PPOTrainer final fallback failed: {e}")
            return PPOStub(config=ppo_config, model=self.policy_model, tokenizer=self.tokenizer, ref_model=self.ref_model)
    
    def generate_responses(self, prompts: List[str]) -> Dict[str, Any]:
        """Generate code responses with proper tensor formatting."""
        self.policy_model.eval()
        
        if not prompts:
            return {"response_tensors": [], "response_texts": [], "prompt_tensors": []}
        
        try:
            # Tokenize prompts
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=getattr(self.config, 'max_prompt_length', 256),
                return_tensors="pt"
            ).to(self.device)
            
            generation_kwargs = {
                'max_new_tokens': getattr(self.config, 'max_response_length', 128),
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'repetition_penalty': 1.1,
            }
            
            with torch.no_grad():
                responses = self.policy_model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **generation_kwargs
                )
            
            # Extract generated code - ensure we return lists of tensors
            response_tensors = []
            response_texts = []
            prompt_tensors = []
            
            for i, response in enumerate(responses):
                input_length = inputs.input_ids[i].shape[0]
                generated_tokens = response[input_length:]
                
                # Ensure we have proper tensor format
                response_tensors.append(generated_tokens.cpu())
                prompt_tensors.append(inputs.input_ids[i].cpu())
                
                response_text = self.tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                response_text = self._clean_generated_code(response_text)
                response_texts.append(response_text)
            
            return {
                "response_tensors": response_tensors,  # List of tensors
                "response_texts": response_texts,
                "prompt_tensors": prompt_tensors,      # List of tensors
            }
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            # Return proper format even in error case
            return {
                "response_tensors": [torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long) for _ in prompts],
                "response_texts": ["# Code generation failed"] * len(prompts),
                "prompt_tensors": [torch.tensor([0]) for _ in prompts],
            }
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean and format generated code."""
        if not code:
            return "# No code generated"
        
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        lines = []
        for line in code.split('\n'):
            line = line.rstrip()
            if line.strip():
                lines.append(line)
        
        return '\n'.join(lines).strip()
    
    def train_epoch(self, dataset, epoch: int) -> Dict[str, float]:
        """Train for one epoch with code-specific evaluation."""
        logger.info(f"Starting code training epoch {epoch}")
        
        train_size = min(4, len(dataset))
        train_dataset = dataset.select(range(train_size))
        
        epoch_stats = {
            "mean_reward": 0.0,
            "syntax_score": 0.0,
            "structure_score": 0.0,
        }
        
        batch_count = 0
        
        for i in range(0, len(train_dataset), max(1, self.config.batch_size)):
            batch = train_dataset[i:i + max(1, self.config.batch_size)]
            
            if isinstance(batch, dict) and 'prompt' in batch:
                prompts = batch['prompt']
                if isinstance(prompts, str):
                    prompts = [prompts]
            else:
                continue
            
            if not prompts:
                continue
                
            batch_stats = self.train_batch({"prompt": prompts}, epoch, batch_count)
            
            for key in epoch_stats:
                if key in batch_stats:
                    epoch_stats[key] += batch_stats[key]
            
            batch_count += 1
        
        if batch_count > 0:
            for key in epoch_stats:
                epoch_stats[key] /= batch_count
        
        logger.info(f"Epoch {epoch} completed: Mean Reward = {epoch_stats['mean_reward']:.4f}")
        return epoch_stats
    
    def train_batch(self, batch: Dict, epoch: int, batch_idx: int) -> Dict[str, float]:
        """Train on a single batch with proper PPO tensor formatting."""
        prompts = batch["prompt"]
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if not prompts:
            return {
                "mean_reward": 0.0, 
                "syntax_score": 0.0, 
                "structure_score": 0.0
            }
        
        batch_metrics = {
            "mean_reward": 0.0,
            "syntax_score": 0.0,
            "structure_score": 0.0,
        }
        
        try:
            # Generate code responses
            generation_output = self.generate_responses(prompts)
            responses = generation_output["response_texts"]
            
            # Compute code-specific rewards
            rewards = self.reward_model.compute_reward(prompts, responses)

            # Normalize rewards to stabilize PPO (batch norm)
            try:
                if isinstance(rewards, torch.Tensor):
                    r_mean = rewards.mean()
                    r_std = rewards.std(unbiased=False)
                    eps = 1e-6
                    rewards = (rewards - r_mean) / (r_std + eps)
                    # Clip to reasonable range
                    rewards = torch.clamp(rewards, -1.0, 1.0)
                else:
                    # Convert list-like to tensor and normalize
                    rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
                    r_mean = rewards.mean()
                    r_std = rewards.std(unbiased=False)
                    eps = 1e-6
                    rewards = (rewards - r_mean) / (r_std + eps)
                    rewards = torch.clamp(rewards, -1.0, 1.0)
            except Exception as e:
                logger.warning(f"Reward normalization failed: {e}")
            
            # Calculate syntax and structure scores
            syntax_scores = []
            structure_scores = []
            for response in responses:
                syntax_scores.append(getattr(self.reward_model, '_check_syntax', lambda x: 0.5)(response))
                structure_scores.append(getattr(self.reward_model, '_check_structure', lambda x: 0.5)(response))
            
            batch_metrics["syntax_score"] = sum(syntax_scores) / len(syntax_scores) if syntax_scores else 0.0
            batch_metrics["structure_score"] = sum(structure_scores) / len(structure_scores) if structure_scores else 0.0
            
            if hasattr(rewards, 'mean') and rewards.numel() > 0:
                batch_metrics["mean_reward"] = rewards.mean().item()
            else:
                batch_metrics["mean_reward"] = 0.1
            
            # Record metrics
            self.metrics_tracker.record_batch_metrics(
                epoch=epoch,
                batch=batch_idx,
                batch_stats=batch_metrics,
                prompts=prompts,
                generated_texts=responses
            )
            
            # PPO training step with proper tensor formatting
            try:
                # Convert rewards for PPO
                rewards_list = [torch.tensor([batch_metrics["mean_reward"]], device=self.device) for _ in prompts]
                
                # Get tensors and ensure they're on correct device
                prompt_tensors = [tensor.to(self.device) for tensor in generation_output["prompt_tensors"]]
                response_tensors = [tensor.to(self.device) for tensor in generation_output["response_tensors"]]
                
                # PPO training step
                stats = self.ppo_trainer.step(
                    prompt_tensors,
                    response_tensors,
                    rewards_list,
                )
                
                logger.info(f"PPO step completed: {stats}")
                
            except Exception as e:
                logger.warning(f"PPO step failed: {e}")
                # Continue without PPO but log the error
                
        except Exception as e:
            logger.error(f"Training step failed: {e}")
        
        batch_stats = {
            "epoch": epoch,
            "batch": batch_idx,
            **batch_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.results.append(batch_stats)
        
        logger.info(f"Epoch {epoch}, Batch {batch_idx}: Reward = {batch_stats['mean_reward']:.4f}")
        
        return batch_stats
    
    def save_final_results(self) -> None:
        """Save final training results and metrics."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_path = os.path.join(self.config.output_dir, "training_results.csv")
            results_df.to_csv(results_path, index=False)
            logger.info(f"Results saved to: {results_path}")
        
        self.metrics_tracker.save_detailed_metrics()
        
        summary = self.metrics_tracker.get_metrics_summary()
        if summary:
            logger.info("Training Metrics Summary:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value:.4f}")
        
        try:
            self.metrics_tracker.plot_metrics()
        except Exception as e:
            logger.warning(f"Could not plot metrics: {e}")
        
        try:
            model_path = os.path.join(self.config.output_dir, "final_model")
            self.policy_model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            logger.info(f"Model saved to: {model_path}")
        except Exception as e:
            logger.error(f"Could not save model: {e}")

# class CodeRLHFTrainer:
#     """RLHF trainer specialized for code generation tasks."""    
#     def __init__(self, config, tokenizer, policy_model, ref_model, reward_model) -> None:
#         self.config = config
#         self.tokenizer = tokenizer
#         self.policy_model = policy_model
#         self.ref_model = ref_model
#         self.reward_model = reward_model
#         self.device = torch.device(config.device)
#         self.metrics_tracker = MetricsTracker(output_dir=config.output_dir)
        
#         self.ppo_trainer = self._setup_ppo_trainer()
#         self.results: List[Dict] = []
    
#     def _setup_ppo_trainer(self) -> PPOTrainer:
#         """Set up PPO trainer with compatible configuration."""
#         try:
#             # Базовая конфигурация для совместимости
#             ppo_config_args = {
#                 'learning_rate': self.config.learning_rate,
#                 'batch_size': self.config.batch_size,
#                 'mini_batch_size': getattr(self.config, 'mini_batch_size', 1),
#                 'ppo_epochs': getattr(self.config, 'ppo_epochs', 2),  # Меньше эпох для стабильности
#             }
            
#             # Добавляем только поддерживаемые параметры
#             optional_params = ['gradient_accumulation_steps', 'max_grad_norm']
#             for param in optional_params:
#                 if hasattr(self.config, param):
#                     ppo_config_args[param] = getattr(self.config, param)
            
#             # Создаем конфиг
#             ppo_config = PPOConfig(**ppo_config_args)
            
#             # Отключаем проблемные параметры
#             if hasattr(ppo_config, 'fp16'):
#                 ppo_config.fp16 = False
#             if hasattr(ppo_config, 'bf16'):
#                 ppo_config.bf16 = False
                
#             logger.info(f"PPO Config created: {ppo_config}")
                
#         except Exception as e:
#             logger.error(f"PPOConfig creation failed: {e}")
#             # Фолбэк с минимальной конфигурацией
#             ppo_config = PPOConfig(
#                 learning_rate=self.config.learning_rate,
#                 batch_size=self.config.batch_size,
#                 mini_batch_size=1,
#                 ppo_epochs=2,
#             )
        
#         logger.info("PPO Config created successfully")
        
#         # Инициализация PPOTrainer с обработкой ошибок
#         try:
#             return PPOTrainer(
#                 config=ppo_config,
#                 model=self.policy_model,
#                 ref_model=self.ref_model,
#                 tokenizer=self.tokenizer,
#             )
#         except Exception as e:
#             logger.error(f"PPOTrainer initialization failed: {e}")
#             # Альтернативная инициализация без ref_model
#             return PPOTrainer(
#                 config=ppo_config,
#                 model=self.policy_model,
#                 tokenizer=self.tokenizer,
#             )
    
#     def generate_responses(self, prompts: List[str]) -> Dict[str, Any]:
#         """Generate code responses with error handling."""
#         self.policy_model.eval()
        
#         if not prompts:
#             return {"response_tensors": [], "response_texts": [], "prompt_tensors": []}
        
#         try:
#             # Tokenize prompts
#             inputs = self.tokenizer(
#                 prompts,
#                 padding=True,
#                 truncation=True,
#                 max_length=getattr(self.config, 'max_prompt_length', 256),  # Уменьшено для стабильности
#                 return_tensors="pt"
#             ).to(self.device)
            
#             # Упрощенные параметры генерации
#             generation_kwargs = {
#                 'max_new_tokens': getattr(self.config, 'max_response_length', 128),  # Уменьшено
#                 'do_sample': True,
#                 'temperature': 0.7,
#                 'top_p': 0.9,
#                 'pad_token_id': self.tokenizer.pad_token_id,
#                 'eos_token_id': self.tokenizer.eos_token_id,
#                 'repetition_penalty': 1.1,
#             }
            
#             with torch.no_grad():
#                 responses = self.policy_model.generate(
#                     inputs.input_ids,
#                     attention_mask=inputs.attention_mask,
#                     **generation_kwargs
#                 )
            
#             # Extract generated code
#             response_tensors = []
#             response_texts = []
            
#             for i, response in enumerate(responses):
#                 input_length = inputs.input_ids[i].shape[0]
#                 generated_tokens = response[input_length:]
#                 response_tensors.append(generated_tokens)
                
#                 response_text = self.tokenizer.decode(
#                     generated_tokens, 
#                     skip_special_tokens=True,
#                     clean_up_tokenization_spaces=True
#                 )
                
#                 # Clean and format generated code
#                 response_text = self._clean_generated_code(response_text)
#                 response_texts.append(response_text)
            
#             return {
#                 "response_tensors": response_tensors,
#                 "response_texts": response_texts,
#                 "prompt_tensors": inputs.input_ids,
#             }
            
#         except Exception as e:
#             logger.error(f"Code generation failed: {e}")
#             # Простой фолбэк
#             return {
#                 "response_tensors": [torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long) for _ in prompts],
#                 "response_texts": ["# Code generation failed"] * len(prompts),
#                 "prompt_tensors": [torch.tensor([0]) for _ in prompts],
#             }
    
#     def _clean_generated_code(self, code: str) -> str:
#         """Clean and format generated code."""
#         if not code:
#             return "# No code generated"
        
#         # Remove markdown code blocks
#         code = re.sub(r'```python\s*', '', code)
#         code = re.sub(r'```\s*', '', code)
        
#         # Remove excessive whitespace but preserve indentation
#         lines = []
#         for line in code.split('\n'):
#             line = line.rstrip()
#             if line.strip():
#                 lines.append(line)
        
#         return '\n'.join(lines).strip()
    
#     def train_epoch(self, dataset, epoch: int) -> Dict[str, float]:
#         """Train for one epoch with code-specific evaluation."""
#         logger.info(f"Starting code training epoch {epoch}")
        
#         # Используем очень маленький датасет для тестирования
#         train_size = min(4, len(dataset))  # Уменьшено для стабильности
#         train_dataset = dataset.select(range(train_size))
        
#         epoch_stats = {
#             "mean_reward": 0.0,
#             "syntax_score": 0.0,
#             "structure_score": 0.0,
#         }
        
#         batch_count = 0
        
#         for i in range(0, len(train_dataset), max(1, self.config.batch_size)):
#             batch = train_dataset[i:i + max(1, self.config.batch_size)]
            
#             if isinstance(batch, dict) and 'prompt' in batch:
#                 prompts = batch['prompt']
#                 if isinstance(prompts, str):
#                     prompts = [prompts]
#             else:
#                 continue
            
#             if not prompts:
#                 continue
                
#             batch_stats = self.train_batch({"prompt": prompts}, epoch, batch_count)
            
#             # Accumulate statistics
#             for key in epoch_stats:
#                 if key in batch_stats:
#                     epoch_stats[key] += batch_stats[key]
            
#             batch_count += 1
        
#         # Average statistics
#         if batch_count > 0:
#             for key in epoch_stats:
#                 epoch_stats[key] /= batch_count
        
#         logger.info(f"Epoch {epoch} completed: Mean Reward = {epoch_stats['mean_reward']:.4f}")
#         return epoch_stats
    
#     def train_batch(self, batch: Dict, epoch: int, batch_idx: int) -> Dict[str, float]:
#         """Train on a single batch of code generation examples."""
#         prompts = batch["prompt"]
#         if isinstance(prompts, str):
#             prompts = [prompts]
        
#         if not prompts:
#             return {
#                 "mean_reward": 0.0, 
#                 "syntax_score": 0.0, 
#                 "structure_score": 0.0
#             }
        
#         batch_metrics = {
#             "mean_reward": 0.0,
#             "syntax_score": 0.0,
#             "structure_score": 0.0,
#         }
        
#         try:
#             # Generate code responses
#             generation_output = self.generate_responses(prompts)
#             responses = generation_output["response_texts"]
            
#             # Compute code-specific rewards
#             rewards = self.reward_model.compute_reward(prompts, responses)
            
#             # Calculate syntax and structure scores
#             syntax_scores = []
#             structure_scores = []
#             for response in responses:
#                 syntax_scores.append(getattr(self.reward_model, '_check_syntax', lambda x: 0.5)(response))
#                 structure_scores.append(getattr(self.reward_model, '_check_structure', lambda x: 0.5)(response))
            
#             batch_metrics["syntax_score"] = sum(syntax_scores) / len(syntax_scores) if syntax_scores else 0.0
#             batch_metrics["structure_score"] = sum(structure_scores) / len(structure_scores) if structure_scores else 0.0
            
#             # Упрощенная логика вознаграждения
#             if hasattr(rewards, 'mean') and rewards.numel() > 0:
#                 batch_metrics["mean_reward"] = rewards.mean().item()
#             else:
#                 batch_metrics["mean_reward"] = 0.1  # Минимальное вознаграждение
            
#             # Record metrics
#             self.metrics_tracker.record_batch_metrics(
#                 epoch=epoch,
#                 batch=batch_idx,
#                 batch_stats=batch_metrics,
#                 prompts=prompts,
#                 generated_texts=responses
#             )
            
#             # Упрощенный PPO шаг
#             try:
#                 # Конвертируем rewards для PPO
#                 rewards_list = [torch.tensor([batch_metrics["mean_reward"]]) for _ in prompts]
                
#                 # PPO training step
#                 stats = self.ppo_trainer.step(
#                     generation_output["prompt_tensors"],
#                     generation_output["response_tensors"],
#                     rewards_list,
#                 )
                
#                 logger.info(f"PPO step completed: {stats}")
                
#             except Exception as e:
#                 logger.warning(f"PPO step failed, continuing without PPO: {e}")
#                 # Продолжаем без PPO, но записываем метрики
                
#         except Exception as e:
#             logger.error(f"Training step failed: {e}")
#             # Записываем метрики для неудачного батча
#             self.metrics_tracker.record_batch_metrics(
#                 epoch=epoch,
#                 batch=batch_idx,
#                 batch_stats=batch_metrics,
#                 prompts=prompts,
#                 generated_texts=["Error"] * len(prompts)
#             )
        
#         batch_stats = {
#             "epoch": epoch,
#             "batch": batch_idx,
#             **batch_metrics,
#             "timestamp": datetime.now().isoformat(),
#         }
        
#         self.results.append(batch_stats)
        
#         logger.info(f"Epoch {epoch}, Batch {batch_idx}: Reward = {batch_stats['mean_reward']:.4f}")
        
#         return batch_stats
    
#     def save_final_results(self) -> None:
#         """Save final training results and metrics."""
#         os.makedirs(self.config.output_dir, exist_ok=True)
        
#         # Save results to CSV
#         if self.results:
#             results_df = pd.DataFrame(self.results)
#             results_path = os.path.join(self.config.output_dir, "training_results.csv")
#             results_df.to_csv(results_path, index=False)
#             logger.info(f"Results saved to: {results_path}")
        
#         # Save metrics
#         self.metrics_tracker.save_detailed_metrics()
        
#         # Generate metrics summary
#         summary = self.metrics_tracker.get_metrics_summary()
#         if summary:
#             logger.info("Training Metrics Summary:")
#             for key, value in summary.items():
#                 logger.info(f"  {key}: {value:.4f}")
        
#         # Try to plot metrics
#         try:
#             self.metrics_tracker.plot_metrics()
#         except Exception as e:
#             logger.warning(f"Could not plot metrics: {e}")
        
#         # Save model
#         try:
#             model_path = os.path.join(self.config.output_dir, "final_model")
#             self.policy_model.save_pretrained(model_path)
#             self.tokenizer.save_pretrained(model_path)
#             logger.info(f"Model saved to: {model_path}")
#         except Exception as e:
#             logger.error(f"Could not save model: {e}")