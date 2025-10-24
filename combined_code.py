

# ------------------------------------------------------------
# FILE: .\evaluate_multiple_datasets.py
# ------------------------------------------------------------

# evaluate_multiple_datasets.py
import sys
import os
import pandas as pd
import torch
from typing import List, Dict, Any
import logging

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import CodeRLHFConfig
from src.models.model_loader import ModelLoader, CodeRewardModel
from src.train.ppo_trainer import CodeRLHFTrainer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiDatasetEvaluator:
    def __init__(self, datasets_path: str):
        self.datasets_path = datasets_path
        self.config = CodeRLHFConfig()
        self.model_loader = ModelLoader(self.config)
        self.tokenizer, self.policy_model, self.ref_model = self.model_loader.load_models()
        self.reward_model = CodeRewardModel(self.config)
        self.trainer = CodeRLHFTrainer(self.config, self.tokenizer, self.policy_model, self.ref_model, self.reward_model)
    
    def get_available_datasets(self) -> List[str]:
        """Получить список всех CSV файлов в директории."""
        csv_files = [f for f in os.listdir(self.datasets_path) if f.endswith('.csv')]
        return sorted(csv_files)
    
    def load_dataset(self, filename: str, sample_size: int = None) -> List[str]:
        """Загрузить датасет и извлечь промпты."""
        file_path = os.path.join(self.datasets_path, filename)
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Загружен датасет {filename} с {len(df)} примерами")
            
            # Поиск колонки с промптами
            prompt_columns = ['prompt', 'instruction', 'question', 'text', 'input', 'code_prompt']
            for col in prompt_columns:
                if col in df.columns:
                    prompts = df[col].dropna().astype(str).tolist()
                    if sample_size:
                        prompts = prompts[:sample_size]
                    logger.info(f"Используется колонка '{col}', найдено {len(prompts)} промптов")
                    return prompts
            
            # Если нет стандартных колонок, используем первую текстовую колонку
            for col in df.columns:
                if df[col].dtype == 'object':
                    prompts = df[col].dropna().astype(str).tolist()
                    if sample_size:
                        prompts = prompts[:sample_size]
                    logger.info(f"Используется колонка '{col}', найдено {len(prompts)} промптов")
                    return prompts
            
            raise ValueError(f"Не найдена подходящая колонка с промптами в {filename}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки {filename}: {e}")
            return []
    
    def evaluate_dataset(self, filename: str, sample_size: int = 10) -> Dict[str, Any]:
        """Оценить модель на одном датасете."""
        logger.info(f"Оценка датасета: {filename}")
        
        prompts = self.load_dataset(filename, sample_size)
        if not prompts:
            return {}
        
        results = []
        total_reward = 0
        total_syntax = 0
        total_execution = 0
        
        for i, prompt in enumerate(prompts):
            try:
                # Генерация кода
                generation_output = self.trainer.generate_responses([prompt])
                response = generation_output["response_texts"][0]
                
                # Вычисление метрик
                reward = self.reward_model.compute_reward([prompt], [response])
                syntax_score = self.reward_model._check_syntax(response)
                execution_score = self.reward_model._check_execution(prompt, response)
                
                result = {
                    'dataset': filename,
                    'prompt_index': i,
                    'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    'generated_code': response[:150] + "..." if len(response) > 150 else response,
                    'reward': reward.item(),
                    'syntax_score': syntax_score,
                    'execution_score': execution_score
                }
                
                results.append(result)
                total_reward += reward.item()
                total_syntax += syntax_score
                total_execution += execution_score
                
                logger.info(f"  Пример {i+1}: reward={reward.item():.3f}, syntax={syntax_score:.3f}")
                
            except Exception as e:
                logger.error(f"Ошибка при оценке примера {i} в {filename}: {e}")
                continue
        
        if not results:
            return {}
        
        # Статистика по датасету
        stats = {
            'dataset': filename,
            'num_examples': len(results),
            'avg_reward': total_reward / len(results),
            'avg_syntax': total_syntax / len(results),
            'avg_execution': total_execution / len(results),
            'results': results
        }
        
        return stats
    
    def evaluate_multiple_datasets(self, dataset_indices: List[int] = None, sample_size: int = 10) -> Dict[str, Any]:
        """Оценить модель на нескольких датасетах."""
        all_datasets = self.get_available_datasets()
        
        if not all_datasets:
            logger.error("CSV файлы не найдены в указанной директории")
            return {}
        
        logger.info(f"Найдены датасеты: {all_datasets}")
        
        # Выбор датасетов для оценки
        if dataset_indices is None:
            selected_datasets = all_datasets
        else:
            selected_datasets = [all_datasets[i] for i in dataset_indices if i < len(all_datasets)]
        
        logger.info(f"Выбраны для оценки: {selected_datasets}")
        
        all_results = []
        dataset_stats = []
        
        for dataset in selected_datasets:
            stats = self.evaluate_dataset(dataset, sample_size)
            if stats:
                dataset_stats.append(stats)
                all_results.extend(stats['results'])
            
            # Сохраняем промежуточные результаты после каждого датасета
            self.save_results(all_results, dataset_stats, "intermediate_results")
        
        # Итоговая статистика
        final_stats = self.calculate_final_stats(dataset_stats)
        
        # Сохранение финальных результатов
        self.save_results(all_results, dataset_stats, "final_results")
        
        return final_stats
    
    def calculate_final_stats(self, dataset_stats: List[Dict]) -> Dict[str, Any]:
        """Вычислить итоговую статистику по всем датасетам."""
        if not dataset_stats:
            return {}
        
        total_examples = sum(stats['num_examples'] for stats in dataset_stats)
        
        # Взвешенное среднее по количеству примеров
        avg_reward = sum(stats['avg_reward'] * stats['num_examples'] for stats in dataset_stats) / total_examples
        avg_syntax = sum(stats['avg_syntax'] * stats['num_examples'] for stats in dataset_stats) / total_examples
        avg_execution = sum(stats['avg_execution'] * stats['num_examples'] for stats in dataset_stats) / total_examples
        
        return {
            'total_datasets': len(dataset_stats),
            'total_examples': total_examples,
            'overall_avg_reward': avg_reward,
            'overall_avg_syntax': avg_syntax,
            'overall_avg_execution': avg_execution,
            'dataset_details': dataset_stats
        }
    
    def save_results(self, all_results: List[Dict], dataset_stats: List[Dict], prefix: str):
        """Сохранить результаты в CSV файлы."""
        output_dir = "./evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Детальные результаты
        if all_results:
            detailed_df = pd.DataFrame(all_results)
            detailed_path = os.path.join(output_dir, f"{prefix}_detailed.csv")
            detailed_df.to_csv(detailed_path, index=False, encoding='utf-8')
            logger.info(f"Детальные результаты сохранены в {detailed_path}")
        
        # Сводная статистика
        if dataset_stats:
            summary_data = []
            for stats in dataset_stats:
                summary_data.append({
                    'dataset': stats['dataset'],
                    'num_examples': stats['num_examples'],
                    'avg_reward': stats['avg_reward'],
                    'avg_syntax': stats['avg_syntax'],
                    'avg_execution': stats['avg_execution']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_dir, f"{prefix}_summary.csv")
            summary_df.to_csv(summary_path, index=False, encoding='utf-8')
            logger.info(f"Сводная статистика сохранена в {summary_path}")

def main():
    """Основная функция для запуска оценки."""
    datasets_path = r"C:\Users\Полина\Desktop\Работа\huawei\rlhf\datasets_for_eval"
    
    # Проверяем существование пути
    if not os.path.exists(datasets_path):
        logger.error(f"Директория {datasets_path} не найдена!")
        return
    
    evaluator = MultiDatasetEvaluator(datasets_path)
    
    # Показываем доступные датасеты
    available_datasets = evaluator.get_available_datasets()
    print("\nДоступные датасеты:")
    for i, dataset in enumerate(available_datasets):
        print(f"  {i}: {dataset}")
    
    # Выбор датасетов для оценки
    print("\nВыберите датасеты для оценки:")
    print("  - 'all' для оценки всех датасетов")
    print("  - Номера через запятую (например: 0,2,5)")
    print("  - Диапазон (например: 0-3)")
    
    choice = input("Ваш выбор: ").strip()
    
    if choice.lower() == 'all':
        dataset_indices = None
    elif '-' in choice:
        # Обработка диапазона
        start, end = map(int, choice.split('-'))
        dataset_indices = list(range(start, end + 1))
    else:
        # Обработка списка номеров
        dataset_indices = [int(x.strip()) for x in choice.split(',')]
    
    # Выбор количества примеров
    sample_size = int(input("Количество примеров для оценки на каждом датасете (по умолчанию 10): ") or "10")
    
    # Запуск оценки
    print(f"\nЗапуск оценки на {len(dataset_indices) if dataset_indices else len(available_datasets)} датасетах...")
    
    final_stats = evaluator.evaluate_multiple_datasets(dataset_indices, sample_size)
    
    # Вывод результатов
    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)
    print(f"Оценено датасетов: {final_stats['total_datasets']}")
    print(f"Всего примеров: {final_stats['total_examples']}")
    print(f"Средний reward: {final_stats['overall_avg_reward']:.4f}")
    print(f"Средний syntax score: {final_stats['overall_avg_syntax']:.4f}")
    print(f"Средний execution score: {final_stats['overall_avg_execution']:.4f}")
    
    print("\nДетали по датасетам:")
    for detail in final_stats['dataset_details']:
        print(f"  {detail['dataset']}:")
        print(f"    Примеры: {detail['num_examples']}, Reward: {detail['avg_reward']:.4f}, "
              f"Syntax: {detail['avg_syntax']:.4f}, Execution: {detail['avg_execution']:.4f}")
    
    print(f"\nРезультаты сохранены в папку: ./evaluation_results/")

if __name__ == "__main__":
    main()

# ------------------------------------------------------------
# FILE: .\scripts\compare_models.py
# ------------------------------------------------------------

#!/usr/bin/env python3
"""Compare trained model with baseline model."""

import sys
import os
import pandas as pd
import torch
import logging
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import CodeRLHFConfig
from src.models.model_loader import ModelLoader
from src.models.reward_model import ImprovedCodeRewardModel

class ModelComparator:
    def __init__(self, baseline_model_name: str, trained_model_path: str, reward_model_path: str):
        self.config = CodeRLHFConfig()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load baseline model
        self.logger.info("Loading baseline model...")
        self.config.model_name = baseline_model_name
        baseline_loader = ModelLoader(self.config)
        self.baseline_tokenizer, self.baseline_model, _ = baseline_loader.load_models()
        
        # Load trained model
        self.logger.info("Loading trained model...")
        self.config.model_name = trained_model_path
        trained_loader = ModelLoader(self.config)
        self.trained_tokenizer, self.trained_model, _ = trained_loader.load_models()
        
        # Load reward model
        # Initialize reward model from config name by default
        self.reward_model = ImprovedCodeRewardModel(self.config.reward_model_name)

        # If a reward model artifact exists, try to load it. Support directory (HF) or .pt state_dict
        try:
            if os.path.exists(reward_model_path):
                if os.path.isdir(reward_model_path):
                    try:
                        self.reward_model = ImprovedCodeRewardModel(reward_model_path)
                        self.logger.info(f"Initialized reward model from directory: {reward_model_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to init ImprovedCodeRewardModel from dir {reward_model_path}: {e}; using default model")
                else:
                    try:
                        state = torch.load(reward_model_path, map_location=self.config.device)
                        try:
                            self.reward_model.load_state_dict(state, strict=False)
                            self.logger.info(f"Loaded reward model state_dict from: {reward_model_path}")
                        except RuntimeError as e:
                            self.logger.warning(f"reward model state_dict load failed (shape/key mismatch): {e}; continuing with default model")
                    except Exception as e:
                        self.logger.warning(f"Failed to load reward model file {reward_model_path}: {e}; continuing with default model")
        except Exception as e:
            self.logger.warning(f"Unexpected error while preparing reward model: {e}; continuing with default model")

        try:
            self.reward_model.to(self.config.device)
        except Exception:
            pass

        self.reward_model.eval()
    
    def _setup_logging(self):
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
    
    def generate_with_model(self, model, tokenizer, prompt: str) -> str:
        """Generate code with specified model."""
        model.eval()
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    def compare_models(self, prompts: List[str]) -> pd.DataFrame:
        """Compare models on multiple prompts."""
        results = []
        
        for i, prompt in enumerate(prompts):
            self.logger.info(f"Comparing models on prompt {i+1}/{len(prompts)}")
            
            # Generate with both models
            baseline_code = self.generate_with_model(self.baseline_model, self.baseline_tokenizer, prompt)
            trained_code = self.generate_with_model(self.trained_model, self.trained_tokenizer, prompt)
            
            # Evaluate both
            with torch.no_grad():
                baseline_reward = self.reward_model.compute_reward([prompt], [baseline_code])
                trained_reward = self.reward_model.compute_reward([prompt], [trained_code])
                
                baseline_quality = self.reward_model.predict_quality(prompt, baseline_code)
                trained_quality = self.reward_model.predict_quality(prompt, trained_code)
            
            result = {
                'prompt': prompt,
                'baseline_code': baseline_code,
                'trained_code': trained_code,
                'baseline_reward': baseline_reward.item(),
                'trained_reward': trained_reward.item(),
                'reward_improvement': trained_reward.item() - baseline_reward.item(),
                'baseline_overall': baseline_quality['overall'],
                'trained_overall': trained_quality['overall'],
                'overall_improvement': trained_quality['overall'] - baseline_quality['overall']
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def calculate_improvement_stats(self, comparison_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate improvement statistics."""
        stats = {
            'avg_baseline_reward': comparison_df['baseline_reward'].mean(),
            'avg_trained_reward': comparison_df['trained_reward'].mean(),
            'avg_reward_improvement': comparison_df['reward_improvement'].mean(),
            'avg_baseline_overall': comparison_df['baseline_overall'].mean(),
            'avg_trained_overall': comparison_df['trained_overall'].mean(),
            'avg_overall_improvement': comparison_df['overall_improvement'].mean(),
            'improvement_rate': (comparison_df['reward_improvement'] > 0).mean(),
            'significant_improvement_rate': (comparison_df['reward_improvement'] > 0.1).mean()
        }
        
        return stats

def main():
    # Model paths
    baseline_model = "microsoft/DialoGPT-medium"  # or "gpt2" for baseline
    trained_model_path = "./outputs/final_model"
    reward_model_path = "./outputs/trained_reward_model.pt"
    
    # Test prompts
    test_prompts = [
        "Write a Python function to calculate factorial",
        "Create a function to reverse a string",
        "Write code to read a CSV file and print its contents",
        "Create a Python class for a simple calculator",
        "Write a function to check if a number is prime",
        "Create code to download a file from URL using requests"
    ]
    
    # Compare models
    comparator = ModelComparator(baseline_model, trained_model_path, reward_model_path)
    comparison_results = comparator.compare_models(test_prompts)
    
    # Calculate stats
    stats = comparator.calculate_improvement_stats(comparison_results)
    
    # Save results
    os.makedirs("./evaluation_results", exist_ok=True)
    comparison_results.to_csv("./evaluation_results/model_comparison.csv", index=False, encoding='utf-8')
    
    # Print results
    comparator.logger.info("\n" + "="*60)
    comparator.logger.info("MODEL COMPARISON RESULTS")
    comparator.logger.info("="*60)
    
    for key, value in stats.items():
        comparator.logger.info(f"{key}: {value:.4f}")
    
    comparator.logger.info(f"\nDetailed results saved to: ./evaluation_results/model_comparison.csv")

if __name__ == "__main__":
    main()

# ------------------------------------------------------------
# FILE: .\scripts\convert_reward_checkpoint.py
# ------------------------------------------------------------

#!/usr/bin/env python3
"""Convert a legacy reward checkpoint (state_dict .pt) into an HF-style folder with
robust handling of embedding size mismatches (vocab/position embeddings).

This script will:
 - Load the legacy checkpoint (state_dict) from --checkpoint
 - Instantiate the current `ImprovedCodeRewardModel` with --model-name
 - Merge compatible keys, and for embedding size mismatches copy the overlapping rows
 - Save the base HF model (bert) and tokenizer to --out-dir/base_model and --out-dir
 - Save `improved_state_dict.pt` in --out-dir with the merged state_dict

Usage:
  python scripts/convert_reward_checkpoint.py --checkpoint outputs/trained_reward_model.pt --out-dir outputs/reward_model_hf_converted
"""
import argparse
import os
import sys
import torch
import logging

# Ensure repo root is on sys.path so `src` imports work when script is run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models.reward_model import ImprovedCodeRewardModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location='cpu')
    # Accept either raw state_dict or dict with keys
    if isinstance(ckpt, dict) and not any(k.startswith('bert.') or k.startswith('consistency_head') for k in ckpt.keys()):
        # If it's wrapper dict, try common keys
        for candidate in ['state_dict', 'model_state_dict', 'improved_state_dict']:
            if candidate in ckpt:
                return ckpt[candidate]
    return ckpt


def merge_state_dicts(base_state, ckpt_state):
    """Merge ckpt_state into base_state, handling embedding size mismatches gracefully."""
    merged = base_state.copy()
    for k, v in ckpt_state.items():
        if k not in base_state:
            logger.info(f"Skipping key not in target model: {k}")
            continue
        base_v = base_state[k]
        if v.shape == base_v.shape:
            merged[k] = v
            continue

        # Handle common embedding mismatches by copying overlap
        if 'embeddings.word_embeddings.weight' in k or 'embeddings.position_embeddings.weight' in k:
            min_rows = min(v.shape[0], base_v.shape[0])
            new_v = base_v.clone()
            try:
                new_v[:min_rows, :] = v[:min_rows, :]
                merged[k] = new_v
                logger.info(f"Merged embedding {k}: ckpt_rows={v.shape[0]} target_rows={base_v.shape[0]} copied={min_rows}")
            except Exception as e:
                logger.warning(f"Could not merge embedding {k}: {e} — leaving target init")
            continue

        # If shapes differ but are compatible on trailing dims, try to copy overlapping prefix
        if v.ndim == base_v.ndim and all(v.shape[i] == base_v.shape[i] for i in range(1, v.ndim)):
            min0 = min(v.shape[0], base_v.shape[0])
            new_v = base_v.clone()
            new_v[:min0] = v[:min0]
            merged[k] = new_v
            logger.info(f"Partially merged prefix for {k}: copied {min0} rows")
            continue

        logger.warning(f"Shape mismatch for {k}: ckpt {v.shape} vs target {base_v.shape}. Skipping key.")

    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Path to legacy state_dict .pt')
    p.add_argument('--out-dir', default='outputs/reward_model_hf_converted', help='Output directory for HF-style model')
    p.add_argument('--model-name', default='microsoft/codebert-base', help='Base pretrained model for reward')
    args = p.parse_args()

    ckpt_path = args.checkpoint
    out_dir = args.out_dir

    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return 2

    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Loading checkpoint from {ckpt_path}")
    ckpt_state = load_checkpoint(ckpt_path)
    if not isinstance(ckpt_state, dict):
        logger.error("Loaded checkpoint is not a state_dict-like mapping")
        return 3

    logger.info(f"Initializing target model ({args.model_name}) to obtain shapes")
    model = ImprovedCodeRewardModel(model_name=args.model_name)
    base_state = model.state_dict()

    logger.info("Merging state dicts (will copy overlapping rows for embeddings)")
    merged = merge_state_dicts(base_state, ckpt_state)

    # Save HF-style components: base bert and tokenizer
    base_out = os.path.join(out_dir, 'base_model')
    os.makedirs(base_out, exist_ok=True)
    try:
        model.bert.save_pretrained(base_out)
        model.tokenizer.save_pretrained(out_dir)
        logger.info(f"Saved base model to {base_out} and tokenizer to {out_dir}")
    except Exception as e:
        logger.warning(f"Could not save base model/tokenizer via transformers API: {e}")

    # Save merged improved state
    improved_state_path = os.path.join(out_dir, 'improved_state_dict.pt')
    torch.save(merged, improved_state_path)
    logger.info(f"Saved merged improved state_dict to {improved_state_path}")

    logger.info("Conversion complete")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


# ------------------------------------------------------------
# FILE: .\scripts\evaluate_model.py
# ------------------------------------------------------------

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
from src.metrics_tracker import MetricsTracker

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
        # Initialize reward model from config name by default
        self.reward_model = ImprovedCodeRewardModel(self.config.reward_model_name)

        # If a reward model artifact exists, try to load it. Support either:
        #  - a Hugging Face saved model directory, or
        #  - a .pt state_dict file. If loading fails, log and continue with the
        #    model created from config to avoid aborting evaluation.
        try:
            if os.path.exists(reward_model_path):
                if os.path.isdir(reward_model_path):
                    # Try to initialize model from HF-style directory
                    try:
                        self.reward_model = ImprovedCodeRewardModel(reward_model_path)
                        self.logger.info(f"Initialized reward model from directory: {reward_model_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to init ImprovedCodeRewardModel from dir {reward_model_path}: {e}; using default model")
                else:
                    # Attempt to load a PyTorch state_dict
                    try:
                        state = torch.load(reward_model_path, map_location=self.config.device)
                        try:
                            self.reward_model.load_state_dict(state, strict=False)
                            self.logger.info(f"Loaded reward model state_dict from: {reward_model_path}")
                        except RuntimeError as e:
                            self.logger.warning(f"reward model state_dict load failed (shape/key mismatch): {e}; continuing with default model")
                    except Exception as e:
                        self.logger.warning(f"Failed to load reward model file {reward_model_path}: {e}; continuing with default model")
        except Exception as e:
            self.logger.warning(f"Unexpected error while preparing reward model: {e}; continuing with default model")

        # Move to device and set to eval
        try:
            self.reward_model.to(self.config.device)
        except Exception:
            # best-effort: if device move fails, continue on CPU
            pass
        self.reward_model.eval()
        # Ensure output dir exists and create metrics tracker
        os.makedirs(self.config.output_dir, exist_ok=True)
        # Metrics tracker for automatic metrics (BERTScore/BLEU/CodeBLEU/ROUGE/RUBY)
        self.metrics_tracker = MetricsTracker(output_dir=os.path.join(self.config.output_dir))
    
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

            # Detect reference column (robust, case-insensitive, substring match)
            ref_column = None
            ref_keywords = ['answer', 'ans', 'reference', 'ref', 'target', 'snippet', 'solution', 'code', 'gold']
            for col in df.columns:
                col_low = str(col).lower()
                if any(k in col_low for k in ref_keywords):
                    ref_column = col
                    break

            if ref_column is not None:
                references = df[ref_column].fillna('').astype(str).tolist()
                self.logger.info(f"Using reference column '{ref_column}' for evaluation")
            else:
                # No clear reference column found. Use empty references instead of
                # falling back to prompts (which artificially deflates e.g. BLEU/CodeBLEU
                # when prompts are not code). Metrics that require references will
                # gracefully return 0.0 in MetricsTracker.
                self.logger.warning(f"No reference column detected in {dataset_path}; automatic metrics that need references will be set to 0.")
                references = [''] * len(prompts)
            
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
                # attach reference
                ref_text = references[i] if i < len(references) else ''
                result['reference'] = ref_text

                # compute automatic metrics for this example
                try:
                    metrics = self.metrics_tracker.calculate_metrics([prompt], [result['generated_code']], [ref_text])
                    # merge metrics into result
                    result.update(metrics)
                except Exception as e:
                    self.logger.error(f"Failed to compute metrics for example {i}: {e}")
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
    results = evaluator.evaluate_dataset(dataset_path, sample_size=10)  # Evaluate on 10 samples for faster run
    
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

# ------------------------------------------------------------
# FILE: .\scripts\eval_utils.py
# ------------------------------------------------------------

import ast
import tempfile
import subprocess
import os
import sys
from difflib import SequenceMatcher


def is_syntax_valid_python(code: str) -> bool:
    """Check Python syntax by parsing with ast."""
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def run_py_compile_file(path: str) -> bool:
    """Try to compile a Python file with py_compile (checks syntax)."""
    try:
        import py_compile

        py_compile.compile(path, doraise=True)
        return True
    except Exception:
        return False


def run_code_in_subprocess(code: str, timeout: int = 5) -> (bool, str):
    """Run code in subprocess safely (best-effort). Returns (success, stdout+stderr).

    Note: Running arbitrary code can be unsafe. This helper uses subprocess with a timeout
    and should be run in a sandbox if you don't trust the inputs.
    """
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(code)
        tmp = f.name

    try:
        proc = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=timeout)
        ok = proc.returncode == 0
        out = (proc.stdout or '') + (proc.stderr or '')
        return ok, out
    except subprocess.TimeoutExpired as e:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def compute_simple_code_similarity(pred: str, ref: str) -> float:
    """Fallback similarity: sequence matcher ratio on whitespace-normalized code.

    This is a lightweight substitute for CodeBLEU / full semantic metrics and works
    well enough for quick feedback in the smoke tests.
    """
    a = "\n".join([ln.strip() for ln in pred.splitlines() if ln.strip()])
    b = "\n".join([ln.strip() for ln in ref.splitlines() if ln.strip()])
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def evaluate_generated_code(pred: str, ref: str, run_tests: bool = False) -> dict:
    """Compute a small set of signals for a generated Python snippet.

    Returns dict with keys: syntax_ok (bool), similarity (0..1), exec_ok (bool), exec_output (str)
    """
    syntax_ok = is_syntax_valid_python(pred)
    similarity = compute_simple_code_similarity(pred, ref)
    exec_ok = False
    exec_out = ""
    if run_tests and syntax_ok:
        exec_ok, exec_out = run_code_in_subprocess(pred)

    return {
        'syntax_ok': syntax_ok,
        'similarity': similarity,
        'exec_ok': exec_ok,
        'exec_output': exec_out,
    }


# ------------------------------------------------------------
# FILE: .\scripts\fill_references.py
# ------------------------------------------------------------

#!/usr/bin/env python3
import os
import re
import pandas as pd
from collections import Counter

DATA_DIR = 'datasets_for_eval'
EVAL_DETAILED = 'evaluation_results/final_evaluation_detailed.csv'

def tokenize(s: str):
    if s is None or s != s:
        return []
    s = re.sub(r"[`\"'\\\[\]{}()<>.,:;!\\/?@#\$%\^&\*\-_=+\\|~]", ' ', str(s).lower())
    toks = [t for t in s.split() if t]
    return toks

def overlap_score(a, b):
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    inter = sa & sb
    # normalize by smaller length to be permissive
    denom = min(len(sa), len(sb))
    return len(inter) / denom if denom > 0 else 0.0

def find_reference_for_prompt(prompt_tokens, dfs_files):
    # search each csv file in chunks
    ref_cols = ['Answer','ModelAnswer','FullModelAnswer','snippet','ans_gt','ans','reference']
    search_cols = ['Prompt','Question','prompt','question','text','clean_question','instruction','input']
    def is_code_like(s: str) -> bool:
        if s is None or s != s:
            return False
        s = str(s)
        if '\n' in s and len(s.splitlines()) > 1:
            return True
        code_tokens = ['def ', 'return ', 'import ', 'class ', 'print(', 'self.', ':']
        hits = sum(1 for t in code_tokens if t in s)
        return hits >= 2
    for f in dfs_files:
        path = os.path.join(DATA_DIR, f)
        try:
            for chunk in pd.read_csv(path, chunksize=500, encoding='utf-8'):
                # for each search col present
                for sc in search_cols:
                    if sc in chunk.columns:
                        texts = chunk[sc].astype(str).fillna('')
                        for idx, text in texts.items():
                            t_tokens = tokenize(text)
                            score = overlap_score(prompt_tokens, t_tokens)
                            if score >= 0.5:
                                # find a ref column with non-empty value
                                # prefer code-like references
                                for rc in ref_cols:
                                    if rc in chunk.columns:
                                        val = chunk.at[idx, rc]
                                        if pd.notna(val) and str(val).strip() and is_code_like(val):
                                            return str(val)
                                # if none code-like, keep first non-empty as last resort
                                for rc in ref_cols:
                                    if rc in chunk.columns:
                                        val = chunk.at[idx, rc]
                                        if pd.notna(val) and str(val).strip():
                                            return str(val)
        except Exception:
            continue
    return None


def main():
    if not os.path.exists(EVAL_DETAILED):
        print('Evaluation detailed CSV not found:', EVAL_DETAILED)
        return

    df = pd.read_csv(EVAL_DETAILED)
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

    filled = 0
    for i, row in df.iterrows():
        if pd.notna(row.get('reference')) and str(row.get('reference')).strip():
            continue
        prompt = row.get('prompt')
        tokens = tokenize(prompt)
        if not tokens:
            continue
        ref = find_reference_for_prompt(tokens, files)
        if ref:
            df.at[i, 'reference'] = ref
            filled += 1
            print(f'Filled row {i} with reference (len {len(ref)} chars)')

    print('Filled', filled, 'references out of', len(df))
    df.to_csv(EVAL_DETAILED, index=False)
    print('Saved updated', EVAL_DETAILED)

if __name__ == '__main__':
    main()


# ------------------------------------------------------------
# FILE: .\scripts\finetune_sft_hf.py
# ------------------------------------------------------------

"""
Supervised fine-tuning for causal models using HuggingFace Transformers.

This script consumes `datasets_for_training/sft_dataset.csv` (question,best_answer)
and runs a standard causal LM finetuning using `Trainer` and `DataCollatorForLanguageModeling`.

If required libraries are missing, exits with code 2 (so CI/tests can assert).
"""
import argparse
import os
import sys
import csv
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fail_missing_libs():
    print("Missing required libraries for HF SFT. Install `torch` and `transformers`.")
    sys.exit(2)


try:
    import torch
    tver = getattr(torch, "__version__", "0.0.0")
    try:
        ver_parts = tver.split("+")[0].split(".")
        tmajor = int(ver_parts[0])
        tminor = int(ver_parts[1]) if len(ver_parts) > 1 else 0
    except Exception:
        tmajor, tminor = 0, 0
    if (tmajor, tminor) < (2, 1):
        fail_missing_libs()

    from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
    from transformers import DataCollatorForLanguageModeling
    from torch.utils.data import Dataset
except Exception:
    fail_missing_libs()


class SFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=512):
        self.examples = []
        for r in rows:
            q = r.get("question", "")
            a = r.get("best_answer", "")
            text = (q + "\n" + a).strip()
            self.examples.append(tokenizer(text, truncation=True, max_length=max_length))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}


def read_sft(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("best_answer"):
                continue
            rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="gpt2")
    p.add_argument("--output_dir", default="outputs/sft_model_hf")
    p.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=256)
    args = p.parse_args()

    data_path = os.path.join(args.root, "datasets_for_training", "sft_dataset.csv")
    if not os.path.exists(data_path):
        logging.error("SFT dataset not found: %s", data_path)
        return 3

    rows = read_sft(data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)

    ds = SFTDataset(rows, tokenizer, max_length=args.max_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.lr,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=ds, data_collator=data_collator)
    trainer.train()
    trainer.save_model(args.output_dir)
    logging.info("Saved SFT model to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ------------------------------------------------------------
# FILE: .\scripts\finetune_supervised.py
# ------------------------------------------------------------

"""Lightweight supervised finetune placeholder.

Reads `datasets_for_training/sft_dataset.csv` and creates a tiny "model"
artifact that contains a simple vocabulary and example mappings. This file
is meant as a fast smoke-test for the SFT pipeline and to be replaced by a
real HuggingFace training run once the environment is ready.

Outputs:
    outputs/sft_model_placeholder.json
"""
import csv
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(ROOT, "datasets_for_training", "sft_dataset.csv")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "sft_model_placeholder.json")


def load_sft(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def build_vocab(texts, max_vocab=2000):
    freq = {}
    for t in texts:
        for tok in (t or "").split():
            freq[tok] = freq.get(tok, 0) + 1
    items = sorted(freq.items(), key=lambda x: -x[1])[:max_vocab]
    vocab = {tok: idx for idx, (tok, _) in enumerate(items, start=1)}
    vocab["<unk>"] = 0
    return vocab


def main():
    if not os.path.exists(IN_PATH):
        logging.error("Required input not found: %s", IN_PATH)
        logging.error("Run scripts/prepare_pairs.py first to generate datasets.")
        return 2

    rows = load_sft(IN_PATH)
    if not rows:
        logging.error("No rows found in %s", IN_PATH)
        return 2

    questions = [r.get("question", "") for r in rows]
    answers = [r.get("best_answer", "") for r in rows]

    vocab = build_vocab(questions + answers, max_vocab=2000)

    artifact = {"meta": {"type": "placeholder_sft", "rows": len(rows)}, "vocab_size": len(vocab), "vocab_sample": dict(list(vocab.items())[:20])}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    logging.info("Saved SFT placeholder model to %s (vocab size=%d)", OUT_PATH, len(vocab))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ------------------------------------------------------------
# FILE: .\scripts\human_evaluation.py
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# FILE: .\scripts\improved_rlhf_training.py
# ------------------------------------------------------------

#!/usr/bin/env python3
"""Improved RLHF training script for meaningful text generation."""

import logging
import random
import numpy as np
import torch
from datetime import datetime
import os
import sys
from typing import List, Dict, Any, Optional
import re

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    # Try to import with datasets
    from src.data.dataset_utils import CodeDatasetLoader
    print("Using datasets version")
except ImportError as e:
    print(f"datasets import failed: {e}")
    print("Using standalone dataset implementation")
    # Fallback to our standalone implementation
    from src.data.dataset_utils import CodeDatasetLoader

from src.config import CodeRLHFConfig
from src.data.dataset_utils import CodeDatasetLoader
from src.models.model_loader import ModelLoader, CodeRewardModel
from src.train.ppo_trainer import CodeRLHFTrainer


def setup_logging(output_dir: str) -> None:
    """Set up logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def main() -> None:
    """Main training function for code generation."""
    # Configuration for code generation
    config = CodeRLHFConfig()
    
    # Setup
    setup_logging(config.output_dir)
    set_seed(config.seed)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Code Generation RLHF Training")
    logger.info(f"Configuration: {config}")
    
    try:
        # Load code dataset
        logger.info("Loading code dataset...")
        dataset_loader = CodeDatasetLoader(config)
        train_dataset = dataset_loader.load_dataset()
        
        # Load models
        logger.info("Loading models...")
        model_loader = ModelLoader(config)
        tokenizer, policy_model, ref_model = model_loader.load_models()
        
        # Initialize code reward model
        reward_model = CodeRewardModel(config)
        
        # Initialize code trainer
        trainer = CodeRLHFTrainer(config, tokenizer, policy_model, ref_model, reward_model)
        
        # Training loop
        logger.info("Starting code training...")
        for epoch in range(config.ppo_epochs):
            epoch_stats = trainer.train_epoch(train_dataset, epoch)
            logger.info(f"Epoch {epoch} statistics: {epoch_stats}")
            
            # Early stopping based on code quality
            if epoch_stats.get('syntax_score', 0) > 0.8 and epoch_stats.get('mean_reward', 0) > 0.6:
                logger.info("Good code quality achieved, stopping early")
                break
        
        # Save final results
        trainer.save_final_results()
        trainer.evaluate_code_quality()
        
        logger.info("Code RLHF training completed successfully!")
        
    except Exception as e:
        logger.error(f"Code training failed: {e}")
        raise


if __name__ == "__main__":
    main()

# ------------------------------------------------------------
# FILE: .\scripts\main_rlhf_training.py
# ------------------------------------------------------------

#!/usr/bin/env python3
"""Main RLHF training script with improved reward model."""

import logging
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import CodeRLHFConfig
from src.data.dataset_loader import CodeDatasetLoader
from src.models.model_loader import ModelLoader
from src.models.reward_model import ImprovedCodeRewardModel
from src.train.ppo_trainer import CodeRLHFTrainer 
import importlib.util
eval_utils = None
try:
    # load scripts/eval_utils.py as a module when running as a script
    spec = importlib.util.spec_from_file_location("scripts.eval_utils", os.path.join(os.path.dirname(__file__), "eval_utils.py"))
    eval_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_utils)
except Exception:
    eval_utils = None
import subprocess
import sys

def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "training.log"))
        ]
    )

def main():
    config = CodeRLHFConfig()
    setup_logging(config.output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Improved RLHF Training Pipeline")
    
    try:
        # 1. Load or train a reward model
        logger.info("Preparing reward model...")
        reward_model = ImprovedCodeRewardModel(config.reward_model_name)
        # Prefer HF-style folder if present, then fall back to single-file state_dict
        hf_dir = os.path.join(config.output_dir, "reward_model_hf")
        reward_model_path = os.path.join(config.output_dir, "trained_reward_model.pt")

        if os.path.exists(hf_dir):
            logger.info(f"Found HF-style reward model directory: {hf_dir}. Initializing from directory and loading improved state if present.")
            try:
                # Initialize model to pick up tokenizer/base model from hf_dir
                reward_model = ImprovedCodeRewardModel(hf_dir)
                # Look for an improved state dict inside the HF dir and load it (non-strict)
                improved_state = os.path.join(hf_dir, "improved_state_dict.pt")
                if os.path.exists(improved_state):
                    state_dict = torch.load(improved_state, map_location=config.device)
                elif os.path.exists(reward_model_path):
                    state_dict = torch.load(reward_model_path, map_location=config.device)
                else:
                    state_dict = None

                if state_dict is not None:
                    keys_to_remove = [k for k in state_dict.keys() if 'position_ids' in k]
                    for k in keys_to_remove:
                        del state_dict[k]
                    try:
                        reward_model.load_state_dict(state_dict, strict=False)
                        logger.info(f"Loaded improved reward model from HF dir, ignored keys: {keys_to_remove}")
                    except RuntimeError as re:
                        logger.warning(f"Could not fully load improved state into reward model (shape mismatch): {re}. Continuing with partially initialized model.")
            except Exception as e:
                logger.warning(f"Failed to initialize reward model from HF dir {hf_dir}: {e}. Falling back to default init.")
        elif os.path.exists(reward_model_path):
            state_dict = torch.load(reward_model_path, map_location=config.device)
            keys_to_remove = [k for k in state_dict.keys() if 'position_ids' in k]
            for k in keys_to_remove:
                del state_dict[k]
            try:
                reward_model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded reward model, ignored keys: {keys_to_remove}")
            except RuntimeError as re:
                # Shape mismatches may occur when switching to a different underlying pretrained model
                logger.warning(f"Could not load saved reward model due to shape mismatch: {re}. Skipping loading and continuing with uninitialized reward model.")
        else:
            # If no reward model - build training TSV from human prefs and run the quick trainer stub
            logger.warning("No trained reward model found. Building preferences TSV and training a quick reward model (stub)...")
            prefs_folder = config.human_eval_path if hasattr(config, 'human_eval_path') else os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluation_results_server')
            script_path = os.path.join(os.path.dirname(__file__), 'pref_convert_and_reward_train.py')
            out_tsv = os.path.join(config.output_dir, 'training_data', 'prefs.tsv')
            # include --pairwise to get a pairwise reward trainer by default
            cmd = [sys.executable, script_path, '--prefs-folder', prefs_folder, '--out-tsv', out_tsv, '--reward-output', os.path.join(config.output_dir, 'reward_model'), '--pairwise', '--pairwise-epochs', '1']
            logger.info('Running reward model train stub: ' + ' '.join(cmd))
            try:
                subprocess.run(cmd, check=True)
                logger.info('Reward model stub training completed.')
                # Try to load the weights from the stub if available
                stub_model_dir = os.path.join(config.output_dir, 'reward_model')
                if os.path.exists(stub_model_dir):
                    try:
                        reward_model = ImprovedCodeRewardModel(stub_model_dir)
                        logger.info('Loaded reward model from stub directory.')
                    except Exception:
                        logger.warning('Could not initialize ImprovedCodeRewardModel from stub dir — will continue with untrained model')
            except subprocess.CalledProcessError as e:
                logger.error(f'Reward model stub failed: {e}. Continuing with untrained reward model')

        reward_model.eval()
        
        # 2. Load dataset
        logger.info("Loading code dataset...")
        dataset_loader = CodeDatasetLoader(config)
        train_dataset = dataset_loader.load_dataset()
        logger.info(f"Loaded dataset with {len(train_dataset)} examples")
        
        # 3. Load policy model
        logger.info("Loading policy model...")
        model_loader = ModelLoader(config)
        tokenizer, policy_model, ref_model = model_loader.load_models()
        
        # 4. Initialize trainer
        logger.info("Initializing RLHF trainer...")
        trainer = CodeRLHFTrainer(config, tokenizer, policy_model, ref_model, reward_model)
        
        # 5. Training loop
        logger.info("Starting RLHF training...")
        best_reward = -float('inf')
        
        for epoch in range(config.ppo_epochs):
            epoch_stats = trainer.train_epoch(train_dataset, epoch)
            
            current_reward = epoch_stats.get('mean_reward', 0)
            logger.info(f"Epoch {epoch} completed:")
            logger.info(f"  Mean Reward: {current_reward:.4f}")
            logger.info(f"  Syntax Score: {epoch_stats.get('syntax_score', 0):.4f}")
            
            # Save best model
            if current_reward > best_reward:
                best_reward = current_reward
                trainer.save_final_results()
                logger.info(f"New best model saved with reward: {best_reward:.4f}")

            # Early stopping (configurable target, default 0.8)
            stop_target = getattr(config, 'early_stop_reward', 0.8)
            if current_reward >= stop_target:
                logger.info(f"Early stopping: reached target mean reward {stop_target}")
                break
        
        logger.info(" RLHF training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()

# ------------------------------------------------------------
# FILE: .\scripts\plot_metrics.py
# ------------------------------------------------------------

import json
import os
from collections import defaultdict

def load_metrics(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)

def aggregate_by_epoch(records):
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    for r in records:
        e = int(r.get('epoch', 0))
        counts[e] += 1
        for key in ['reward', 'bertscore', 'bleu', 'codebleu', 'rouge', 'syntax_score', 'structure_score']:
            if key in r and r[key] is not None:
                try:
                    sums[e][key] += float(r[key])
                except Exception:
                    pass

    epochs = sorted(counts.keys())
    out = []
    for e in epochs:
        row = {'epoch': e}
        for key in ['reward', 'bertscore', 'bleu', 'codebleu', 'rouge', 'syntax_score', 'structure_score']:
            row[key] = (sums[e].get(key, 0.0) / counts[e]) if counts[e] > 0 else 0.0
        out.append(row)
    return out

def save_csv(agg, out_path):
    import csv
    keys = ['epoch','reward','syntax_score','structure_score','bertscore','bleu','codebleu','rouge']
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in agg:
            writer.writerow({k: r.get(k, '') for k in keys})

def plot(agg, out_png):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print('matplotlib not available, will not plot PNG:', e)
        return False

    epochs = [r['epoch'] for r in agg]
    def series(k):
        return [r.get(k, 0.0) for r in agg]

    plt.figure(figsize=(10,6))
    plt.plot(epochs, series('reward'), label='reward')
    plt.plot(epochs, series('bertscore'), label='bertscore')
    plt.plot(epochs, series('bleu'), label='bleu')
    plt.plot(epochs, series('codebleu'), label='codebleu')
    plt.plot(epochs, series('rouge'), label='rouge')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    return True

if __name__ == '__main__':
    metrics_path = os.path.join('outputs', 'detailed_metrics.json')
    if not os.path.exists(metrics_path):
        print('Could not find', metrics_path)
        raise SystemExit(1)

    records = load_metrics(metrics_path)
    agg = aggregate_by_epoch(records)

    os.makedirs('outputs', exist_ok=True)
    csv_out = os.path.join('outputs', 'metrics_by_epoch.csv')
    save_csv(agg, csv_out)
    print('Saved CSV to', csv_out)

    png_out = os.path.join('outputs', 'metrics_by_epoch.png')
    ok = plot(agg, png_out)
    if ok:
        print('Saved plot to', png_out)
    else:
        print('Plot not created')


# ------------------------------------------------------------
# FILE: .\scripts\ppo_stability_test.py
# ------------------------------------------------------------

#!/usr/bin/env python3
"""Quick PPO stability test: run a single short epoch with conservative generation/gym settings
to observe KL behavior and batch ratio warnings. This uses the same trainer wrapper but overrides
config values for a short run.
"""
import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import CodeRLHFConfig
from src.data.dataset_loader import CodeDatasetLoader
from src.models.model_loader import ModelLoader
from src.models.reward_model import ImprovedCodeRewardModel
from src.train.ppo_trainer import CodeRLHFTrainer

def main():
    cfg = CodeRLHFConfig()
    # Conservative overrides
    cfg.ppo_epochs = 1
    cfg.batch_size = 1
    cfg.mini_batch_size = 1
    cfg.learning_rate = 2e-5
    cfg.max_prompt_length = 128
    cfg.max_response_length = 64
    cfg.device = 'cuda' if 'cuda' in cfg.device else cfg.device

    # Lower temperature and disable sampling to reduce KL surprises
    cfg.temperature = 0.3
    cfg.do_sample = False
    cfg.top_p = 0.95

    # Logging
    logging.basicConfig(level=logging.INFO)

    reward_model = ImprovedCodeRewardModel(cfg.reward_model_name)
    dataset_loader = CodeDatasetLoader(cfg)
    dataset = dataset_loader.load_dataset()
    tokenizer, policy_model, ref_model = ModelLoader(cfg).load_models()

    trainer = CodeRLHFTrainer(cfg, tokenizer, policy_model, ref_model, reward_model)

    # Run one epoch and print summary
    stats = trainer.train_epoch(dataset, epoch=0)
    print('Stability test stats:', stats)

if __name__ == '__main__':
    main()


# ------------------------------------------------------------
# FILE: .\scripts\pref_convert_and_reward_train.py
# ------------------------------------------------------------

import os
import json
from glob import glob
from typing import List, Tuple
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch


def load_pref_json_files(folder: str) -> List[dict]:
    files = glob(os.path.join(folder, '*.json'))
    results = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                j = json.load(fh)
                results.append(j)
        except Exception:
            continue
    return results


def extract_pairs_from_json(j: dict) -> List[Tuple[str, str, int]]:
    """Extract (A, B, label) where label=1 if left preferred else 0.

    The JSON structure in evaluation_results_server contains two answers per question in `questions_df`.
    We will interpret compare results using `comparison_slider` or the per-field scores.
    This is a heuristic extractor adapted to the observed structure.
    """
    pairs = []
    qdf = j.get('questions_df', []) or []

    # Group entries by question ID or index
    groups = {}
    for item in qdf:
        key = item.get('ID') or item.get('index')
        if key is None:
            continue
        groups.setdefault(key, []).append(item)

    for key, items in groups.items():
        if len(items) < 2:
            continue
        left = items[0].get('Answer', '')
        right = items[1].get('Answer', '')
        prompt = items[0].get('Question', '') or ''

        # Determine label using available per-file or per-item scores if present
        # Prefer file-level aggregated comparison_slider if provided
        file_slider = j.get('comparison_slider')
        if file_slider is not None:
            label = 1 if file_slider >= 0 else 0
        else:
            # Sum left vs right per-item annotated scores if present
            left_score = 0
            right_score = 0
            for k in ['consistent_L', 'correct_L', 'useful_L']:
                left_score += j.get(k, 0)
            for k in ['consistent_R', 'correct_R', 'useful_R']:
                right_score += j.get(k, 0)
            if left_score >= right_score:
                label = 1
            else:
                label = 0

        pairs.append((prompt, left, right, int(label)))
    return pairs


def build_dataset_from_folder(folder: str, out_path: str):
    all_json = load_pref_json_files(folder)
    records = []
    for j in all_json:
        recs = extract_pairs_from_json(j)
        records.extend(recs)

    # Save TSV with columns: prompt \t left \t right \t label
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        for prompt, left, right, lbl in records:
            fh.write(prompt.replace('\n', '\\n') + '\t' + left.replace('\n', '\\n') + '\t' + right.replace('\n', '\\n') + '\t' + str(lbl) + '\n')


class RewardModelTrainerStub:
    """Very small reward model trainer stub using transformers classification head.

    This is a quick way to get a model that scores generations for the smoke tests. For production
    you should train a proper pairwise reward model using ranking or pairwise losses. We bias the
    default to a code-aware model (CodeBERT) and better preprocessing for code prompts.
    """
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        self.model_name = model_name

    def train(self, tsv_path: str, output_dir: str):
        # Read tsv
        examples = []
        with open(tsv_path, 'r', encoding='utf-8') as fh:
            for ln in fh:
                parts = ln.rstrip('\n').split('\t')
                if len(parts) < 4:
                    # Old format: left \t right \t label
                    continue
                prompt, left, right, lbl = parts[0], parts[1], parts[2], parts[3]
                examples.append((prompt.replace('\\n', '\n'), left.replace('\\n', '\n'), right.replace('\\n', '\n'), int(lbl)))

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        texts = [ (l + tokenizer.sep_token + r) for _, l, r, _ in examples]
        labels = [lbl for _, _, _, lbl in examples]
        enc = tokenizer(texts, truncation=True, padding=True, max_length=512)

        class DS(torch.utils.data.Dataset):
            def __init__(self, enc, labels):
                self.enc = enc
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

        ds = DS(enc, labels)

        # Use a small number of epochs for the stub but a code-aware initialization.
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        # Use minimal batch size and epochs to fit small GPUs for smoke/baseline runs
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_strategy='no',
            logging_strategy='no',
            learning_rate=2e-5,
        )
        trainer = Trainer(model=model, args=args, train_dataset=ds)
        trainer.train()
        trainer.save_model(output_dir)

class PairwiseRewardTrainer:
    """Train a simple pairwise reward model by classifying which answer is preferred."""
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        self.model_name = model_name

    def train(self, tsv_path: str, output_dir: str, epochs: int = 1, per_device_batch_size: int = 1):
        # Read tsv expecting prompt \t left \t right \t label
        examples = []
        with open(tsv_path, 'r', encoding='utf-8') as fh:
            for ln in fh:
                parts = ln.rstrip('\n').split('\t')
                if len(parts) < 4:
                    continue
                prompt, left, right, lbl = parts[0].replace('\\n','\n'), parts[1].replace('\\n','\n'), parts[2].replace('\\n','\n'), int(parts[3])
                examples.append((prompt, left, right, lbl))

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        texts = [ (p + tokenizer.sep_token + l + tokenizer.sep_token + r) for p,l,r,_ in examples]
        labels = [lbl for _,_,_,lbl in examples]
        enc = tokenizer(texts, truncation=True, padding=True, max_length=512)

        class DS(torch.utils.data.Dataset):
            def __init__(self, enc, labels):
                self.enc = enc
                self.labels = labels
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k,v in self.enc.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

        ds = DS(enc, labels)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        # Respect small per-device batch sizes to avoid OOM on 4GB GPUs
        args = TrainingArguments(output_dir=output_dir, num_train_epochs=epochs, per_device_train_batch_size=per_device_batch_size, save_strategy='no', logging_strategy='no')
        trainer = Trainer(model=model, args=args, train_dataset=ds)
        trainer.train()
        trainer.save_model(output_dir)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--prefs-folder', type=str, required=True)
    p.add_argument('--out-tsv', type=str, default='training_data/prefs.tsv')
    p.add_argument('--reward-output', type=str, default='outputs/reward_model')
    p.add_argument('--pairwise', action='store_true', help='Train pairwise reward model instead of stub')
    p.add_argument('--pairwise-epochs', type=int, default=1)
    args = p.parse_args()

    build_dataset_from_folder(args.prefs_folder, args.out_tsv)
    print('Built TSV:', args.out_tsv)
    if getattr(args, 'pairwise', False):
        print('Running pairwise trainer...')
        trainer = PairwiseRewardTrainer()
        trainer.train(args.out_tsv, args.reward_output, epochs=args.pairwise_epochs)
    else:
        trainer = RewardModelTrainerStub()
        trainer.train(args.out_tsv, args.reward_output)


# ------------------------------------------------------------
# FILE: .\scripts\prepare_pairs.py
# ------------------------------------------------------------

"""
Prepare pairwise preference and SFT datasets from human comparison JSONs.

This script scans the `evaluation_results_server/` directory for JSON files
containing human comparisons. For each record it expects a top-level set of
ratings for left/right (e.g., `consistent_L`, `correct_L`, `useful_L`) and a
`questions_df` list with two entries (left then right) holding `Question` and
`Answer` and metadata. It computes which side is preferred by summing the
left/right scores and emits two CSVs:

- datasets_for_training/pairwise_prefs.csv:
    question,preferred_answer,other_answer,preferred_model_tag,other_model_tag,preference,source_json,datetime

- datasets_for_training/sft_dataset.csv:
    question,best_answer,model_tag,source_json,datetime

Assumptions made when converting:
- `questions_df` contains exactly two entries: index 0 -> LEFT, index 1 -> RIGHT.
- Preference is decided by sum(consistent, correct, useful) for left vs right.
- Ties are skipped in pairwise output but not in SFT.

If some files don't match the expected format the script will log warnings and
skip those entries.
"""
import os
import glob
import json
import csv
import logging


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(ROOT, "evaluation_results_server")
OUT_DIR = os.path.join(ROOT, "datasets_for_training")
os.makedirs(OUT_DIR, exist_ok=True)

PAIRWISE_OUT = os.path.join(OUT_DIR, "pairwise_prefs.csv")
SFT_OUT = os.path.join(OUT_DIR, "sft_dataset.csv")


def parse_record(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dt = data.get("datetime")
    src = os.path.basename(path)

    # get top-level L/R scores
    try:
        score_L = sum(int(data.get(k, 0)) for k in ("consistent_L", "correct_L", "useful_L"))
        score_R = sum(int(data.get(k, 0)) for k in ("consistent_R", "correct_R", "useful_R"))
    except Exception:
        logging.warning("Invalid score fields in %s", src)
        return []

    qlist = data.get("questions_df") or []
    if not isinstance(qlist, list) or len(qlist) < 2:
        logging.warning("Skipping %s: questions_df missing or <2 entries", src)
        return []

    # assume first is LEFT, second is RIGHT
    left = qlist[0]
    right = qlist[1]

    question = left.get("Question") or right.get("Question") or ""
    left_ans = left.get("Answer", "")
    right_ans = right.get("Answer", "")
    left_tag = left.get("MODEL_TAG", left.get("CSV_PATH", "left"))
    right_tag = right.get("MODEL_TAG", right.get("CSV_PATH", "right"))

    preferred = None
    if score_L > score_R:
        preferred = "left"
    elif score_R > score_L:
        preferred = "right"
    else:
        preferred = "tie"

    recs = []
    # pairwise entry (skip ties)
    if preferred != "tie":
        if preferred == "left":
            recs.append({
                "question": question,
                "preferred_answer": left_ans,
                "other_answer": right_ans,
                "preferred_model_tag": left_tag,
                "other_model_tag": right_tag,
                "preference": "left",
                "source_json": src,
                "datetime": dt,
            })
        else:
            recs.append({
                "question": question,
                "preferred_answer": right_ans,
                "other_answer": left_ans,
                "preferred_model_tag": right_tag,
                "other_model_tag": left_tag,
                "preference": "right",
                "source_json": src,
                "datetime": dt,
            })

    # SFT entry: include best answer even if tie (choose left in ties)
    best = left_ans if (preferred != "right") else right_ans
    best_tag = left_tag if (preferred != "right") else right_tag
    recs.append({
        "question": question,
        "best_answer": best,
        "model_tag": best_tag,
        "source_json": src,
        "datetime": dt,
    })

    return recs


def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    if not files:
        logging.error("No JSON files found in %s", INPUT_DIR)
        return

    pairwise_fields = [
        "question",
        "preferred_answer",
        "other_answer",
        "preferred_model_tag",
        "other_model_tag",
        "preference",
        "source_json",
        "datetime",
    ]

    sft_fields = ["question", "best_answer", "model_tag", "source_json", "datetime"]

    pair_count = 0
    sft_count = 0

    with open(PAIRWISE_OUT, "w", encoding="utf-8", newline="") as pf, open(SFT_OUT, "w", encoding="utf-8", newline="") as sf:
        pw = csv.DictWriter(pf, fieldnames=pairwise_fields)
        sw = csv.DictWriter(sf, fieldnames=sft_fields)
        pw.writeheader()
        sw.writeheader()

        for p in files:
            recs = parse_record(p)
            for r in recs:
                # pairwise rows have 'preferred_answer' key
                if "preferred_answer" in r:
                    pw.writerow(r)
                    pair_count += 1
                else:
                    sw.writerow(r)
                    sft_count += 1

    logging.info("Wrote %d pairwise rows to %s", pair_count, PAIRWISE_OUT)
    logging.info("Wrote %d SFT rows to %s", sft_count, SFT_OUT)


if __name__ == "__main__":
    main()


# ------------------------------------------------------------
# FILE: .\scripts\recompute_metrics.py
# ------------------------------------------------------------

import json
import os
import sys
sys.path.insert(0, os.getcwd())
from src.metrics_tracker import MetricsTracker

def recompute(detailed_metrics_path: str):
    if not os.path.exists(detailed_metrics_path):
        print('Could not find', detailed_metrics_path)
        return

    with open(detailed_metrics_path, 'r', encoding='utf-8') as fh:
        records = json.load(fh)

    mt = MetricsTracker(output_dir='outputs')

    # We'll recompute codebleu and ruby for each record if generated_texts and prompts exist
    new_records = []
    for r in records:
        prompts = r.get('prompt')
        gen = r.get('generated_code') or r.get('generated_texts')

        # Normalize to lists
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(gen, str):
            gen = [gen]

        if prompts is None or gen is None:
            # keep existing
            new_records.append(r)
            continue

        # Use prompts as references (best-effort fallback)
        refs = prompts

        try:
            metrics = mt.calculate_metrics(prompts, gen, refs)
            r.update(metrics)
        except Exception as e:
            print('Failed to compute metrics for a record:', e)

        new_records.append(r)

    # Save back
    with open(detailed_metrics_path, 'w', encoding='utf-8') as fh:
        json.dump(new_records, fh, indent=2, ensure_ascii=False)

    print('Recomputed metrics and updated', detailed_metrics_path)

if __name__ == '__main__':
    recompute(os.path.join('outputs', 'detailed_metrics.json'))


# ------------------------------------------------------------
# FILE: .\scripts\run_full_pipeline.py
# ------------------------------------------------------------

#!/usr/bin/env python3
"""Run full RLHF pipeline: prefs -> reward training -> PPO -> eval -> recompute -> plot

Usage examples:
  python scripts/run_full_pipeline.py --pairwise-epochs 8 --ppo-steps 1000
  python scripts/run_full_pipeline.py --smoke
"""
import argparse
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run(cmd, desc=None, check=True):
    if desc:
        print('\n>>', desc)
    print('Running:', cmd)
    rc = subprocess.run(cmd, shell=True)
    if check and rc.returncode != 0:
        print(f'Command failed (rc={rc.returncode}):', cmd)
        sys.exit(rc.returncode)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pairwise-epochs', type=int, default=8, help='Epochs for pairwise reward training')
    p.add_argument('--ppo-steps', type=int, default=1000, help='Steps for PPO training (or episodes depending on script)')
    p.add_argument('--device', default='cpu', help='Device for training (cpu or cuda)')
    p.add_argument('--smoke', action='store_true', help='Run a quick smoke test instead of full training')
    args = p.parse_args()

    # Smoke defaults
    if args.smoke:
        pw_pairwise_epochs = 1
        ppo_steps = 10
        sample_arg = '--sample-size 20'
    else:
        pw_pairwise_epochs = args.pairwise_epochs
        ppo_steps = args.ppo_steps
        sample_arg = ''

    python = sys.executable
    prefs_folder = os.path.join(ROOT, 'evaluation_results_server')
    if not os.path.exists(prefs_folder):
        print('Preferences folder not found:', prefs_folder)
        print('Please ensure human preference JSONs are in evaluation_results_server')
        sys.exit(1)

    # 1) Convert prefs and train reward model (pairwise)
    trained_reward_path = os.path.join(ROOT, 'outputs', 'trained_reward_model.pt')
    if os.path.exists(trained_reward_path):
        print('\n>> Skipping reward training: existing trained reward model found at', trained_reward_path)
    else:
        cmd_reward = f'{python} "{os.path.join(ROOT, "scripts", "pref_convert_and_reward_train.py")}" --prefs-folder "{prefs_folder}" --pairwise --pairwise-epochs {pw_pairwise_epochs}'
        run(cmd_reward, desc='Training pairwise reward model')

    # 2) Run PPO training (main_rlhf_training.py assumed to accept --ppo-steps/--device)
    cmd_ppo = f'{python} "{os.path.join(ROOT, "scripts", "main_rlhf_training.py")}" --ppo-steps {ppo_steps} --device {args.device} '
    run(cmd_ppo, desc='Running PPO RLHF training')

    # 3) Evaluation (quick or full, depending on smoke)
    cmd_eval = f'{python} "{os.path.join(ROOT, "scripts", "evaluate_model.py")}" {sample_arg}'
    run(cmd_eval, desc='Evaluating model')

    # 4) Recompute metrics
    cmd_recompute = f'{python} "{os.path.join(ROOT, "scripts", "recompute_metrics.py")}"'
    run(cmd_recompute, desc='Recomputing metrics')

    # 5) Plot metrics
    cmd_plot = f'{python} "{os.path.join(ROOT, "scripts", "plot_metrics.py")}"'
    run(cmd_plot, desc='Plotting metrics by epoch')

    print('\nFull pipeline finished successfully. Outputs in outputs/ and evaluation_results/')

if __name__ == '__main__':
    main()


# ------------------------------------------------------------
# FILE: .\scripts\run_ppo_rlhf.py
# ------------------------------------------------------------

"""
PPO orchestration script using TRL + Accelerate.

This script is a wrapper that will run PPO training once the SFT checkpoint
and a reward model are available. It requires `trl` (from huggingface/trl),
`transformers`, `accelerate`, and `torch`.

If dependencies are missing, exits with code 2 (so CI/tests can assert).

This file contains a high-level orchestration; fill trainer hyperparams via
command-line args.
"""
import argparse
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fail_missing_libs():
    print("Missing required libraries for PPO RLHF. Install `trl`, `transformers`, `accelerate`, and `torch`.")
    sys.exit(2)


try:
    import torch
    tver = getattr(torch, "__version__", "0.0.0")
    try:
        ver_parts = tver.split("+")[0].split(".")
        tmajor = int(ver_parts[0])
        tminor = int(ver_parts[1]) if len(ver_parts) > 1 else 0
    except Exception:
        tmajor, tminor = 0, 0
    if (tmajor, tminor) < (2, 1):
        fail_missing_libs()
    # trl may not be installed; import lazily where used
    import transformers
except Exception:
    fail_missing_libs()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_model_dir", default="outputs/sft_model_hf")
    p.add_argument("--reward_model_dir", default="outputs/reward_model_hf")
    p.add_argument("--output_dir", default="outputs/ppo_model")
    p.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = p.parse_args()

    # Basic checks
    if not os.path.exists(args.sft_model_dir):
        logging.error("SFT model directory not found: %s", args.sft_model_dir)
        return 3
    if not os.path.exists(args.reward_model_dir):
        logging.error("Reward model directory not found: %s", args.reward_model_dir)
        return 3

    # If TRL is available, run a minimal PPO loop. Otherwise, exit with code 2
    try:
        from transformers import AutoTokenizer
        from transformers import AutoModel 
        # TRL imports (may be optional)
        import numpy as np
        from trl import PPOTrainer, PPOConfig
    except Exception:
        fail_missing_libs()

    # Minimal example using TRL's PPOTrainer (this is a high-level template).
    # Real runs should configure dataset, sampling/evaluation loops, and metrics.
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_dir)
    model = AutoModel.from_pretrained(args.sft_model_dir)

    # Load reward model as a callable: reward_fn(prompts, responses) -> np.array
    def reward_fn(prompts, responses):
        # This wrapper should load the HF reward model and score each prompt+response.
        # For the production script, we expect a directory with AutoModelForSequenceClassification
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_dir)
        reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_dir, num_labels=1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        scores = []
        for p, r in zip(prompts, responses):
            inp = reward_tokenizer(p + "\n" + r, return_tensors='pt', truncation=True, max_length=512).to(reward_model.device)
            with torch.no_grad():
                s = reward_model(**inp).logits.squeeze(-1).cpu().numpy().item()
            scores.append(s)
        return np.array(scores)

    # Build a minimal PPO config
    ppo_config = PPOConfig(
        model_name=args.sft_model_dir,
        learning_rate=1.41e-5,
        batch_size=16,
    )

    # Initialize PPO trainer (the real initialization requires many arguments; this is illustrative)
    ppo_trainer = PPOTrainer(model=model, tokenizer=tokenizer, config=ppo_config, reward_fn=reward_fn)

    # Example training loop (toy): sample prompts, generate responses, compute rewards, step
    # WARNING: Real PPO runs are complex and require robust sampling/evaluation and checkpointing.
    prompts = ["# Write a function that reverses a string\n"] * 8
    for epoch in range(1):
        responses = []
        for prompt in prompts:
            # generate via model
            out = tokenizer(prompt, return_tensors="pt")
            gen = model.generate(**{k: v.to(model.device) for k, v in out.items()}, max_length=128)
            responses.append(tokenizer.decode(gen[0], skip_special_tokens=True))
        rewards = reward_fn(prompts, responses)
        # ppo step (illustrative)
        stats = ppo_trainer.step(prompts, responses, rewards)
        logging.info("PPO step stats: %s", stats)

    os.makedirs(args.output_dir, exist_ok=True)
    marker = os.path.join(args.output_dir, "ppo_complete_marker.txt")
    with open(marker, "w", encoding="utf-8") as f:
        f.write("PPO run complete (toy example).\n")
    logging.info("Wrote PPO marker to %s", marker)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ------------------------------------------------------------
# FILE: .\scripts\smoke_train.py
# ------------------------------------------------------------

import os
import argparse
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import torch


def create_small_dataset():
    # A tiny synthetic dataset of prompt -> reference code (python)
    return [
        ("Write a function that returns the sum of a list of integers.", "def sum_list(lst):\n    return sum(lst)\n"),
        ("Check if a string is a palindrome.", "def is_pal(s):\n    s2 = s.replace(' ', '').lower()\n    return s2 == s2[::-1]\n"),
    ]


def run_sft_one_epoch(model_name: str, output_dir: str):
    data = create_small_dataset()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', use_fast=True)
    model = AutoModel.from_pretrained(model_name)

    texts = ["Prompt: " + p + "\nCode: " + r for p, r in data]
    enc = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')

    class DS(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.enc = enc

        def __len__(self):
            return self.enc['input_ids'].size(0)

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.enc.items()}

    ds = DS(enc)

    args = TrainingArguments(output_dir=output_dir, num_train_epochs=1, per_device_train_batch_size=1, save_strategy='no', logging_steps=1)
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()
    trainer.save_model(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2')
    parser.add_argument('--out', default='outputs/sft_model')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run_sft_one_epoch(args.model, args.out)
    print('SFT smoke training done. Model saved to', args.out)


if __name__ == '__main__':
    main()


# ------------------------------------------------------------
# FILE: .\scripts\train_reward_model.py
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# FILE: .\scripts\train_reward_model_hf.py
# ------------------------------------------------------------

"""
Reward model training using HuggingFace Transformers (pairwise loss).

This script implements a simple pairwise reward model trainer that expects
`datasets_for_training/pairwise_prefs.csv` with columns:
  question, preferred_answer, other_answer, preference

It requires: torch, transformers, datasets.

Behavior:
- If required libraries are missing, exits with code 2 and prints an
  informational message (so CI/tests can assert this behavior).
- If libraries are present, performs a small training loop using
  AutoModelForSequenceClassification (num_labels=1) computing
  pairwise logistic loss: -log(sigmoid(score_pref - score_other)).

This is a general-purpose trainer; tune model name and hyperparams via
command-line args.
"""
import argparse
import csv
import os
import sys
import math
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fail_missing_libs():
    print("Missing required libraries for HF training. Install `torch`, `transformers`, and `datasets`.")
    sys.exit(2)


try:
    import torch
    # require a reasonably recent torch to avoid subtle incompatibilities
    tver = getattr(torch, "__version__", "0.0.0")
    try:
        ver_parts = tver.split("+")[0].split(".")
        tmajor = int(ver_parts[0])
        tminor = int(ver_parts[1]) if len(ver_parts) > 1 else 0
    except Exception:
        tmajor, tminor = 0, 0
    if (tmajor, tminor) < (2, 1):
        # treat older torch as missing to force user to install matching version
        fail_missing_libs()

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader, Dataset
except Exception:
    # avoid stack trace in normal flows; tests assert this exit
    fail_missing_libs()


class PairwiseDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=512):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        prompt = r.get("question", "")
        pref = r.get("preferred_answer", "")
        other = r.get("other_answer", "")
        # Prepare two tokenized examples
        a = self.tokenizer(prompt + "\n" + pref, truncation=True, max_length=self.max_length, return_tensors="pt")
        b = self.tokenizer(prompt + "\n" + other, truncation=True, max_length=self.max_length, return_tensors="pt")
        return {"a": a, "b": b}


def read_pairs(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("preferred_answer") or not r.get("other_answer"):
                continue
            rows.append(r)
    return rows


def pairwise_loss(score_pref, score_other):
    # uses -log(sigmoid(s_pref - s_other))
    x = score_pref - score_other
    return -torch.log(torch.sigmoid(x) + 1e-12).mean()


def collate_batch(batch):
    # batch is list of dicts with tok tensors inside; we will stack input_ids and attention_mask
    def stack(key):
        return torch.cat([item[key] for item in batch], dim=0)

    # Each item has a and b tokenized dicts with tensors of shape (1, seq_len)
    a_input_ids = torch.cat([item["a"]["input_ids"] for item in batch], dim=0)
    a_attn = torch.cat([item["a"]["attention_mask"] for item in batch], dim=0)
    b_input_ids = torch.cat([item["b"]["input_ids"] for item in batch], dim=0)
    b_attn = torch.cat([item["b"]["attention_mask"] for item in batch], dim=0)
    return {
        "a_input_ids": a_input_ids,
        "a_attention_mask": a_attn,
        "b_input_ids": b_input_ids,
        "b_attention_mask": b_attn,
    }


def train(args):
    data_path = os.path.join(args.root, "datasets_for_training", "pairwise_prefs.csv")
    if not os.path.exists(data_path):
        logging.error("Pairwise dataset not found at %s", data_path)
        return 3

    rows = read_pairs(data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)

    ds = PairwiseDataset(rows, tokenizer, max_length=args.max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in dl:
            a_in = {"input_ids": batch["a_input_ids"].to(device), "attention_mask": batch["a_attention_mask"].to(device)}
            b_in = {"input_ids": batch["b_input_ids"].to(device), "attention_mask": batch["b_attention_mask"].to(device)}
            optim.zero_grad()
            out_a = model(**a_in).logits.squeeze(-1)
            out_b = model(**b_in).logits.squeeze(-1)
            loss = pairwise_loss(out_a, out_b)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        logging.info("Epoch %d loss=%.6f", epoch + 1, total_loss / len(dl))

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "reward_model_hf")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logging.info("Saved reward model to %s", save_path)
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA available")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        return train(args)
    except SystemExit as e:
        raise
    except Exception as e:
        logging.exception("Training failed: %s", e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())


# ------------------------------------------------------------
# FILE: .\scripts\train_reward_model_hf_prod.py
# ------------------------------------------------------------

"""
Production-ready reward model trainer (pairwise) using HuggingFace/Transformers.

Features:
- YAML/CLI configuration
- logging with rotating file handler
- checkpointing per epoch and best-checkpoint-by-metric
- evaluation hooks computing Pearson/Spearman correlation on a validation split
- graceful failure with exit code 2 when heavy libs (torch>=2.1, transformers, datasets) are missing

Notes:
- Expects `datasets_for_training/pairwise_prefs.csv` with columns: question, preferred_answer, other_answer, preference
- Writes checkpoints to `output_dir` + `reward_train_checkpoints/`.

This script is intended to be run on a machine with a configured HF environment.
"""
import argparse
import os
import sys
import yaml
import logging
from logging.handlers import RotatingFileHandler


def fail_missing_libs(msg=None):
    if msg:
        print(msg)
    print("Missing required heavy libraries for production HF reward training. Install `torch>=2.1`, `transformers`, and `datasets`.")
    sys.exit(2)


try:
    import math
    import random
    import csv
    import numpy as np
    import torch
    tver = getattr(torch, "__version__", "0.0.0")
    try:
        ver_parts = tver.split("+")[0].split(".")
        tmajor = int(ver_parts[0]); tminor = int(ver_parts[1]) if len(ver_parts) > 1 else 0
    except Exception:
        tmajor, tminor = 0, 0
    if (tmajor, tminor) < (2, 1):
        fail_missing_libs()
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import pearsonr, spearmanr
except Exception:
    fail_missing_libs()


def setup_logger(logpath=None):
    logger = logging.getLogger("reward_trainer")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logpath:
        fh = RotatingFileHandler(logpath, maxBytes=10_000_000, backupCount=3)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class PairwiseDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=512):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        prompt = r.get("question", "")
        pref = r.get("preferred_answer", "")
        other = r.get("other_answer", "")
        # pack prompt + answer as single text
        a = self.tokenizer(prompt + "\n" + pref, truncation=True, max_length=self.max_length, return_tensors="pt")
        b = self.tokenizer(prompt + "\n" + other, truncation=True, max_length=self.max_length, return_tensors="pt")
        label = 1 if r.get("preference") == "left" else 0
        return {"a": a, "b": b, "label": label}


def read_pairs(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("preferred_answer") or not r.get("other_answer"):
                continue
            out.append(r)
    return out


def collate_batch(batch):
    import torch
    a_input_ids = torch.cat([item["a"]["input_ids"] for item in batch], dim=0)
    a_attn = torch.cat([item["a"]["attention_mask"] for item in batch], dim=0)
    b_input_ids = torch.cat([item["b"]["input_ids"] for item in batch], dim=0)
    b_attn = torch.cat([item["b"]["attention_mask"] for item in batch], dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    return {"a_input_ids": a_input_ids, "a_attention_mask": a_attn, "b_input_ids": b_input_ids, "b_attention_mask": b_attn, "labels": labels}


def pairwise_loss(score_pref, score_other):
    import torch
    x = score_pref - score_other
    return -torch.log(torch.sigmoid(x) + 1e-12).mean()


def evaluate_model(model, tokenizer, rows, device, max_length=256):
    # Compute score difference for each pair and compute correlations with label
    model.eval()
    diffs = []
    labels = []
    with torch.no_grad():
        for r in rows:
            prompt = r.get("question", "")
            pref = r.get("preferred_answer", "")
            other = r.get("other_answer", "")
            a = tokenizer(prompt + "\n" + pref, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            b = tokenizer(prompt + "\n" + other, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out_a = model(**a).logits.squeeze(-1).cpu().numpy().item()
            out_b = model(**b).logits.squeeze(-1).cpu().numpy().item()
            diffs.append(out_a - out_b)
            labels.append(1 if r.get("preference") == "left" else 0)
    # correlation expects two continuous arrays; convert labels to -1/1 or use diffs vs labels
    try:
        pearson = pearsonr(diffs, labels)[0]
    except Exception:
        pearson = float("nan")
    try:
        spearman = spearmanr(diffs, labels)[0]
    except Exception:
        spearman = float("nan")
    return {"pearson": pearson, "spearman": spearman}


def save_checkpoint(model, tokenizer, outdir, epoch, metric=None):
    import os
    path = os.path.join(outdir, f"checkpoint_epoch{epoch}")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    if metric is not None:
        with open(os.path.join(path, "metric.txt"), "w", encoding="utf-8") as f:
            f.write(str(metric))
    return path


def train_from_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logpath = os.path.join(cfg.get("output_dir", "outputs"), "train_reward.log")
    logger = setup_logger(logpath)

    data_path = os.path.join(cfg.get("root", "."), "datasets_for_training", "pairwise_prefs.csv")
    if not os.path.exists(data_path):
        logger.error("Pairwise data not found: %s", data_path)
        return 3

    rows = read_pairs(data_path)
    random_seed = cfg.get("seed", 42)
    import random
    random.seed(random_seed)
    random.shuffle(rows)
    n = len(rows)
    val_frac = cfg.get("val_fraction", 0.1)
    nval = max(1, int(n * val_frac))
    val_rows = rows[:nval]
    train_rows = rows[nval:]

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("model_name", "bert-base-uncased"))
    model = AutoModelForSequenceClassification.from_pretrained(cfg.get("model_name", "bert-base-uncased"), num_labels=1)
    model.to(device)

    train_ds = PairwiseDataset(train_rows, tokenizer, max_length=cfg.get("max_length", 256))
    val_ds = val_rows
    from torch.utils.data import DataLoader
    dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8), shuffle=True, collate_fn=collate_batch)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 5e-5))
    total_steps = len(dl) * cfg.get("epochs", 1)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=cfg.get("warmup_steps", 0), num_training_steps=total_steps)

    best_metric = -float("inf")
    ckpt_dir = os.path.join(cfg.get("output_dir", "outputs"), "reward_train_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, cfg.get("epochs", 1) + 1):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(dl, start=1):
            optim.zero_grad()
            a = {"input_ids": batch["a_input_ids"].to(device), "attention_mask": batch["a_attention_mask"].to(device)}
            b = {"input_ids": batch["b_input_ids"].to(device), "attention_mask": batch["b_attention_mask"].to(device)}
            out_a = model(**a).logits.squeeze(-1)
            out_b = model(**b).logits.squeeze(-1)
            loss = pairwise_loss(out_a, out_b)
            loss.backward()
            optim.step()
            scheduler.step()
            running_loss += loss.item()
            if i % cfg.get("log_every", 10) == 0:
                logger.info("Epoch %d step %d loss=%.6f", epoch, i, running_loss / i)

        # evaluation
        metrics = evaluate_model(model, tokenizer, val_ds, device, max_length=cfg.get("max_length", 256))
        logger.info("Epoch %d eval: pearson=%.4f spearman=%.4f", epoch, metrics.get("pearson"), metrics.get("spearman"))

        # checkpoint
        ckpt_path = save_checkpoint(model, tokenizer, ckpt_dir, epoch, metric=metrics.get("pearson"))
        if metrics.get("pearson") and metrics.get("pearson") > best_metric:
            best_metric = metrics.get("pearson")
            best_path = os.path.join(cfg.get("output_dir", "outputs"), "best_reward_model")
            # copy latest ckpt to best
            import shutil
            if os.path.exists(best_path):
                shutil.rmtree(best_path)
            shutil.copytree(ckpt_path, best_path)
            logger.info("Saved best model to %s (pearson=%.4f)", best_path, best_metric)

    # final save
    final_path = os.path.join(cfg.get("output_dir", "outputs"), "final_reward_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("Training complete. Final model saved to %s", final_path)
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/reward_train.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        print(f"Config not found: {cfg_path}")
        return 2
    try:
        return train_from_config(cfg_path)
    except SystemExit:
        raise
    except Exception as e:
        print("Training failed:", e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())


# ------------------------------------------------------------
# FILE: .\scripts\update_eval_metrics.py
# ------------------------------------------------------------

#!/usr/bin/env python3
import os
import pandas as pd
from src.metrics_tracker import MetricsTracker

EVAL_DETAILED = 'evaluation_results/final_evaluation_detailed.csv'


def is_code_like(text: str) -> bool:
    if text is None or text != text:
        return False
    s = str(text)
    # heuristics: presence of typical code tokens or multiple lines
    tokens = ['def ', 'return ', 'import ', 'from ', 'class ', '\n', '():', '):', 'print(', 'self.', ':']
    hits = sum(1 for t in tokens if t in s)
    # multi-line or at least two token hits
    return ('\n' in s and len(s.splitlines()) > 1) or hits >= 2


def main():
    if not os.path.exists(EVAL_DETAILED):
        print('File not found:', EVAL_DETAILED)
        return

    df = pd.read_csv(EVAL_DETAILED)
    mt = MetricsTracker(output_dir='outputs')

    updated = 0
    for i, row in df.iterrows():
        prompt = row.get('prompt')
        gen = row.get('generated_code')
        ref = row.get('reference')
        if not pd.isna(ref) and is_code_like(ref):
            # compute metrics
            try:
                metrics = mt.calculate_metrics([prompt], [gen], [ref])
                for k, v in metrics.items():
                    df.at[i, k] = v
                updated += 1
            except Exception as e:
                print('Failed metrics for row', i, e)
        else:
            # attempt to mark codebleu/ruby as NaN explicitly to show missing reference or non-code ref
            df.at[i, 'codebleu'] = 0.0
            df.at[i, 'ruby'] = 0.0

    print('Updated metrics for', updated, 'rows out of', len(df))
    df.to_csv(EVAL_DETAILED, index=False)
    print('Saved updated', EVAL_DETAILED)


if __name__ == '__main__':
    main()


# ------------------------------------------------------------
# FILE: .\src\config.py
# ------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class CodeRLHFConfig:
    """Configuration for improved code generation RLHF."""
    
    # Model settings
    model_name: str = "microsoft/CodeGPT-small-py"
    # Use a smaller distilled model by default to fit low-memory GPUs during reward training
    reward_model_name: str = "distilbert-base-uncased"
    
    # Dataset settings
    dataset_path: str = "./datasets_for_eval"
    human_eval_path: str = "./evaluation_results_server"
    
    # Training settings
    learning_rate: float = 2e-5
    # Keep default batch sizes small to accommodate 4GB GPUs
    batch_size: int = 1
    ppo_epochs: int = 10
    reward_training_epochs: int = 5
    mini_batch_size: int = 1
    
    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.2
    
    # Code-specific settings
    max_prompt_length: int = 256
    max_response_length: int = 512
    min_code_length: int = 20
    
    # Hardware settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging and saving
    output_dir: str = "./outputs"
    save_steps: int = 500
    logging_steps: int = 100
    # Early stopping target mean reward for RLHF training (0-1 scale)
    early_stop_reward: float = 0.8
    
    # Reproducibility
    seed: int = 42

# ------------------------------------------------------------
# FILE: .\src\metrics_tracker.py
# ------------------------------------------------------------

#!/usr/bin/env python3
"""Track training metrics and save to file."""

import json
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Track and save training metrics."""
    
    def __init__(self, output_dir: str = "./training_metrics"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Metrics file
        self.metrics_file = os.path.join(output_dir, "training_metrics.csv")
        self.detailed_metrics_file = os.path.join(output_dir, "detailed_metrics.json")
        
        # Initialize metrics storage
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.metrics_file):
            pd.DataFrame(columns=[
                'timestamp', 'epoch', 'batch', 'reward', 'syntax_score', 
                'structure_score', 'bertscore', 'bleu', 'codebleu', 'rouge', 'ruby',
                'kl_divergence', 'policy_loss', 'value_loss'
            ]).to_csv(self.metrics_file, index=False)
    
    def calculate_bertscore(self, generated_texts: List[str], reference_texts: List[str]) -> float:
        """Calculate BERTScore metric."""
        try:
            from bert_score import BERTScorer
            scorer = BERTScorer(lang="en")
            P, R, F1 = scorer.score(generated_texts, reference_texts)
            return F1.mean().item()
        except ImportError:
            logger.warning("BERTScore not available, installing...")
            os.system("pip install bert-score")
            try:
                from bert_score import BERTScorer
                scorer = BERTScorer(lang="en")
                P, R, F1 = scorer.score(generated_texts, reference_texts)
                return F1.mean().item()
            except:
                return 0.0
        except Exception as e:
            logger.error(f"Error calculating BERTScore: {e}")
            return 0.0
    
    def calculate_bleu(self, generated_texts: List[str], reference_texts: List[str]) -> float:
        """Calculate BLEU score."""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            smoothie = SmoothingFunction().method4
            
            scores = []
            for gen, ref in zip(generated_texts, reference_texts):
                # Tokenize
                gen_tokens = gen.split()
                ref_tokens = ref.split()
                
                if len(gen_tokens) > 0 and len(ref_tokens) > 0:
                    score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)
                    scores.append(score)
            
            return sum(scores) / len(scores) if scores else 0.0
            
        except ImportError:
            logger.warning("NLTK not available, installing...")
            os.system("pip install nltk")
            try:
                import nltk
                nltk.download('punkt', quiet=True)
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                smoothie = SmoothingFunction().method4
                
                scores = []
                for gen, ref in zip(generated_texts, reference_texts):
                    gen_tokens = gen.split()
                    ref_tokens = ref.split()
                    
                    if len(gen_tokens) > 0 and len(ref_tokens) > 0:
                        score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)
                        scores.append(score)
                
                return sum(scores) / len(scores) if scores else 0.0
            except:
                return 0.0
        except Exception as e:
            logger.error(f"Error calculating BLEU: {e}")
            return 0.0
    
    def calculate_codebleu(self, generated_codes: List[str], reference_codes: List[str]) -> float:
        """Calculate CodeBLEU score."""
        try:
            # Some codebleu implementations expect files; to be robust we write temporary files
            # Defensive: filter out pairs where reference is empty and ensure equal lengths
            paired = [(p, r) for p, r in zip(generated_codes, reference_codes) if (r and r.strip())]
            if not paired:
                logger.warning("No non-empty references available for CodeBLEU calculation")
                return 0.0
            preds_filtered, refs_filtered = zip(*paired)

            try:
                from codebleu import calc_codebleu
            except Exception as imp_e:
                logger.exception(f"Failed to import codebleu.calc_codebleu: {imp_e}")
                raise
            import tempfile
            import json

            with tempfile.TemporaryDirectory() as td:
                refs_path = os.path.join(td, 'refs.txt')
                preds_path = os.path.join(td, 'preds.txt')
                # Write one code per line
                with open(refs_path, 'w', encoding='utf-8') as fr:
                    for r in refs_filtered:
                        fr.write(r.replace('\n', '\\n') + '\n')
                with open(preds_path, 'w', encoding='utf-8') as fp:
                    for p in preds_filtered:
                        fp.write(p.replace('\n', '\\n') + '\n')

                # Call calc_codebleu - prefer list API, but fall back to file-based or weighted ngram match
                try:
                    # Prefer list API when available. Provide references as a list of
                    # single-reference lists: [[ref1], [ref2], ...] and predictions
                    # as a flat list so shapes match typical list-based APIs.
                    results = calc_codebleu(references=[[r] for r in refs_filtered], predictions=list(preds_filtered), lang="python")
                except Exception as e:
                    logger.warning(f"calc_codebleu failed, attempting fallback: {e}")
                    # Try file-based invocation if supported
                    try:
                        results = calc_codebleu(refs_path, preds_path, lang="python")
                    except Exception as e2:
                        logger.warning(f"file-based calc_codebleu also failed: {e2}")
                        # Try weighted_ngram_match as a lightweight fallback
                        try:
                            from codebleu.codebleu import weighted_ngram_match
                            # weighted_ngram_match expects a list of reference-lists
                            # and a list of tokenized predictions. We provide refs as
                            # [[tokens_of_ref1], [tokens_of_ref2], ...]
                            refs_tokenized = [[r.split()] for r in refs_filtered]
                            preds_tokenized = [p.split() for p in preds_filtered]
                            wg = weighted_ngram_match(refs_tokenized, preds_tokenized, lang='python')
                            results = {'codebleu': wg}
                        except Exception as e3:
                            logger.warning(f"weighted_ngram_match fallback failed: {e3}")
                            # Final fallback: compute simple BLEU average
                            try:
                                bleu_fallback = self.calculate_bleu(list(preds_filtered), list(refs_filtered))
                                results = {'codebleu': bleu_fallback}
                            except Exception as e4:
                                logger.warning(f"Final BLEU fallback failed: {e4}")
                                results = None

            # codebleu implementations may return dict or float
            if isinstance(results, dict):
                return results.get('codebleu', 0.0) or results.get('total', 0.0)
            elif isinstance(results, (float, int)):
                return float(results)
            else:
                return 0.0
        except Exception as e:
            # Log full traceback for debugging any failures during import or execution
            logger.exception("CodeBLEU calculation failed")
            return 0.0
    
    def calculate_rouge(self, generated_texts: List[str], reference_texts: List[str]) -> float:
        """Calculate ROUGE score."""
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = []
            
            for gen, ref in zip(generated_texts, reference_texts):
                score = scorer.score(ref, gen)
                scores.append(score['rougeL'].fmeasure)
            
            return sum(scores) / len(scores) if scores else 0.0
            
        except ImportError:
            logger.warning("ROUGE not available, installing...")
            os.system("pip install rouge-score")
            try:
                from rouge_score import rouge_scorer
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                scores = []
                
                for gen, ref in zip(generated_texts, reference_texts):
                    score = scorer.score(ref, gen)
                    scores.append(score['rougeL'].fmeasure)
                
                return sum(scores) / len(scores) if scores else 0.0
            except:
                return 0.0
        except Exception as e:
            logger.error(f"Error calculating ROUGE: {e}")
            return 0.0
    
    def calculate_metrics(self, prompts: List[str], generated_texts: List[str], reference_texts: List[str] = None) -> Dict[str, float]:
        """Calculate all metrics for generated texts."""
        if reference_texts is None:
            # Use prompts as simple references for demonstration
            reference_texts = prompts
        # Defensive: many metric implementations crash or return invalid results
        # when reference strings are empty. Compute each metric on the subset
        # of (gen, ref) pairs where ref is non-empty. If no valid pairs exist,
        # return 0.0 for that metric.

        def _filter_nonempty_refs(gens, refs):
            paired = [(g, r) for g, r in zip(gens, refs) if r and r.strip()]
            if not paired:
                return [], []
            gs, rs = zip(*paired)
            return list(gs), list(rs)

        # BERTScore
        gs_bs, rs_bs = _filter_nonempty_refs(generated_texts, reference_texts)
        if gs_bs:
            bscore = self.calculate_bertscore(gs_bs, rs_bs)
        else:
            logger.warning("Empty reference sentence detected; setting raw BERTScores to 0.")
            bscore = 0.0

        # BLEU
        gs_bleu, rs_bleu = _filter_nonempty_refs(generated_texts, reference_texts)
        if gs_bleu:
            bleu = self.calculate_bleu(gs_bleu, rs_bleu)
        else:
            bleu = 0.0

        # CodeBLEU (function already filters internally but keep count logging)
        gs_cb, rs_cb = _filter_nonempty_refs(generated_texts, reference_texts)
        if gs_cb:
            codebleu = self.calculate_codebleu(gs_cb, rs_cb)
        else:
            logger.warning("No non-empty references available for CodeBLEU calculation")
            codebleu = 0.0

        # ROUGE
        gs_rouge, rs_rouge = _filter_nonempty_refs(generated_texts, reference_texts)
        if gs_rouge:
            rouge = self.calculate_rouge(gs_rouge, rs_rouge)
        else:
            rouge = 0.0

        metrics = {
            'bertscore': bscore,
            'bleu': bleu,
            'codebleu': codebleu,
            'rouge': rouge,
        }
        # Try to compute RUBY metric if available; surface exceptions to logs
        try:
            from src.metrics.ruby_metric import RUBYMetric
            ruby = RUBYMetric()
            # compute average ruby only over non-empty reference pairs
            gs_ruby, rs_ruby = _filter_nonempty_refs(generated_texts, reference_texts)
            ruby_scores = []
            for gen, ref in zip(gs_ruby, rs_ruby):
                try:
                    score = ruby.compute_ruby(ref, gen)
                    if score is not None:
                        ruby_scores.append(score)
                except Exception as e:
                    logger.warning(f"RUBY compute failed for one pair: {e}")
                    continue
            metrics['ruby'] = sum(ruby_scores) / len(ruby_scores) if ruby_scores else 0.0
        except Exception as e:
            logger.warning(f"RUBY metric not available or failed: {e}")
            metrics['ruby'] = 0.0
        
        return metrics
    
    def record_batch_metrics(self, epoch: int, batch: int, batch_stats: Dict[str, float], 
                           prompts: List[str] = None, generated_texts: List[str] = None):
        """Record metrics for a training batch."""
        timestamp = datetime.now().isoformat()
        
        # Basic metrics from batch_stats
        metrics_record = {
            'timestamp': timestamp,
            'epoch': epoch,
            'batch': batch,
            'reward': batch_stats.get('mean_reward', 0),
            'syntax_score': batch_stats.get('syntax_score', 0),
            'structure_score': batch_stats.get('structure_score', 0),
            'kl_divergence': batch_stats.get('kl_divergence', 0),
            'policy_loss': batch_stats.get('policy_loss', 0),
            'value_loss': batch_stats.get('value_loss', 0),
        }
        
        # Calculate additional metrics if texts are provided
        if prompts is not None and generated_texts is not None:
            try:
                additional_metrics = self.calculate_metrics(prompts, generated_texts)
                metrics_record.update(additional_metrics)
            except Exception as e:
                logger.error(f"Error calculating additional metrics: {e}")
                # Set default values for failed metrics
                metrics_record.update({
                    'bertscore': 0.0,
                    'bleu': 0.0,
                    'codebleu': 0.0,
                    'rouge': 0.0,
                })
        
        # Add to history
        self.metrics_history.append(metrics_record)
        
        # Save to CSV
        try:
            df = pd.DataFrame([metrics_record])
            df.to_csv(self.metrics_file, mode='a', header=False, index=False)
        except Exception as e:
            logger.error(f"Error saving metrics to CSV: {e}")
        
        # Log metrics
        logger.info(f"Metrics - Epoch {epoch}, Batch {batch}: "
                   f"Reward: {metrics_record['reward']:.4f}, "
                   f"Syntax: {metrics_record['syntax_score']:.4f}, "
                   f"BERTScore: {metrics_record.get('bertscore', 0):.4f}, "
                   f"BLEU: {metrics_record.get('bleu', 0):.4f}")
        
        return metrics_record
    
    def save_detailed_metrics(self):
        """Save detailed metrics to JSON file."""
        try:
            with open(self.detailed_metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
            logger.info(f"Detailed metrics saved to: {self.detailed_metrics_file}")
        except Exception as e:
            logger.error(f"Error saving detailed metrics: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all metrics."""
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        summary = {}
        
        for column in df.columns:
            if column not in ['timestamp', 'epoch', 'batch']:
                summary[f'avg_{column}'] = df[column].mean()
                summary[f'std_{column}'] = df[column].std()
                summary[f'min_{column}'] = df[column].min()
                summary[f'max_{column}'] = df[column].max()
        
        return summary
    
    def plot_metrics(self, metrics_to_plot: List[str] = None):
        """Plot training metrics (optional)."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.metrics_history:
                logger.warning("No metrics data to plot")
                return
            
            df = pd.DataFrame(self.metrics_history)
            
            if metrics_to_plot is None:
                metrics_to_plot = ['reward', 'syntax_score', 'bertscore', 'bleu']
            
            # Create subplots
            n_metrics = len(metrics_to_plot)
            fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3*n_metrics))
            
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics_to_plot):
                if metric in df.columns:
                    axes[i].plot(df['batch'], df[metric], label=metric, alpha=0.7)
                    axes[i].set_ylabel(metric)
                    axes[i].legend()
                    
                    # Add epoch separators
                    epoch_changes = df[df['epoch'].diff() != 0].index
                    for change_idx in epoch_changes:
                        axes[i].axvline(x=change_idx, color='red', alpha=0.3, linestyle='--')
            
            plt.xlabel('Batch')
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, "training_metrics_plot.png")
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Metrics plot saved to: {plot_path}")
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting metrics: {e}")

# ------------------------------------------------------------
# FILE: .\src\__init__.py
# ------------------------------------------------------------



# ------------------------------------------------------------
# FILE: .\src\data\dataset_loader.py
# ------------------------------------------------------------

from typing import List, Dict, Any
import pandas as pd
import os
from glob import glob
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)

class CodeDatasetLoader:
    """Improved dataset loader for code generation tasks."""
    
    def __init__(self, config):
        self.config = config
    
    def load_dataset(self) -> Dataset:
        """Load dataset from CSV files."""
        dataset_path = self.config.dataset_path
        csv_files = glob(os.path.join(dataset_path, "*.csv"))
        
        if not csv_files:
            logger.warning("No CSV files found, using synthetic dataset")
            return self._load_synthetic_dataset()
        
        all_prompts = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"Loaded dataset: {os.path.basename(csv_file)} with {len(df)} rows")
                
                # Find prompt column
                prompt_column = None
                for col in ['Question', 'Prompt', 'prompt', 'instruction', 'input', 'text']:
                    if col in df.columns:
                        prompt_column = col
                        break
                
                if prompt_column is None:
                    prompt_column = df.columns[0]
                
                prompts = df[prompt_column].dropna().astype(str).tolist()
                all_prompts.extend(prompts)
                
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                continue
        
        if not all_prompts:
            raise ValueError("No valid prompts found in any CSV files")
        
        dataset = Dataset.from_dict({"prompt": all_prompts})
        logger.info(f"Successfully loaded {len(all_prompts)} prompts")
        
        return dataset
    
    def _load_synthetic_dataset(self) -> Dataset:
        """Create synthetic dataset as fallback."""
        synthetic_prompts = [
            "Write a Python function to calculate factorial",
            "Create a function to reverse a string in Python",
            "Write code to read a CSV file and print its contents",
            "Create a Python class for a simple calculator",
            "Write a function to check if a number is prime",
            "Create code to download a file from URL using requests",
            "Write a Python script to parse JSON data",
            "Create a function to sort a list of dictionaries by key"
        ]
        
        return Dataset.from_dict({"prompt": synthetic_prompts})

# ------------------------------------------------------------
# FILE: .\src\data\dataset_utils.py
# ------------------------------------------------------------

from typing import Tuple, List, Dict, Any
import sys
from datasets import Dataset, load_dataset
import logging
import re
import pandas as pd
import os
from glob import glob

logger = logging.getLogger(__name__)

class CodeDatasetLoader:
    """Improved dataset loader for code generation tasks."""
    
    def __init__(self, config) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate dataset configuration."""
        required_fields = ['dataset_name', 'max_prompt_length']
        for field in required_fields:
            if not hasattr(self.config, field):
                raise ValueError(f"Config missing required field: {field}")

    def _load_custom_eval_dataset(self) -> Dataset:
        """Load custom evaluation dataset from CSV files."""
        try:
            dataset_path = self.config.dataset_path
            csv_files = glob(os.path.join(dataset_path, "*.csv"))
            
            if not csv_files:
                logger.warning("No CSV files found, using synthetic dataset")
                return self._load_synthetic_code_dataset()
            
            all_prompts = []
            all_codes = []
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    logger.info(f"Loaded dataset: {os.path.basename(csv_file)} with {len(df)} rows")
                    
                    # Extract prompts from various possible columns
                    prompt_column = None
                    for col in ['Question', 'Prompt', 'prompt', 'instruction', 'input']:
                        if col in df.columns:
                            prompt_column = col
                            break
                    
                    if prompt_column is None:
                        logger.warning(f"No prompt column found in {csv_file}, using first column")
                        prompt_column = df.columns[0]
                    
                    prompts = df[prompt_column].dropna().astype(str).tolist()
                    all_prompts.extend(prompts)
                    
                    # Use empty strings for code as we're generating it
                    all_codes.extend([""] * len(prompts))
                    
                except Exception as e:
                    logger.error(f"Error loading {csv_file}: {e}")
                    continue
            
            if not all_prompts:
                raise ValueError("No valid prompts found in any CSV files")
                
            dataset = Dataset.from_dict({
                "prompt": all_prompts,
                "code": all_codes
            })
            
            logger.info(f"Successfully loaded {len(all_prompts)} prompts from {len(csv_files)} files")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load custom eval dataset: {e}")
            return self._load_synthetic_code_dataset()

    def _load_synthetic_code_dataset(self) -> Dataset:
        """Create synthetic code generation prompts optimized for the dataset style."""
        synthetic_prompts = [
            "Write Python code to send a signal to the current process",
            "How to decode a hex string to UTF-8 in Python?",
            "Remove None values from a dictionary in Python",
            "Capture output of system commands using subprocess",
            "Find intersection between two pandas Series",
            "Send HTTP headers to a client",
            "Format datetime string to extract date only",
            "Split multi-line string into separate lines",
            "Concatenate list elements with a colon",
            "Get first object from Django model queryset",
            "Calculate sum of 2D numpy array rows",
            "Run Python script with arguments using subprocess",
            "Parse time string with milliseconds",
            "Convert string with commas to float",
            "Set Python path in script",
            "Split string using regex pattern",
            "Open file in append mode",
            "Download file from URL and save locally"
        ]
        
        return Dataset.from_dict({
            "prompt": synthetic_prompts,
            "code": [""] * len(synthetic_prompts)
        })

    def load_dataset(self) -> Dataset:
        """Main method to load and prepare the dataset."""
        logger.info(f"Loading code dataset: {self.config.dataset_name}")
        
        try:
            if self.config.dataset_name == "code_search_net":
                dataset = self._load_code_search_net()
            elif self.config.dataset_name == "synthetic_code":
                dataset = self._load_synthetic_code_dataset()
            elif self.config.dataset_name == "custom_code":
                dataset = self._load_custom_eval_dataset()
            else:
                dataset = self._load_custom_dataset()
            
            return self._format_code_dataset(dataset)
        except Exception as e:
            logger.error(f"Failed to load code dataset: {e}")
            raise

    def _format_code_dataset(self, dataset: Dataset) -> Dataset:
        """Format dataset for code generation training."""
        def format_code_prompts(batch: Dict) -> Dict:
            """Format batch of code prompts."""
            prompts = []
            for prompt in batch["prompt"]:
                prompt = str(prompt).strip()
                
                # Clean and standardize prompts
                if prompt.startswith('"') and prompt.endswith('"'):
                    prompt = prompt[1:-1]
                
                # Ensure prompt is properly formatted
                if not prompt.endswith((".", "?", "!")):
                    prompt += "."
                
                # Add Python context if missing
                if not any(keyword in prompt.lower() for keyword in 
                          ["python", "code", "function", "def ", "import"]):
                    prompt = "Write Python code to " + prompt.lower()
                
                prompts.append(prompt)
            
            return {"prompt": prompts}
        
        return dataset.map(format_code_prompts, batched=True)

# ------------------------------------------------------------
# FILE: .\src\data\human_eval_processor.py
# ------------------------------------------------------------

import json
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def process_human_evaluations(json_dir: str) -> pd.DataFrame:
    """Process human evaluations for reward model training."""
    evaluations = []
    json_files = Path(json_dir).glob("*.json")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for q in data.get('questions_df', []):
                # Normalize scores from -2..2 to 0..1
                def normalize_score(score):
                    return (score + 2) / 4.0
                
                evaluation = {
                    'question': q['Question'],
                    'answer': q['Answer'],
                    'model_tag': q['MODEL_TAG'],
                    'consistent_score': normalize_score(q.get('consistent_L', 0) + q.get('consistent_R', 0)),
                    'correct_score': normalize_score(q.get('correct_L', 0) + q.get('correct_R', 0)),
                    'useful_score': normalize_score(q.get('useful_L', 0) + q.get('useful_R', 0)),
                    'total_score': normalize_score(
                        q.get('consistent_L', 0) + q.get('consistent_R', 0) +
                        q.get('correct_L', 0) + q.get('correct_R', 0) +
                        q.get('useful_L', 0) + q.get('useful_R', 0)
                    )
                }
                evaluations.append(evaluation)
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    df = pd.DataFrame(evaluations)
    logger.info(f"Processed {len(df)} human evaluations")
    return df

# ------------------------------------------------------------
# FILE: .\src\data\__init__.py
# ------------------------------------------------------------



# ------------------------------------------------------------
# FILE: .\src\metrics\ruby_metric.py
# ------------------------------------------------------------

import ast
from typing import Optional, List
try:
    import networkx as nx
except Exception:
    nx = None

class RUBYMetric:
    def __init__(self):
        self.available_representations = ['PDG', 'AST', 'text']

    def compute_ruby(self, reference_code: str, translated_code: str) -> float:
        # Try GRS (PDG level)
        grs_score = self.compute_grs(reference_code, translated_code)
        if grs_score is not None:
            return grs_score

        # Try TRS (AST level)
        trs_score = self.compute_trs(reference_code, translated_code)
        if trs_score is not None:
            return trs_score

        # Fallback to STS (string/token level)
        return self.compute_sts(reference_code, translated_code)

    def compute_grs(self, reference_code: str, translated_code: str) -> Optional[float]:
        if nx is None:
            return None
        try:
            pdg_ref = self.build_pdg(reference_code)
            pdg_trans = self.build_pdg(translated_code)
            if pdg_ref is None or pdg_trans is None:
                return None
            ged = self.graph_edit_distance(pdg_ref, pdg_trans)
            pdg_size = len(pdg_ref.nodes) + len(pdg_ref.edges) + len(pdg_trans.nodes) + len(pdg_trans.edges)
            return 1.0 - (ged / pdg_size) if pdg_size > 0 else 0.0
        except Exception:
            return None

    def compute_trs(self, reference_code: str, translated_code: str) -> Optional[float]:
        try:
            ast_ref = self.parse_ast(reference_code)
            ast_trans = self.parse_ast(translated_code)
            if ast_ref is None or ast_trans is None:
                return None
            ted = self.tree_edit_distance(ast_ref, ast_trans)
            tree_size = self.count_ast_nodes(ast_ref) + self.count_ast_nodes(ast_trans)
            return 1.0 - (ted / tree_size) if tree_size > 0 else 0.0
        except Exception:
            return None

    def compute_sts(self, reference_code: str, translated_code: str) -> float:
        tokens_ref = self.tokenize_code(reference_code)
        tokens_trans = self.tokenize_code(translated_code)
        sed = self.string_edit_distance(tokens_ref, tokens_trans)
        max_length = max(len(tokens_ref), len(tokens_trans))
        return 1.0 - (sed / max_length) if max_length > 0 else 1.0

    def build_pdg(self, code: str):
        try:
            tree = self.parse_ast(code)
            if tree is None:
                return None
            pdg = nx.DiGraph()
            # simplified: add one node per ast node
            for i, node in enumerate(ast.walk(tree)):
                pdg.add_node(i, type=type(node).__name__)
            return pdg
        except Exception:
            return None

    def parse_ast(self, code: str):
        try:
            return ast.parse(code)
        except SyntaxError:
            return None

    def tokenize_code(self, code: str) -> List[str]:
        tokens = []
        current_token = ""
        for char in code:
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            elif char in '(){}[];.,=+-*/<>!&|':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            else:
                current_token += char
        if current_token:
            tokens.append(current_token)
        return tokens

    def string_edit_distance(self, tokens1: List[str], tokens2: List[str]) -> int:
        m, n = len(tokens1), len(tokens2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif tokens1[i-1] == tokens2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n]

    def tree_edit_distance(self, tree1, tree2) -> int:
        return abs(self.count_ast_nodes(tree1) - self.count_ast_nodes(tree2))

    def count_ast_nodes(self, node) -> int:
        count = 1
        for child in ast.iter_child_nodes(node):
            count += self.count_ast_nodes(child)
        return count

    def graph_edit_distance(self, graph1, graph2) -> int:
        return abs(len(graph1.nodes) - len(graph2.nodes)) + abs(len(graph1.edges) - len(graph2.edges))


# ------------------------------------------------------------
# FILE: .\src\models\model_loader.py
# ------------------------------------------------------------

import logging
import ast
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List

logger = logging.getLogger(__name__)

class ModelLoader:
    """Improved model loader with better error handling."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
    
    def load_models(self):
        """Load tokenizer and policy model."""
        logger.info(f"Loading models with device: {self.device}")
        
        # Load tokenizer
        tokenizer = self._load_tokenizer()
        
        # Load policy model
        policy_model = self._load_policy_model()
        
        # Reference model (can be None for PPOTrainer)
        ref_model = None
        
        logger.info("Models loaded successfully!!")
        return tokenizer, policy_model, ref_model
    
    def _load_tokenizer(self):
        """Load and configure tokenizer."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"First attempt failed: {e}, trying fallback...")
            try:
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                logger.info("Using GPT-2 tokenizer as fallback")
            except Exception as e2:
                logger.error(f"Failed to load tokenizer: {e2}")
                raise
        
        # Configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        return tokenizer
    
    def _load_policy_model(self):
        """Load policy model with value head."""
        try:
            # Try to use TRL's AutoModelWithValueHead if available.
            try:
                from trl import AutoModelWithValueHead
                model = AutoModelWithValueHead.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                model = model.to(self.device)
                return model
            except Exception:
                # If TRL is not available or incompatible, fall back to a standard causal LM.
                logger.warning("TRL AutoModelWithValueHead not available or failed to import; falling back to AutoModel")
                model = AutoModel.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                model = model.to(self.device)
                return model
        except Exception as e:
            logger.error(f"Failed to load policy model: {e}")
            raise

class CodeRewardModel:
    """Enhanced reward model for code generation with better metrics."""
    
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device(config.device)
        
        # Optimized metric weights for code quality
        self.metric_weights = {
            'syntax': 0.4,
            'structure': 0.3,
            'relevance': 0.2,
            'completeness': 0.1
        }
        
        # Enhanced code quality indicators
        self.good_practices = [
            'def ', 'return ', 'import ', 'from ', 'class ', 'try:', 'except ',
            'if __name__', 'with open', 'isinstance', 'len(', 'range(', 'subprocess.',
            'datetime.', 'pandas.', 'numpy.', 'os.', 'sys.'
        ]
        
        self.bad_practices = [
            'eval(', 'exec(', 'input()', 'while True:', 'import *',
            'except:', 'except Exception:', 'print(', 'exit()'
        ]

    def compute_reward(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute enhanced rewards for code generation."""
        rewards = []
        
        for prompt, code in zip(prompts, responses):
            if not code or len(code.strip()) < self.config.min_code_length:
                rewards.append(-1.0)
                continue
            
            reward = 0.0
            
            # 1. Syntax validity (most important)
            syntax_score = self._check_syntax(code)
            reward += syntax_score * self.metric_weights['syntax']
            
            # 2. Code structure and best practices
            structure_score = self._check_structure(code)
            reward += structure_score * self.metric_weights['structure']
            
            # 3. Relevance to prompt
            relevance_score = self._check_relevance(prompt, code)
            reward += relevance_score * self.metric_weights['relevance']
            
            # 4. Code completeness
            completeness_score = self._check_completeness(code)
            reward += completeness_score * self.metric_weights['completeness']
            
            # Penalties for bad practices
            penalties = self._check_bad_practices(code)
            reward -= penalties
            
            # Normalize and ensure reasonable range
            reward = max(min(reward, 1.0), -0.5)
            
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def _check_syntax(self, code: str) -> float:
        """Enhanced syntax checking."""
        try:
            ast.parse(code)
            
            lines = code.strip().split('\n')
            if len(lines) >= 2:
                return 0.9
            else:
                return 0.7
                
        except SyntaxError as e:
            error_msg = str(e)
            if 'unexpected EOF' in error_msg or 'parenthesis' in error_msg:
                return 0.3
            return 0.0

    def _check_structure(self, code: str) -> float:
        """Check code structure and best practices."""
        score = 0.0
        lines = [line for line in code.split('\n') if line.strip()]
        
        if any('import ' in line for line in lines):
            score += 0.3
        
        if any('def ' in line for line in lines):
            score += 0.4
        
        good_count = sum(1 for practice in self.good_practices if practice in code)
        score += min(good_count * 0.05, 0.2)
        
        if len(lines) >= 2 and len(lines) <= 15:
            score += 0.1
        
        return min(score, 1.0)

    def _check_relevance(self, prompt: str, code: str) -> float:
        """Enhanced relevance checking."""
        prompt_lower = prompt.lower()
        code_lower = code.lower()
        
        relevance = 0.0
        
        keyword_mappings = {
            'signal': ['signal', 'kill', 'pid'],
            'decode': ['decode', 'hex', 'utf'],
            'dictionary': ['dict', 'kwargs', 'items'],
            'subprocess': ['subprocess', 'call', 'check_output'],
            'pandas': ['pandas', 'series', 'dataframe'],
            'http': ['http', 'header', 'client'],
            'datetime': ['datetime', 'strptime', 'date'],
            'split': ['split', 'string', 'lines'],
            'concatenate': ['join', 'concatenate'],
            'django': ['django', 'model', 'queryset'],
            'numpy': ['numpy', 'array', 'sum'],
            'file': ['file', 'open', 'write']
        }
        
        for prompt_key, code_keys in keyword_mappings.items():
            if prompt_key in prompt_lower:
                if any(key in code_lower for key in code_keys):
                    relevance += 0.3
                    break
        
        prompt_words = set(prompt_lower.split())
        code_words = set(code_lower.split())
        if prompt_words and code_words:
            overlap = len(prompt_words.intersection(code_words))
            relevance += min(overlap / len(prompt_words) * 0.3, 0.3)
        
        return min(relevance, 1.0)

    def _check_completeness(self, code: str) -> float:
        """Check if code appears complete and executable."""
        score = 0.0
        
        if code.strip().endswith((')', ']', '}', '"', "'")):
            score += 0.3
        
        if code.count('(') == code.count(')') and code.count('[') == code.count(']'):
            score += 0.3
        
        if any(mod in code for mod in ['subprocess', 'datetime', 'pandas']):
            if 'import' in code:
                score += 0.2
        else:
            score += 0.2
        
        return score

    def _check_bad_practices(self, code: str) -> float:
        """Check for bad coding practices."""
        penalty = 0.0
        
        danger_count = sum(1 for practice in self.bad_practices if practice in code)
        penalty += danger_count * 0.1
        
        try:
            ast.parse(code)
        except:
            penalty += 0.2
        
        return min(penalty, 0.3)

    def _check_execution(self, prompt: str, code: str) -> float:
        """Basic execution check - simplified version."""
        try:
            # Safe compilation check
            compile(code, '<string>', 'exec')
            return 0.5
        except:
            return 0.0

class ImprovedCodeRewardModel:
    """Enhanced reward model for code generation with better metrics."""
    
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device(config.device)
        
        # Optimized metric weights for code quality
        self.metric_weights = {
            'syntax': 0.4,        # Increased importance of syntax
            'structure': 0.3,     # Code structure and practices
            'relevance': 0.2,     # Relevance to prompt
            'completeness': 0.1   # Code completeness
        }
        
        # Enhanced code quality indicators
        self.good_practices = [
            'def ', 'return ', 'import ', 'from ', 'class ', 'try:', 'except ',
            'if __name__', 'with open', 'isinstance', 'len(', 'range(', 'subprocess.',
            'datetime.', 'pandas.', 'numpy.', 'os.', 'sys.'
        ]
        
        self.bad_practices = [
            'eval(', 'exec(', 'input()', 'while True:', 'import *',
            'except:', 'except Exception:', 'print(', 'exit()'
        ]

    def compute_reward(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute enhanced rewards for code generation."""
        rewards = []
        
        for prompt, code in zip(prompts, responses):
            if not code or len(code.strip()) < self.config.min_code_length:
                rewards.append(-1.0)
                continue
            
            reward = 0.0
            
            # 1. Syntax validity (most important)
            syntax_score = self._check_syntax(code)
            reward += syntax_score * self.metric_weights['syntax']
            
            # 2. Code structure and best practices
            structure_score = self._check_structure(code)
            reward += structure_score * self.metric_weights['structure']
            
            # 3. Relevance to prompt
            relevance_score = self._check_relevance(prompt, code)
            reward += relevance_score * self.metric_weights['relevance']
            
            # 4. Code completeness
            completeness_score = self._check_completeness(code)
            reward += completeness_score * self.metric_weights['completeness']
            
            # Penalties for bad practices
            penalties = self._check_bad_practices(code)
            reward -= penalties
            
            # Normalize and ensure reasonable range
            reward = max(min(reward, 1.0), -0.5)
            
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def _check_syntax(self, code: str) -> float:
        """Enhanced syntax checking."""
        try:
            # Try to parse the code
            ast.parse(code)
            
            # Additional quality checks
            lines = code.strip().split('\n')
            if len(lines) >= 2:  # Multi-line code gets higher score
                return 0.9
            else:
                return 0.7
                
        except SyntaxError as e:
            error_msg = str(e)
            # Partial credit for common syntax errors that are close to correct
            if 'unexpected EOF' in error_msg or 'parenthesis' in error_msg:
                return 0.3
            return 0.0

    def _check_structure(self, code: str) -> float:
        """Check code structure and best practices."""
        score = 0.0
        lines = [line for line in code.split('\n') if line.strip()]
        
        # Check for imports and function definitions
        if any('import ' in line for line in lines):
            score += 0.3
        
        if any('def ' in line for line in lines):
            score += 0.4
        
        # Check for good practices
        good_count = sum(1 for practice in self.good_practices if practice in code)
        score += min(good_count * 0.05, 0.2)  # Reduced weight per practice
        
        # Check code organization
        if len(lines) >= 2 and len(lines) <= 15:  # Reasonable length
            score += 0.1
        
        return min(score, 1.0)

    def _check_relevance(self, prompt: str, code: str) -> float:
        """Enhanced relevance checking."""
        prompt_lower = prompt.lower()
        code_lower = code.lower()
        
        relevance = 0.0
        
        # Keyword matching with context
        keyword_mappings = {
            'signal': ['signal', 'kill', 'pid'],
            'decode': ['decode', 'hex', 'utf'],
            'dictionary': ['dict', 'kwargs', 'items'],
            'subprocess': ['subprocess', 'call', 'check_output'],
            'pandas': ['pandas', 'series', 'dataframe'],
            'http': ['http', 'header', 'client'],
            'datetime': ['datetime', 'strptime', 'date'],
            'split': ['split', 'string', 'lines'],
            'concatenate': ['join', 'concatenate'],
            'django': ['django', 'model', 'queryset'],
            'numpy': ['numpy', 'array', 'sum'],
            'file': ['file', 'open', 'write']
        }
        
        for prompt_key, code_keys in keyword_mappings.items():
            if prompt_key in prompt_lower:
                if any(key in code_lower for key in code_keys):
                    relevance += 0.3
                    break
        
        # Basic word overlap
        prompt_words = set(prompt_lower.split())
        code_words = set(code_lower.split())
        if prompt_words and code_words:
            overlap = len(prompt_words.intersection(code_words))
            relevance += min(overlap / len(prompt_words) * 0.3, 0.3)
        
        return min(relevance, 1.0)

    def _check_completeness(self, code: str) -> float:
        """Check if code appears complete and executable."""
        score = 0.0
        
        # Check for proper endings
        if code.strip().endswith((')', ']', '}', '"', "'")):
            score += 0.3
        
        # Check for balanced brackets
        if code.count('(') == code.count(')') and code.count('[') == code.count(']'):
            score += 0.3
        
        # Check for imports if needed
        if any(mod in code for mod in ['subprocess', 'datetime', 'pandas']):
            if 'import' in code:
                score += 0.2
        else:
            score += 0.2  # Bonus for not needing imports
        
        return score

    def _check_bad_practices(self, code: str) -> float:
        """Check for bad coding practices."""
        penalty = 0.0
        
        # Check for dangerous functions
        danger_count = sum(1 for practice in self.bad_practices if practice in code)
        penalty += danger_count * 0.1  # Reduced penalty
        
        # Check for syntax errors in the structure
        try:
            ast.parse(code)
        except:
            penalty += 0.2
        
        return min(penalty, 0.3)

# ------------------------------------------------------------
# FILE: .\src\models\reward_model.py
# ------------------------------------------------------------

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import logging
from typing import List

logger = logging.getLogger(__name__)


import ast
import re
from typing import List

class ImprovedCodeRewardModel(nn.Module):
    """Improved reward model trained on human evaluations."""
    
    def __init__(self, model_name="microsoft/codebert-base"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Multi-head evaluation
        self.consistency_head = nn.Linear(768, 1)
        self.correctness_head = nn.Linear(768, 1)
        self.usefulness_head = nn.Linear(768, 1)
        self.overall_head = nn.Linear(768, 1)
        
        # Code quality indicators (for compatibility)
        self.good_practices = [
            'def ', 'return ', 'import ', 'from ', 'class ', 'try:', 'except ',
            'if __name__', 'with open', 'isinstance', 'len(', 'range(', 'subprocess.',
            'datetime.', 'pandas.', 'numpy.', 'os.', 'sys.'
        ]
        
        self.bad_practices = [
            'eval(', 'exec(', 'input()', 'while True:', 'import *',
            'except:', 'except Exception:', 'print(', 'exit()'
        ]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    # ДОБАВЛЯЕМ МЕТОДЫ ДЛЯ СОВМЕСТИМОСТИ С СТАРЫМ КОДОМ
    def _check_syntax(self, code: str) -> float:
        """Check Python syntax validity."""
        try:
            ast.parse(code)
            lines = code.strip().split('\n')
            if len(lines) >= 2:
                return 0.9
            else:
                return 0.7
        except SyntaxError:
            return 0.0
    
    def _check_structure(self, code: str) -> float:
        """Check code structure and best practices."""
        score = 0.0
        lines = [line for line in code.split('\n') if line.strip()]
        
        if any('import ' in line for line in lines):
            score += 0.3
        
        if any('def ' in line for line in lines):
            score += 0.4
        
        good_count = sum(1 for practice in self.good_practices if practice in code)
        score += min(good_count * 0.05, 0.2)
        
        if len(lines) >= 2 and len(lines) <= 15:
            score += 0.1
        
        return min(score, 1.0)
    
    def _check_execution(self, prompt: str, code: str) -> float:
        """Basic execution check."""
        try:
            compile(code, '<string>', 'exec')
            return 0.5
        except:
            return 0.0
    
    def _check_relevance(self, prompt: str, code: str) -> float:
        """Check relevance to prompt."""
        prompt_lower = prompt.lower()
        code_lower = code.lower()
        
        keyword_mappings = {
            'signal': ['signal', 'kill', 'pid'],
            'decode': ['decode', 'hex', 'utf'],
            'dictionary': ['dict', 'kwargs', 'items'],
            'subprocess': ['subprocess', 'call', 'check_output'],
            'pandas': ['pandas', 'series', 'dataframe'],
            'http': ['http', 'header', 'client'],
            'datetime': ['datetime', 'strptime', 'date'],
            'split': ['split', 'string', 'lines'],
            'concatenate': ['join', 'concatenate'],
        }
        
        for prompt_key, code_keys in keyword_mappings.items():
            if prompt_key in prompt_lower:
                if any(key in code_lower for key in code_keys):
                    return 0.7
        
        return 0.3
    
    def _check_completeness(self, code: str) -> float:
        """Check if code appears complete."""
        score = 0.0
        
        if code.strip().endswith((')', ']', '}', '"', "'")):
            score += 0.3
        
        if code.count('(') == code.count(')') and code.count('[') == code.count(']'):
            score += 0.3
        
        if any(mod in code for mod in ['subprocess', 'datetime', 'pandas']):
            if 'import' in code:
                score += 0.2
        else:
            score += 0.2
        
        return score
    
    def _check_bad_practices(self, code: str) -> float:
        """Check for bad coding practices."""
        penalty = 0.0
        danger_count = sum(1 for practice in self.bad_practices if practice in code)
        penalty += danger_count * 0.1
        
        try:
            ast.parse(code)
        except:
            penalty += 0.2
        
        return min(penalty, 0.3)
        
    def forward(self, questions: List[str], answers: List[str]):
        # Encode questions and answers
        inputs = self.tokenizer(
            questions, answers,
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.bert(**inputs)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Get scores for different aspects
        consistency = torch.sigmoid(self.consistency_head(pooled_output))
        correctness = torch.sigmoid(self.correctness_head(pooled_output))
        usefulness = torch.sigmoid(self.usefulness_head(pooled_output))
        overall = torch.sigmoid(self.overall_head(pooled_output))
        
        return {
            'consistency': consistency,
            'correctness': correctness, 
            'usefulness': usefulness,
            'overall': overall
        }
    
    def compute_reward(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute reward for PPO training."""
        with torch.no_grad():
            results = self.forward(prompts, responses)
            # Use overall score as reward
            return results['overall'].squeeze()
    
    def predict_quality(self, prompt: str, response: str) -> dict:
        """Predict quality scores for a single prompt-response pair."""
        with torch.no_grad():
            results = self.forward([prompt], [response])
            return {
                'consistency': results['consistency'].item(),
                'correctness': results['correctness'].item(),
                'usefulness': results['usefulness'].item(),
                'overall': results['overall'].item()
            }

# ------------------------------------------------------------
# FILE: .\src\models\__init__.py
# ------------------------------------------------------------



# ------------------------------------------------------------
# FILE: .\src\train\ppo_trainer.py
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# FILE: .\src\train\__init__.py
# ------------------------------------------------------------



# ------------------------------------------------------------
# FILE: .\tests\test_hf_trainers_mock.py
# ------------------------------------------------------------

"""CI-style unit tests for HF-heavy training scripts.

These tests assert that when heavy libraries are missing the scripts exit
with the documented status code (2). This allows CI to run quickly without
GPU/HF and ensures the scripts fail loudly with actionable messages.
"""
import runpy
import sys


def run_and_expect_exit(script_path, expected_code=2):
    try:
        runpy.run_path(script_path, run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        assert code == expected_code, f"Expected exit {expected_code} got {code} for {script_path}"
        return
    # If no SystemExit raised, that's unexpected in minimal CI environment
    raise AssertionError(f"Script {script_path} did not exit as expected")


def test_train_reward_model_hf_missing_libs():
    run_and_expect_exit("scripts/train_reward_model_hf.py", expected_code=2)


def test_finetune_sft_hf_missing_libs():
    run_and_expect_exit("scripts/finetune_sft_hf.py", expected_code=2)


def test_run_ppo_rlhf_missing_libs():
    run_and_expect_exit("scripts/run_ppo_rlhf.py", expected_code=2)


if __name__ == "__main__":
    test_train_reward_model_hf_missing_libs()
    test_finetune_sft_hf_missing_libs()
    test_run_ppo_rlhf_missing_libs()
    print("HF trainer mock tests passed")


# ------------------------------------------------------------
# FILE: .\tests\test_metrics_tracker.py
# ------------------------------------------------------------

import sys
import types
import pytest
from src.metrics_tracker import MetricsTracker


def test_empty_references_return_zeros():
    mt = MetricsTracker(output_dir="./outputs_test_metrics")
    generated = ["print(1)", "print(2)"]
    refs = ["", "  "]  # empty/whitespace references

    metrics = mt.calculate_metrics(prompts=["p1", "p2"], generated_texts=generated, reference_texts=refs)

    assert metrics['bertscore'] == 0.0
    assert metrics['bleu'] == 0.0
    assert metrics['codebleu'] == 0.0
    assert metrics['rouge'] == 0.0
    assert metrics['ruby'] == 0.0


def test_mixed_references_use_only_nonempty_and_respect_monkeypatched_metrics():
    mt = MetricsTracker(output_dir="./outputs_test_metrics")
    generated = ["a", "b", "c"]
    refs = ["", "REF_B", ""]

    # monkeypatch internal metric calculators to deterministic values
    mt.calculate_bertscore = lambda g, r: 0.71
    mt.calculate_bleu = lambda g, r: 0.42
    mt.calculate_codebleu = lambda g, r: 0.33
    mt.calculate_rouge = lambda g, r: 0.55

    metrics = mt.calculate_metrics(prompts=["p1", "p2", "p3"], generated_texts=generated, reference_texts=refs)

    # Only one non-empty ref => metrics should equal the monkeypatched return values
    assert pytest.approx(metrics['bertscore'], rel=1e-6) == 0.71
    assert pytest.approx(metrics['bleu'], rel=1e-6) == 0.42
    assert pytest.approx(metrics['codebleu'], rel=1e-6) == 0.33
    assert pytest.approx(metrics['rouge'], rel=1e-6) == 0.55


def test_calculate_codebleu_list_api_shape_handling(monkeypatch):
    mt = MetricsTracker(output_dir="./outputs_test_metrics")

    # Create a fake codebleu module with a calc_codebleu function that validates shapes
    fake_mod = types.ModuleType("codebleu")

    def fake_calc_codebleu(references=None, predictions=None, lang=None):
        # references should be list of single-reference lists
        assert isinstance(references, list)
        assert all(isinstance(r, list) for r in references)
        # predictions should be a flat list with same length as references
        assert isinstance(predictions, list)
        assert len(predictions) == len(references)
        return {'codebleu': 0.95}

    fake_mod.calc_codebleu = fake_calc_codebleu
    sys.modules['codebleu'] = fake_mod

    preds = ["print(1)", "print(2)"]
    refs = ["print(1)", "print(2)"]

    score = mt.calculate_codebleu(preds, refs)
    assert pytest.approx(score, rel=1e-6) == 0.95

    # cleanup
    del sys.modules['codebleu']


# ------------------------------------------------------------
# FILE: .\tests\test_pipeline_smoke.py
# ------------------------------------------------------------

"""Smoke tests for data-prep and placeholder training scripts.

This test runs the prepare_pairs (already executed), then runs the
placeholder reward and SFT trainers and checks that their output artifacts
exist. It's intentionally lightweight and should pass on a minimal CPU-only
environment.
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
REWARD_CKPT = os.path.join(OUT_DIR, "reward_model_placeholder.json")
SFT_CKPT = os.path.join(OUT_DIR, "sft_model_placeholder.json")


def run(script):
    path = os.path.join(ROOT, script)
    print("Running", path)
    runpy.run_path(path, run_name="__main__")


def test_train_placeholders():
    # run the two scripts
    run("scripts/train_reward_model.py")
    run("scripts/finetune_supervised.py")

    assert os.path.exists(REWARD_CKPT), "reward placeholder not created"
    assert os.path.exists(SFT_CKPT), "sft placeholder not created"


if __name__ == "__main__":
    test_train_placeholders()
    print("Smoke test succeeded")
