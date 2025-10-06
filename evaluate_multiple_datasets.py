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