# Modern RLHF Project Explanation

## 1. **Какие данные использовались**

В проекте использовались **три основных источника данных**:

### **1.1 Supervised Fine-Tuning (SFT) данные**
- **Файл**: `datasets_for_training/sft_dataset.csv` (2025 строк)
- **Формат**: CSV с колонками `question`, `best_answer`, `model_tag`, `source_json`, `datetime`
- **Содержание**: Реальные вопросы и ответы о программировании на Python, собранные из различных источников (Stack Overflow, GitHub и т.д.)
- **Примеры**:
  - Вопрос: *"How to plot maximal intensity projection of images in same directory..."*
  - Ответ: Подробный код на Python для обработки изображений

### **1.2 Human Feedback данные**
- **Папка**: `evaluation_results_server/`
- **Формат**: JSON файлы с оценками ответов (рейтинг 1-5)
- **Количество**: 200+ файлов с именами вида `2022-08-16-13-24-32-Anonymous.json`
- **Содержание**: Человеческие оценки качества ответов на вопросы о коде

### **1.3 Синтетические данные**
- **Генерация**: 11 hardcoded примеров кода в коде (факториал, сортировка, палиндромы и т.д.)
- **Назначение**: Дополнение к реальным данным для базового обучения

## 2. **Описание проекта**

Проект представляет собой **современную реализацию RLHF (Reinforcement Learning from Human Feedback)** для генерации кода на Python. Давайте разберем каждую строчку кода досконально.

### **2.1 Основной скрипт (`run_modern_rlhf.py`)**

```python
#!/usr/bin/env python3
"""
Quick Start Script for Modern RLHF
==================================

Simple script to run the modern RLHF framework with your existing data.
"""
```

**Строка 1**: Shebang для запуска как исполняемого файла в Unix-системах.

**Строки 2-7**: Докстринг с описанием - простой скрипт для запуска RLHF фреймворка с существующими данными.

```python
import sys
import os
from pathlib import Path
import json
import random
import time
import argparse
```

**Строки 9-16**: Импорты стандартных библиотек:
- `sys` - для работы с системой (пути, аргументы)
- `os` - операции с файловой системой
- `Path` - современный способ работы с путями
- `json` - работа с JSON данными
- `random` - генерация случайных чисел
- `time` - работа со временем
- `argparse` - парсинг командной строки

```python
# Add modern_rlhf to path
sys.path.insert(0, str(Path(__file__).parent / "modern_rlhf"))
```

**Строки 18-19**: Добавляем папку `modern_rlhf` в путь поиска модулей, чтобы импортировать локальные модули.

```python
# Ensure stdout/stderr use UTF-8 where possible to avoid console encoding errors
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    # Older Python / environments may not support reconfigure; ignore
    pass
```

**Строки 21-26**: Настраиваем кодировку вывода на UTF-8 для корректного отображения русских символов и эмодзи.

```python
import warnings
import logging
import os

# Suppress transformers warnings about uninitialized weights
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
```

**Строки 28-35**: Настраиваем логирование:
- Подавляем предупреждения transformers о неинициализированных весах (нормально для fine-tuning)
- Устанавливаем уровень логирования для transformers на ERROR

```python
from modern_rlhf import ModernRLHFPipeline, get_research_config, get_production_config, get_fast_config, get_cpu_test_config
from modern_rlhf.config import ModernRLHFConfig
```

**Строка 44**: Импортируем основные компоненты фреймворка.

```python
def main():
    """Quick start function."""
    parser = argparse.ArgumentParser(description="Run Modern RLHF")
    parser.add_argument('--config', type=str, default='fast', choices=['research', 'production', 'fast', 'cpu-test'], help='Config type')
```

**Строки 47-51**: Функция main() с парсингом аргументов командной строки.

```python
# Create configuration
if args.config == 'research':
    config = get_research_config()
elif args.config == 'production':
    config = get_production_config()
elif args.config == 'fast':
    config = get_fast_config()
elif args.config == 'cpu-test':
    config = get_cpu_test_config()
```

**Строки 73-82**: Выбираем конфигурацию на основе аргумента `--config`.

```python
# Use specified data paths
config.data.train_data_path = str(Path(__file__).parent / "datasets_for_training")
config.data.eval_data_path = str(Path(__file__).parent / "datasets_for_eval")
config.data.human_feedback_path = str(Path(__file__).parent / "evaluation_results_server")
config.data.output_path = str(Path(__file__).parent / "modern_outputs")
```

**Строки 89-92**: Устанавливаем пути к данным, используя абсолютные пути от скрипта.

```python
# MULTI-STAGE RLHF CONFIGURATION WITH CUSTOM REWARD MODEL
config.training.use_multi_stage = True
config.training.sft_epochs = 3  # SFT stage epochs (better foundation)
config.training.reward_modeling_epochs = 5  # Reward modeling stage epochs (better reward model)
config.training.rlhf_epochs = 20  # RLHF stage epochs (matches your successful run)
```

**Строки 100-106**: Включаем многоэтапное обучение: SFT → Reward Modeling → RLHF.

```python
# Custom reward model settings (code-specific for better results)
config.reward.use_custom_reward = True
config.reward.custom_reward_backbone = "microsoft/codebert-base"  # Code-specific model
config.reward.custom_reward_heads = 3  # Multiple heads for different aspects
config.reward.custom_reward_pretrain = True
config.reward.custom_reward_freeze_backbone = True  # Freeze backbone to save memory
```

**Строки 109-113**: Настраиваем кастомную reward model с несколькими головами для разных аспектов кода.

```python
# Target metrics (matching your successful run)
config.evaluation.target_bertscore = 0.90  # High target
config.evaluation.target_codebleu = 0.80   # Your target: 0.800
config.evaluation.target_bleu = 0.60       # Your target: 0.600
config.evaluation.target_rouge = 0.70      # Your target: 0.700
config.evaluation.target_ruby = 0.98       # Your target: 0.982
```

**Строки 154-158**: Устанавливаем целевые метрики, которые нужно достичь.

```python
# Check device requirements
import torch
is_cpu_test = "cpu" in config.tags or "test" in config.tags

if not torch.cuda.is_available():
    if is_cpu_test:
        print("⚠️  Using CPU for testing - performance will be very slow!")
        config.hardware.device = "cpu"
    else:
        raise RuntimeError("CUDA GPU is not available! Training requires GPU.")
```

**Строки 183-191**: Проверяем доступность GPU - RLHF требует GPU для обучения.

```python
# Create pipeline
print("Initializing Modern RLHF Pipeline...")
pipeline = ModernRLHFPipeline(config)

# Run pipeline
if config.training.use_multi_stage:
    print("🔄 Using Multi-Stage Pipeline (SFT → Reward → RLHF)")
    results = pipeline.run_multi_stage_pipeline()
```

**Строки 376-384**: Создаем pipeline и запускаем многоэтапное обучение.

### **2.2 Конфигурация (`modern_rlhf/config.py`)**

```python
@dataclass
class ModelConfig:
    """Configuration for model settings."""

    # Base model settings
    base_model_name: str = "microsoft/CodeGPT-small-py"
    reward_model_name: str = "microsoft/codebert-base"
```

**Строки 15-21**: Определяем конфигурацию моделей - базовую модель для генерации и reward model.

```python
@dataclass
class TrainingConfig:
    """Configuration for training settings."""

    # PPO specific settings
    ppo_epochs: int = 4
    ppo_clip_ratio: float = 0.2
    ppo_value_loss_coef: float = 0.1
    ppo_entropy_coef: float = 0.01
    ppo_kl_penalty: float = 0.02
```

**Строки 40-55**: Параметры PPO (Proximal Policy Optimization) - основной RL алгоритм.

```python
# Multi-stage training
use_multi_stage: bool = True  # Enable SFT -> Reward -> RLHF pipeline
sft_learning_rate: float = 5e-5  # Learning rate for supervised fine-tuning
sft_epochs: int = 1  # Epochs for SFT stage
reward_modeling_epochs: int = 2  # Epochs for reward modeling stage
rlhf_epochs: int = 3  # Epochs for RLHF stage
```

**Строки 72-77**: Настройки многоэтапного обучения - сначала supervised fine-tuning, потом reward model, потом RLHF.

### **2.3 Data Loader (`modern_rlhf/data_loader.py`)**

```python
@dataclass
class DataSample:
    """Container for a single data sample."""
    prompt: str
    response: str
    reference: Optional[str] = None
    rating: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
```

**Строки 43-49**: Структура данных для одного сэмпла - вопрос, ответ, референс, рейтинг, метаданные.

```python
def load_training_data(self) -> List[DataSample]:
    """Load training data from various sources."""
    logger.info("Loading training data...")

    all_samples: List[DataSample] = []

    # Skip CoNaLa dataset - use only local datasets
    logger.info("Skipping CoNaLa dataset loading - using only local datasets")

    # Load from different sources
    sources = [
        self._load_sft_data,
        self._load_preference_data,
        self._load_synthetic_data
    ]

    for source_func in sources:
        try:
            samples = source_func()
            all_samples.extend(samples)
            logger.info(f"Loaded {len(samples)} samples from {source_func.__name__}")
        except Exception as e:
            logger.warning(f"Failed to load from {source_func.__name__}: {e}")
```

**Строки 339-362**: Основная функция загрузки данных - загружает из разных источников (SFT, preference, synthetic).

```python
def _load_sft_data(self) -> List[DataSample]:
    """Load supervised fine-tuning data."""
    samples = []

    sft_path = Path(self.data_config.train_data_path) / "sft_dataset.csv"

    if sft_path.exists():
        df = pd.read_csv(sft_path)

        # Find appropriate columns
        prompt_col = self._find_column(df, ['prompt', 'instruction', 'question', 'input'])
        response_col = self._find_column(df, ['response', 'answer', 'output', 'completion', 'best_answer'])
```

**Строки 760-771**: Загружает SFT данные из CSV, ищет подходящие колонки для prompt и response.

```python
def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """Find a column with one of the possible names."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None
```

**Строки 872-877**: Вспомогательная функция для поиска колонок по списку возможных имен.

### **2.4 Pipeline (`modern_rlhf/pipeline.py`)**

```python
class ModernRLHFPipeline:
    """Main RLHF pipeline class."""

    def __init__(self, config: Optional[ModernRLHFConfig] = None):
        self.config = config or get_research_config()

        # Check device requirements
        if not torch.cuda.is_available():
            if is_test_config:
                self.config.hardware.device = "cpu"
                self.device = torch.device("cpu")
            else:
                raise RuntimeError("CUDA GPU is not available! Pipeline requires GPU for training.")
```

**Строки 52-69**: Конструктор pipeline с проверкой GPU - RLHF требует мощного GPU.

```python
def run_multi_stage_pipeline(self):
    """Run the complete multi-stage RLHF pipeline."""

    # Stage 1: Supervised Fine-Tuning (SFT)
    print("[Stage 1/3] Supervised Fine-Tuning...")
    sft_results = self._run_sft_stage()

    # Stage 2: Reward Model Training
    print("[Stage 2/3] Reward Model Training...")
    reward_results = self._run_reward_modeling_stage()

    # Stage 3: RLHF Training
    print("[Stage 3/3] RLHF Training...")
    rlhf_results = self._run_rlhf_stage()
```

**Строки 200-212**: Основная функция многоэтапного pipeline - SFT → Reward Model → RLHF.

### **2.5 Reward Model (`modern_rlhf/reward_model.py`)**

```python
class CustomRewardModel(nn.Module):
    """Custom reward model with multiple heads for different aspects."""

    def __init__(self, backbone_name="microsoft/codebert-base", num_heads=3, freeze_backbone=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.num_heads = num_heads

        # Multiple heads for different reward aspects
        self.heads = nn.ModuleList([
            nn.Linear(self.backbone.config.hidden_size, 1) for _ in range(num_heads)
        ])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
```

**Строки 50-66**: Кастомная reward model с несколькими головами для разных аспектов оценки кода (синтаксис, семантика, предпочтения).

### **2.6 Trainer (`modern_rlhf/trainer.py`)**

```python
class PPOTrainer:
    """PPO Trainer for RLHF."""

    def __init__(self, config, policy_model, reward_model, tokenizer, device):
        self.config = config
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device

        # PPO hyperparameters
        self.clip_ratio = config.training.ppo_clip_ratio
        self.value_loss_coef = config.training.ppo_value_loss_coef
        self.entropy_coef = config.training.ppo_entropy_coef
```

**Строки 50-65**: PPO тренер - реализует алгоритм Proximal Policy Optimization для RLHF.

## **3. Архитектура проекта**

```
📁 modern_rlhf/
├── 📄 config.py          # Конфигурации всех компонентов
├── 📄 pipeline.py        # Основной pipeline (SFT → Reward → RLHF)
├── 📄 data_loader.py     # Загрузка и preprocessing данных
├── 📄 reward_model.py    # Reward model с custom heads
├── 📄 trainer.py         # PPO/DPO тренеры
├── 📄 metrics.py         # Оценка метрик (CodeBLEU, BLEU, ROUGE, RUBY)
└── 📄 utils.py           # Вспомогательные функции

📁 datasets_for_training/
└── 📄 sft_dataset.csv    # 2023 примера Q&A по Python

📁 evaluation_results_server/
└── 📄 *.json             # 200+ файлов с human feedback

📄 run_modern_rlhf.py     # Основной скрипт запуска
```

## **4. Ключевые особенности**

1. **Многоэтапное обучение**: SFT → Reward Modeling → RLHF
2. **Кастомная Reward Model**: Много-головый подход для оценки разных аспектов кода
3. **Современные метрики**: CodeBLEU, BLEU, ROUGE, RUBY, BERTScore
4. **Гибкая конфигурация**: Разные режимы (research, production, fast, cpu-test)
5. **Real-time monitoring**: Прогресс-бары, логирование метрик
6. **GPU-first**: Оптимизирован для GPU обучения

Проект представляет собой production-ready реализацию RLHF для генерации кода с акцентом на качество и воспроизводимость результатов.
