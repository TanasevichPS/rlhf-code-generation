# 🚀 Modern RLHF Framework - Полное Руководство

## 📋 Обзор

Я создал для вас **современную, чистую и эффективную** систему RLHF для генерации кода, которая решает все проблемы вашего старого проекта:

### ✅ Что исправлено:
- **Чистая архитектура** - модульная структура без запутанного кода
- **Современные методы** - поддержка PPO и DPO (Direct Preference Optimization)
- **Комплексные метрики** - BERTScore, CodeBLEU, BLEU, ROUGE, Ruby
- **Интеграция human feedback** - поддержка human logits в последнем слое трансформера
- **GPU оптимизация** - эффективное использование GPU с mixed precision
- **Целевые метрики** - настройка для достижения ваших целей (>0.7 для разных метрик)

## 🏗️ Структура Проекта

```
modern_rlhf/
├── __init__.py          # Основной модуль
├── config.py            # Управление конфигурацией
├── metrics.py           # Все метрики оценки
├── reward_model.py      # Reward модель с human feedback
├── trainer.py           # PPO/DPO тренировщики
├── pipeline.py          # Основной пайплайн обучения
├── data_loader.py       # Загрузка и обработка данных
├── main.py             # CLI интерфейс
├── requirements.txt     # Зависимости
└── README.md           # Документация
```

## 🚀 Быстрый Старт

### 1. Установка зависимостей
```bash
cd modern_rlhf
pip install -r requirements.txt
```

### 2. Простой запуск
```bash
# Быстрый тест
python ../run_modern_rlhf.py

# Или через CLI
python main.py --mode fast --epochs 2 --steps 500
```

### 3. Полное обучение
```bash
# Исследовательский режим
python main.py --mode research --epochs 10 --steps 2000

# Продакшн режим
python main.py --mode production --device cuda --batch-size 8
```

## 🎯 Целевые Метрики

Система настроена для достижения ваших целей:

| Метрика | Цель | Описание |
|---------|------|----------|
| **BERTScore** | ≥ 0.7 | Семантическое сходство |
| **CodeBLEU** | ≥ 0.6 | Специфичная для кода оценка |
| **BLEU** | ≥ 0.4 | N-gram overlap |
| **ROUGE** | ≥ 0.5 | Метрики суммаризации |
| **Ruby** | ≥ 0.3 | Кастомная метрика качества кода |

## 🔧 Ключевые Особенности

### 1. Human Feedback Integration
```python
# Поддержка human logits в последнем слое
config.reward.use_human_logits = True
config.reward.human_logits_layer = "last"  # или "second_last"
config.reward.human_feedback_weight = 0.3
```

### 2. Современные Методы Обучения
- **PPO** - классический RLHF
- **DPO** - Direct Preference Optimization (новый метод)
- **Mixed Precision** - для ускорения обучения
- **Gradient Checkpointing** - для экономии памяти

### 3. Комплексная Reward Модель
```python
# Компоненты reward модели:
- Syntax correctness (20%)
- Execution success (30%) 
- Semantic similarity (30%)
- Human preferences (20%)
```

### 4. Автоматическая Оценка
- Проверка синтаксиса кода
- Тестирование выполнения
- Анализ сложности кода
- Оценка стиля кода

## 📊 Использование

### Базовое использование
```python
from modern_rlhf import ModernRLHFPipeline, get_research_config

# Создать конфигурацию
config = get_research_config()

# Создать пайплайн
pipeline = ModernRLHFPipeline(config)

# Запустить обучение
results = pipeline.run_full_pipeline()

# Проверить результаты
if results.success:
    print(f"BERTScore: {results.evaluation_metrics['bertscore']:.3f}")
    print(f"CodeBLEU: {results.evaluation_metrics['codebleu']:.3f}")
```

### Кастомная конфигурация
```python
from modern_rlhf.config import ModernRLHFConfig

config = ModernRLHFConfig()

# Настройка модели
config.model.base_model_name = "microsoft/CodeGPT-small-py"
config.model.reward_model_name = "microsoft/codebert-base"

# Настройка обучения
config.training.learning_rate = 1e-5
config.training.batch_size = 4
config.training.ppo_epochs = 10

# Целевые метрики
config.evaluation.target_bertscore = 0.8
config.evaluation.target_codebleu = 0.7

# Human feedback
config.reward.use_human_logits = True
config.reward.human_feedback_weight = 0.3
```

## 🎛️ CLI Команды

### Основные команды
```bash
# Быстрый прототип
python main.py --mode fast

# Исследовательский эксперимент
python main.py --mode research --epochs 10

# Продакшн обучение
python main.py --mode production --device cuda

# Кастомная конфигурация
python main.py \
    --learning-rate 1e-5 \
    --batch-size 4 \
    --model-name microsoft/CodeGPT-small-py \
    --target-bertscore 0.8 \
    --target-codebleu 0.7
```

### Параметры
- `--mode`: research/production/fast
- `--learning-rate`: скорость обучения
- `--batch-size`: размер батча
- `--epochs`: количество эпох
- `--device`: cpu/cuda/auto
- `--target-*`: целевые метрики

## 📈 Мониторинг и Визуализация

Система автоматически создает:
- Графики метрик обучения
- Визуализацию достижения целей
- Детальные логи
- Сохранение чекпоинтов

## 🔬 Исследовательские Возможности

### 1. Эксперименты с DPO
```python
# Включить DPO вместо PPO
config.training.use_dpo = True
config.training.dpo_beta = 0.1
config.training.dpo_loss_type = "sigmoid"
```

### 2. Абляционные исследования
```python
# Отключить компоненты reward модели
config.reward.syntax_reward_weight = 0.0
config.reward.execution_reward_weight = 0.0
config.reward.semantic_reward_weight = 1.0
```

### 3. Различные модели
```python
# Попробовать разные базовые модели
config.model.base_model_name = "microsoft/CodeGPT-small-py"
# или
config.model.base_model_name = "Salesforce/codegen-350M-mono"
```

## 🚀 Преимущества Новой Системы

### По сравнению со старым проектом:

1. **Чистота кода**: Модульная архитектура, легко понимать и модифицировать
2. **Современность**: Использует последние методы (DPO, современные метрики)
3. **Эффективность**: Оптимизировано для GPU, mixed precision
4. **Гибкость**: Легко настраивать под разные задачи
5. **Мониторинг**: Встроенная визуализация и логирование
6. **Надежность**: Обработка ошибок, валидация данных

## 📝 Следующие Шаги

1. **Установите зависимости**:
   ```bash
   pip install -r modern_rlhf/requirements.txt
   ```

2. **Запустите тест**:
   ```bash
   python test_modern_rlhf.py
   ```

3. **Запустите быстрый эксперимент**:
   ```bash
   python run_modern_rlhf.py
   ```

4. **Настройте под ваши данные**:
   - Обновите пути к данным в конфигурации
   - Настройте целевые метрики
   - Выберите подходящую модель

5. **Запустите полное обучение**:
   ```bash
   python modern_rlhf/main.py --mode research --epochs 10
   ```

## 🎉 Результат

Теперь у вас есть:
- ✅ **Чистая, современная система RLHF**
- ✅ **Поддержка всех необходимых метрик**
- ✅ **Интеграция human feedback**
- ✅ **Эффективное использование GPU**
- ✅ **Гибкая конфигурация**
- ✅ **Готовность к исследовательской работе**

Система готова для достижения ваших целей по метрикам и проведения качественных исследований в области RLHF для генерации кода!
