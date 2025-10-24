# 🚀 Simplified RLHF - Полное Руководство

## ✅ Новая система готова!

Я создал **полностью новую, упрощенную систему RLHF** с современными методами и чистой архитектурой.

## 🏗️ Новая архитектура

```
rlhf_code_project/
├── config/
│   └── training_config.py    # ✅ Простая конфигурация
├── data/
│   └── preference_dataset.py # ✅ Загрузка данных
├── training/
│   └── dpo_trainer.py        # ✅ DPO тренировщик
├── evaluation/
│   └── metrics_calculator.py # ✅ Все метрики
├── scripts/
│   └── train.py              # ✅ Главный скрипт
└── requirements.txt          # ✅ Минимальные зависимости
```

## 🎯 Ключевые улучшения

### 1. **Современные методы**
- ✅ **DPO (Direct Preference Optimization)** - современная альтернатива PPO
- ✅ **Простая архитектура** - только 8 основных модулей
- ✅ **Эффективное обучение** - оптимизировано для быстрых экспериментов

### 2. **Все целевые метрики**
- ✅ **BERTScore** - семантическое сходство
- ✅ **CodeBLEU** - специфичная для кода оценка
- ✅ **BLEU** - n-gram overlap
- ✅ **ROUGE** - метрики суммаризации
- ✅ **Ruby** - кастомная метрика качества кода

### 3. **Готовность к исследованию**
- ✅ **Human feedback integration** - готово для человеческой обратной связи
- ✅ **Гибкая конфигурация** - легко настраивать под ваши нужды
- ✅ **Быстрое тестирование** - режим быстрого прототипирования

## 🚀 Быстрый запуск

### 1. **Проверка системы**
```bash
python quick_start_simple.py
```

### 2. **Установка зависимостей**
```bash
pip install -r rlhf_code_project/requirements.txt
```

### 3. **Полное обучение**
```bash
# Быстрый тест (2 эпохи)
python rlhf_code_project/scripts/train.py --fast

# Полное DPO обучение
python rlhf_code_project/scripts/train.py --method dpo --epochs 5 --batch-size 8

# Исследовательский режим
python rlhf_code_project/scripts/train.py --method dpo --epochs 10 --batch-size 4
```

## 📊 Целевые метрики

Система настроена для достижения:
- **BERTScore**: ≥ 0.7 (семантическое сходство)
- **CodeBLEU**: ≥ 0.6 (специфичная для кода оценка)
- **BLEU**: ≥ 0.4 (n-gram overlap)
- **ROUGE**: ≥ 0.5 (метрики суммаризации)
- **Ruby**: ≥ 0.3 (кастомная метрика качества кода)

## 🔧 Конфигурация

### Предустановленные конфигурации
```python
from rlhf_code_project.config import get_fast_config, get_dpo_config, get_research_config

# Быстрое прототипирование
config = get_fast_config()

# Продакшн DPO обучение
config = get_dpo_config()

# Исследовательские эксперименты
config = get_research_config()
```

### Кастомная конфигурация
```python
from rlhf_code_project.config import RLHFConfig

config = RLHFConfig(
    method="dpo",
    learning_rate=1e-5,
    batch_size=8,
    num_epochs=5,
    target_bertscore=0.8,
    target_codebleu=0.7,
    use_human_feedback=True
)
```

## 📈 Методы обучения

### DPO (Direct Preference Optimization)
- **Преимущества**: Проще PPO, стабильное обучение
- **Использование**: Современное RLHF обучение
- **Статус**: ✅ Готово к использованию

### PPO (Proximal Policy Optimization)
- **Преимущества**: Хорошо изучен, подходит для сложных reward
- **Использование**: Традиционное RLHF обучение
- **Статус**: 🔄 В разработке

## 🔬 Исследовательские возможности

### Интеграция человеческой обратной связи
```python
config.use_human_feedback = True
config.human_feedback_dim = 64
config.human_feedback_weight = 0.3
```

### Отслеживание экспериментов
```python
# Добавить в requirements.txt: wandb>=0.15.0
# Включить в конфиге: use_wandb = True
```

## 📝 Формат данных

### Данные предпочтений (для DPO)
```csv
prompt,chosen_response,rejected_response
"Write a function to calculate factorial","def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)","def factorial(n):\n    return 1"
```

### Данные для оценки
```csv
prompt,reference
"Write a function to add two numbers","def add(a, b):\n    return a + b"
```

## 🎉 Почему это лучше

1. **Простота**: 8 модулей вместо 20+ файлов
2. **Современность**: DPO вместо сложного PPO
3. **Скорость**: Оптимизировано для быстрых экспериментов
4. **Чистота**: Легко понимать и модифицировать
5. **Эффективность**: Создано для достижения ваших целевых метрик

## 🔬 Научная основа

- **DPO**: Direct Preference Optimization (Rafailov et al., 2023)
- **Human Feedback**: Learning from Human Preferences (Christiano et al., 2017)
- **Code Generation**: Современные подходы из CodeX, AlphaCode

## 📞 Поддержка

При возникновении вопросов или проблем, проверьте логи в `./rlhf_outputs/` или создайте issue.

---

## 🚀 Готово к использованию!

**Новая система RLHF готова для достижения ваших исследовательских целей!**

### Следующие шаги:
1. **Запустите проверку**: `python quick_start_simple.py`
2. **Установите зависимости**: `pip install -r rlhf_code_project/requirements.txt`
3. **Запустите полное обучение**: `python rlhf_code_project/scripts/train.py --method dpo --epochs 10`

**Теперь у вас есть современная, эффективная система RLHF для генерации кода!** 🎉
