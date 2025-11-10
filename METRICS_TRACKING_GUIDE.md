# 📊 Metrics Tracking Guide

## Обзор

Система отслеживания метрик автоматически логирует и визуализирует прогресс обучения RLHF модели. После каждой эпохи создаются:

- **CSV файлы** с метриками для анализа
- **JSON файлы** с полной историей
- **Графики PNG** для визуализации
- **Консольный вывод** с summary по эпохе

---

## 📁 Структура выходных файлов

После запуска обучения создается структура:

```
modern_outputs/
└── metrics/
    ├── training_metrics.csv          # CSV со всеми метриками
    ├── training_metrics.json         # JSON с полной историей
    └── plots/
        ├── training_progress.png     # Loss и Reward
        ├── policy_metrics.png        # KL Divergence и Entropy
        └── evaluation_metrics.png    # BERTScore, CodeBLEU, etc.
```

---

## 📈 Отслеживаемые метрики

### Training Metrics (каждая эпоха)

| Метрика | Описание |
|---------|----------|
| `loss` | Training loss (Policy + Value + Entropy) |
| `reward` | Средний reward от reward model |
| `kl_divergence` | KL divergence между policy и reference model |
| `entropy` | Энтропия policy (exploration measure) |
| `learning_rate` | Текущий learning rate |
| `epoch_time` | Время выполнения эпохи (секунды) |
| `samples_per_second` | Скорость обработки samples |

### Evaluation Metrics (если включена evaluation)

| Метрика | Описание |
|---------|----------|
| `bertscore` | Semantic similarity (BERTScore) |
| `codebleu` | Code-specific BLEU score |
| `bleu` | Standard BLEU score |
| `rouge` | ROUGE-L score |
| `ruby` | Code quality metric |

---

## 🎯 Прогресс-бары

### Прогресс-бар эпохи

Показывает прогресс внутри эпохи с live метриками:

```
Epoch 3/10 |████████████████████| 100/100 [05:32<00:00, 3.32s/batch]
  loss: 2.3456  reward: 0.7821  lr: 1.00e-05  step: 300/1000
```

**Показывает:**
- Текущий batch / Всего batches
- Прошедшее время / Оставшееся время
- Скорость обработки (batches/sec)
- Текущие метрики (loss, reward, lr, step)

### Прогресс-бар Evaluation

Показывает прогресс оценки модели:

```
Evaluating |████████████████| 20/20 [02:15<00:00, 6.75s/batch]
  processed: 160  avg_reward: 0.7234
```

---

## 📊 Summary после каждой эпохи

После завершения эпохи печатается detailed summary:

```
================================================================================
EPOCH 3 SUMMARY
================================================================================

[Training Metrics]
  Loss:          2.345678
  Reward:        0.782145
  KL Divergence: 0.012345
  Entropy:       2.567890
  Learning Rate: 1.00e-05

[Evaluation Metrics]
  BERTScore:  0.5234
  CodeBLEU:   0.3891
  BLEU:       0.2678
  ROUGE:      0.3456
  RUBY:       0.2134

[Performance]
  Epoch Time:             332.50s
  Samples/sec:            4.82
  Estimated Time Remaining: 38.9 min (2334s)

================================================================================
```

---

## 📈 Автоматические графики

### 1. Training Progress (`training_progress.png`)

Два графика:
- **Loss over Epochs**: Как уменьшается loss
- **Reward over Epochs**: Как растет reward

### 2. Policy Metrics (`policy_metrics.png`)

Два графика:
- **KL Divergence**: Расхождение с reference model
- **Entropy**: Exploration/exploitation balance

### 3. Evaluation Metrics (`evaluation_metrics.png`)

Все evaluation метрики на одном графике:
- BERTScore (красный)
- CodeBLEU (синий)
- BLEU (зеленый)
- ROUGE (оранжевый)
- RUBY (фиолетовый)

---

## 💻 Программный доступ к метрикам

### Чтение CSV

```python
import pandas as pd

# Загрузить все метрики
df = pd.read_csv('modern_outputs/metrics/training_metrics.csv')

# Посмотреть последние 5 эпох
print(df.tail())

# Найти лучшие метрики
best_reward_epoch = df.loc[df['reward'].idxmax()]
print(f"Best reward: {best_reward_epoch['reward']} at epoch {best_reward_epoch['epoch']}")
```

### Чтение JSON

```python
import json

# Загрузить полную историю
with open('modern_outputs/metrics/training_metrics.json', 'r') as f:
    history = json.load(f)

# Последняя эпоха
last_epoch = history[-1]
print(f"Epoch {last_epoch['epoch']}: Loss={last_epoch['loss']}, Reward={last_epoch['reward']}")

# Все rewards
rewards = [epoch['reward'] for epoch in history]
print(f"Average reward: {sum(rewards)/len(rewards)}")
```

### Через MetricsTracker API

```python
from modern_rlhf.metrics_tracker import MetricsTracker

# Создать tracker
tracker = MetricsTracker(output_dir="./modern_outputs/metrics")

# Получить лучшие метрики
best = tracker.get_best_metrics()
print(f"Best reward: {best['best_reward']}")
print(f"Lowest loss: {best['lowest_loss']}")

# Получить summary
summary = tracker.export_summary()
print(f"Total epochs: {summary['total_epochs']}")
print(f"Total training time: {summary['total_training_time']:.2f}s")
```

---

## ⚙️ Настройка

### Отключить графики (если нет matplotlib)

Графики автоматически отключаются, если matplotlib не установлен:

```bash
# Удалить matplotlib из requirements.txt
# Или не устанавливать: pip install torch transformers trl
```

### Изменить директорию вывода

В `config.py`:

```python
@dataclass
class DataConfig:
    output_path: str = "./my_custom_output"  # Изменить здесь
```

Метрики будут в `./my_custom_output/metrics/`

### Изменить частоту сохранения

По умолчанию метрики сохраняются после каждой эпохи. Это настраивается в `MetricsTracker`:

```python
# В modern_rlhf/metrics_tracker.py
class MetricsTracker:
    def __init__(self, output_dir: str = "./modern_outputs/metrics", save_every_n_epochs: int = 1):
        self.save_every_n_epochs = save_every_n_epochs
```

---

## 📋 Примеры использования

### Мониторинг во время обучения

```bash
# Запустить обучение
python fix_training.py

# В другом терминале - смотреть метрики в real-time
watch -n 5 tail -n 20 modern_outputs/metrics/training_metrics.csv
```

### Анализ после обучения

```python
import pandas as pd
import matplotlib.pyplot as plt

# Загрузить метрики
df = pd.read_csv('modern_outputs/metrics/training_metrics.csv')

# Построить custom график
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['epoch'], df['reward'], marker='o', label='Reward')
ax.plot(df['epoch'], df['loss'], marker='s', label='Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True)
plt.savefig('my_custom_plot.png')
plt.show()
```

### Сравнение нескольких запусков

```python
import pandas as pd

# Загрузить метрики из разных runs
run1 = pd.read_csv('run1/metrics/training_metrics.csv')
run2 = pd.read_csv('run2/metrics/training_metrics.csv')

# Сравнить
print(f"Run 1 best reward: {run1['reward'].max()}")
print(f"Run 2 best reward: {run2['reward'].max()}")

# Построить сравнение
import matplotlib.pyplot as plt
plt.plot(run1['epoch'], run1['reward'], label='Run 1')
plt.plot(run2['epoch'], run2['reward'], label='Run 2')
plt.legend()
plt.savefig('comparison.png')
```

---

## 🚀 Best Practices

### 1. Мониторинг в Real-time

Открывайте графики в background во время обучения:

```bash
# Linux/Mac
watch -n 30 eog modern_outputs/metrics/plots/training_progress.png

# Windows
# Откройте файл и включите auto-refresh в вашем image viewer
```

### 2. Регулярные Checkpoints

```python
# В config.py
save_steps: int = 100  # Сохранять checkpoint каждые 100 steps
```

### 3. Early Stopping

Следите за `reward` и `loss`:
- Если `reward` не растет 5+ эпох → возможно переобучение
- Если `loss` не падает → возможно проблема с LR

### 4. Backup Metrics

```bash
# Периодически сохраняйте metrics в backup
cp -r modern_outputs/metrics backups/metrics_$(date +%Y%m%d_%H%M%S)
```

---

## 🐛 Troubleshooting

### Проблема: Графики не создаются

**Решение**: Установите matplotlib:
```bash
pip install matplotlib seaborn
```

### Проблема: CSV файл пустой

**Решение**: Проверьте, что обучение запустилось корректно. Метрики логируются только после завершения первой эпохи.

### Проблема: Прогресс-бар не отображается

**Решение** (Windows):
```python
# В modern_rlhf/trainer.py уже настроено:
tqdm_kwargs = {'ascii': True, 'ncols': 100}  # ASCII режим для Windows
```

### Проблема: Metrics файлы слишком большие

**Решение**: Логируйте метрики реже или очищайте старые файлы:
```bash
# Оставить только последние 10 эпох
tail -n 11 modern_outputs/metrics/training_metrics.csv > temp.csv
mv temp.csv modern_outputs/metrics/training_metrics.csv
```

---

## 📚 Дополнительные ресурсы

- **Metrics API**: `modern_rlhf/metrics_tracker.py`
- **Trainer Integration**: `modern_rlhf/trainer.py` (lines 103-106, 814-830, 1247-1261)
- **Config**: `modern_rlhf/config.py`

---

**✅ Готово!** Система мониторинга полностью интегрирована и работает автоматически. Просто запустите обучение, и метрики будут логироваться в real-time! 🚀

