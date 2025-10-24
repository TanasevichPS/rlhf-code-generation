# 🚀 RLHF Code Project - Simplified & Modern

A clean, efficient RLHF implementation for code generation with modern methods and simple architecture.

## ✨ Key Features

- **Direct Preference Optimization (DPO)** - Modern alternative to PPO
- **Simple Architecture** - Only 8 core modules, easy to understand
- **Comprehensive Metrics** - BERTScore, CodeBLEU, BLEU, ROUGE, Ruby
- **Human Feedback Integration** - Ready for human preference data
- **Fast Training** - Optimized for quick experiments

## 🏗️ Architecture

```
rlhf_code_project/
├── config/
│   └── training_config.py    # Simple configuration
├── data/
│   └── preference_dataset.py # Data loading
├── training/
│   └── dpo_trainer.py        # DPO training
├── evaluation/
│   └── metrics_calculator.py # All metrics
├── scripts/
│   └── train.py              # Main training script
└── requirements.txt          # Minimal dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
# Fast test (2 epochs, small batch)
python scripts/train.py --fast

# Full DPO training
python scripts/train.py --method dpo --epochs 5 --batch-size 8

# Custom configuration
python scripts/train.py --epochs 10 --batch-size 4 --device cuda
```

### 3. Check Results
Results are saved to `./rlhf_outputs/` with:
- Trained model
- Evaluation metrics
- Training logs

## 🎯 Target Metrics

The system is designed to achieve:
- **BERTScore**: ≥ 0.7 (semantic similarity)
- **CodeBLEU**: ≥ 0.6 (code-specific evaluation)
- **BLEU**: ≥ 0.4 (n-gram overlap)
- **ROUGE**: ≥ 0.5 (summarization metrics)
- **Ruby**: ≥ 0.3 (custom code quality metric)

## 🔧 Configuration

### Predefined Configs
```python
from config import get_fast_config, get_dpo_config, get_research_config

# Fast prototyping
config = get_fast_config()

# Production DPO training
config = get_dpo_config()

# Research experiments
config = get_research_config()
```

### Custom Configuration
```python
from config import RLHFConfig

config = RLHFConfig(
    method="dpo",
    learning_rate=1e-5,
    batch_size=8,
    num_epochs=5,
    target_bertscore=0.8,
    target_codebleu=0.7
)
```

## 📊 Training Methods

### DPO (Direct Preference Optimization)
- **Advantages**: Simpler than PPO, more stable training
- **Use case**: Modern RLHF training
- **Paper**: https://arxiv.org/abs/2305.18290

### PPO (Proximal Policy Optimization)
- **Advantages**: Well-established, good for complex rewards
- **Use case**: Traditional RLHF training
- **Status**: Coming soon

## 📈 Evaluation

The system automatically evaluates on:
- **Synthetic data** (if no real data available)
- **Your existing datasets** (from `datasets_for_eval/`)
- **All target metrics** with automatic comparison

## 🔬 Research Features

### Human Feedback Integration
```python
config.use_human_feedback = True
config.human_feedback_dim = 64
config.human_feedback_weight = 0.3
```

### Experiment Tracking
```python
# Add to requirements.txt: wandb>=0.15.0
# Enable in config: use_wandb = True
```

## 📝 Data Format

### Preference Data (for DPO)
```csv
prompt,chosen_response,rejected_response
"Write a function to calculate factorial","def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)","def factorial(n):\n    return 1"
```

### Evaluation Data
```csv
prompt,reference
"Write a function to add two numbers","def add(a, b):\n    return a + b"
```

## 🎉 Why This is Better

1. **Simplicity**: 8 modules vs 20+ files
2. **Modern**: DPO instead of complex PPO
3. **Fast**: Optimized for quick experiments
4. **Clean**: Easy to understand and modify
5. **Effective**: Designed to achieve your target metrics

## 🔬 Scientific Foundation

- **DPO**: Direct Preference Optimization (Rafailov et al., 2023)
- **Human Feedback**: Learning from Human Preferences (Christiano et al., 2017)
- **Code Generation**: Modern approaches from CodeX, AlphaCode

## 📞 Support

For questions or issues, please check the logs in `./rlhf_outputs/` or open an issue.

---

**Ready to achieve your research goals with modern RLHF!** 🚀
