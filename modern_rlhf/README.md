# Modern RLHF Framework for Code Generation

A state-of-the-art, clean implementation of RLHF (Reinforcement Learning from Human Feedback) specifically designed for code generation tasks with comprehensive evaluation metrics and modern training methods.

## 🚀 Features

- **Modern Architecture**: Clean, modular design with separation of concerns
- **Multiple Training Methods**: Support for both PPO and DPO (Direct Preference Optimization)
- **Comprehensive Metrics**: BERTScore, CodeBLEU, BLEU, ROUGE, and custom Ruby metric
- **Human Feedback Integration**: Advanced reward modeling with human preference integration
- **GPU Optimization**: Efficient training with mixed precision and gradient checkpointing
- **Experiment Tracking**: Built-in support for Weights & Biases and TensorBoard
- **Flexible Configuration**: Easy configuration management with predefined setups

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.6+ (for GPU training)
- 8GB+ GPU memory (recommended)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd modern_rlhf
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install the package** (optional):
```bash
pip install -e .
```

## 🎯 Quick Start

### Research Experiment
```bash
python main.py --mode research --epochs 10 --steps 2000
```

### Production Training
```bash
python main.py --mode production --device cuda --batch-size 8
```

### Fast Prototype
```bash
python main.py --mode fast --epochs 2 --steps 500
```

### Custom Configuration
```bash
python main.py \
    --learning-rate 1e-5 \
    --batch-size 4 \
    --model-name microsoft/CodeGPT-small-py \
    --target-bertscore 0.8 \
    --target-codebleu 0.7
```

## 📊 Target Metrics

The framework is designed to achieve the following target metrics:

- **BERTScore**: ≥ 0.7 (semantic similarity)
- **CodeBLEU**: ≥ 0.6 (code-specific evaluation)
- **BLEU**: ≥ 0.4 (n-gram overlap)
- **ROUGE**: ≥ 0.5 (summarization metrics)
- **Ruby**: ≥ 0.3 (custom code quality metric)

## 🏗️ Architecture

```
modern_rlhf/
├── config.py          # Configuration management
├── metrics.py         # Evaluation metrics
├── reward_model.py    # Reward modeling with human feedback
├── trainer.py         # PPO/DPO training
├── pipeline.py        # Main training pipeline
├── data_loader.py     # Data loading and preprocessing
├── main.py           # Command-line interface
└── requirements.txt   # Dependencies
```

## 🔧 Configuration

### Predefined Configurations

- **Research**: Optimized for experiments and research
- **Production**: Stable settings for production deployment
- **Fast**: Quick prototyping and testing

### Custom Configuration

Create a custom configuration file:

```json
{
  "model": {
    "base_model_name": "microsoft/CodeGPT-small-py",
    "reward_model_name": "microsoft/codebert-base"
  },
  "training": {
    "learning_rate": 5e-6,
    "batch_size": 4,
    "ppo_epochs": 10
  },
  "evaluation": {
    "target_bertscore": 0.7,
    "target_codebleu": 0.6
  }
}
```

## 📈 Training Pipeline

1. **Data Loading**: Load training, evaluation, and human feedback data
2. **Reward Model Training**: Train reward model with human preferences
3. **RLHF Training**: Train policy model using PPO or DPO
4. **Evaluation**: Comprehensive evaluation with multiple metrics
5. **Visualization**: Generate plots and analysis

## 🎛️ Command Line Options

### Training Parameters
- `--learning-rate`: Learning rate for training
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs
- `--steps`: Total number of training steps

### Model Parameters
- `--model-name`: Base model name
- `--reward-model-name`: Reward model name

### Hardware Parameters
- `--device`: Device to use (cpu/cuda/auto)

### Data Parameters
- `--train-data-path`: Path to training data
- `--eval-data-path`: Path to evaluation data
- `--human-feedback-path`: Path to human feedback data

### Output Parameters
- `--output-dir`: Output directory for results
- `--experiment-name`: Name of the experiment

## 📊 Evaluation Metrics

### BERTScore
Semantic similarity between generated and reference code using BERT embeddings.

### CodeBLEU
Code-specific evaluation metric that considers:
- N-gram match
- Weighted n-gram match
- Syntax match
- Semantic match

### BLEU
N-gram overlap between generated and reference text.

### ROUGE
Recall-Oriented Understudy for Gisting Evaluation, useful for code summarization.

### Ruby (Custom Metric)
Custom metric combining:
- Syntax correctness (40%)
- Code complexity (20%)
- Code style (20%)
- Execution success (20%)

## 🔬 Human Feedback Integration

The framework supports advanced human feedback integration:

- **Human Logits**: Integration of human-provided logits in the last transformer layer
- **Preference Data**: Support for pairwise preference data
- **Rating Integration**: Incorporation of human ratings into reward computation
- **Feedback Weighting**: Configurable weighting of human feedback vs. automated metrics

## 🚀 Advanced Features

### Direct Preference Optimization (DPO)
Alternative to PPO that directly optimizes human preferences without explicit reward modeling.

### Mixed Precision Training
Automatic mixed precision for faster training and reduced memory usage.

### Gradient Checkpointing
Memory-efficient training for large models.

### Experiment Tracking
Built-in support for Weights & Biases and TensorBoard for experiment monitoring.

## 📁 Data Format

### Training Data
```csv
prompt,response,rating
"Write a function to calculate factorial","def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",1.0
```

### Evaluation Data
```csv
prompt,reference
"Write a function to reverse a string","def reverse_string(s):\n    return s[::-1]"
```

### Human Feedback Data
```json
[
  {
    "prompt": "Write a function to calculate factorial",
    "response": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
    "rating": 5.0,
    "logits": [0.1, 0.2, 0.3, ...]
  }
]
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Slow Training**: Enable mixed precision training
3. **Poor Metrics**: Check data quality and adjust target thresholds

### Debug Mode
```bash
python main.py --debug --verbose
```

## 📚 Examples

### Basic Training
```bash
python main.py --mode research --epochs 5 --batch-size 2
```

### High-Performance Training
```bash
python main.py \
    --mode production \
    --device cuda \
    --batch-size 8 \
    --learning-rate 1e-6 \
    --epochs 20
```

### Custom Model Training
```bash
python main.py \
    --model-name microsoft/CodeGPT-small-py \
    --reward-model-name microsoft/codebert-base \
    --target-bertscore 0.8 \
    --target-codebleu 0.7
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers
- OpenAI for RLHF research
- Stanford for DPO research
- Microsoft for CodeGPT and CodeBERT models

## 📞 Support

For questions and support, please open an issue on GitHub or contact the development team.

---

**Note**: This framework is designed for research and educational purposes. For production use, please ensure proper testing and validation.
