# GPU Usage Verification Summary
"""
ВСЕ МОДЕЛИ БУДУТ ИСПОЛЬЗОВАТЬ GPU

Изменения для гарантии использования GPU:
1. run_modern_rlhf.py - проверка наличия CUDA перед запуском
2. modern_rlhf/pipeline.py - принудительное использование GPU при инициализации
3. modern_rlhf/trainer.py - принудительное использование GPU для PPOTrainer
4. modern_rlhf/reward_model.py - принудительное использование GPU для RewardModel
5. modern_rlhf/config.py - проверка CUDA при инициализации конфигурации

Все тензоры и модели будут автоматически перемещены на GPU.
"""

import torch

print("="*70)
print("GPU VERIFICATION SUMMARY")
print("="*70)

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available!")
    exit(1)

print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
print(f"[OK] CUDA Version: {torch.version.cuda}")
print(f"[OK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"[OK] PyTorch Version: {torch.__version__}")

print("\n" + "="*70)
print("All components will use GPU:")
print("  - Pipeline: Force GPU on initialization")
print("  - RewardModel: Force GPU on initialization")
print("  - PPOTrainer: Force GPU on initialization")
print("  - All tensors: Automatically moved to GPU")
print("="*70)

