# Проверка статуса обучения
import sys
import os
import json
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("DIAGNOSTICS")
print("=" * 60)

# 1. Проверка конфигурации
config_path = Path("modern_outputs/config.json")
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    training = config.get('training', {})
    print(f"\n[CONFIG] Training parameters:")
    print(f"   Total steps: {training.get('total_steps', 'N/A')}")
    print(f"   PPO epochs: {training.get('ppo_epochs', 'N/A')}")
    print(f"   Batch size: {training.get('batch_size', 'N/A')}")
    print(f"   Gradient accumulation: {training.get('gradient_accumulation_steps', 'N/A')}")

# 2. Проверка чекпоинтов
checkpoints = sorted(Path("modern_outputs").glob("checkpoint-*"), key=lambda x: x.stat().st_mtime)
if checkpoints:
    latest = checkpoints[-1]
    mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
    age = datetime.now() - mod_time
    print(f"\n[CHECKPOINTS] Latest checkpoint: {latest.name}")
    print(f"   Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Age: {age}")
    
    # Читаем training_state если есть
    state_file = latest / "training_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        print(f"   Current step: {state.get('step', 'N/A')}")
        print(f"   Current epoch: {state.get('epoch', 'N/A')}")
else:
    print("\n[WARNING] No checkpoints found")

# 3. Проверка результатов
results_path = Path("modern_outputs/pipeline_results.json")
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
    timestamp = results.get('timestamp', '')
    success = results.get('success', False)
    print(f"\n[RESULTS] From {timestamp[:19]}:")
    print(f"   Status: {'SUCCESS' if success else 'FAILED'}")
    if results.get('total_time'):
        print(f"   Total time: {results['total_time']:.1f} sec")
    if results.get('training_time'):
        print(f"   Training time: {results['training_time']:.1f} sec")

# 4. Оценка времени
if checkpoints and config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    total_steps = config.get('training', {}).get('total_steps', 500)
    
    if latest:
        state_file = latest / "training_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            current_step = state.get('step', 0)
            progress = (current_step / total_steps) * 100 if total_steps > 0 else 0
            print(f"\n[PROGRESS]")
            print(f"   Step {current_step} of {total_steps} ({progress:.1f}%)")
            
            # Оценка оставшегося времени (если есть данные о времени)
            if results_path.exists():
                with open(results_path) as f:
                    results = json.load(f)
                prev_time = results.get('training_time', 0)
                if prev_time > 0 and current_step > 0:
                    time_per_step = prev_time / max(current_step, 1)
                    remaining_steps = total_steps - current_step
                    eta_seconds = time_per_step * remaining_steps
                    eta_minutes = eta_seconds / 60
                    print(f"   Estimated time remaining: ~{eta_minutes:.1f} minutes")

# 5. Проверка процесса Python
print(f"\n[PROCESSES] Active Python processes:")
try:
    import psutil
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'create_time']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.cmdline())[:80]
                if 'rlhf' in cmdline.lower() or 'modern' in cmdline.lower():
                    runtime = datetime.now() - datetime.fromtimestamp(proc.info['create_time'])
                    mem_mb = proc.info['memory_info'].rss / 1024 / 1024
                    print(f"   • PID {proc.info['pid']}: CPU {proc.info['cpu_percent']:.1f}%, "
                          f"RAM {mem_mb:.0f}MB, работает {runtime}")
                    print(f"     Команда: {cmdline}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
except ImportError:
    print("   ⚠️  psutil не установлен (pip install psutil)")

print("\n" + "=" * 60)

