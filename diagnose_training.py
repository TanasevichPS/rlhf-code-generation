"""
Training Diagnostics Script
===========================

Quick diagnostic tool to check if training setup is correct.
Run this before starting training to catch configuration issues.
"""

import sys
import torch
from pathlib import Path
import json

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def check_gpu():
    """Check GPU availability and specs."""
    print_header("GPU CHECK")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available")
        print("   Training requires GPU!")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ CUDA is available")
    print(f"   GPU count: {gpu_count}")
    
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"   GPU {i}: {name}")
        print(f"           Memory: {memory_gb:.1f} GB")
        print(f"           Compute: {props.major}.{props.minor}")
    
    return True

def check_config():
    """Check configuration settings."""
    print_header("CONFIGURATION CHECK")
    
    try:
        from modern_rlhf import ModernRLHFConfig
        config = ModernRLHFConfig()
        
        # Critical: do_sample must be True for training
        if not config.generation.do_sample:
            print("❌ CRITICAL: do_sample=False")
            print("   This will break PPO training!")
            print("   Fix: Set do_sample=True in config.py")
            return False
        else:
            print("✅ do_sample=True (correct for PPO)")
        
        # Check generation params
        print(f"   temperature: {config.generation.temperature}")
        print(f"   top_p: {config.generation.top_p}")
        print(f"   max_new_tokens: {config.generation.max_new_tokens}")
        
        # Check training params
        print(f"\n   Learning rate: {config.training.learning_rate}")
        print(f"   Batch size: {config.training.batch_size}")
        print(f"   PPO epochs: {config.training.ppo_epochs}")
        print(f"   Total steps: {config.training.total_steps}")
        
        # Check targets
        print(f"\n   Target BERTScore: {config.evaluation.target_bertscore}")
        print(f"   Target CodeBLEU: {config.evaluation.target_codebleu}")
        
        if config.evaluation.target_bertscore > 0.6:
            print("   ⚠️  Warning: BERTScore target might be too high")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False

def check_conala():
    """Check CoNaLa dataset."""
    print_header("CONALA DATASET CHECK")
    
    conala_path = Path("./conala-corpus")
    
    if not conala_path.exists():
        print(f"❌ CoNaLa directory not found: {conala_path}")
        return False
    
    print(f"✅ CoNaLa directory exists: {conala_path}")
    
    # Check train file
    train_file = conala_path / "conala-train.json"
    if not train_file.exists():
        print(f"❌ Training file not found: {train_file}")
        return False
    
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        train_count = len(train_data) if isinstance(train_data, list) else 0
        print(f"✅ Training file: {train_file}")
        print(f"   Samples: {train_count}")
    except Exception as e:
        print(f"❌ Failed to read training file: {e}")
        return False
    
    # Check test file
    test_file = conala_path / "conala-test.json"
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        test_count = len(test_data) if isinstance(test_data, list) else 0
        print(f"✅ Test file: {test_file}")
        print(f"   Samples: {test_count}")
    except Exception as e:
        print(f"❌ Failed to read test file: {e}")
        return False
    
    return True

def check_directories():
    """Check output directories."""
    print_header("DIRECTORIES CHECK")
    
    dirs = {
        "Output": "./modern_outputs",
        "Human feedback": "./evaluation_results_server",
        "Training data": "./datasets_for_training",
        "Eval data": "./datasets_for_eval"
    }
    
    all_ok = True
    for name, path in dirs.items():
        p = Path(path)
        if p.exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"⚠️  {name}: {path} (will be created)")
            try:
                p.mkdir(parents=True, exist_ok=True)
                print(f"   Created: {path}")
            except Exception as e:
                print(f"   ❌ Failed to create: {e}")
                all_ok = False
    
    return all_ok

def check_dependencies():
    """Check required dependencies."""
    print_header("DEPENDENCIES CHECK")
    
    required = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'datasets': 'Hugging Face Datasets',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'bert_score': 'BERTScore',
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} not installed")
            print(f"   Install: pip install {module}")
            all_ok = False
    
    return all_ok

def check_model_access():
    """Check if we can access required models."""
    print_header("MODEL ACCESS CHECK")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Try to load base model
        model_name = "microsoft/CodeGPT-small-py"
        print(f"Checking access to {model_name}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"✅ Can access {model_name}")
        except Exception as e:
            print(f"❌ Cannot access {model_name}: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model check failed: {e}")
        return False

def main():
    """Run all diagnostics."""
    print("\n" + "="*60)
    print("  MODERN RLHF TRAINING DIAGNOSTICS")
    print("="*60)
    
    results = {
        "GPU": check_gpu(),
        "Configuration": check_config(),
        "CoNaLa Dataset": check_conala(),
        "Directories": check_directories(),
        "Dependencies": check_dependencies(),
        "Model Access": check_model_access(),
    }
    
    # Summary
    print_header("SUMMARY")
    
    all_passed = True
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED")
        print("\nYou can now run training:")
        print("  python fix_training.py")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before training.")
        print("See CRITICAL_FIXES_SUMMARY.md for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

