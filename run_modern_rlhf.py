#!/usr/bin/env python3
"""
Quick Start Script for Modern RLHF
==================================

Simple script to run the modern RLHF framework with your existing data.
"""

import sys
import os
from pathlib import Path
import json
import random
import time

# Add modern_rlhf to path
sys.path.insert(0, str(Path(__file__).parent / "modern_rlhf"))

# Ensure stdout/stderr use UTF-8 where possible to avoid console encoding errors
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    # Older Python / environments may not support reconfigure; ignore
    pass

from modern_rlhf import ModernRLHFPipeline, get_research_config
from modern_rlhf.config import ModernRLHFConfig

def main():
    """Quick start function."""
    print("Modern RLHF Framework - Quick Start")
    print("=" * 50)
    
    # Create configuration
    config = get_research_config()
    
    # Adjust paths to use existing data
    config.data.train_data_path = r"C:\Users\Полина\Desktop\Работа\huawei\rlhf\conala-corpus\conala-train.json"
    config.data.eval_data_path = r"C:\Users\Полина\Desktop\Работа\huawei\rlhf\conala-corpus\conala-test.json"
    config.data.human_feedback_path = "./evaluation_results_server"
    config.data.output_path = "./modern_outputs"
    config.data.min_prompt_length = 0
    config.data.min_response_length = 0
    # Force local CoNaLa corpus (preferred over Hub)
    #config.data.conala_local_path = r"C:\Users\Полина\Desktop\Работа\huawei\rlhf\conala-corpus"

    config.data.train_data_path = "./datasets_for_training"
    config.data.eval_data_path = "./datasets_for_eval"
    config.data.human_feedback_path = "./evaluation_results_server"
    config.data.output_path = "./modern_outputs"
    
    # Set experiment name
    config.experiment_name = "modern_rlhf_experiment"
    
    # ============================================
    # FULL TRAINING CONFIGURATION (20 epochs)
    # ============================================
    # Training parameters for full training run
    config.training.ppo_epochs = 20  # Full training run
    config.training.total_steps = 5000  # More steps for better convergence
    config.training.learning_rate = 1e-5  # Stable learning rate
    config.training.batch_size = 4
    config.training.gradient_accumulation_steps = 4  # Effective batch size = 16
    config.training.early_stopping_patience = 7  # Allow more epochs before early stopping
    config.training.save_steps = 250  # Save checkpoint every 250 steps
    config.training.eval_steps = 100  # Evaluate every 100 steps
    config.training.logging_steps = 10  # Log every 10 steps
    
    # Evaluation settings
    config.evaluation.eval_samples = 100  # Evaluate on 100 samples
    
    # Set target metrics
    config.evaluation.target_bertscore = 0.7
    config.evaluation.target_codebleu = 0.6
    config.evaluation.target_bleu = 0.4
    config.evaluation.target_rouge = 0.5
    config.evaluation.target_ruby = 0.3
    
    # Reward model training
    config.reward.reward_epochs = 5  # Train reward model for 5 epochs
    
    # Prefer local CoNaLa corpus if it exists in the repo (you downloaded JSONs into conala-corpus/)
    config.data.conala_local_path = "./conala-corpus"
    
    # Enable model-based synthetic human feedback generation using a small model
    config.data.use_model_for_synth_feedback = True
    config.data.synth_feedback_model_name = "gpt2"
    
    # Force GPU usage - must use GPU
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available! Training requires GPU. Please ensure CUDA is installed and GPU is accessible.")
    
    config.hardware.device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    print(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB) - Using GPU for training")
    print(f"CUDA version: {torch.version.cuda}")

    # Diagnostic prints and optional pre-generation of synthetic human feedback using model
    #  - show local conala files
    conala_dir = Path(config.data.conala_local_path)
    if conala_dir.exists():
        files = sorted([p.name for p in conala_dir.glob('*.json*')])
        if files:
            print(f"Local Conala files found: {files}")
        else:
            print("No local Conala JSON files found in conala-corpus")
    else:
        print(f"Conala local corpus path does not exist: {conala_dir}")

    # Pre-generate synthetic human feedback with model if requested
    if getattr(config.data, 'use_model_for_synth_feedback', False):
        try:
            from transformers import pipeline, set_seed
            try:
                import torch
                device = 0 if torch.cuda.is_available() else -1
            except Exception:
                device = -1

            model_name = config.data.synth_feedback_model_name
            print(f"Initializing model for synthetic human feedback: {model_name} (device={device})")
            gen = pipeline('text-generation', model=model_name, device=device)
            set_seed(42)

            # Try to load some examples from local conala to rate
            examples = []
            if conala_dir.exists():
                # prefer test file
                test_candidates = list(conala_dir.glob('*test*.json*'))
                files_to_read = test_candidates if test_candidates else list(conala_dir.glob('*.json*'))
                for p in files_to_read[:50]:
                    try:
                        with open(p, 'r', encoding='utf-8') as f:
                            if p.suffix == '.jsonl' or p.name.lower().endswith('.jsonl'):
                                for line in f:
                                    if not line.strip():
                                        continue
                                    obj = json.loads(line)
                                    examples.append((obj.get('rewritten_intent') or obj.get('intent') or obj.get('question') or '', obj.get('snippet') or obj.get('code') or obj.get('response') or ''))
                            else:
                                obj = json.load(f)
                                records = []
                                if isinstance(obj, list):
                                    records = obj
                                elif isinstance(obj, dict):
                                    for key in ['test', 'data', 'examples', 'items']:
                                        if key in obj and isinstance(obj[key], list):
                                            records = obj[key]
                                            break
                                for item in records:
                                    if isinstance(item, dict):
                                        examples.append((item.get('rewritten_intent') or item.get('intent') or item.get('question') or '', item.get('snippet') or item.get('code') or item.get('response') or ''))
                    except Exception:
                        continue

            # If no examples found, synthesize simple examples
            if not examples:
                for i in range(20):
                    examples.append((f"Example prompt {i}", f"Example response {i}"))

            # Generate ratings
            out_dir = Path(config.data.human_feedback_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            items = []
            for i, (prompt_text, resp_text) in enumerate(examples[:50]):
                instruction = f"Rate the following response on a scale 1-5 for correctness and usefulness. Response: '''{resp_text}'''\nRating:"
                try:
                    res = gen(instruction, max_new_tokens=8, do_sample=False)
                    text = res[0].get('generated_text', '')
                    # try to find first digit 1-5
                    import re
                    m = re.search(r'([1-5])', text)
                    rating = int(m.group(1)) if m else random.randint(1, 5)
                except Exception:
                    rating = random.randint(1, 5)

                items.append({'id': f'model_synth_{i}', 'prompt': prompt_text, 'response': resp_text, 'rating': rating, 'comment': f'Generated by {model_name}'})

            fname = out_dir / f"model_synthetic_human_feedback_{int(time.time())}.json"
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(items, f, indent=2)
            print(f"Wrote model-based synthetic human feedback to {fname}")
        except Exception as e:
            print(f"Model-based synthetic human feedback generation failed: {e}")

    print(f"Training data: {config.data.train_data_path}")
    print(f"Evaluation data: {config.data.eval_data_path}")
    print(f"Human feedback: {config.data.human_feedback_path}")
    print(f"Output directory: {config.data.output_path}")
    if getattr(config.data, 'conala_local_path', None):
        print(f"CoNaLa local corpus: {config.data.conala_local_path}")
    
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"PPO Epochs: {config.training.ppo_epochs}")
    print(f"Total Steps: {config.training.total_steps}")
    print(f"Batch Size: {config.training.batch_size} (x{config.training.gradient_accumulation_steps} accumulation = {config.training.batch_size * config.training.gradient_accumulation_steps} effective)")
    print(f"Learning Rate: {config.training.learning_rate}")
    print(f"Early Stopping Patience: {config.training.early_stopping_patience} epochs")
    print(f"Reward Model Epochs: {config.reward.reward_epochs}")
    print(f"Evaluation Samples: {config.evaluation.eval_samples}")
    print("="*70 + "\n")
    
    print(f"Target BERTScore: {config.evaluation.target_bertscore}")
    print(f"Target CodeBLEU: {config.evaluation.target_codebleu}")
    print(f"Target BLEU: {config.evaluation.target_bleu}")
    print(f"Target ROUGE: {config.evaluation.target_rouge}")
    print(f"Target Ruby: {config.evaluation.target_ruby}")
    print()
    try:
        hf_is_ok = False
        try:
            from datasets import load_dataset
            try:
                # attempt to load curated test split; this may raise if dataset scripts are unsupported
                _ = load_dataset('neulab/conala', 'curated', split='test')
                hf_is_ok = True
            except RuntimeError as re:
                if 'Dataset scripts are no longer supported' in str(re):
                    hf_is_ok = False
                else:
                    hf_is_ok = False
            except Exception:
                hf_is_ok = False
        except Exception:
            hf_is_ok = False

        if not hf_is_ok:
            # attempt to download curated parquet directly via huggingface_hub
            try:
                from huggingface_hub import hf_hub_download
                # Save downloaded parquet into conala-corpus/ so loader can use it as a local corpus
                conala_dir = Path(config.data.conala_local_path) if getattr(config.data, 'conala_local_path', None) else Path('conala-corpus')
                conala_dir.mkdir(parents=True, exist_ok=True)
                candidates = [
                    'curated/test-00000-of-00001.parquet',
                    'curated/test.parquet',
                    'test-00000-of-00001.parquet',
                    'test.parquet'
                ]
                downloaded = False
                for fname in candidates:
                    try:
                        local_path = hf_hub_download(repo_id='neulab/conala', filename=fname, repo_type='dataset', local_dir=str(conala_dir))
                        if Path(local_path).exists():
                            downloaded = True
                            break
                    except Exception:
                        continue
            except Exception as e:
                print(f"HF parquet download fallback failed: {e}")

            # After attempting download, set local path if parquet was obtained
            if downloaded:
                # point loader to local conala-corpus dir so it picks up parquet without using dataset script
                config.data.conala_local_path = str(conala_dir)
                print(f"Downloaded CoNaLa curated parquet to {conala_dir}; loader will use local parquet.")
            else:
                print("HF curated parquet not found via hf_hub_download; loader will attempt HF and fallbacks at runtime.")
    except Exception:
        pass

    print(f"Target BERTScore: {config.evaluation.target_bertscore}")
    print(f"Target CodeBLEU: {config.evaluation.target_codebleu}")
    print(f"Target BLEU: {config.evaluation.target_bleu}")
    print(f"Target ROUGE: {config.evaluation.target_rouge}")
    print(f"Target Ruby: {config.evaluation.target_ruby}")
    print()
    
    # Create output directory
    os.makedirs(config.data.output_path, exist_ok=True)
    
    # Create pipeline
    print("Initializing Modern RLHF Pipeline...")
    pipeline = ModernRLHFPipeline(config)

    # Run pipeline (let exceptions bubble up so we see full trace during debugging)
    print("Starting training pipeline...")
    results = pipeline.run_full_pipeline()

    # Create visualizations
    print("Creating visualizations...")
    pipeline.visualize_results()

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    if results.success:
        print("Pipeline completed successfully!")
        print(f"Total time: {results.total_time:.2f} seconds")
        print(f"Training time: {results.training_time:.2f} seconds")

        print("\nFinal Metrics:")
        for metric, value in results.final_metrics.items():
            print(f"  {metric}: {value}")

        print("\nEvaluation Metrics:")
        for metric, value in results.evaluation_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")

        # Check targets
        if 'targets_met' in results.evaluation_metrics:
            targets_met = results.evaluation_metrics['targets_met']
            met_count = sum(targets_met.values())
            total_count = len(targets_met)
            print(f"\nTargets Met: {met_count}/{total_count}")

            if met_count == total_count:
                print("All targets achieved!")
            else:
                print("Some targets not met:")
                for metric, met in targets_met.items():
                    status = "✅" if met else "❌"
                    print(f"  {status} {metric}")

        print(f"\nResults saved to: {config.data.output_path}")
    else:
        print("Pipeline failed!")
        print(f"Error: {results.error_message}")

if __name__ == "__main__":
    main()
