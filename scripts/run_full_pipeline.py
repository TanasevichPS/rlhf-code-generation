#!/usr/bin/env python3
"""Run full RLHF pipeline: prefs -> reward training -> PPO -> eval -> recompute -> plot

Usage examples:
  python scripts/run_full_pipeline.py --pairwise-epochs 8 --ppo-steps 1000
  python scripts/run_full_pipeline.py --smoke
"""
import argparse
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run(cmd, desc=None, check=True):
    if desc:
        print('\n>>', desc)
    print('Running:', cmd)
    rc = subprocess.run(cmd, shell=True)
    if check and rc.returncode != 0:
        print(f'Command failed (rc={rc.returncode}):', cmd)
        sys.exit(rc.returncode)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pairwise-epochs', type=int, default=8, help='Epochs for pairwise reward training')
    p.add_argument('--ppo-steps', type=int, default=1000, help='Steps for PPO training (or episodes depending on script)')
    p.add_argument('--device', default='cpu', help='Device for training (cpu or cuda)')
    p.add_argument('--smoke', action='store_true', help='Run a quick smoke test instead of full training')
    args = p.parse_args()

    # Smoke defaults
    if args.smoke:
        pw_pairwise_epochs = 1
        ppo_steps = 10
        sample_arg = '--sample-size 20'
    else:
        pw_pairwise_epochs = args.pairwise_epochs
        ppo_steps = args.ppo_steps
        sample_arg = ''

    python = sys.executable
    prefs_folder = os.path.join(ROOT, 'evaluation_results_server')
    if not os.path.exists(prefs_folder):
        print('Preferences folder not found:', prefs_folder)
        print('Please ensure human preference JSONs are in evaluation_results_server')
        sys.exit(1)

    # 1) Convert prefs and train reward model (pairwise)
    trained_reward_path = os.path.join(ROOT, 'outputs', 'trained_reward_model.pt')
    if os.path.exists(trained_reward_path):
        print('\n>> Skipping reward training: existing trained reward model found at', trained_reward_path)
    else:
        cmd_reward = f'{python} "{os.path.join(ROOT, "scripts", "pref_convert_and_reward_train.py")}" --prefs-folder "{prefs_folder}" --pairwise --pairwise-epochs {pw_pairwise_epochs}'
        run(cmd_reward, desc='Training pairwise reward model')

    # 2) Run PPO training (main_rlhf_training.py assumed to accept --ppo-steps/--device)
    cmd_ppo = f'{python} "{os.path.join(ROOT, "scripts", "main_rlhf_training.py")}" --ppo-steps {ppo_steps} --device {args.device} '
    run(cmd_ppo, desc='Running PPO RLHF training')

    # 3) Evaluation (quick or full, depending on smoke)
    cmd_eval = f'{python} "{os.path.join(ROOT, "scripts", "evaluate_model.py")}" {sample_arg}'
    run(cmd_eval, desc='Evaluating model')

    # 4) Recompute metrics
    cmd_recompute = f'{python} "{os.path.join(ROOT, "scripts", "recompute_metrics.py")}"'
    run(cmd_recompute, desc='Recomputing metrics')

    # 5) Plot metrics
    cmd_plot = f'{python} "{os.path.join(ROOT, "scripts", "plot_metrics.py")}"'
    run(cmd_plot, desc='Plotting metrics by epoch')

    print('\nFull pipeline finished successfully. Outputs in outputs/ and evaluation_results/')

if __name__ == '__main__':
    main()
