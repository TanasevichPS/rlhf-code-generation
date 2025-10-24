"""
PPO orchestration script using TRL + Accelerate.

This script is a wrapper that will run PPO training once the SFT checkpoint
and a reward model are available. It requires `trl` (from huggingface/trl),
`transformers`, `accelerate`, and `torch`.

If dependencies are missing, exits with code 2 (so CI/tests can assert).

This file contains a high-level orchestration; fill trainer hyperparams via
command-line args.
"""
import argparse
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fail_missing_libs():
    print("Missing required libraries for PPO RLHF. Install `trl`, `transformers`, `accelerate`, and `torch`.")
    sys.exit(2)


try:
    import torch
    tver = getattr(torch, "__version__", "0.0.0")
    try:
        ver_parts = tver.split("+")[0].split(".")
        tmajor = int(ver_parts[0])
        tminor = int(ver_parts[1]) if len(ver_parts) > 1 else 0
    except Exception:
        tmajor, tminor = 0, 0
    if (tmajor, tminor) < (2, 1):
        fail_missing_libs()
    # trl may not be installed; import lazily where used
    import transformers
except Exception:
    fail_missing_libs()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_model_dir", default="outputs/sft_model_hf")
    p.add_argument("--reward_model_dir", default="outputs/reward_model_hf")
    p.add_argument("--output_dir", default="outputs/ppo_model")
    p.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = p.parse_args()

    # Basic checks
    if not os.path.exists(args.sft_model_dir):
        logging.error("SFT model directory not found: %s", args.sft_model_dir)
        return 3
    if not os.path.exists(args.reward_model_dir):
        logging.error("Reward model directory not found: %s", args.reward_model_dir)
        return 3

    # If TRL is available, run a minimal PPO loop. Otherwise, exit with code 2
    try:
        from transformers import AutoTokenizer
        from transformers import AutoModelForCausalLM
        # TRL imports (may be optional)
        import numpy as np
        from trl import PPOTrainer, PPOConfig
    except Exception:
        fail_missing_libs()

    # Minimal example using TRL's PPOTrainer (this is a high-level template).
    # Real runs should configure dataset, sampling/evaluation loops, and metrics.
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.sft_model_dir)

    # Load reward model as a callable: reward_fn(prompts, responses) -> np.array
    def reward_fn(prompts, responses):
        # This wrapper should load the HF reward model and score each prompt+response.
        # For the production script, we expect a directory with AutoModelForSequenceClassification
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_dir)
        reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_dir, num_labels=1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        scores = []
        for p, r in zip(prompts, responses):
            inp = reward_tokenizer(p + "\n" + r, return_tensors='pt', truncation=True, max_length=512).to(reward_model.device)
            with torch.no_grad():
                s = reward_model(**inp).logits.squeeze(-1).cpu().numpy().item()
            scores.append(s)
        return np.array(scores)

    # Build a minimal PPO config
    ppo_config = PPOConfig(
        model_name=args.sft_model_dir,
        learning_rate=1.41e-5,
        batch_size=16,
    )

    # Initialize PPO trainer (the real initialization requires many arguments; this is illustrative)
    ppo_trainer = PPOTrainer(model=model, tokenizer=tokenizer, config=ppo_config, reward_fn=reward_fn)

    # Example training loop (toy): sample prompts, generate responses, compute rewards, step
    # WARNING: Real PPO runs are complex and require robust sampling/evaluation and checkpointing.
    prompts = ["# Write a function that reverses a string\n"] * 8
    for epoch in range(1):
        responses = []
        for prompt in prompts:
            # generate via model
            out = tokenizer(prompt, return_tensors="pt")
            gen = model.generate(**{k: v.to(model.device) for k, v in out.items()}, max_length=128)
            responses.append(tokenizer.decode(gen[0], skip_special_tokens=True))
        rewards = reward_fn(prompts, responses)
        # ppo step (illustrative)
        stats = ppo_trainer.step(prompts, responses, rewards)
        logging.info("PPO step stats: %s", stats)

    os.makedirs(args.output_dir, exist_ok=True)
    marker = os.path.join(args.output_dir, "ppo_complete_marker.txt")
    with open(marker, "w", encoding="utf-8") as f:
        f.write("PPO run complete (toy example).\n")
    logging.info("Wrote PPO marker to %s", marker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
