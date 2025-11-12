"""Quick launcher for Simple DPO mock training (legacy pipeline)."""
import os
from pathlib import Path

import rlhf_code_project.training as training
from rlhf_code_project.config import get_fast_config
from rlhf_code_project.scripts.train import main


def run():
    # Force fallback to SimpleDPOTrainer by disabling the full DPO availability flag.
    training.DPO_AVAILABLE = False  # type: ignore[attr-defined]

    root = Path(__file__).resolve().parent
    config = get_fast_config()
    config.train_data_path = str(root / "datasets_for_training")
    config.eval_data_path = str(root / "datasets_for_eval")
    config.human_feedback_path = str(root / "evaluation_results_server")
    config.output_dir = str(root / "rlhf_outputs_fast")

    # Use mock identifiers so that the trainer definitely falls back to the simple implementation.
    config.policy_model_name = "mock-policy"
    config.reward_model_name = "mock-reward"
    config.device = "cpu"
    config.mixed_precision = False

    os.makedirs(config.output_dir, exist_ok=True)

    results = main(config)
    print("Run completed. Keys:", results.keys())


if __name__ == "__main__":
    run()
