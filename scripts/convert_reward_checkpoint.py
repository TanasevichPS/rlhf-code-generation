#!/usr/bin/env python3
"""Convert a legacy reward checkpoint (state_dict .pt) into an HF-style folder with
robust handling of embedding size mismatches (vocab/position embeddings).

This script will:
 - Load the legacy checkpoint (state_dict) from --checkpoint
 - Instantiate the current `ImprovedCodeRewardModel` with --model-name
 - Merge compatible keys, and for embedding size mismatches copy the overlapping rows
 - Save the base HF model (bert) and tokenizer to --out-dir/base_model and --out-dir
 - Save `improved_state_dict.pt` in --out-dir with the merged state_dict

Usage:
  python scripts/convert_reward_checkpoint.py --checkpoint outputs/trained_reward_model.pt --out-dir outputs/reward_model_hf_converted
"""
import argparse
import os
import sys
import torch
import logging

# Ensure repo root is on sys.path so `src` imports work when script is run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models.reward_model import ImprovedCodeRewardModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location='cpu')
    # Accept either raw state_dict or dict with keys
    if isinstance(ckpt, dict) and not any(k.startswith('bert.') or k.startswith('consistency_head') for k in ckpt.keys()):
        # If it's wrapper dict, try common keys
        for candidate in ['state_dict', 'model_state_dict', 'improved_state_dict']:
            if candidate in ckpt:
                return ckpt[candidate]
    return ckpt


def merge_state_dicts(base_state, ckpt_state):
    """Merge ckpt_state into base_state, handling embedding size mismatches gracefully."""
    merged = base_state.copy()
    for k, v in ckpt_state.items():
        if k not in base_state:
            logger.info(f"Skipping key not in target model: {k}")
            continue
        base_v = base_state[k]
        if v.shape == base_v.shape:
            merged[k] = v
            continue

        # Handle common embedding mismatches by copying overlap
        if 'embeddings.word_embeddings.weight' in k or 'embeddings.position_embeddings.weight' in k:
            min_rows = min(v.shape[0], base_v.shape[0])
            new_v = base_v.clone()
            try:
                new_v[:min_rows, :] = v[:min_rows, :]
                merged[k] = new_v
                logger.info(f"Merged embedding {k}: ckpt_rows={v.shape[0]} target_rows={base_v.shape[0]} copied={min_rows}")
            except Exception as e:
                logger.warning(f"Could not merge embedding {k}: {e} — leaving target init")
            continue

        # If shapes differ but are compatible on trailing dims, try to copy overlapping prefix
        if v.ndim == base_v.ndim and all(v.shape[i] == base_v.shape[i] for i in range(1, v.ndim)):
            min0 = min(v.shape[0], base_v.shape[0])
            new_v = base_v.clone()
            new_v[:min0] = v[:min0]
            merged[k] = new_v
            logger.info(f"Partially merged prefix for {k}: copied {min0} rows")
            continue

        logger.warning(f"Shape mismatch for {k}: ckpt {v.shape} vs target {base_v.shape}. Skipping key.")

    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Path to legacy state_dict .pt')
    p.add_argument('--out-dir', default='outputs/reward_model_hf_converted', help='Output directory for HF-style model')
    p.add_argument('--model-name', default='microsoft/codebert-base', help='Base pretrained model for reward')
    args = p.parse_args()

    ckpt_path = args.checkpoint
    out_dir = args.out_dir

    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return 2

    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Loading checkpoint from {ckpt_path}")
    ckpt_state = load_checkpoint(ckpt_path)
    if not isinstance(ckpt_state, dict):
        logger.error("Loaded checkpoint is not a state_dict-like mapping")
        return 3

    logger.info(f"Initializing target model ({args.model_name}) to obtain shapes")
    model = ImprovedCodeRewardModel(model_name=args.model_name)
    base_state = model.state_dict()

    logger.info("Merging state dicts (will copy overlapping rows for embeddings)")
    merged = merge_state_dicts(base_state, ckpt_state)

    # Save HF-style components: base bert and tokenizer
    base_out = os.path.join(out_dir, 'base_model')
    os.makedirs(base_out, exist_ok=True)
    try:
        model.bert.save_pretrained(base_out)
        model.tokenizer.save_pretrained(out_dir)
        logger.info(f"Saved base model to {base_out} and tokenizer to {out_dir}")
    except Exception as e:
        logger.warning(f"Could not save base model/tokenizer via transformers API: {e}")

    # Save merged improved state
    improved_state_path = os.path.join(out_dir, 'improved_state_dict.pt')
    torch.save(merged, improved_state_path)
    logger.info(f"Saved merged improved state_dict to {improved_state_path}")

    logger.info("Conversion complete")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
