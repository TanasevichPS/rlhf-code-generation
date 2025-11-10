"""
Simple script to generate only synthetic human feedback dataset.
Run this before training if you need to recreate the synthetic data.
"""

import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("\n" + "="*70)
    print("  SYNTHETIC HUMAN FEEDBACK GENERATOR")
    print("="*70)
    
    # Import modules
    try:
        from modern_rlhf.config import ModernRLHFConfig
        from modern_rlhf.data_loader import ModernDataLoader
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        logger.error("Make sure you're in the project root directory")
        return 1
    
    # Initialize
    logger.info("\n[1/3] Initializing configuration...")
    config = ModernRLHFConfig()
    
    # Check CoNaLa dataset
    conala_path = Path(config.data.conala_local_path or "./conala-corpus")
    if not conala_path.exists():
        logger.error(f"\n[ERROR] CoNaLa dataset not found at: {conala_path}")
        logger.error("Please download the CoNaLa dataset first!")
        return 1
    
    logger.info(f"[OK] CoNaLa dataset found: {conala_path}")
    
    # Create data loader
    logger.info("\n[2/3] Initializing data loader...")
    data_loader = ModernDataLoader(config)
    
    # Generate synthetic feedback
    logger.info("\n[3/3] Generating synthetic human feedback...")
    target_size = config.data.target_feedback_size  # Fixed: correct attribute name
    logger.info(f"  - Target samples: {target_size}")
    logger.info(f"  - Output directory: {config.data.human_feedback_path}")
    logger.info(f"  - Based on: CoNaLa dataset")
    
    try:
        feedback_items = data_loader.generate_synthetic_human_feedback(
            n=target_size,
            output_dir=config.data.human_feedback_path
        )
        
        print("\n" + "="*70)
        print("  [SUCCESS] SYNTHETIC DATASET CREATED!")
        print("="*70)
        logger.info(f"\n  Generated: {len(feedback_items)} feedback entries")
        logger.info(f"  Location: {config.data.human_feedback_path}/")
        
        # Show sample
        if feedback_items:
            print("\n--- Sample Entry ---")
            sample = feedback_items[0]
            print(f"Prompt: {sample.get('prompt', '')[:80]}...")
            print(f"Response: {sample.get('response', '')[:80]}...")
            print(f"Rating: {sample.get('rating', 'N/A')}/5")
        
        print("\n[OK] Ready for RLHF training!")
        print("Next step: python fix_training.py\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n[ERROR] Failed to generate synthetic feedback: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

