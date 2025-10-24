"""
Training module for RLHF Code Project
"""

try:
    from .dpo_trainer import DPOTrainer
    DPO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Full DPO trainer not available: {e}")
    DPO_AVAILABLE = False

from .simple_dpo_trainer import SimpleDPOTrainer

__all__ = ['DPOTrainer', 'SimpleDPOTrainer', 'DPO_AVAILABLE']
