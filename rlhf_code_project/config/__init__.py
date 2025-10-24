"""
Configuration module for RLHF Code Project
"""

from .training_config import (
    RLHFConfig,
    get_dpo_config,
    get_ppo_config,
    get_fast_config,
    get_research_config
)

__all__ = [
    'RLHFConfig',
    'get_dpo_config',
    'get_ppo_config',
    'get_fast_config',
    'get_research_config'
]
