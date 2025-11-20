"""
ClassifMLP - Neural Code Quality Classification Pipeline
"""

from .integrated_system import (
    EnhancedClassifierWithFeatures,
    IntegratedTrainingPipeline,
    InferencePipeline,
    train_integrated_system,
    run_inference_demo
)
from .model import MultiHeadClassifier, build_feature_vector
from .embedding import EmbeddingEncoder, EmbeddingConfig
from .dataset import FeedbackClassificationDataset, iter_feedback_samples

__version__ = "1.0.0"
__all__ = [
    'EnhancedClassifierWithFeatures',
    'IntegratedTrainingPipeline',
    'InferencePipeline',
    'MultiHeadClassifier',
    'EmbeddingEncoder',
    'FeedbackClassificationDataset',
    'train_integrated_system',
    'run_inference_demo'
]
