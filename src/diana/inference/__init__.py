"""
Diana Inference Module

This module handles feature extraction and prediction for new aDNA samples.
It uses MUSET's methodology to compute unitig-based features consistent with training data.
"""

from .feature_extractor import FeatureExtractor, extract_features_from_fastq
from .predictor import Predictor

__all__ = [
    'FeatureExtractor',
    'extract_features_from_fastq',
    'Predictor',
]
