"""Evaluation metrics for classification."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report as sklearn_report
)
from typing import Dict, List


def compute_metrics(y_true: np.ndarray, 
                   y_pred: np.ndarray,
                   average: str = "weighted") -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multi-class
        
    Returns:
        Dictionary of metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def classification_report(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         target_names: List[str] = None) -> str:
    """Generate classification report."""
    return sklearn_report(y_true, y_pred, target_names=target_names)


def compute_confusion_matrix(y_true: np.ndarray,
                            y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)
