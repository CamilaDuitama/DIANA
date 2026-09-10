"""Evaluation metrics for classification."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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


# --- Canonical classification metrics for the v9 evaluation ---------------------
#
# One definition, used by both diana-test and scripts/evaluation/09_baselines_v9.py,
# so a model number and a baseline number are always comparable. They were not
# before: the baselines passed `labels=seen` while diana-test passed no `labels` at
# all, which makes sklearn average over the UNION of true and predicted classes.
# Under a BioProject-disjoint split the model routinely predicts classes absent from
# the test rows, so the two macro-F1s were measuring different things.
#
# The two metrics that matter here are balanced accuracy and f1_macro_eligible.
# Plain accuracy is misleading: predicting `Homo sapiens` for everything scores
# 0.851 on sample_host.

def load_eligible_classes(eligibility_path, target: str) -> set:
    """Classes evaluable out-of-project, i.e. present in >=2 BioProjects.

    A class living in a single BioProject falls entirely on one side of any
    group-disjoint split, so it can never be tested out-of-project and must not sit
    in the denominator of a macro average.
    """
    import pandas as pd
    tbl = pd.read_csv(eligibility_path, sep="\t")
    return set(tbl[(tbl.target == target) & tbl.evaluable]["class"])


def classification_metrics(y_true, y_pred, eligible: set | None = None) -> Dict:
    """Accuracy, balanced accuracy, and macro-F1 over seen and eligible classes."""
    seen = sorted(set(y_true))
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro_seen": float(f1_score(y_true, y_pred, labels=seen,
                                        average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=seen,
                                      average="weighted", zero_division=0)),
        "n_classes_seen": len(seen),
    }
    if eligible is not None:
        elig = sorted(c for c in seen if c in eligible)
        out["f1_macro_eligible"] = (float(f1_score(y_true, y_pred, labels=elig,
                                                   average="macro", zero_division=0))
                                    if elig else float("nan"))
        out["n_classes_eligible"] = len(elig)
    return out


def bootstrap_ci(y_true, y_pred, eligible: set, n_boot: int = 1000,
                 seed: int = 42, alpha: float = 0.05) -> Dict:
    """Percentile bootstrap CI for f1_macro_eligible (R1.10).

    The eligible class set is fixed from the full sample, not recomputed per
    resample, so every resample scores the same denominator.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    elig = sorted(c for c in set(y_true.tolist()) if c in eligible)
    if not elig:
        return {"f1_macro_eligible_ci_low": float("nan"),
                "f1_macro_eligible_ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(f1_score(y_true[idx], y_pred[idx], labels=elig,
                             average="macro", zero_division=0))
    return {"f1_macro_eligible_ci_low": float(np.percentile(vals, 100 * alpha / 2)),
            "f1_macro_eligible_ci_high": float(np.percentile(vals, 100 * (1 - alpha / 2)))}
