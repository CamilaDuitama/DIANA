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


def bootstrap_ci(y_true, y_pred, eligible: set, groups, n_boot: int = 1000,
                 seed: int = 42, alpha: float = 0.05) -> Dict:
    """Percentile CIs for f1_macro_eligible and balanced accuracy, resampling GROUPS (R1.10).

    `groups` is required, and is the BioProject of each run. Whole projects are
    drawn with replacement and every run in a drawn project comes with it.

    Resampling individual runs instead treats two runs from one study as two
    independent observations. They are not: they share extraction protocol, library
    prep, platform and sometimes the physical specimen. Measured here, that mistake
    narrows the interval by 2.2-5.1x -- `community_type` reads 0.510-0.582 instead
    of 0.362-0.725 -- which is not a better measurement, just a false one. Passing
    groups is therefore mandatory rather than an option, because this function had
    already been wired into diana-test while resampling runs.

    The eligible class set is fixed from the full sample, not recomputed per
    resample, so every resample scores the same denominator.
    """
    if groups is None:
        raise ValueError(
            "bootstrap_ci requires `groups` (the BioProject of each run). Resampling "
            "runs understates the interval by 2.2-5.1x here; see PROJECT.md rule 6.")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    if groups.shape[0] != y_true.shape[0]:
        raise ValueError(f"groups has {groups.shape[0]} entries for "
                         f"{y_true.shape[0]} predictions")
    elig = sorted(c for c in set(y_true.tolist()) if c in eligible)
    if not elig:
        return {"f1_macro_eligible_ci_low": float("nan"),
                "f1_macro_eligible_ci_high": float("nan"),
                "ci_resampling_unit": "group", "n_groups": 0}

    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    f1s, bals = [], []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in drawn])
        f1s.append(f1_score(y_true[idx], y_pred[idx], labels=elig,
                            average="macro", zero_division=0))
        bals.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"f1_macro_eligible_ci_low": float(np.percentile(f1s, lo)),
            "f1_macro_eligible_ci_high": float(np.percentile(f1s, hi)),
            "balanced_accuracy_ci_low": float(np.percentile(bals, lo)),
            "balanced_accuracy_ci_high": float(np.percentile(bals, hi)),
            "ci_resampling_unit": "group", "n_groups": int(len(uniq))}
