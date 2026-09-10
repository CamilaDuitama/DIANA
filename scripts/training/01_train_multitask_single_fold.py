
#!/usr/bin/env python3
"""
Multi-Task MLP Hyperparameter Optimization - Single Fold for SLURM Array Jobs
==============================================================================

Performs nested cross-validation with Optuna-based Bayesian hyperparameter optimization
for multi-task learning on ancient DNA samples.

SUPPORTED TASKS (Config-Driven):
---------------------------------
This script handles both classification and regression tasks simultaneously:

Classification (outputs class probabilities, optimized with cross-entropy):
  - Sample-level: sample_type, sample_host
  - Community-level: community_type, material
  - Any custom categorical target

Regression (outputs continuous value in [0,1], optimized with MAE):
  - Temporal: sample_age (log1p-transformed, then normalized)
  - Spatial: latitude, longitude (min-max normalized)
  - Any custom continuous target (auto-scaled to [0,1])

The script automatically:
  - Detects task types from config (task_types dict)
  - Applies appropriate normalization for regression
  - Masks NaN values in regression targets (missing labels)
  - Combines classification + regression losses in multi-task objective

DEPENDENCIES:
-------------
Python packages:
  - numpy, pandas, polars (data manipulation)
  - torch (PyTorch for neural networks)
  - scikit-learn (metrics, cross-validation, label encoding)
  - optuna (Bayesian hyperparameter optimization)

Internal modules:
  - diana.models.multitask_mlp: MultiTaskMLP, MultiTaskLoss
  - diana.data.loader: MatrixLoader (polars-based fast loading)

Input files:
  - K-mer matrix (.frac.mat or .pa.mat): Feature matrix with sample IDs
  - Metadata (.tsv): TSV with Run_accession and task target columns

INPUT DATA (v7 example):
------------------------
- Features: data/matrices/matrix_v7_3190/unitigs.frac.mat (3,190 samples × 78,430 unitigs)
- Metadata: data/splits_v7/train_metadata.tsv (2,838 samples)
- Tasks: 3 classification + 3 regression (configured in JSON)

OUTPUT STRUCTURE:
-----------------
<output_dir>/cv_results/fold_{fold_id}/
├── multitask_fold_{fold_id}_results_{timestamp}.json    # Metrics + hyperparameters
├── best_model.pth                                       # Best model (lowest val loss)
├── final_model.pth                                      # Final model (early stopped)
├── run_config.json                                      # Copy of run config (reproducibility)
└── fold_{fold_id}_training_log_{timestamp}.txt          # Detailed training log

Metrics in results JSON:
  - Classification: accuracy, f1_macro, f1_weighted, balanced_accuracy
  - Regression: mae_normalised, score (1 - MAE, comparable to classification score)

USAGE:
------
# SLURM GPU array job (5-fold CV):
sbatch --array=0-4 scripts/training/run_hyperopt_bioproject_v7.sbatch

# Manual run (single fold):
python scripts/training/01_train_multitask_single_fold.py \\
    --run-config configs/train_config_bioproject_v7.json \\
    --fold_id 0 --use_gpu

WORKFLOW:
---------
1. Load feature matrix and metadata from config paths
2. Create outer CV split (stratified by first 2 classification tasks)
3. Optuna hyperparameter search with inner CV:
   - Search space: hidden_dims, dropout, lr, batch_size, task_weights, etc.
   - Objective: Mean of per-task scores
     - Classification: (balanced_acc + macro_f1) / 2
     - Regression: 1 - MAE (both in [0,1] range)
   - Each trial evaluated on ALL inner folds (no pruning for stability)
4. Train final model on full outer train set with best hyperparameters
5. Evaluate on held-out test fold (never used for model selection)
6. Save model weights, hyperparameters, and metrics

REGRESSION NORMALIZATION:
-------------------------
Regression targets are transformed before training:
  - sample_age: log1p(x), then scaled to [0,1] using [log1p(100), log1p(2M)]
  - latitude: scaled to [0,1] using [-90, 90]
  - longitude: scaled to [0,1] using [-180, 180]

To denormalize predictions after inference:
  - sample_age: exp(pred * (max-min) + min) - 1
  - latitude/longitude: pred * (max-min) + min

See scripts/evaluation/denormalize_predictions.py for utility functions.
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score, classification_report
)

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from diana.models.multitask_mlp import IGNORE_INDEX, MultiTaskMLP, MultiTaskLoss
from diana.data.loader import MatrixLoader
from diana.config import ConfigManager
from diana.utils.config import setup_logging
from diana.utils.checkpointing import CheckpointManager

# Logger will be initialized in main() after args are parsed
logger = None


def load_matrix_data(matrix_path: str, metadata_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load k-mer matrix and metadata using MatrixLoader class.
    
    Args:
        matrix_path: Path to .pa.mat file
        metadata_path: Path to metadata TSV
        
    Returns:
        Tuple of (features_matrix, metadata_df as pandas for sklearn compatibility)
    """
    # Use MatrixLoader class for loading
    loader = MatrixLoader(Path(matrix_path))
    features, metadata_pl = loader.load_with_metadata(
        metadata_path=Path(metadata_path),
        align_to_matrix=True,
        # Refuse to train on fewer runs than the split claims.
        require_all_metadata=True
    )
    
    # Convert metadata to pandas for sklearn compatibility
    metadata = metadata_pl.to_pandas()
    
    logger.info(f"Loaded {features.shape[0]} samples with {features.shape[1]} features")
    
    return features, metadata


def denormalize_regression_predictions(
    preds_normalized: np.ndarray,
    task: str,
) -> Tuple[np.ndarray, str]:
    """
    Denormalize regression predictions back to original units.

    Args:
        preds_normalized: Predictions in [0, 1] range
        task: Task name (sample_age, latitude, longitude)

    Returns:
        Tuple of (denormalized predictions, unit label)

    Normalization was:
      - sample_age: log1p(x) then scaled to [0,1]
      - latitude: linear scale [-90, 90] → [0,1]
      - longitude: linear scale [-180, 180] → [0,1]
    """
    if task == "sample_age":
        vmin, vmax, _ = np.log1p(100.0), np.log1p(2_000_000.0), True
        # Reverse: x = exp(pred * (vmax - vmin) + vmin) - 1
        denormalized = np.exp(preds_normalized * (vmax - vmin) + vmin) - 1
        return denormalized, "years"
    elif task == "latitude":
        vmin, vmax = -90.0, 90.0
        denormalized = preds_normalized * (vmax - vmin) + vmin
        return denormalized, "degrees"
    elif task == "longitude":
        vmin, vmax = -180.0, 180.0
        denormalized = preds_normalized * (vmax - vmin) + vmin
        return denormalized, "degrees"
    else:
        # Unknown task: return as-is
        return preds_normalized, "normalized"


def prepare_labels(
    metadata: pd.DataFrame,
    task_names: List[str],
    task_types: Dict[str, str],
) -> Tuple[Dict[str, np.ndarray], Dict, Dict[str, int]]:
    """
    Prepare labels for all tasks.

    Classification tasks: LabelEncoder → int array.
    Regression tasks:     normalize to [0, 1] (log1p for sample_age), keep NaN as np.nan.

    Returns:
        labels:      {task: np.ndarray} — int64 for classification, float32 (with NaN) for regression
        encoders:    {task: LabelEncoder} for classification;
                     {task: {"min", "max", "log_transform"}} for regression
        num_classes: {task: int} — classification tasks only
    """
    # Fixed normalization bounds for known regression tasks (domain-aware)
    REGRESSION_BOUNDS = {
        "sample_age": (np.log1p(100.0), np.log1p(2_000_000.0), True),   # (min, max, log_transform)
        "latitude":   (-90.0, 90.0, False),
        "longitude":  (-180.0, 180.0, False),
    }

    labels = {}
    encoders = {}
    num_classes = {}

    for task in task_names:
        if task_types.get(task, "classification") == "regression":
            raw = metadata[task].values.astype(float)
            if task in REGRESSION_BOUNDS:
                vmin, vmax, do_log = REGRESSION_BOUNDS[task]
            else:
                do_log = False
                vmin = float(np.nanmin(raw))
                vmax = float(np.nanmax(raw))
            transformed = np.log1p(raw) if do_log else raw.copy()
            normalized = (transformed - vmin) / (vmax - vmin)
            np.clip(normalized, 0.0, 1.0, out=normalized)
            # Restore NaN (masked out in loss)
            normalized[np.isnan(raw)] = np.nan
            labels[task] = normalized.astype(np.float32)
            encoders[task] = {"min": vmin, "max": vmax, "log_transform": do_log}
            logger.info(f"{task} (regression): {(~np.isnan(raw)).sum()} valid / {len(raw)} samples, "
                        f"range [{np.nanmin(raw):.1f}, {np.nanmax(raw):.1f}]")
        else:
            # Absent labels are MASKED, never encoded as a class. Fitting the encoder
            # on raw values turns NaN into a real trainable class -- on v9 that is a
            # 'nan' class for all four heads, including 2,155 rows (78 % of training)
            # for `feature`. diana-test masks those same rows out, so the train and
            # test label spaces would silently disagree and every metric would be
            # computed against a different vocabulary than was trained.
            vals = metadata[task].values
            present = pd.notna(vals)
            encoder = LabelEncoder()
            encoder.fit(vals[present])
            enc = np.full(len(vals), IGNORE_INDEX, dtype=np.int64)
            enc[present] = encoder.transform(vals[present])
            labels[task] = enc
            encoders[task] = encoder
            num_classes[task] = len(encoder.classes_)
            logger.info(f"{task}: {num_classes[task]} classes, {int(present.sum())} labelled, "
                        f"{int((~present).sum())} masked - {list(encoder.classes_[:5])}...")

    return labels, encoders, num_classes


def compute_class_priors(labels_dict: Dict[str, np.ndarray],
                         num_classes: Dict[str, int]) -> Dict[str, "torch.Tensor"]:
    """Per-class training frequencies for logit adjustment (masked rows excluded).

    Pass only the rows the model is fitted on. Passing the whole dataset leaks the
    test half into the prior and, worse, hands a finite prior to classes with no
    training examples in this fold: the loss then gives their logits no gradient
    while `_adjust` subtracts a large constant, so their raw logits drift and win
    the argmax at inference. That is how `feature` scored exactly 0.000 on three
    folds while predicting `thermokarst`, a class absent from those folds' training
    rows.

    Logit adjustment needs the *natural* class distribution. Inverse-frequency
    class weighting and logit adjustment correct for the same thing, so exactly one
    of them should be active -- MultiTaskLoss raises if both are passed.
    """
    priors = {}
    for task_name in num_classes:
        labels = labels_dict[task_name]
        labels = labels[labels != IGNORE_INDEX]
        counts = np.bincount(labels, minlength=num_classes[task_name]).astype(np.float32)
        priors[task_name] = torch.from_numpy(counts)
        logger.info(f"{task_name} priors: {counts.astype(int).tolist()}")
    return priors


def compute_class_weights(labels_dict: Dict[str, np.ndarray], num_classes: Dict[str, int], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Compute class weights for imbalanced classification tasks.
    Regression tasks (not in num_classes) are automatically skipped.
    """
    class_weights = {}

    for task_name in num_classes:  # num_classes only contains classification tasks
        labels = labels_dict[task_name]
        labels = labels[labels != IGNORE_INDEX]   # masked rows must not skew the weights
        n_total_classes = num_classes[task_name]
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        # Initialize weights for all classes with 1.0 (neutral weight for missing classes)
        weights = np.ones(n_total_classes, dtype=np.float32)
        
        # Compute weights for classes present in this split
        for cls_idx, count in zip(unique, counts):
            weights[cls_idx] = total / (len(unique) * count)
        
        class_weights[task_name] = torch.FloatTensor(weights).to(device)
        
        logger.info(f"{task_name} class weights (present classes): {dict(zip(unique, weights[unique]))}")
    
    return class_weights



def train_outer_fold(
    fold_id: int,
    total_folds: int,
    features: np.ndarray,
    metadata: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Train one outer CV fold with hyperparameter optimization.
    
    Args:
        fold_id: Fold identifier (0-indexed)
        total_folds: Total number of folds
        features: Feature matrix
        metadata: Metadata DataFrame
        config: Configuration dictionary
        output_dir: Output directory
        
    Returns:
        Results dictionary
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=== FOLD {fold_id}/{total_folds-1} ===")
    logger.info(f"Output directory: {fold_dir}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
    logger.info(f"Using device: {device}")

    # Task configuration from config
    task_names = config["task_names"]
    task_types = config.get("task_types", {t: "classification" for t in task_names})
    regression_tasks = [t for t in task_names if task_types.get(t) == "regression"]

    # Exactly one imbalance correction may be active. The v9 config asks for logit
    # adjustment; applying it on top of inverse-frequency weights double-corrects and
    # MultiTaskLoss raises. Default to class weights so older configs are unchanged.
    _imb = config.get("class_imbalance", {}) or {}
    LOGIT_TAU = float(_imb.get("logit_adjust_tau", 0.0) or 0.0)
    USE_PRIORS = LOGIT_TAU > 0
    SEARCH_TAU = bool(_imb.get("search_tau", False))
    SEARCH_ONLY = bool(config.get("search_only", False))
    if SEARCH_TAU and not USE_PRIORS:
        raise ValueError("search_tau needs logit adjustment active; set a non-zero "
                         "logit_adjust_tau as the starting value.")
    if SEARCH_TAU:
        logger.info("Class imbalance: logit adjustment, tau SEARCHED in [0.05, 0.8]")
    else:
        logger.info("Class imbalance: %s",
                f"logit adjustment, tau={LOGIT_TAU:.2f} (class weights disabled)"
                if USE_PRIORS else "inverse-frequency class weights")
    classification_tasks = [t for t in task_names if task_types.get(t, "classification") == "classification"]

    # Prepare labels
    labels_dict, encoders, num_classes = prepare_labels(metadata, task_names, task_types)

    # Outer CV split: stratify by the first two (or one) classification tasks
    def _make_stratify_key(idx_array):
        if len(classification_tasks) >= 2:
            t1, t2 = classification_tasks[0], classification_tasks[1]
            return [f"{labels_dict[t1][i]}_{labels_dict[t2][i]}" for i in idx_array]
        elif len(classification_tasks) == 1:
            t1 = classification_tasks[0]
            return [str(labels_dict[t1][i]) for i in idx_array]
        else:
            return list(idx_array)  # No stratification possible

    all_indices = np.arange(len(metadata))
    stratify_key = _make_stratify_key(all_indices)

    # Runs from one BioProject must never be split across folds. They share
    # extraction protocol, library prep, sequencing platform, lab contamination
    # signature and sometimes the physical specimen, so a random split lets the
    # model tune on fold A having already seen fold B's batch -- the confounder
    # R3.2 is about. v7 used a plain StratifiedKFold here, which means its
    # hyperparameters were selected under exactly that leakage.
    #
    # archive_project is the BioProject accession. A run without one becomes its
    # own group (keyed on the run accession) so it is never silently pooled with
    # unrelated runs.
    if "archive_project" in metadata.columns:
        group_col = "archive_project"
    elif "project_name" in metadata.columns:
        group_col = "project_name"
        logger.warning("archive_project absent; grouping CV on project_name instead")
    else:
        raise ValueError(
            "Metadata carries neither archive_project nor project_name, so CV "
            "folds cannot be made BioProject-disjoint. Refusing to run: ungrouped "
            "folds silently reproduce the leakage referee 3 flagged (R3.2)."
        )

    acc_col = "Run_accession" if "Run_accession" in metadata.columns else metadata.columns[0]
    groups = (metadata[group_col].astype("object")
              .fillna(pd.Series("__ungrouped_" + metadata[acc_col].astype(str),
                                index=metadata.index))
              .to_numpy())
    n_groups = len(set(groups))
    logger.info(f"Grouping CV folds on '{group_col}': {n_groups} groups "
                f"over {len(metadata)} runs")
    if n_groups < total_folds:
        raise ValueError(f"{n_groups} groups is fewer than {total_folds} folds; "
                         "cannot build BioProject-disjoint folds.")

    # Prefer the canonical dev folds. The internal StratifiedGroupKFold stratifies on
    # classification_tasks[0:2], so a 4-head config and a 1-head config generate
    # DIFFERENT folds -- measured as 51 vs 64 scored `feature` rows on fold 0, which
    # makes a multi-task vs single-task comparison meaningless. dev_folds.tsv fixes
    # one fold assignment for every arm, and it is already asserted BioProject-
    # disjoint and free of held-out runs when it is written.
    dev_folds_path = config.get("dev_folds_path")
    if SEARCH_ONLY:
        # The outer folds already gave the generalisation estimate. This is the
        # nested-CV endpoint: one search over ALL of train, so the chosen
        # hyperparameters are not selected on whichever fold happened to be easiest
        # (fold 4 scored 0.453 against 0.154-0.238 purely because it had 98.8 %
        # in-vocabulary coverage and a 75.3 % majority class).
        train_idx = np.arange(len(metadata))
        test_idx = np.array([], dtype=int)
    elif dev_folds_path and Path(dev_folds_path).exists():
        dev = pd.read_csv(dev_folds_path, sep="\t")
        fold_of = dict(zip(dev["Run_accession"], dev["fold"]))
        acc_series = metadata[acc_col].astype(str)
        unassigned = set(acc_series) - set(fold_of)
        if unassigned:
            raise ValueError(
                f"{len(unassigned)} run(s) have no entry in {dev_folds_path}, e.g. "
                f"{sorted(unassigned)[:5]}. Regenerate the dev folds for this split.")
        assigned = acc_series.map(fold_of).to_numpy()
        n_dev = len(set(assigned))
        if fold_id >= n_dev:
            raise ValueError(f"fold_id {fold_id} but {dev_folds_path} has {n_dev} folds.")
        test_idx  = np.where(assigned == fold_id)[0]
        train_idx = np.where(assigned != fold_id)[0]
        logger.info("Outer fold %d taken from %s (%d dev folds)",
                    fold_id, dev_folds_path, n_dev)
    else:
        skf_outer = StratifiedGroupKFold(n_splits=total_folds, shuffle=True, random_state=42)
        splits = list(skf_outer.split(features, stratify_key, groups=groups))
        train_idx, test_idx = splits[fold_id]

    def _assert_group_disjoint(a_idx, b_idx, label: str) -> None:
        shared = set(groups[a_idx]) & set(groups[b_idx])
        if shared:
            raise AssertionError(
                f"{label}: {len(shared)} {group_col} value(s) appear on both sides "
                f"of the split, e.g. {sorted(shared)[:5]}. Folds are not "
                f"BioProject-disjoint."
            )

    def _support_mask(idx_array, task: str) -> "torch.Tensor":
        """Classes this fold actually trains on.

        A head keeps one output per class in the global label space, but a
        BioProject-grouped fold may contain no example of some of them. Those units
        get no gradient, so their logits are arbitrary and can win the argmax --
        `feature` predicted `thermokarst`, absent from that fold's training rows, on
        every test sample. Restrict the decision to the classes the model has seen.
        """
        lab = labels_dict[task][idx_array]
        lab = lab[lab != IGNORE_INDEX]
        m = torch.zeros(num_classes[task], dtype=torch.bool)
        if lab.size:
            m[torch.from_numpy(np.unique(lab).astype(np.int64))] = True
        return m

    def _masked_argmax(logits, mask):
        z = logits.clone()
        z[:, ~mask.to(z.device)] = float("-inf")
        return torch.argmax(z, dim=1)

    if SEARCH_ONLY:
        logger.info("search-only: %d training runs, no outer fold held back",
                    len(train_idx))
    else:
        _assert_group_disjoint(train_idx, test_idx, f"outer fold {fold_id}")
        logger.info(f"Train: {len(train_idx)}, Test: {len(test_idx)} "
                    f"(outer fold verified {group_col}-disjoint)")
    
    # Inner CV for hyperparameter optimization
    n_inner_splits = config.get("n_inner_splits", 3)
    # The inner loop selects the hyperparameters, so it must be grouped too --
    # leaking here is what actually biases the chosen configuration.
    skf_inner = StratifiedGroupKFold(n_splits=n_inner_splits, shuffle=True, random_state=42)

    if SEARCH_ONLY and dev_folds_path and Path(dev_folds_path).exists():
        # Use the audited dev folds as the inner CV: already asserted
        # BioProject-disjoint, free of held-out runs, and chosen for class coverage.
        # Every candidate is therefore scored on all 5 folds, easy and hard alike.
        _dev = pd.read_csv(dev_folds_path, sep="\t")
        _fold_of = dict(zip(_dev["Run_accession"], _dev["fold"]))
        _assigned = metadata[acc_col].astype(str).map(_fold_of).to_numpy()
        if pd.isna(_assigned).any():
            raise ValueError(f"some runs have no entry in {dev_folds_path}")
        inner_cv_splits = [(np.where(_assigned != k)[0], np.where(_assigned == k)[0])
                           for k in sorted(set(_assigned))]
        n_inner_splits = len(inner_cv_splits)
        logger.info("inner CV = the %d dev folds from %s",
                    n_inner_splits, dev_folds_path)
    else:
        inner_cv_splits = list(skf_inner.split(
            features[train_idx],
            _make_stratify_key(train_idx),
            groups=groups[train_idx],
        ))
    for i, (a, b) in enumerate(inner_cv_splits):
        _assert_group_disjoint(train_idx[a], train_idx[b], f"inner fold {i}")
    logger.info(f"{n_inner_splits} inner folds verified {group_col}-disjoint")
    
    logger.info(f"Starting Optuna optimization with {config.get('n_trials', 50)} trials...")
    logger.info(f"Each trial will be evaluated on {n_inner_splits} inner CV folds")
    
    def objective_with_cv(trial: optuna.Trial) -> float:
        """
        Objective function that evaluates each trial on ALL inner CV folds.
        Returns the average performance across folds for robust hyperparameter selection.
        """
        # Hyperparameters to optimize
        n_layers = trial.suggest_int("n_layers", 2, 4)
        hidden_dims = []
        
        for i in range(n_layers):
            hidden_dim = trial.suggest_int(f"hidden_dim_{i}", 64, 512, step=64)
            hidden_dims.append(hidden_dim)
        
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        activation = trial.suggest_categorical("activation", ["relu", "gelu", "leaky_relu"])

        # Logit-adjustment strength. tau=1 is the parameter-free default in Menon et
        # al., but at this imbalance it over-corrects badly: a class with n=2 gets a
        # +5.3 logit offset, so the model picks it whenever unsure. Measured on dev
        # fold 2, single-task `feature`: tau=1.00 -> accuracy 0.009, tau=0.25 ->
        # 0.484. It is a hyperparameter here, not a constant, and is reported as one.
        trial_tau = (trial.suggest_float("logit_adjust_tau", 0.05, 0.8)
                     if SEARCH_TAU else LOGIT_TAU)
        
        # Task loss weights (one weight per task, all tasks)
        # With one head, task_weight only rescales the loss, which learning_rate
        # already does -- searching it would spend a dimension on a no-op and
        # handicap the single-task control against the multi-task arm.
        if len(task_names) > 1:
            task_weights = {
                t: trial.suggest_float(f"task_weight_{t}", 0.5, 2.0)
                for t in task_names
            }
        else:
            task_weights = {t: 1.0 for t in task_names}

        # Per-task label smoothing for classification tasks only
        if config.get('no_label_smoothing', False):
            label_smoothing_per_task = {t: 0.0 for t in classification_tasks}
        else:
            label_smoothing_per_task = {
                t: trial.suggest_float(f"ls_{t}", 0.0, 0.15)
                for t in classification_tasks
            }

        # Evaluate on ALL inner CV folds
        fold_scores = []

        for inner_fold_idx, (inner_train, inner_val) in enumerate(inner_cv_splits):
            # Get actual indices
            fold_train_idx = train_idx[inner_train]
            fold_val_idx = train_idx[inner_val]

            # Prepare data as tensors
            X_train = torch.FloatTensor(features[fold_train_idx])
            X_val = torch.FloatTensor(features[fold_val_idx])

            y_train = {}
            y_val = {}
            for task in task_names:
                if task in regression_tasks:
                    y_train[task] = torch.FloatTensor(labels_dict[task][fold_train_idx])
                    y_val[task]   = torch.FloatTensor(labels_dict[task][fold_val_idx])
                else:
                    y_train[task] = torch.LongTensor(labels_dict[task][fold_train_idx])
                    y_val[task]   = torch.LongTensor(labels_dict[task][fold_val_idx])

            # Create DataLoaders with proper batch_size
            train_dataset = TensorDataset(
                X_train,
                *[y_train[task] for task in task_names]
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                # A trailing batch of 1 makes BatchNorm raise in train mode
                # ("Expected more than 1 value per channel"). The search varies
                # batch_size, so this fires whenever n_train % batch_size == 1 and
                # would kill those trials rather than score them.
                drop_last=True
            )
            
            val_dataset = TensorDataset(
                X_val,
                *[y_val[task] for task in task_names]
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            # Create model
            model = MultiTaskMLP(
                input_dim=features.shape[1],
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                regression_tasks=regression_tasks,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                activation=activation
            ).to(device)

            inner_support = {t: _support_mask(fold_train_idx, t) for t in classification_tasks}

            # Class weights only when logit adjustment is off -- computing them under
            # logit adjustment logs weights that are never applied.
            class_weights = None if USE_PRIORS else compute_class_weights(
                {task: labels_dict[task][fold_train_idx] for task in num_classes.keys()},
                num_classes,
                device
            )

            # Loss and optimizer
            criterion = MultiTaskLoss(
                task_names=task_names,
                regression_tasks=regression_tasks,
                task_weights=task_weights,
                class_weights=None if USE_PRIORS else class_weights,
                class_priors=compute_class_priors(
                    {t: labels_dict[t][fold_train_idx] for t in num_classes},
                    num_classes) if USE_PRIORS else None,
                logit_adjust_tau=trial_tau,
                label_smoothing=label_smoothing_per_task,
            ).to(device)
            
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            # Training
            max_epochs = config.get("max_epochs", 100)
            patience = config.get("patience", 15)
            best_val_score = 0
            patience_counter = 0
            
            for epoch in range(max_epochs):
                model.train()

                # Training loop with mini-batches
                for batch_data in train_loader:
                    X_batch = batch_data[0].to(device)
                    y_batch = {task: batch_data[i+1].to(device) for i, task in enumerate(task_names)}

                    # Forward pass
                    outputs = model(X_batch)
                    total_loss, task_losses = criterion(outputs, y_batch)

                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                # Validation every 5 epochs
                if epoch % 5 == 0:
                    model.eval()
                    task_scores = []

                    with torch.no_grad():
                        all_preds  = {task: [] for task in classification_tasks}
                        all_true   = {task: [] for task in classification_tasks}
                        reg_abs_err = {task: [] for task in regression_tasks}

                        for batch_data in val_loader:
                            X_batch = batch_data[0].to(device)
                            y_batch = {task: batch_data[i+1].to(device) for i, task in enumerate(task_names)}

                            val_outputs = model(X_batch)

                            for task in classification_tasks:
                                preds = _masked_argmax(
                                    val_outputs[task],
                                    inner_support[task]).cpu().numpy()
                                true  = y_batch[task].cpu().numpy()
                                keep  = true != IGNORE_INDEX   # absent labels are not predictions to score
                                all_preds[task].extend(preds[keep])
                                all_true[task].extend(true[keep])

                            for task in regression_tasks:
                                pred = val_outputs[task].cpu()
                                tgt  = y_batch[task].cpu()
                                valid = ~torch.isnan(tgt)
                                if valid.sum() > 0:
                                    reg_abs_err[task].extend(
                                        torch.abs(pred[valid] - tgt[valid]).numpy().tolist()
                                    )

                        # Classification: (balanced_acc + macro_f1) / 2
                        for task in classification_tasks:
                            if not all_true[task]:
                                continue   # no labelled rows for this task in this inner fold
                            bal_acc   = balanced_accuracy_score(all_true[task], all_preds[task])
                            macro_f1  = f1_score(all_true[task], all_preds[task], average='macro', zero_division=0)
                            task_scores.append((bal_acc + macro_f1) / 2.0)

                        # Regression: 1 - mean_absolute_error (both in [0,1])
                        for task in regression_tasks:
                            if reg_abs_err[task]:
                                mae = float(np.mean(reg_abs_err[task]))
                                task_scores.append(max(0.0, 1.0 - mae))

                        avg_score = np.mean(task_scores) if task_scores else 0.0

                    # Early stopping
                    if avg_score > best_val_score:
                        best_val_score = avg_score
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        break

            # Store this fold's best score
            fold_scores.append(best_val_score)

        # Return average score across all inner folds
        avg_score = np.mean(fold_scores)
        return avg_score
    
    # A fixed-hyperparameter arm skips the search entirely. Used to compare the
    # multi-task net against a single-task control on identical settings: with no
    # search, any difference between the arms is task sharing rather than a luckier
    # draw from the search space.
    fixed_params = config.get("fixed_hyperparameters")
    if fixed_params:
        best_params = dict(fixed_params)
        n_layers_fixed = int(best_params.get("n_layers", 0))
        for i in range(n_layers_fixed):
            if f"hidden_dim_{i}" not in best_params:
                raise ValueError(
                    f"fixed_hyperparameters says n_layers={n_layers_fixed} but "
                    f"hidden_dim_{i} is missing.")
        for t in task_names:
            best_params.setdefault(f"task_weight_{t}", 1.0)
        logger.info("Fixed hyperparameters, no search: %s", best_params)
    else:
        # Create study and optimize
        # Note: Using NopPruner because each trial is evaluated on multiple folds
        # and pruning mid-trial would discard incomplete fold evaluations
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.NopPruner()  # No pruning for multi-fold CV
        )

        study.optimize(
            objective_with_cv,
            n_trials=config.get('n_trials', 50),
            show_progress_bar=False
        )

        best_params = study.best_params
        # aggregate_cv_results reads task_names off the task_weight_* keys, so they
        # must be present even for a single-task run that never searched them.
        for t in task_names:
            best_params.setdefault(f"task_weight_{t}", 1.0)
        logger.info(f"Best hyperparameters (averaged over {n_inner_splits} folds): {best_params}")
        logger.info(f"Best CV score: {study.best_value:.4f}")
    
    if SEARCH_ONLY:
        out = output_dir / "search_all_train_best_params.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"best_params": best_params,
                   "n_train": int(len(train_idx)),
                   "n_inner_folds": int(n_inner_splits),
                   "note": "single search over all of train; the nested-CV endpoint. "
                           "Feed to 02_train_final_model.py via aggregate_cv_results.py."},
                  open(out, "w"), indent=2)
        logger.info("search-only complete -> %s", out)
        return {"best_params": best_params, "search_only": True}

    # Train final model with best hyperparameters
    # Split train_idx into sub-train and sub-val to avoid test set leakage during early stopping
    logger.info("Training final model with proper train/val split...")
    
    outer_support = {t: _support_mask(train_idx, t) for t in classification_tasks}
    for t in classification_tasks:
        n_sup = int(outer_support[t].sum())
        if n_sup < num_classes[t]:
            logger.info("%s: %d/%d classes have training support in this fold",
                        t, n_sup, num_classes[t])

    # Class weights only when logit adjustment is off (see note above).
    class_weights_full = None if USE_PRIORS else compute_class_weights(
        {task: labels_dict[task][train_idx] for task in num_classes.keys()},
        num_classes,
        device
    )

    sub_train_idx, sub_val_idx = train_test_split(
        train_idx,
        test_size=0.1,
        random_state=42
    )
    
    logger.info(f"Final training split: {len(sub_train_idx)} train, {len(sub_val_idx)} validation, {len(test_idx)} test")
    
    # Build model with best params
    hidden_dims = [best_params[f"hidden_dim_{i}"] for i in range(best_params["n_layers"])]

    model = MultiTaskMLP(
        input_dim=features.shape[1],
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        regression_tasks=regression_tasks,
        dropout=best_params["dropout"],
        use_batch_norm=best_params["use_batch_norm"],
        activation=best_params["activation"]
    ).to(device)

    # Prepare final training data (train/val for early stopping, test for final eval only)
    def _make_label_tensor(task, idx):
        vals = labels_dict[task][idx]
        return torch.FloatTensor(vals) if task in regression_tasks else torch.LongTensor(vals)

    X_train = torch.FloatTensor(features[sub_train_idx])
    X_val   = torch.FloatTensor(features[sub_val_idx])
    X_test  = torch.FloatTensor(features[test_idx])

    y_train = {task: _make_label_tensor(task, sub_train_idx) for task in task_names}
    y_val   = {task: _make_label_tensor(task, sub_val_idx)   for task in task_names}
    y_test  = {task: _make_label_tensor(task, test_idx)      for task in task_names}

    # Create DataLoaders with best batch_size
    train_dataset = TensorDataset(
        X_train,
        *[y_train[task] for task in task_names]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(round(best_params["batch_size"])),
        shuffle=True,
        drop_last=True   # see note above: a trailing batch of 1 breaks BatchNorm
    )

    val_dataset = TensorDataset(
        X_val,
        *[y_val[task] for task in task_names]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(round(best_params["batch_size"])),
        shuffle=False
    )

    test_dataset = TensorDataset(
        X_test,
        *[y_test[task] for task in task_names]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(round(best_params["batch_size"])),
        shuffle=False
    )
    
    # Task weights and label smoothing from best Optuna params
    task_weights = {t: float(best_params.get(f"task_weight_{t}", 1.0)) for t in task_names}
    label_smoothing_per_task = {
        t: best_params.get(f"ls_{t}", 0.0)
        for t in classification_tasks
    }
    criterion = MultiTaskLoss(
        task_names=task_names,
        regression_tasks=regression_tasks,
        task_weights=task_weights,
        class_weights=None if USE_PRIORS else class_weights_full,
        class_priors=compute_class_priors(
            {t: labels_dict[t][train_idx] for t in num_classes},
            num_classes) if USE_PRIORS else None,
        logit_adjust_tau=float(best_params.get("logit_adjust_tau", LOGIT_TAU)),
        label_smoothing=label_smoothing_per_task,
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"]
    )
    
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(
        output_dir=fold_dir,
        save_best=True,
        save_frequency=config.get("checkpoint_freq", 10),
        save_final=True,
        keep_last_n=5
    )
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if config.get("resume_from"):
        try:
            start_epoch = checkpoint_mgr.resume_from_checkpoint(
                model, optimizer, config["resume_from"]
            )
            logger.info(f"Resumed from checkpoint at epoch {start_epoch}")
        except Exception as e:
            logger.warning(f"Could not resume from checkpoint: {e}")
    
    # Train
    max_epochs = config.get("max_epochs", 200)
    patience = config.get("patience", 20)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(start_epoch, max_epochs):
        model.train()
        
        # Training loop with mini-batches
        train_loss_sum = 0.0
        for batch_data in train_loader:
            X_batch = batch_data[0].to(device)
            y_batch = {task: batch_data[i+1].to(device) for i, task in enumerate(task_names)}

            outputs = model(X_batch)
            total_loss, task_losses = criterion(outputs, y_batch)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss_sum += total_loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # Validation (use proper validation set, NOT test set)
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch_data in val_loader:
                X_batch = batch_data[0].to(device)
                y_batch = {task: batch_data[i+1].to(device) for i, task in enumerate(task_names)}

                val_outputs = model(X_batch)
                val_loss, _ = criterion(val_outputs, y_batch)

                val_loss_sum += val_loss.item()
        
        avg_val_loss = val_loss_sum / len(val_loader)
        
        # Save checkpoint
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        checkpoint_mgr.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics={"train_loss": avg_train_loss, "val_loss": avg_val_loss},
            hyperparams=best_params,
            is_best=is_best
        )
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs (patience={patience})")
            break
    
    # Load best model for final evaluation
    best_checkpoint = checkpoint_mgr.load_best_model()
    if best_checkpoint:
        model.load_state_dict(best_checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {best_checkpoint.get('epoch', '?')} (val_loss: {best_checkpoint['metrics']['val_loss']:.4f})")
    
    # Evaluate on test set ONCE (never used for model selection)
    logger.info("Evaluating on held-out test set...")
    model.eval()

    with torch.no_grad():
        all_preds  = {task: [] for task in classification_tasks}
        all_true   = {task: [] for task in classification_tasks}
        reg_preds  = {task: [] for task in regression_tasks}
        reg_true   = {task: [] for task in regression_tasks}

        for batch_data in test_loader:
            X_batch = batch_data[0].to(device)
            y_batch = {task: batch_data[i+1].to(device) for i, task in enumerate(task_names)}

            test_outputs = model(X_batch)

            for task in classification_tasks:
                preds = _masked_argmax(test_outputs[task], outer_support[task]).cpu().numpy()
                true  = y_batch[task].cpu().numpy()
                keep  = true != IGNORE_INDEX   # absent labels are not predictions to score
                all_preds[task].extend(preds[keep])
                all_true[task].extend(true[keep])

            for task in regression_tasks:
                pred  = test_outputs[task].cpu()
                tgt   = y_batch[task].cpu()
                valid = ~torch.isnan(tgt)
                reg_preds[task].extend(pred[valid].numpy().tolist())
                reg_true[task].extend(tgt[valid].numpy().tolist())

        test_metrics = {}
        for task in classification_tasks:
            if not all_true[task]:
                logger.warning(f"{task}: no labelled rows in the held-out set; metrics are undefined")
                test_metrics[task] = {"accuracy": None, "f1_weighted": None,
                                      "f1_macro": None, "balanced_accuracy": None,
                                      "n_scored": 0}
                continue
            test_metrics[task] = {
                "n_scored":          int(len(all_true[task])),
                "accuracy":          float(accuracy_score(all_true[task], all_preds[task])),
                "f1_weighted":       float(f1_score(all_true[task], all_preds[task], average='weighted', zero_division=0)),
                "f1_macro":          float(f1_score(all_true[task], all_preds[task], average='macro', zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(all_true[task], all_preds[task]))
            }
        for task in regression_tasks:
            if reg_true[task]:
                pred_arr = np.array(reg_preds[task])
                true_arr = np.array(reg_true[task])

                # Normalized MAE (for comparison with classification scores)
                mae_norm = float(np.mean(np.abs(pred_arr - true_arr)))

                # Denormalize for interpretable metrics
                pred_denorm, unit = denormalize_regression_predictions(pred_arr, task)
                true_denorm, _ = denormalize_regression_predictions(true_arr, task)
                mae_actual = float(np.mean(np.abs(pred_denorm - true_denorm)))

                # Also compute R² score (coefficient of determination)
                ss_res = np.sum((true_denorm - pred_denorm) ** 2)
                ss_tot = np.sum((true_denorm - np.mean(true_denorm)) ** 2)
                r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

                test_metrics[task] = {
                    "mae_normalised": mae_norm,
                    "mae_actual": mae_actual,
                    "unit": unit,
                    "r2": r2,
                    "score": max(0.0, 1.0 - mae_norm)
                }
            else:
                test_metrics[task] = {"mae_normalised": None, "mae_actual": None, "unit": None, "r2": None, "score": None}

        for task in task_names:
            logger.info(f"{task} test metrics: {test_metrics[task]}")
    
    # Save final model with checkpoint manager
    final_model_path = checkpoint_mgr.save_final_model(
        model=model,
        metrics=test_metrics,
        hyperparams=best_params
    )
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save results
    results = {
        "fold_id": fold_id,
        "total_folds": total_folds,
        "n_outer_train": len(train_idx),
        "n_sub_train": len(sub_train_idx),
        "n_sub_val": len(sub_val_idx),
        "n_test": len(test_idx),
        "num_classes": num_classes,
        "best_params": best_params,
        "test_metrics": test_metrics,
        "best_model_path": str(fold_dir / "best_model.pth"),
        "final_model_path": str(final_model_path),
        "timestamp": timestamp,
        "note": "Test set ONLY used for final evaluation, never for model selection"
    }
    
    results_path = fold_dir / f"multitask_fold_{fold_id}_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    logger.info("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Multi-task MLP hyperparameter optimization — single CV fold.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run-config JSON schema (all settings in one file):
  features_path     : path to .frac.mat feature matrix
  metadata_path     : path to train_metadata.tsv
  output_dir        : base output directory
  task_names        : list of task column names
  task_types        : {task: "classification"|"regression"} — defaults all to classification
  n_folds           : total CV folds (default 5)
  n_trials          : Optuna trials per fold (default 50)
  max_epochs        : max training epochs (default 100)
  patience          : early stopping patience (default 15)
  n_inner_splits    : inner CV folds (default 3)
  random_seed       : RNG seed (default 42)
  no_label_smoothing: bool — fix label smoothing at 0 (default true)

Example:
  python scripts/training/01_train_multitask_single_fold.py \\
      --run-config configs/train_config_bioproject_v7.json \\
      --fold_id $SLURM_ARRAY_TASK_ID --use_gpu
""")

    # Primary: single config file drives everything
    parser.add_argument('--run-config', type=Path, dest='run_config', required=True,
                        help='Path to JSON run-config file. All settings read from here.')

    # SLURM / hardware — must remain CLI args because they vary per job
    parser.add_argument('--fold_id', type=int, required=True,
                        help='Fold ID (0-indexed, set by SLURM_ARRAY_TASK_ID)')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU if available')

    # Optional CLI overrides (rarely needed; prefer editing the JSON)
    parser.add_argument('--n_trials', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--resume_from', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=None,
                        help='override random_seed. Used by the architecture paired '
                             'test, where several seeds per fold separate the effect '
                             'of the architecture from run-to-run noise.')

    args = parser.parse_args()

    # ── Load run-config JSON ────────────────────────────────────────────────
    if not args.run_config.exists():
        print(f"ERROR: run-config not found: {args.run_config}", file=sys.stderr)
        sys.exit(1)

    with open(args.run_config) as fh:
        cfg = json.load(fh)

    # Required fields
    features_path  = cfg['features_path']
    metadata_path  = cfg['metadata_path']
    output_dir     = Path(cfg['output_dir'])
    task_names     = cfg['task_names']
    task_types     = cfg.get('task_types', {t: 'classification' for t in task_names})

    # Optional fields with defaults
    total_folds        = cfg.get('n_folds', 5)
    n_trials           = args.n_trials   or cfg.get('n_trials', 50)
    max_epochs         = args.max_epochs or cfg.get('max_epochs', 100)
    patience           = args.patience   or cfg.get('patience', 15)
    n_inner_splits     = cfg.get('n_inner_splits', 3)
    random_seed        = args.seed if args.seed is not None else cfg.get('random_seed', 42)
    no_label_smoothing = cfg.get('no_label_smoothing', True)
    checkpoint_freq    = cfg.get('checkpoint_freq', 10)

    # ── Logging ─────────────────────────────────────────────────────────────
    fold_dir = output_dir / 'cv_results' / (
        f'fold_{args.fold_id}' if args.seed is None
        else f'fold_{args.fold_id}_seed{args.seed}')
    fold_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        log_file=fold_dir / f'fold_{args.fold_id}_training.log',
        level='INFO',
        log_to_console=True,
        log_to_file=True
    )

    global logger
    logger = logging.getLogger(__name__)

    logger.info('=' * 80)
    logger.info('DIANA Multi-Task Hyperparameter Optimization')
    logger.info('=' * 80)
    logger.info(f'Run config:   {args.run_config}')
    logger.info(f'Features:     {features_path}')
    logger.info(f'Metadata:     {metadata_path}')
    logger.info(f'Output dir:   {output_dir}')
    logger.info(f'Tasks:        {task_names}')
    logger.info(f'Task types:   {task_types}')
    logger.info(f'Fold:         {args.fold_id} / {total_folds}')
    logger.info(f'Trials:       {n_trials}  Max epochs: {max_epochs}  Patience: {patience}')
    logger.info(f'GPU:          {args.use_gpu}  Seed: {random_seed}')

    # Save a copy of the config used (for reproducibility)
    import shutil
    shutil.copy(args.run_config, fold_dir / 'run_config.json')

    # ── Random seeds ────────────────────────────────────────────────────────
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    # ── Load data ────────────────────────────────────────────────────────────
    try:
        logger.info(f'Loading data from {features_path}')
        features, metadata = load_matrix_data(features_path, metadata_path)
        logger.info(f'Loaded {features.shape[0]} samples × {features.shape[1]} features')
    except Exception as e:
        logger.error(f'Failed to load data: {e}', exc_info=True)
        sys.exit(1)

    # ── Train config dict (passed into train_outer_fold) ────────────────────
    #
    # Start from the run config so every key reaches train_outer_fold, then override
    # only what the CLI and defaults resolve. This used to be a hand-maintained
    # whitelist, and anything omitted was read as absent rather than as an error:
    # `class_imbalance` went missing (logit adjustment silently became v7 class
    # weights) and so did `fixed_hyperparameters` (the no-search arm silently ran a
    # full Optuna search). Spreading cfg makes that class of bug impossible.
    train_config = {
        **cfg,
        'task_names':         task_names,
        'task_types':         task_types,
        'n_trials':           n_trials,
        'max_epochs':         max_epochs,
        'patience':           patience,
        'use_gpu':            args.use_gpu,
        'n_inner_splits':     n_inner_splits,
        'checkpoint_freq':    checkpoint_freq,
        'resume_from':        args.resume_from,
        'no_label_smoothing': no_label_smoothing,
        # Read by train_outer_fold to choose logit adjustment over class weights.
        # Omitting it silently reverted v9 to v7's inverse-frequency weighting.
        'class_imbalance':    cfg.get('class_imbalance', {}) or {},
        # Same trap as class_imbalance: read by train_outer_fold, so it must be
        # copied here or the fixed-hyperparameter arm silently runs a full search.
        'fixed_hyperparameters': cfg.get('fixed_hyperparameters') or None,
    }

    # ── Train fold ───────────────────────────────────────────────────────────
    try:
        results = train_outer_fold(
            fold_id=args.fold_id,
            total_folds=total_folds,
            features=features,
            metadata=metadata,
            config=train_config,
            output_dir=output_dir / 'cv_results',
        )

        logger.info('=' * 80)
        logger.info('=== FOLD COMPLETE ===')
        logger.info('=' * 80)
        if results.get("search_only"):
            logger.info("Search-only run: best params written, no test metrics.")
        else:
            logger.info(f'Results: {results["test_metrics"]}')

    except Exception as e:
        logger.error(f'Training failed: {e}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
