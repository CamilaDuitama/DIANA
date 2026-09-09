#!/usr/bin/env python
"""
Train final model using configuration from JSON file.
Called by run_final_training_gpu.sbatch.
"""
import argparse
import torch
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from diana.data.loader import MatrixLoader
from diana.models.multitask_mlp import IGNORE_INDEX, MultiTaskMLP
from diana.training.trainer import MultiTaskTrainer

def main():
    # P1: Use argparse for robust CLI argument parsing
    parser = argparse.ArgumentParser(description='Train final multi-task model')
    parser.add_argument('config_file', type=Path, help='Path to training configuration JSON')
    args = parser.parse_args()
    
    config_file = args.config_file
    
    # Load configuration
    with open(config_file) as f:
        config = json.load(f)

    # Save config to output directory for reproducibility
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_copy_path = output_dir / 'final_training_config.json'
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f'Config saved to: {config_copy_path}')

    # Set random seeds for reproducibility
    random_seed = config.get('random_seed', 42)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    logger.info(f'Random seed set to: {random_seed}')

    # P1: Hyperparameters now come pre-formatted in nested structure
    hyperparams = config['hyperparameters']
    
    logger.info('='*50)
    logger.info('Training final model with best hyperparameters')
    logger.info('='*50)
    logger.info(f'Model params: {json.dumps(hyperparams["model_params"], indent=2)}')
    logger.info(f'Trainer params: {json.dumps(hyperparams["trainer_params"], indent=2)}')
    logger.info(f'Batch size: {hyperparams["batch_size"]}')

    # Load data
    logger.info(f"Loading matrix from {config['features_path']}")
    loader = MatrixLoader(Path(config['features_path']))
    X_all, metadata_pl = loader.load_with_metadata(
        metadata_path=Path(config['metadata_path']),
        align_to_matrix=True,
        # Refuse to train on fewer runs than the split claims.
        require_all_metadata=True
    )
    
    # Convert metadata to pandas for compatibility
    metadata_all = metadata_pl.to_pandas()

    logger.info(f'Full matrix shape: {X_all.shape}')
    logger.info(f'Total samples in matrix: {len(X_all)}')
    
    # Filter to train samples only (exclude test set)
    train_ids_path = Path(config.get('train_ids_path', 'data/splits_v5/train_ids.txt'))
    logger.info(f"Loading train IDs from {train_ids_path}")
    with open(train_ids_path, 'r') as f:
        train_ids = set(line.strip() for line in f if line.strip())
    
    train_mask = metadata_all['Run_accession'].isin(train_ids)
    X_full = X_all[train_mask]
    metadata = metadata_all[train_mask].reset_index(drop=True)
    
    # Verify train_ids match
    matched_ids = set(metadata['Run_accession'].values)
    missing_from_matrix = train_ids - matched_ids
    if missing_from_matrix:
        logger.warning(f'WARNING: {len(missing_from_matrix)} train IDs not found in matrix:')
        logger.warning(f'  First 10: {list(missing_from_matrix)[:10]}')
    logger.info(f'Matched {len(matched_ids)}/{len(train_ids)} train IDs to matrix')
    
    logger.info(f'Training data shape (after filtering to train set): {X_full.shape}')
    logger.info(f'Train samples: {len(X_full)}')

    # Get task info and labels
    task_names      = config['task_names']
    task_types      = config.get('task_types', {t: 'classification' for t in task_names})
    regression_tasks = [t for t in task_names if task_types.get(t) == 'regression']
    task_info  = {}   # classification tasks only: {task: n_classes}
    y_full     = {}   # all tasks: np arrays (int for cls, float for reg)
    label_encoders = {}

    # Fixed normalization bounds for regression tasks (domain-aware)
    REGRESSION_BOUNDS = {
        'sample_age': (np.log1p(100.0), np.log1p(2_000_000.0), True),
        'latitude':   (-90.0, 90.0, False),
        'longitude':  (-180.0, 180.0, False),
    }

    for task_name in task_names:
        if task_name in regression_tasks:
            raw = metadata[task_name].values.astype(float)
            if task_name in REGRESSION_BOUNDS:
                vmin, vmax, do_log = REGRESSION_BOUNDS[task_name]
            else:
                do_log = False
                vmin = float(np.nanmin(raw))
                vmax = float(np.nanmax(raw))
            transformed = np.log1p(raw) if do_log else raw.copy()
            normalized  = (transformed - vmin) / (vmax - vmin)
            np.clip(normalized, 0.0, 1.0, out=normalized)
            normalized[np.isnan(raw)] = np.nan
            y_full[task_name] = normalized.astype(np.float32)
            label_encoders[task_name] = {'min': vmin, 'max': vmax, 'log_transform': do_log}
            logger.info(f"Task '{task_name}' (regression): "
                        f"{(~np.isnan(raw)).sum()} valid / {len(raw)} samples, "
                        f"range [{np.nanmin(raw):.1f}, {np.nanmax(raw):.1f}]")
        else:
            # Absent labels are MASKED, not encoded as a class. Fitting on raw values
            # makes NaN a trainable class -- on v9 that is 2,155 rows (78 % of
            # training) for `feature` alone -- while diana-test masks the same rows
            # out, so the trained and evaluated label spaces would disagree.
            vals = metadata[task_name].values
            present = pd.notna(vals)
            encoder = LabelEncoder()
            encoder.fit(vals[present])
            enc = np.full(len(vals), IGNORE_INDEX, dtype=np.int64)
            enc[present] = encoder.transform(vals[present])
            y_full[task_name] = enc
            label_encoders[task_name] = encoder
            n_classes = len(encoder.classes_)
            task_info[task_name] = n_classes
            logger.info(f"Task '{task_name}': {n_classes} classes, "
                        f"{int(present.sum())} labelled, {int((~present).sum())} masked")

    # Save label encoders BEFORE training (in case training crashes)
    encoders_path = Path(config['output_dir']) / 'label_encoders.json'
    encoders_data = {}
    for task, enc in label_encoders.items():
        if task in regression_tasks:
            encoders_data[task] = enc  # already a plain dict
        else:
            encoders_data[task] = {'classes': enc.classes_.tolist()}
    with open(encoders_path, 'w') as f:
        json.dump(encoders_data, f, indent=2)
    logger.info(f'Label encoders saved to: {encoders_path}')

    # Compute class weights BEFORE split (classification tasks only)
    logger.info('Computing class weights for handling class imbalance...')
    class_weights = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for task_name in task_names:
        if task_name in regression_tasks:
            continue  # no class weights for regression
        _lab = y_full[task_name]
        _lab = _lab[_lab != IGNORE_INDEX]   # masked rows must not skew the weights
        unique, counts = np.unique(_lab, return_counts=True)
        total = len(_lab)
        n_total_classes = task_info[task_name]

        weights = np.ones(n_total_classes, dtype=np.float32)
        for cls_idx, count in zip(unique, counts):
            weights[cls_idx] = total / (len(unique) * count)

        class_weights[task_name] = torch.FloatTensor(weights).to(device)
        logger.info(f"{task_name} - Classes: {len(unique)}, Weights (present): {dict(zip(unique.astype(int), weights[unique]))}")

    # Split into sub-train and validation for early stopping
    validation_split = config.get('validation_split', 0.1)
    logger.info(f'Creating validation split: {validation_split * 100:.0f}% for validation')

    # Stratify by first two classification tasks (or first one, or none for pure regression)
    classification_tasks = [t for t in task_names if t not in regression_tasks]
    stratify_key = None
    if len(classification_tasks) >= 2:
        t1, t2 = classification_tasks[0], classification_tasks[1]
        stratify_key = np.array([
            f"{metadata.iloc[i][t1]}_{metadata.iloc[i][t2]}"
            for i in range(len(metadata))
        ])
        logger.info(f'Stratifying by {t1} + {t2}')
    elif len(classification_tasks) == 1:
        stratify_key = y_full[classification_tasks[0]]
        logger.info(f'Stratifying by {classification_tasks[0]}')
    else:
        logger.info('No classification tasks — no stratification')

    indices = np.arange(len(X_full))
    # stratify_key was computed and logged but never passed, so the log asserted the
    # opposite of what happened. Stratify where it is usable: every class needs at
    # least 2 members, and masked rows (IGNORE_INDEX) must not form a stratum.
    strat = None
    if stratify_key is not None:
        sk = pd.Series(stratify_key).astype(str)
        vc = sk.value_counts()
        sk = sk.where(sk.map(vc) >= 2, "__rare__")
        if sk.nunique() > 1 and sk.value_counts().min() >= 2:
            strat = sk.to_numpy()
        else:
            logger.warning("stratification not possible for this split; proceeding unstratified")
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_split,
        random_state=random_seed,
        stratify=strat,
    )

    X_train, X_val = X_full[train_idx], X_full[val_idx]
    y_train = {task: y_full[task][train_idx] for task in task_names}
    y_val   = {task: y_full[task][val_idx]   for task in task_names}

    logger.info(f'Sub-train samples: {len(X_train)}')
    logger.info(f'Validation samples: {len(X_val)}')

    # P1: Initialize model with clean nested params (use unpacking)
    logger.info(f'Using device: {device}')

    model = MultiTaskMLP(
        input_dim=X_full.shape[1],
        num_classes=task_info,
        regression_tasks=regression_tasks,
        **hyperparams['model_params']
    )

    # P1: Initialize trainer with clean nested params + class weights
    # label_smoothing can be a float (shared) or dict (per-task)
    label_smoothing = config.get(
        'label_smoothing_per_task',
        config.get('label_smoothing', 0.0)
    )
    # One imbalance correction only: the v9 config asks for logit adjustment, which
    # needs no post-hoc tuning. Configs without the block keep inverse-frequency
    # weighting, so v7 behaviour is unchanged.
    _imb = config.get('class_imbalance', {}) or {}
    logit_tau = float(_imb.get('logit_adjust_tau', 0.0) or 0.0)
    class_priors = None
    if logit_tau > 0:
        class_priors = {}
        for task_name in task_info:
            lab = y_full[task_name]
            lab = lab[lab != IGNORE_INDEX]
            counts = np.bincount(lab, minlength=task_info[task_name]).astype(np.float32)
            class_priors[task_name] = torch.from_numpy(counts)
        logger.info('Class imbalance: logit adjustment, tau=%.2f (class weights disabled)', logit_tau)
    else:
        logger.info('Class imbalance: inverse-frequency class weights')

    trainer = MultiTaskTrainer(
        model=model,
        task_names=task_names,
        device=device,
        learning_rate=hyperparams['trainer_params']['learning_rate'],
        weight_decay=hyperparams['trainer_params']['weight_decay'],
        task_weights=hyperparams['trainer_params']['task_weights'],
        class_weights=None if logit_tau > 0 else class_weights,
        class_priors=class_priors,
        logit_adjust_tau=logit_tau,
        label_smoothing=label_smoothing,
        regression_tasks=regression_tasks,
    )

    # Train with early stopping
    logger.info('Starting training with validation-based early stopping...')
    history = trainer.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_epochs=config.get('max_epochs', 200),
        batch_size=hyperparams['batch_size'],
        patience=config.get('early_stopping_patience', config.get('patience', 20)),
        checkpoint_dir=Path(config['output_dir']),
        verbose=True
    )

    # The best model is already saved by trainer.fit() to checkpoint_dir/best_model.pth
    # and loaded back into the trainer. Just verify it exists.
    model_path = Path(config['output_dir']) / 'best_model.pth'
    if model_path.exists():
        logger.info(f'Final model saved to: {model_path}')
    else:
        logger.error(f'ERROR: Model file not found at {model_path}')
        raise FileNotFoundError(f'Training completed but model not saved to {model_path}')

    # P2: Save training history with proper type conversion
    history_path = Path(config['output_dir']) / 'training_history.json'
    
    # Convert NumPy types to Python types for JSON serialization
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    history_serializable = convert_to_python_types(history)
    
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    logger.info(f'Training history saved to: {history_path}')

    logger.info('='*50)
    logger.info('Final training complete!')
    logger.info('='*50)

if __name__ == '__main__':
    main()
