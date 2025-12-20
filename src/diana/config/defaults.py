"""
Default Configuration for DIANA Multi-Task Training
====================================================

This configuration can be loaded and overridden by user-provided YAML files.
"""

DEFAULT_CONFIG = {
    # Data paths
    "data": {
        "train_matrix": "data/splits/train_matrix.pa.mat",
        "train_metadata": "data/splits/train_metadata.tsv",
        "test_matrix": "data/splits/test_matrix.pa.mat",
        "test_metadata": "data/splits/test_metadata.tsv",
    },
    
    # Output directories
    "output": {
        "base_dir": "results/experiments/multitask",
        "experiment_name": None,  # Auto-generated if None
        "save_checkpoints": True,
        "checkpoint_frequency": 10,  # Save every N epochs
    },
    
    # Training parameters
    "training": {
        "n_folds": 5,
        "n_trials": 50,
        "max_epochs": 200,
        "batch_size": 32,
        "n_inner_splits": 3,
        "random_seed": 42,
        "use_gpu": True,
        "early_stopping_patience": 20,
    },
    
    # Model architecture (default/starting point for Optuna)
    "model": {
        "hidden_dims": [256, 128, 64],
        "activation": "relu",
        "dropout": 0.3,
        "batch_norm": True,
        "task_weights": {
            "sample_type": 1.0,
            "community_type": 1.0,
            "sample_host": 1.0,
            "material": 1.0,
        },
    },
    
    # Optimization
    "optimizer": {
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "optimizer_type": "adam",
    },
    
    # Hyperparameter search space (for Optuna)
    "optuna": {
        "n_layers": [2, 5],
        "hidden_dim_min": 64,
        "hidden_dim_max": 512,
        "learning_rate_min": 1e-5,
        "learning_rate_max": 1e-2,
        "dropout_min": 0.0,
        "dropout_max": 0.5,
        "activations": ["relu", "gelu", "selu"],
        "batch_sizes": [16, 32, 64, 128, 256],
    },
    
    # Logging
    "logging": {
        "level": "INFO",
        "log_to_file": True,
        "log_to_console": True,
        "tensorboard": False,
    },
    
    # SLURM settings
    "slurm": {
        "partition": "gpu",
        "mem": "64G",
        "cpus_per_task": 8,
        "gres": "gpu:1",
        "qos": "gpu",
        "time": "48:00:00",
    },
}
