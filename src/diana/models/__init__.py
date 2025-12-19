"""Model architectures for classification."""

from .multitask_mlp import MultiTaskMLP
from .single_task_mlp import SingleTaskMLP
from .sparse_autoencoder import SparseAutoencoder

__all__ = ["MultiTaskMLP", "SingleTaskMLP", "SparseAutoencoder"]
