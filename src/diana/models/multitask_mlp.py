"""Multi-task MLP with shared backbone and task-specific heads."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union


def task_info_from_encoders(encoders_data: Dict[str, dict]) -> Dict[str, int]:
    """Flat {task: n_outputs} mapping from a ``label_encoders.json`` payload.

    Classification tasks carry a ``classes`` list; regression tasks carry the
    normalisation bounds (``min``/``max``/``log_transform``) instead and get a
    single output.
    """
    return {
        task: (len(data["classes"]) if isinstance(data, dict) and "classes" in data else 1)
        for task, data in encoders_data.items()
    }



def denormalize_regression(value: float, bounds: Dict[str, Union[float, bool]]) -> float:
    """Invert the [0,1] normalisation applied to a regression target during training.

    Training (scripts/training/02_train_final_model.py) applies::

        transformed = log1p(raw) if log_transform else raw
        normalized  = clip((transformed - min) / (max - min), 0, 1)

    so ``min``/``max`` in ``label_encoders.json`` live in *transformed* space. This
    inverts it, returning the value in the original units (years BP, degrees).
    """
    vmin = float(bounds["min"])
    vmax = float(bounds["max"])
    transformed = float(value) * (vmax - vmin) + vmin
    if bounds.get("log_transform"):
        return float(np.expm1(transformed))
    return transformed


def split_tasks(
    task_info: Dict[str, int],
    task_types: Optional[Dict[str, str]] = None,
) -> tuple:
    """Split a flat {task: n_outputs} mapping into MultiTaskMLP constructor args.

    Returns ``(num_classes, regression_tasks)`` ready to pass to :class:`MultiTaskMLP`.

    ``task_types`` (the ``task_types`` block of a training config) is authoritative
    when supplied. Without it, a task with a single output is a regression task --
    unambiguous because :class:`MultiTaskMLP` rejects single-class classification.

    Passing the flat mapping straight to ``num_classes=`` is the bug this exists to
    prevent: it rebuilds regression heads as 1-class classification heads, which
    ``load_state_dict`` accepts silently because the final ``Sigmoid`` holds no
    parameters, and every regression prediction then collapses to a constant 1.0.
    """
    if task_types is not None:
        unknown = set(task_types) - set(task_info)
        if unknown:
            raise ValueError(f"task_types names tasks absent from task_info: {sorted(unknown)}")
        regression_tasks = [t for t in task_info if task_types.get(t) == "regression"]
    else:
        regression_tasks = [t for t, n in task_info.items() if n == 1]

    num_classes = {t: n for t, n in task_info.items() if t not in regression_tasks}
    return num_classes, regression_tasks


class MultiTaskMLP(nn.Module):
    """
    Multi-task MLP for simultaneous prediction of classification and regression targets.

    Classification tasks: output n_classes logits → CrossEntropyLoss.
    Regression tasks:     output 1 scalar (sigmoid-bounded [0,1]) → SmoothL1Loss.
                          Continuous targets are normalized to [0,1] before training.

    Architecture:
        Input → Shared Encoder → Task-specific Heads → Outputs

    Used by: scripts/training/01_train_multitask_single_fold.py
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 num_classes: Dict[str, int] = None,
                 regression_tasks: List[str] = None,
                 dropout: float = 0.5,
                 use_batch_norm: bool = True,
                 activation: str = "relu"):
        """
        Initialize multi-task MLP.

        Args:
            input_dim: Number of input features (k-mers)
            hidden_dims: List of hidden layer dimensions
            num_classes: Dict mapping classification task names → number of classes.
                         Regression tasks are NOT included here.
            regression_tasks: List of regression task names (output 1 sigmoid scalar each).
                              Targets must be pre-normalised to [0, 1] before training.
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'gelu', 'leaky_relu')
        """
        super().__init__()

        if num_classes is None:
            num_classes = {
                "sample_type": 2,
                "community_type": 6,
                "sample_host": 12,
                "material": 13
            }
        if regression_tasks is None:
            regression_tasks = []

        # A regression head is a Linear(...,1) followed by a parameterless Sigmoid, so a
        # regression task mistakenly passed in `num_classes` builds a 1-logit
        # classification head that load_state_dict accepts without complaint -- and
        # every prediction for it then collapses to softmax([x]) == 1.0. Refuse the
        # ambiguous construction instead of failing silently. See split_tasks().
        degenerate = sorted(t for t, n in num_classes.items() if n < 2)
        if degenerate:
            raise ValueError(
                f"num_classes assigns fewer than 2 classes to {degenerate}. "
                "Classification heads need >=2 classes; regression tasks belong in "
                "regression_tasks=. Use split_tasks() to derive both arguments."
            )
        overlap = sorted(set(num_classes) & set(regression_tasks))
        if overlap:
            raise ValueError(f"tasks declared as both classification and regression: {overlap}")

        self.num_classes = num_classes
        self.regression_tasks = regression_tasks
        self.targets = list(num_classes.keys()) + regression_tasks
        
        # Select activation
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "leaky_relu":
            act_fn = lambda: nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Shared backbone
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
            
        self.backbone = nn.Sequential(*layers)
        
        # Task-specific heads (with additional hidden layer for better task separation)
        self.heads = nn.ModuleDict()
        for target, n_classes in num_classes.items():
            self.heads[target] = nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
                act_fn(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[-1] // 2, n_classes)
            )
        for target in regression_tasks:
            # Regression head: single sigmoid-bounded output (target normalised to [0, 1])
            self.heads[target] = nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
                act_fn(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[-1] // 2, 1),
                nn.Sigmoid()
            )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary mapping target names to logits
        """
        # Shared features
        features = self.backbone(x)
        
        # Task-specific predictions
        outputs = {}
        for target in self.targets:
            out = self.heads[target](features)
            # Regression heads output (batch, 1) → squeeze to (batch,)
            if target in self.regression_tasks:
                out = out.squeeze(-1)
            outputs[target] = out

        return outputs
        
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions.

        Classification tasks: returns class probabilities (softmax).
        Regression tasks:     returns normalised scalar in [0, 1] (sigmoid already applied).

        Args:
            x: Input tensor

        Returns:
            Dictionary mapping target names to predictions
        """
        self.eval()
        with torch.no_grad():
            raw = self.forward(x)
            out = {}
            for target in self.targets:
                if target in self.regression_tasks:
                    out[target] = raw[target]  # already (batch,) in [0,1]
                else:
                    out[target] = torch.softmax(raw[target], dim=1)
            return out
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss combining losses from all tasks.

    Classification tasks: CrossEntropyLoss (optionally with class weights + label smoothing).
    Regression tasks:     Masked SmoothL1Loss (Huber). NaN values in the target tensor are
                          automatically excluded from the loss so samples with missing
                          continuous metadata do not harm training.
    """
    
    def __init__(
        self,
        task_names: List[str],
        regression_tasks: Optional[List[str]] = None,
        task_weights: Optional[Dict[str, float]] = None,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        label_smoothing: Union[float, Dict[str, float]] = 0.0,
    ):
        """
        Initialize multi-task loss.

        Args:
            task_names: List of all task names (classification + regression).
            regression_tasks: Names of regression tasks (must be a subset of task_names).
                              All other tasks are treated as classification.
            task_weights: Dictionary mapping task names to loss weights (default: equal).
            class_weights: Class weight tensors for classification tasks.
            label_smoothing: Label smoothing for classification tasks.
        """
        super().__init__()

        self.task_names = task_names
        self.regression_tasks = set(regression_tasks or [])
        self.classification_tasks = [t for t in task_names if t not in self.regression_tasks]

        # Default to equal task weights
        if task_weights is None:
            task_weights = {name: 1.0 for name in task_names}
        self.task_weights = task_weights

        # Normalise label_smoothing to per-task dict (classification only)
        if isinstance(label_smoothing, dict):
            ls_per_task = {name: label_smoothing.get(name, 0.0) for name in self.classification_tasks}
        else:
            ls_per_task = {name: float(label_smoothing) for name in self.classification_tasks}

        # Create loss functions
        self.criterions = nn.ModuleDict()
        for task_name in self.classification_tasks:
            weight = class_weights.get(task_name) if class_weights else None
            self.criterions[task_name] = nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=ls_per_task[task_name],
            )
        for task_name in self.regression_tasks:
            # SmoothL1Loss (Huber) per-element so we can apply a NaN mask
            self.criterions[task_name] = nn.SmoothL1Loss(reduction='none')
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> tuple:
        """
        Compute weighted multi-task loss.

        Args:
            predictions: Task predictions.
                         Classification: logits (batch, n_classes).
                         Regression:     scalar (batch,) in [0, 1].
            targets: Task targets.
                     Classification: class indices LongTensor (batch,).
                     Regression:     FloatTensor (batch,) in [0, 1], may contain NaN.

        Returns:
            Tuple of (total_loss, task_losses_dict)
        """
        task_losses = {}
        total_loss = 0.0

        for task_name in self.task_names:
            if task_name in self.regression_tasks:
                pred = predictions[task_name]          # (batch,)
                tgt  = targets[task_name]              # (batch,) float, possibly NaN
                valid = ~torch.isnan(tgt)
                if valid.sum() == 0:
                    # No valid samples: contribute zero loss
                    loss = pred.sum() * 0.0
                else:
                    loss = self.criterions[task_name](pred[valid], tgt[valid]).mean()
            else:
                loss = self.criterions[task_name](predictions[task_name], targets[task_name])

            task_losses[task_name] = loss
            total_loss = total_loss + self.task_weights[task_name] * loss

        return total_loss, task_losses
