"""Classes a network may learn and predict: those present in its fit rows (v19, C1).

The label encoder spans every training label, so a network fitted on part of the
training set (a dev fold, an early-stopping split) has output units for classes it
has no example of. Left in, those units leak the evaluation fold's class list into
the predictions: label smoothing gives them target mass, a clamped prior gives them a
large logit-adjustment offset, and they end up winning the argmax. Restricting the
loss, the argmax and the probabilities to the active classes makes the network behave
as if it had been built for its fit rows only.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from ..models.multitask_mlp import IGNORE_INDEX


def active_masks(labels: Mapping[str, np.ndarray], n_classes: Mapping[str, int]) -> dict[str, np.ndarray]:
    """Per task, a boolean mask of the classes with at least one labelled fit row."""
    out = {}
    for task, n in n_classes.items():
        lab = np.asarray(labels[task])
        lab = lab[lab != IGNORE_INDEX]
        if not lab.size:
            raise ValueError(f"{task}: no labelled fit row")
        if lab.min() < 0 or lab.max() >= n:
            raise ValueError(f"{task}: label index outside 0..{n - 1}")
        mask = np.zeros(n, dtype=bool)
        mask[np.unique(lab)] = True
        out[task] = mask
    return out


def mask_logits(logits: torch.Tensor, active: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Logits with every inactive class set to -inf."""
    act = torch.as_tensor(np.asarray(active), dtype=torch.bool, device=logits.device)
    if act.shape[0] != logits.shape[1]:
        raise ValueError(f"mask has {act.shape[0]} classes, logits {logits.shape[1]}")
    return logits.masked_fill(~act, float("-inf"))


def masked_argmax(logits: torch.Tensor, active: torch.Tensor | np.ndarray) -> torch.Tensor:
    return mask_logits(logits, active).argmax(dim=1)


def masked_softmax(logits: torch.Tensor, active: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Softmax over the active classes; inactive classes get probability exactly 0."""
    return torch.softmax(mask_logits(logits, active), dim=1)
