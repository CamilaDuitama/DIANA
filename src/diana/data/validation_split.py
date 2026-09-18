"""Grouped validation split for early stopping.

Early stopping needs a validation set the model has not effectively already seen.
Runs from one BioProject share protocol, lab and sometimes the specimen, so a random
split puts near-duplicates on both sides: the criterion then rewards recognising the
study rather than generalising to a new one, and it stops late.

Every split in the v9 outer structure is grouped by ``archive_project`` and asserted in
code (``06_create_bioproject_splits_v9.py``, ``08_rebuild_dev_folds_v9.py``). The inner
early-stopping splits were plain random splits until 2026-09-18, which is what this
module fixes. Disjointness is asserted here rather than claimed in a config file,
because a claim nothing checks is the v7 defect this project exists to correct.

Grouping alone is not enough, and two failures on the first attempt show why:

* ``StratifiedGroupKFold`` balances folds by class, not by size. With 68 very uneven
  BioProjects its first fold held **35 of 2,716 runs (1.3 %)** where 10 % was asked
  for, which is too thin to stop on.
* A task can lose its validation set entirely. ``feature`` is labelled on only 561 of
  2,716 training runs, concentrated in a few projects, so a grouped split gave it
  **0 labelled validation runs**, the multi-task model had no criterion for that head,
  and the fit finished without ever saving a checkpoint.
* A class can end up only on the validation side. ``Arabidopsis thaliana`` lives in one
  BioProject, so grouping put all 32 of its runs in validation and none in training.
  This was first blamed for the NaN validation loss, wrongly: the priors are built from
  the **full** training labels, so no class has a zero prior and ``log(0)`` never
  occurs. The NaN came from ``CrossEntropyLoss`` over an all-masked batch (fixed in
  ``training/trainer.py``). Such a class is therefore only preferred against here, not
  rejected, because rejecting it made the four-task arm infeasible at every fraction
  from 0.10 to 0.40.
* The validation side can collapse onto one label. At a 10 % fraction ``sample_host``
  had **1 of 24 classes**, so its macro-F1 was 1.0000 at epoch 0 and no criterion could
  rank epochs; 14 of its 24 classes live in a single BioProject. Requiring a minimum
  number of distinct validation classes fixes this at the same 10 % fraction, giving 4
  classes over 309 runs, so no training data has to be sacrificed.

So the split is chosen by a deterministic search over candidate grouped splits, scored
on how close the validation fraction lands to the target, subject to hard floors on the
number of validation projects and on per-task labelled support.
"""
from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)

GROUP_COL = "archive_project"

#: Fewest whole BioProjects allowed on the validation side. Below this the criterion is
#: one or two studies and early stopping tracks their idiosyncrasies.
MIN_VAL_GROUPS = 3

#: Fewest labelled validation rows a task needs to contribute a loss. A task under this
#: is reported and the split is rejected, rather than training on an undefined criterion.
MIN_TASK_SUPPORT = 10

#: Fewest distinct classes a task needs on the validation side. With whole studies
#: held out the validation set can collapse onto a single label: at a 10 % fraction
#: `sample_host` had **1 of 24 classes**, so its macro-F1 was 1.0000 at epoch 0 and the
#: criterion could not rank epochs at all. 14 of that task's 24 classes live in one
#: BioProject, which is the same fact that removes 22 of 60 classes from the
#: ``f1_macro_eligible`` denominator.
MIN_VAL_CLASSES = 3

#: Candidate grouped splits to score. Deterministic given ``random_state``.
N_CANDIDATES = 300


def _support_counts(
    labels: Mapping[str, np.ndarray], rows: np.ndarray, ignore_index: int
) -> Dict[str, int]:
    """Labelled rows per task within ``rows``.

    Classification tasks mask absent labels with ``ignore_index``; regression tasks use
    NaN. Both conventions are in use in this repo, so both are honoured here.
    """
    counts: Dict[str, int] = {}
    for task, y in labels.items():
        arr = np.asarray(y)[rows]
        if arr.dtype.kind == "f":
            counts[task] = int(np.count_nonzero(~np.isnan(arr)))
        else:
            counts[task] = int(np.count_nonzero(arr != ignore_index))
    return counts


def _orphan_classes(
    labels: Mapping[str, np.ndarray],
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    ignore_index: int,
) -> Dict[str, set]:
    """Classes present in validation but absent from training, per task.

    Only classification tasks can have these; regression targets are continuous, so a
    float array is skipped.
    """
    orphans: Dict[str, set] = {}
    for task, y in labels.items():
        arr = np.asarray(y)
        if arr.dtype.kind == "f":
            continue
        tr = set(np.unique(arr[train_rows])) - {ignore_index}
        va = set(np.unique(arr[val_rows])) - {ignore_index}
        missing = va - tr
        if missing:
            orphans[task] = missing
    return orphans


def _thin_classes(
    labels: Mapping[str, np.ndarray],
    val_rows: np.ndarray,
    ignore_index: int,
    minimum: int,
) -> Dict[str, int]:
    """Tasks whose validation side carries too few distinct classes.

    The floor is capped at the number of classes the task has overall, so a genuinely
    binary task is not rejected for having 2.
    """
    thin: Dict[str, int] = {}
    for task, y in labels.items():
        arr = np.asarray(y)
        if arr.dtype.kind == "f":
            continue
        overall = len(set(np.unique(arr)) - {ignore_index})
        present = len(set(np.unique(arr[val_rows])) - {ignore_index})
        if present < min(minimum, overall):
            thin[task] = present
    return thin


def grouped_validation_split(
    indices: Sequence[int],
    groups: Sequence,
    validation_split: float = 0.1,
    stratify: Optional[Sequence] = None,
    random_state: int = 42,
    task_labels: Optional[Mapping[str, np.ndarray]] = None,
    ignore_index: int = -100,
    min_val_groups: int = MIN_VAL_GROUPS,
    min_task_support: int = MIN_TASK_SUPPORT,
    min_val_classes: int = MIN_VAL_CLASSES,
    n_candidates: int = N_CANDIDATES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split ``indices`` into (train, val) with no BioProject on both sides.

    Args:
        indices: Row positions to split, in the coordinate system of ``groups``.
        groups: Group label per row of ``indices``, aligned element-wise.
        validation_split: Target fraction of *runs* on the validation side.
        stratify: Accepted for call compatibility and used only to break ties by class
            coverage. Class balance cannot be guaranteed alongside grouping, so it is
            never a hard constraint.
        random_state: Seed; the candidate search is deterministic given this.
        task_labels: Optional per-task label arrays, indexed like ``groups``. When
            given, a split is rejected unless every task keeps ``min_task_support``
            labelled validation rows.
        ignore_index: Mask value for absent classification labels.
        min_val_groups: Hard floor on validation projects.
        min_task_support: Hard floor on labelled validation rows per task.
        min_val_classes: Hard floor on distinct classes per classification task on the
            validation side, capped at what the task actually has. Without it the
            validation set can hold a single label and no criterion can rank epochs.
        n_candidates: Candidate splits to score.

    Returns:
        ``(train_idx, val_idx)``, both holding values drawn from ``indices``.

    Raises:
        ValueError: On misaligned inputs, a single group, or when no candidate split
            satisfies the floors.
    """
    idx = np.asarray(indices)
    grp = np.asarray(groups)
    if len(idx) != len(grp):
        raise ValueError(f"indices and groups misaligned: {len(idx)} vs {len(grp)}")
    if not 0.0 < validation_split < 1.0:
        raise ValueError(f"validation_split must be in (0, 1), got {validation_split}")
    n_groups = len(np.unique(grp))
    if n_groups < 2:
        raise ValueError(
            "cannot build a grouped validation split from a single BioProject; "
            "early stopping would have no held-back group"
        )
    if n_groups < min_val_groups + 1:
        raise ValueError(
            f"{n_groups} BioProjects cannot yield {min_val_groups} validation projects "
            "and a non-empty training side"
        )

    strat = None if stratify is None else np.asarray(stratify)
    if strat is not None and len(strat) != len(idx):
        raise ValueError(f"stratify misaligned: {len(strat)} vs {len(idx)}")

    splitter = GroupShuffleSplit(
        n_splits=n_candidates, test_size=validation_split, random_state=random_state
    )
    target = validation_split * len(idx)

    best = None
    best_score = None
    rejected_groups = 0
    rejected_support: Dict[str, int] = {}
    rejected_thin: Dict[str, int] = {}

    for train_pos, val_pos in splitter.split(idx, groups=grp):
        val_groups = set(grp[val_pos])
        if len(val_groups) < min_val_groups:
            rejected_groups += 1
            continue
        if task_labels is not None:
            counts = _support_counts(task_labels, val_pos, ignore_index)
            short = {t: c for t, c in counts.items() if c < min_task_support}
            if short:
                for t, c in short.items():
                    rejected_support[t] = max(rejected_support.get(t, 0), c)
                continue
            # No class may appear in validation but not in training. Otherwise its
            # training prior is 0 and logit adjustment takes log(0), which makes the
            # validation loss NaN and silently prevents any checkpoint being saved.
            # Classes present in validation but not in training are PREFERRED AGAINST,
            # not rejected. Such a row simply cannot be predicted correctly, which
            # lowers the score without breaking anything: the class priors are built
            # from the full training labels, so none is ever zero and logit adjustment
            # never takes log(0). Rejecting them outright made the four-task multi-task
            # arm infeasible at every validation fraction from 0.10 to 0.40, because
            # with four targets almost any held-out project carries some class that
            # appears nowhere else.
            orphan = _orphan_classes(task_labels, train_pos, val_pos, ignore_index)
            n_orphan = sum(len(c) for c in orphan.values())
            thin = _thin_classes(task_labels, val_pos, ignore_index, min_val_classes)
            if thin:
                for t, n in thin.items():
                    rejected_thin[t] = max(rejected_thin.get(t, 0), n)
                continue
        # Primary objective: land on the requested size. Tie-break on class coverage,
        # which is a preference rather than a constraint.
        score = abs(len(val_pos) - target) / max(target, 1.0)
        if task_labels is not None and n_orphan:
            score += 0.05 * n_orphan
        if strat is not None:
            coverage = len(np.unique(strat[val_pos])) / max(len(np.unique(strat)), 1)
            score += 0.1 * (1.0 - coverage)
        if best_score is None or score < best_score:
            best_score, best = score, (train_pos, val_pos)

    if best is None:
        detail = []
        if rejected_groups:
            detail.append(f"{rejected_groups} candidates had < {min_val_groups} validation projects")
        if rejected_support:
            worst = ", ".join(f"{t} (best {c} labelled)" for t, c in sorted(rejected_support.items()))
            detail.append(f"tasks never reached {min_task_support} labelled validation rows: {worst}")
        if rejected_thin:
            names = ", ".join(f"{t} (best {n} classes)" for t, n in sorted(rejected_thin.items()))
            detail.append(
                f"validation side never carried {min_val_classes} distinct classes, so no "
                f"criterion could rank epochs: {names}"
            )
        raise ValueError(
            f"no grouped validation split among {n_candidates} candidates met the floors; "
            + "; ".join(detail)
            + ". Raise validation_split, lower min_task_support, or drop the unsupported task."
        )

    train_pos, val_pos = best
    train_idx, val_idx = idx[train_pos], idx[val_pos]
    train_groups, val_groups = set(grp[train_pos]), set(grp[val_pos])
    shared = train_groups & val_groups
    if shared:
        raise ValueError(
            f"{len(shared)} BioProject(s) on both sides of the validation split: "
            f"{sorted(shared)[:5]}"
        )

    logger.info(
        "grouped validation split on %s: %d train runs / %d val runs (%.1f %%, target %.1f %%); "
        "%d train projects / %d val projects, 0 shared",
        GROUP_COL, len(train_idx), len(val_idx),
        100.0 * len(val_idx) / len(idx), 100.0 * validation_split,
        len(train_groups), len(val_groups),
    )
    if task_labels is not None:
        counts = _support_counts(task_labels, val_pos, ignore_index)
        logger.info("  labelled validation rows per task: %s",
                    ", ".join(f"{t}={c}" for t, c in sorted(counts.items())))
        orphan = _orphan_classes(task_labels, train_pos, val_pos, ignore_index)
        n_orph = sum(len(c) for c in orphan.values())
        if n_orph:
            logger.info("  %d validation class(es) have no training rows and cannot be "
                        "predicted: %s", n_orph,
                        ", ".join(f"{t}={sorted(c)}" for t, c in sorted(orphan.items())))
        else:
            logger.info("  every validation class has training rows: yes")
        classes = {t: len(set(np.unique(np.asarray(y)[val_pos])) - {ignore_index})
                   for t, y in task_labels.items() if np.asarray(y).dtype.kind != "f"}
        logger.info("  distinct validation classes per task: %s",
                    ", ".join(f"{t}={c}" for t, c in sorted(classes.items())))
    return train_idx, val_idx
