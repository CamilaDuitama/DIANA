"""The inner early-stopping split must not put a BioProject on both sides."""
import numpy as np
import pytest

from diana.data.validation_split import grouped_validation_split


# GroupShuffleSplit's test_size is a fraction of GROUPS, not of samples. A fixture with
# too few projects therefore cannot satisfy min_val_groups and the function refuses, so
# these fixtures use project counts in the range the real data has (v9 train: 68).
def _groups(n_projects=20, per_project=5):
    return np.repeat([f"PRJ{i:03d}" for i in range(n_projects)], per_project)


def test_no_project_on_both_sides():
    groups = _groups()
    idx = np.arange(len(groups))
    train, val = grouped_validation_split(idx, groups, validation_split=0.2)
    assert not set(groups[train]) & set(groups[val])


def test_partition_is_exact():
    groups = _groups()
    idx = np.arange(len(groups))
    train, val = grouped_validation_split(idx, groups, validation_split=0.2)
    assert sorted(np.concatenate([train, val])) == list(idx)
    assert len(set(train) & set(val)) == 0


def test_returns_original_index_values_not_positions():
    """Callers pass a subset (a fold's train_idx), so values must map back."""
    groups = _groups()
    idx = np.arange(100, 100 + len(groups))  # deliberately offset
    train, val = grouped_validation_split(idx, groups, validation_split=0.2)
    assert train.min() >= 100 and val.min() >= 100
    assert set(np.concatenate([train, val])) == set(idx)


def test_stratification_preserved_when_possible():
    groups = _groups(n_projects=40, per_project=5)
    idx = np.arange(len(groups))
    labels = np.array([i % 2 for i in range(len(groups))])
    train, val = grouped_validation_split(idx, groups, 0.2, stratify=labels)
    assert not set(groups[train]) & set(groups[val])
    assert set(np.unique(labels[val])) == {0, 1}


def test_class_confined_to_one_project_is_not_a_hard_failure():
    """A class living in a single project cannot be both stratified and grouped, so
    class balance is a tie-break preference here, never a constraint."""
    groups = _groups(n_projects=30, per_project=5)
    idx = np.arange(len(groups))
    labels = np.zeros(len(groups), dtype=int)
    labels[:5] = 1  # class 1 lives entirely in PRJ000
    train, val = grouped_validation_split(idx, groups, 0.2, stratify=labels)
    assert not set(groups[train]) & set(groups[val])
    assert len(val) > 0


def test_single_project_refuses():
    groups = np.array(["PRJ000"] * 20)
    with pytest.raises(ValueError, match="single BioProject"):
        grouped_validation_split(np.arange(20), groups, 0.2)


def test_misaligned_inputs_refuse():
    with pytest.raises(ValueError, match="misaligned"):
        grouped_validation_split(np.arange(10), _groups(2, 5)[:5], 0.2)


def test_deterministic_for_a_seed():
    groups = _groups()
    idx = np.arange(len(groups))
    a = grouped_validation_split(idx, groups, 0.2, random_state=7)
    b = grouped_validation_split(idx, groups, 0.2, random_state=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_ungrouped_split_would_have_failed():
    """Guard the premise: a plain random split does share projects, so the
    assertion in grouped_validation_split is testing something real."""
    from sklearn.model_selection import train_test_split
    groups = _groups()
    idx = np.arange(len(groups))
    tr, va = train_test_split(idx, test_size=0.2, random_state=42)
    assert set(groups[tr]) & set(groups[va]), "premise broken: random split was already disjoint"


# --- regressions for the two failures the first grouped refit produced (2026-09-18) ---

def test_hits_the_requested_size_with_uneven_projects():
    """StratifiedGroupKFold gave 35 of 2,716 runs where 10% was asked for, because it
    balances folds by class rather than by size. Uneven projects must not do that."""
    sizes = [400, 300, 250, 200, 150] + [40] * 10 + [5] * 50
    groups = np.concatenate([[f"PRJ{i:03d}"] * n for i, n in enumerate(sizes)])
    idx = np.arange(len(groups))
    train, val = grouped_validation_split(idx, groups, validation_split=0.1)
    frac = len(val) / len(idx)
    assert 0.04 < frac < 0.20, f"validation fraction {frac:.3f} far from the 0.10 asked for"
    assert not set(groups[train]) & set(groups[val])


def test_sparse_task_keeps_validation_support():
    """`feature` is labelled on 561 of 2,716 runs and got 0 validation rows, so the
    multi-task fit had no criterion for that head and saved no checkpoint."""
    groups = _groups(n_projects=40, per_project=20)
    idx = np.arange(len(groups))
    dense = np.zeros(len(groups), dtype=np.int64)
    sparse = np.full(len(groups), -100, dtype=np.int64)   # IGNORE_INDEX
    sparse[:200] = 1                                      # labelled in 10 projects only
    train, val = grouped_validation_split(
        idx, groups, 0.1, task_labels={"dense": dense, "sparse": sparse},
        ignore_index=-100, min_task_support=10,
    )
    assert np.count_nonzero(sparse[val] != -100) >= 10
    assert not set(groups[train]) & set(groups[val])


def test_regression_task_nan_mask_counted():
    groups = _groups(n_projects=40, per_project=20)
    idx = np.arange(len(groups))
    y = np.full(len(groups), np.nan, dtype=np.float32)
    y[:300] = 1.5
    train, val = grouped_validation_split(
        idx, groups, 0.1, task_labels={"age": y}, min_task_support=10)
    assert np.count_nonzero(~np.isnan(y[val])) >= 10


def test_minimum_validation_projects_enforced():
    groups = _groups(n_projects=40, per_project=20)
    idx = np.arange(len(groups))
    _, val = grouped_validation_split(idx, groups, 0.1, min_val_groups=3)
    assert len(set(groups[val])) >= 3


def test_refuses_when_a_task_can_never_be_supported():
    """Better to fail loudly than to early-stop on an undefined criterion."""
    groups = _groups(n_projects=60, per_project=20)
    idx = np.arange(len(groups))
    impossible = np.full(len(groups), -100, dtype=np.int64)
    impossible[:3] = 1  # 3 labelled rows in total, floor is 10
    with pytest.raises(ValueError, match="never reached"):
        grouped_validation_split(idx, groups, 0.1,
                                 task_labels={"tiny": impossible},
                                 min_task_support=10)


def test_class_confined_to_one_project_is_not_orphaned_into_validation():
    """`Arabidopsis thaliana` lives in one BioProject, so grouping put all 32 of its
    runs in validation and none in training. Its training prior was then 0 and
    logit adjustment took log(0), making the validation loss NaN at every epoch."""
    groups = _groups(n_projects=40, per_project=20)
    idx = np.arange(len(groups))
    y = np.zeros(len(groups), dtype=np.int64)
    y[:20] = 7          # class 7 exists only in PRJ000
    y[20:40] = 3        # class 3 only in PRJ001
    train, val = grouped_validation_split(
        idx, groups, 0.1, task_labels={"host": y}, ignore_index=-100)
    tr_cls = set(np.unique(y[train])) - {-100}
    va_cls = set(np.unique(y[val])) - {-100}
    assert va_cls <= tr_cls, f"classes only in validation: {va_cls - tr_cls}"


def test_reports_orphan_classes_when_unavoidable():
    """Every project carries a unique class, so no grouped split can avoid orphans."""
    groups = _groups(n_projects=40, per_project=20)
    idx = np.arange(len(groups))
    y = np.repeat(np.arange(40), 20).astype(np.int64)  # one class per project
    with pytest.raises(ValueError, match="no training rows"):
        grouped_validation_split(idx, groups, 0.1,
                                 task_labels={"host": y}, ignore_index=-100)


def test_regression_target_not_treated_as_classes():
    groups = _groups(n_projects=40, per_project=20)
    idx = np.arange(len(groups))
    y = np.random.default_rng(0).normal(size=len(groups)).astype(np.float32)
    train, val = grouped_validation_split(
        idx, groups, 0.1, task_labels={"age": y}, min_task_support=10)
    assert len(val) > 0 and not set(groups[train]) & set(groups[val])
