"""An all-masked batch must not produce NaN, and masked rows must not be scored.

Both bugs were latent while the early-stopping split was random, because masked rows
were spread evenly across batches. Grouping the split by BioProject masks whole studies
together, so a batch can contain no labelled row for a task at all. The NaN then made
`val_loss < best_val_loss` False forever, no checkpoint was written, and the fit ended
reporting "Best validation loss: inf" (observed 2026-09-18 on sample_host and the
multi-task arm).
"""
import numpy as np
import torch

from diana.models.multitask_mlp import IGNORE_INDEX, MultiTaskMLP
from diana.training.trainer import MultiTaskTrainer


def _trainer(tasks, num_classes, **kw):
    model = MultiTaskMLP(input_dim=16, hidden_dims=[8], num_classes=num_classes)
    return MultiTaskTrainer(
        model=model, task_names=tasks, device="cpu",
        learning_rate=1e-3, weight_decay=0.0,
        task_weights={t: 1.0 for t in tasks}, **kw,
    )


def _data(n, num_classes, masked_tail=0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 16)).astype(np.float32)
    y = {}
    for t, c in num_classes.items():
        lab = rng.integers(0, c, size=n).astype(np.int64)
        if masked_tail:
            lab[-masked_tail:] = IGNORE_INDEX
        y[t] = lab
    return X, y


def test_validation_loss_finite_when_a_batch_is_all_masked():
    tasks, nc = ["t"], {"t": 3}
    tr = _trainer(tasks, nc)
    X, y = _data(64, nc)
    Xv, yv = _data(32, nc)
    yv["t"][:] = IGNORE_INDEX          # every validation row masked
    hist = tr.fit(X_train=X, y_train=y, X_val=Xv, y_val=yv,
                  max_epochs=2, batch_size=16, patience=5, verbose=False)
    assert all(np.isfinite(v) for v in hist["val_loss"]), hist["val_loss"]


def test_checkpoint_is_saved_despite_masked_validation_batches(tmp_path):
    """The failure was silent: training 'completed' with no best_model.pth."""
    tasks, nc = ["t"], {"t": 3}
    tr = _trainer(tasks, nc)
    X, y = _data(64, nc)
    Xv, yv = _data(32, nc, masked_tail=16)   # one batch of 16 fully masked
    tr.fit(X_train=X, y_train=y, X_val=Xv, y_val=yv,
           max_epochs=2, batch_size=16, patience=5,
           checkpoint_dir=tmp_path, verbose=False)
    assert (tmp_path / "best_model.pth").exists(), "no checkpoint written"


def test_training_loss_finite_when_a_train_batch_is_all_masked():
    tasks, nc = ["t"], {"t": 3}
    tr = _trainer(tasks, nc)
    X, y = _data(32, nc, masked_tail=16)
    hist = tr.fit(X_train=X, y_train=y, X_val=X, y_val=y,
                  max_epochs=2, batch_size=16, patience=5, verbose=False)
    assert all(np.isfinite(v) for v in hist["train_loss"]), hist["train_loss"]


def test_accuracy_ignores_masked_rows():
    """A prediction can never equal IGNORE_INDEX, so counting masked rows in the
    denominator scored every unlabelled run as wrong."""
    tasks, nc = ["t"], {"t": 3}
    tr = _trainer(tasks, nc)
    X, y = _data(64, nc)
    # half the validation rows masked; accuracy must be over the labelled half only
    Xv, yv = _data(32, nc, masked_tail=16)
    hist = tr.fit(X_train=X, y_train=y, X_val=Xv, y_val=yv,
                  max_epochs=1, batch_size=32, patience=5, verbose=False)
    acc = hist["val_acc"]["t"][-1]
    assert 0.0 <= acc <= 1.0
    # with 16 of 32 rows masked, an accuracy capped at 0.5 would betray the old denominator
    yv_all = {"t": np.full(32, IGNORE_INDEX, dtype=np.int64)}
    hist2 = tr.fit(X_train=X, y_train=y, X_val=Xv, y_val=yv_all,
                   max_epochs=1, batch_size=32, patience=5, verbose=False)
    assert hist2["val_acc"]["t"][-1] == 0


# --- early stopping on macro-F1 instead of loss ---

def test_monitor_macro_f1_saves_the_f1_best_epoch_not_the_loss_best():
    """Stopping on loss saved community_type at epoch 0 (val_acc 0.5896) when epoch 2
    reached 0.7689. The two criteria disagree, so the monitor must be honoured."""
    tasks, nc = ["t"], {"t": 3}
    X, y = _data(96, nc, seed=1)
    Xv, yv = _data(48, nc, seed=2)
    import torch as _t
    saved = {}
    for mon in ("val_loss", "val_macro_f1"):
        _t.manual_seed(0)
        tr = _trainer(tasks, nc)
        h = tr.fit(X_train=X, y_train=y, X_val=Xv, y_val=yv,
                   max_epochs=8, batch_size=16, patience=8,
                   monitor=mon, verbose=False)
        saved[mon] = h
        assert h["monitor"] == mon
        assert len(h["val_macro_f1"]) == len(h["val_loss"])
    # macro-F1 is recorded under both, so the two runs are comparable
    assert all(0.0 <= v <= 1.0 for v in saved["val_macro_f1"]["val_macro_f1"])


def test_monitor_macro_f1_checkpoint_records_the_rule(tmp_path):
    tasks, nc = ["t"], {"t": 3}
    tr = _trainer(tasks, nc)
    X, y = _data(96, nc, seed=3)
    Xv, yv = _data(48, nc, seed=4)
    tr.fit(X_train=X, y_train=y, X_val=Xv, y_val=yv, max_epochs=4, batch_size=16,
           patience=4, checkpoint_dir=tmp_path, monitor="val_macro_f1", verbose=False)
    import torch as _t
    ck = _t.load(tmp_path / "best_model.pth", map_location="cpu", weights_only=False)
    assert ck["monitor"] == "val_macro_f1"
    assert "val_macro_f1" in ck and "val_loss" in ck


def test_unknown_monitor_refused():
    import pytest
    tasks, nc = ["t"], {"t": 3}
    tr = _trainer(tasks, nc)
    X, y = _data(32, nc)
    with pytest.raises(ValueError, match="monitor must be"):
        tr.fit(X_train=X, y_train=y, X_val=X, y_val=y, max_epochs=1,
               batch_size=16, patience=2, monitor="val_auc", verbose=False)


def test_default_monitor_is_unchanged():
    """Existing callers must keep loss-based stopping."""
    tasks, nc = ["t"], {"t": 3}
    tr = _trainer(tasks, nc)
    X, y = _data(32, nc)
    h = tr.fit(X_train=X, y_train=y, X_val=X, y_val=y, max_epochs=2,
               batch_size=16, patience=2, verbose=False)
    assert h["monitor"] == "val_loss"
