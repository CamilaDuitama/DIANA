"""
Patch results/baseline_comparison_bioproject/metrics.json with current DIANA results.

Use this instead of rerunning 08_test_set_baseline_comparison.py when only the
DIANA model has changed (e.g. after v3 retraining).  Baselines are unchanged
so there is no need to retrain them (~30 min) just to update two keys.

Runtime: ~5 seconds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

# ── Paths ─────────────────────────────────────────────────────────────────────
METRICS_JSON    = Path("results/baseline_comparison_bioproject/metrics.json")
TEST_METRICS    = Path("results/test_evaluation_bioproject_v3/test_metrics.json")
TEST_PREDS      = Path("results/test_evaluation_bioproject_v3/test_predictions.tsv")
VAL_PRED_DIR    = Path("results/validation_predictions_bioproject_v3")
VAL_META        = Path("data/splits_bioproject/validation_metadata.tsv")
LABEL_ENCODERS  = Path("results/training_bioproject_v3/label_encoders.json")

TASKS   = ["sample_type", "community_type", "sample_host", "material"]
N_BOOT  = 1_000
RNG     = np.random.default_rng(42)

# ── Helpers ───────────────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = N_BOOT,
) -> dict:
    seen = set(y_true)
    scores_ba, scores_f1 = [], []
    n = len(y_true)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(set(yt)) < 2:
            continue
        scores_ba.append(balanced_accuracy_score(yt, yp))
        labels = [l for l in seen if l in set(yt)]
        scores_f1.append(
            f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
        )
    return {
        "balanced_accuracy_ci_low":  float(np.percentile(scores_ba, 2.5)),
        "balanced_accuracy_ci_high": float(np.percentile(scores_ba, 97.5)),
        "f1_macro_seen_ci_low":      float(np.percentile(scores_f1, 2.5)),
        "f1_macro_seen_ci_high":     float(np.percentile(scores_f1, 97.5)),
    }


# ── Load existing metrics (baselines stay untouched) ─────────────────────────
print(f"Loading {METRICS_JSON} …")
with open(METRICS_JSON) as fh:
    metrics = json.load(fh)

# ── 1. Update diana (test set) ────────────────────────────────────────────────
print("Patching diana (test set) …")
with open(TEST_METRICS) as fh:
    test_raw = json.load(fh)

df_test = pd.read_csv(TEST_PREDS, sep="\t")
diana_new: dict = {}
for task in TASKS:
    tm     = test_raw.get(task, {})
    y_true = df_test[f"{task}_true"].values
    y_pred = df_test[f"{task}_pred"].values
    ci     = bootstrap_ci(y_true, y_pred)
    diana_new[task] = {
        "accuracy":                  tm.get("accuracy"),
        "balanced_accuracy":         tm.get("balanced_accuracy"),
        "f1_macro":                  tm.get("f1_macro"),
        "f1_macro_seen":             tm.get("f1_macro"),
        "f1_weighted":               tm.get("f1_weighted"),
        "n_samples":                 int(tm.get("n_samples", len(y_true))),
        **ci,
    }
    print(
        f"  {task}: bal_acc={diana_new[task]['balanced_accuracy']:.3f} "
        f"CI [{ci['balanced_accuracy_ci_low']:.3f}, {ci['balanced_accuracy_ci_high']:.3f}]"
    )

# ── 2. Update diana_val (validation set) ─────────────────────────────────────
print("Patching diana_val (validation set) …")
with open(LABEL_ENCODERS) as fh:
    encoders = json.load(fh)

val_meta = pd.read_csv(VAL_META, sep="\t")
diana_val_new: dict = {}
for task in TASKS:
    class_names = encoders[task]["classes"]
    rows = []
    for _, row in val_meta.iterrows():
        sid       = row["Run_accession"]
        pred_file = VAL_PRED_DIR / sid / f"{sid}_predictions.json"
        if not pred_file.exists():
            continue
        with open(pred_file) as fh:
            pred = json.load(fh)
        pred_class = pred["predictions"].get(task, {}).get("predicted_class")
        true_label = row.get(task)
        if pd.isna(true_label) or pred_class is None:
            continue
        rows.append({"true": true_label, "pred": pred_class})

    df        = pd.DataFrame(rows)
    y_true    = df["true"].values
    y_pred    = df["pred"].values
    seen_lbls = [l for l in class_names if l in set(y_true)]
    ba        = balanced_accuracy_score(y_true, y_pred)
    f1s       = f1_score(y_true, y_pred, labels=seen_lbls, average="macro", zero_division=0)
    ci        = bootstrap_ci(y_true, y_pred)
    diana_val_new[task] = {
        "balanced_accuracy": float(ba),
        "f1_macro_seen":     float(f1s),
        "n_samples":         len(rows),
        **ci,
    }
    print(f"  {task}: bal_acc={ba:.3f}  f1†={f1s:.3f}  (n={len(rows)})")

# ── 3. Write patched file ─────────────────────────────────────────────────────
metrics["diana"]                    = diana_new
metrics["diana_val"]                = diana_val_new
metrics.setdefault("metadata", {})["model"] = "v3 (per-task label smoothing)"

with open(METRICS_JSON, "w") as fh:
    json.dump(metrics, fh, indent=2)

print(f"\n✓ Patched {METRICS_JSON} with v3 DIANA results (baselines unchanged)")
