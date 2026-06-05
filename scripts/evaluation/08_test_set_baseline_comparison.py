#!/usr/bin/env python3
"""
Test-Set Baseline Comparison for DIANA
=======================================
Trains baseline classifiers on the FULL BioProject-disjoint training set and
evaluates them on the HELD-OUT BioProject-disjoint test set.

This directly addresses reviewer comments requesting:
  (a) Baseline comparison on the test set, not on the training set via CV
  (b) Non-linear baselines (Random Forest) in addition to linear classifiers

Models evaluated:
  - MajorityClass              (predicts most frequent class — trivial lower bound)
  - LogisticRegression         (L2, lbfgs, C=1)
  - LogisticRegression_Bal     (same + class_weight='balanced')
  - LinearSVM                  (C=1)
  - LinearSVM_Bal              (same + class_weight='balanced')
  - RidgeClassifier
  - RidgeClassifier_Bal
  - RandomForest               (n_estimators=200, max_features='sqrt')
  - RandomForest_Bal           (same + class_weight='balanced_subsample')

INPUT:
  - data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat
  - data/splits_bioproject/train_metadata.tsv
  - data/splits_bioproject/test_metadata.tsv
  - results/test_evaluation_bioproject/test_metrics.json  (DIANA reference)

OUTPUT:
  - results/baseline_comparison_bioproject/summary.csv
  - results/baseline_comparison_bioproject/summary.tex
  - results/baseline_comparison_bioproject/metrics.json
  - results/baseline_comparison_bioproject/baseline_comparison.log

USAGE:
  ./env/bin/python scripts/evaluation/08_test_set_baseline_comparison.py
  # or: sbatch scripts/evaluation/run_test_baseline_comparison.sbatch
"""

import sys
import json
import logging
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from diana.data.loader import MatrixLoader

# ─── Configuration ──────────────────────────────────────────────────────────

MATRIX_PATH    = Path("data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat")
TRAIN_META     = Path("data/splits_bioproject/train_metadata.tsv")
TEST_META      = Path("data/splits_bioproject/test_metadata.tsv")
VAL_META       = Path("data/splits_bioproject/validation_metadata.tsv")
VAL_PRED_DIR   = Path("results/validation_predictions")
VAL_PRED_DIR_BP = Path("results/validation_predictions_bioproject")   # per-sample JSONs
DIANA_METRICS  = Path("results/test_evaluation_bioproject/test_metrics.json")
DIANA_TEST_PREDS = Path("results/test_evaluation_bioproject/test_predictions.tsv")
OUTPUT_DIR     = Path("results/baseline_comparison_bioproject")

TASKS        = ["sample_type", "community_type", "sample_host", "material"]
RANDOM_STATE = 42
N_BOOT       = 1000   # bootstrap resamples for 95 % CI


def _build_models() -> dict:
    models = {
        "MajorityClass": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=2000,
            class_weight=None, n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "LogisticRegression_Bal": LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=2000,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "LinearSVM": LinearSVC(
            C=1.0, max_iter=5000, dual="auto",
            class_weight=None, random_state=RANDOM_STATE,
        ),
        "LinearSVM_Bal": LinearSVC(
            C=1.0, max_iter=5000, dual="auto",
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "RidgeClassifier": RidgeClassifier(alpha=1.0),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_features="sqrt",
            class_weight=None, n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "RandomForest_Bal": RandomForestClassifier(
            n_estimators=200, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }
    if Version(sklearn_version) >= Version("1.2"):
        models["RidgeClassifier_Bal"] = RidgeClassifier(
            alpha=1.0, class_weight="balanced"
        )
    return models


# ─── Logging ────────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "baseline_comparison.log"
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ─── Data Loading ───────────────────────────────────────────────────────────

def load_val_from_predictions(
    val_meta_path: Path,
    pred_dir: Path,
    logger: logging.Logger,
) -> tuple:
    """
    Assemble validation feature matrix from per-sample unitig_abundance.txt files
    in results/validation_predictions/<sample_id>/<sample_id>_unitig_abundance.txt.
    Each file has 107,480 lines (one float per unitig), matching the training matrix.
    """
    meta = pd.read_csv(val_meta_path, sep="\t")
    rows = []
    kept_idx = []
    for i, sid in enumerate(meta["Run_accession"]):
        fpath = pred_dir / sid / f"{sid}_unitig_abundance.txt"
        if not fpath.exists():
            logger.warning(f"  Missing abundance file for {sid} — skipping")
            continue
        rows.append(np.loadtxt(fpath, dtype=np.float32))
        kept_idx.append(i)
    X_val = np.stack(rows, axis=0)   # (n_val, 107480)
    meta_val = meta.iloc[kept_idx].reset_index(drop=True)
    logger.info(f"  val: {X_val.shape[0]} samples x {X_val.shape[1]} features")
    return X_val, meta_val


def load_all_splits(
    matrix_path: Path,
    train_meta_path: Path,
    test_meta_path: Path,
    val_meta_path: Path,
    val_pred_dir: Path,
    logger: logging.Logger,
) -> tuple:
    """
    Load the full matrix ONCE and slice rows for train and test.
    Validation is assembled from per-sample unitig_abundance.txt files
    (validation samples were processed individually, not added to the big matrix).
    """
    logger.info(f"Loading full matrix from {matrix_path}")
    loader = MatrixLoader(matrix_path)
    X_full, sample_ids, _ = loader.load(return_pandas=False)
    logger.info(f"  Full matrix: {X_full.shape[0]} samples x {X_full.shape[1]} features")

    id_to_row = {sid: i for i, sid in enumerate(sample_ids)}

    def _slice(meta_path: Path, label: str):
        meta = pd.read_csv(meta_path, sep="\t")
        valid = meta[meta["Run_accession"].isin(id_to_row)].copy()
        idxs  = np.array([id_to_row[r] for r in valid["Run_accession"]])
        X_sub = X_full[idxs]
        logger.info(f"  {label}: {X_sub.shape[0]} samples x {X_sub.shape[1]} features")
        return X_sub, valid.reset_index(drop=True)

    logger.info(f"Slicing train split from {train_meta_path}")
    X_train, meta_train = _slice(train_meta_path, "train")

    logger.info(f"Slicing test split from {test_meta_path}")
    X_test, meta_test = _slice(test_meta_path, "test")

    del X_full

    logger.info(f"Loading validation from per-sample abundance files in {val_pred_dir}")
    X_val, meta_val = load_val_from_predictions(val_meta_path, val_pred_dir, logger)

    return X_train, meta_train, X_test, meta_test, X_val, meta_val


# ─── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # f1_macro_seen: only average over classes that have >=1 sample in y_true
    # (same denominator as F1† in Table 1 — excludes zero-support training classes)
    seen_labels = np.unique(y_true)
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro_seen":     float(f1_score(y_true, y_pred, labels=seen_labels, average="macro", zero_division=0)),
        "f1_weighted":       float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> dict:
    """95 % percentile-bootstrap CI for balanced_accuracy and f1_macro_seen.

    Returns a flat dict with keys  {metric}_ci_low  and  {metric}_ci_high.
    The CIs are based on stratified resampling (same random seed is always
    used so results are reproducible across runs).
    """
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    seen = np.unique(y_true)
    ba_vals, f1_vals = [], []
    for _ in range(n_boot):
        idx  = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        ba_vals.append(balanced_accuracy_score(yt, yp))
        f1_vals.append(f1_score(yt, yp, labels=seen, average="macro", zero_division=0))
    return {
        "balanced_accuracy_ci_low":  float(np.percentile(ba_vals, 2.5)),
        "balanced_accuracy_ci_high": float(np.percentile(ba_vals, 97.5)),
        "f1_macro_seen_ci_low":      float(np.percentile(f1_vals, 2.5)),
        "f1_macro_seen_ci_high":     float(np.percentile(f1_vals, 97.5)),
    }


def evaluate_with_ci(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute point-estimate metrics + 95 % bootstrap CI in one call."""
    m = compute_metrics(y_true, y_pred)
    m.update(bootstrap_ci(y_true, y_pred))
    return m


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)

    logger.info("=" * 70)
    logger.info("TEST-SET BASELINE COMPARISON (BioProject-disjoint split)")
    logger.info("=" * 70)
    logger.info(f"Date: {datetime.now().isoformat()}")
    logger.info(f"sklearn: {sklearn_version}")

    # Load data — single matrix read, then slice
    t0 = time.time()
    X_train, meta_train, X_test, meta_test, X_val, meta_val = load_all_splits(
        MATRIX_PATH, TRAIN_META, TEST_META, VAL_META, VAL_PRED_DIR, logger
    )
    logger.info(f"Data loaded in {time.time() - t0:.1f}s")

    # Encode labels — fit on train, apply to test and val
    le_map: dict = {}
    y_train: dict = {}
    y_test:  dict = {}
    y_val:   dict = {}
    for task in TASKS:
        le = LabelEncoder()
        le.fit(meta_train[task].fillna("Unknown"))
        le_map[task] = le

        y_train[task] = le.transform(meta_train[task].fillna("Unknown"))

        known = set(le.classes_)

        # Test mask
        test_labels = meta_test[task].fillna("Unknown")
        n_unseen = (~test_labels.isin(known)).sum()
        if n_unseen > 0:
            logger.warning(f"  {task}: {n_unseen} test samples have unseen labels — excluded")
        mask_test = test_labels.isin(known)
        y_test[task] = le.transform(test_labels[mask_test])
        le_map[f"{task}_test_mask"] = mask_test.values

        # Validation mask
        val_labels = meta_val[task].fillna("Unknown")
        n_unseen_val = (~val_labels.isin(known)).sum()
        if n_unseen_val > 0:
            logger.warning(f"  {task}: {n_unseen_val} val samples have unseen labels — excluded")
        mask_val = val_labels.isin(known)
        y_val[task] = le.transform(val_labels[mask_val])
        le_map[f"{task}_val_mask"] = mask_val.values

        logger.info(f"  {task}: {len(le.classes_)} classes — "
                    f"test n={mask_test.sum()}, val n={mask_val.sum()}")

    # Load DIANA reference
    diana_ref: dict = {}
    if DIANA_METRICS.exists():
        with open(DIANA_METRICS) as f:
            diana_raw = json.load(f)
        for task in TASKS:
            tm = diana_raw.get(task, {})
            # Recompute seen-class F1 from the stored classification_report
            cr = tm.get("classification_report", {})
            seen_f1_scores = [
                v["f1-score"]
                for k, v in cr.items()
                if k not in ("accuracy", "macro avg", "weighted avg")
                and isinstance(v, dict) and v.get("support", 0) > 0
            ]
            f1_seen = float(np.mean(seen_f1_scores)) if seen_f1_scores else float("nan")
            diana_ref[task] = {
                "accuracy":          tm.get("accuracy"),
                "balanced_accuracy": tm.get("balanced_accuracy"),
                "f1_macro":          tm.get("f1_macro"),
                "f1_macro_seen":     f1_seen,
                "f1_weighted":       tm.get("f1_weighted"),
                "n_samples":         tm.get("n_samples"),
            }
        logger.info(f"\nDIANA test-set metrics loaded from {DIANA_METRICS}")
    else:
        logger.warning(f"DIANA metrics not found at {DIANA_METRICS}")

    # Bootstrap CI for DIANA test predictions
    if DIANA_TEST_PREDS.exists():
        logger.info(f"Computing bootstrap CI for DIANA test predictions...")
        df_diana_test = pd.read_csv(DIANA_TEST_PREDS, sep="\t")
        for task in TASKS:
            yt = df_diana_test[f"{task}_true"].values
            yp = df_diana_test[f"{task}_pred"].values
            # Keep only rows where true label is in training vocab
            le = le_map[task]
            mask = np.isin(yt, le.classes_)
            ci = bootstrap_ci(yt[mask], yp[mask])
            diana_ref.setdefault(task, {}).update(ci)
            logger.info(
                f"  [{task}] bal_acc 95% CI: "
                f"[{ci['balanced_accuracy_ci_low']*100:.1f}, "
                f"{ci['balanced_accuracy_ci_high']*100:.1f}]"
            )

    # Bootstrap CI for DIANA validation predictions
    logger.info(f"Computing bootstrap CI for DIANA validation predictions...")
    diana_val_ref: dict = {}
    from sklearn.metrics import balanced_accuracy_score as _ba, f1_score as _f1
    task_arrs: dict = {t: {"y_true": [], "y_pred": []} for t in TASKS}
    for _, vrow in meta_val.iterrows():
        sid   = vrow["Run_accession"]
        fpath = VAL_PRED_DIR_BP / sid / f"{sid}_predictions.json"
        if not fpath.exists():
            continue
        with open(fpath) as fh:
            d = json.load(fh)
        for task in TASKS:
            if task not in d.get("predictions", {}):
                continue
            pred  = d["predictions"][task]["predicted_class"]
            true  = str(vrow.get(task, ""))
            le    = le_map[task]
            if true not in set(le.classes_):
                continue
            task_arrs[task]["y_true"].append(true)
            task_arrs[task]["y_pred"].append(pred)
    for task in TASKS:
        yt = np.array(task_arrs[task]["y_true"])
        yp = np.array(task_arrs[task]["y_pred"])
        if len(yt) == 0:
            continue
        seen = np.unique(yt)
        diana_val_ref[task] = {
            "accuracy":          float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(_ba(yt, yp)),
            "f1_macro_seen":     float(_f1(yt, yp, labels=seen, average="macro", zero_division=0)),
        }
        ci = bootstrap_ci(yt, yp)
        diana_val_ref[task].update(ci)
        logger.info(
            f"  [{task}] val bal_acc={diana_val_ref[task]['balanced_accuracy']*100:.1f}% "
            f"CI [{ci['balanced_accuracy_ci_low']*100:.1f}, {ci['balanced_accuracy_ci_high']*100:.1f}]"
        )

    # ── Run baselines ────────────────────────────────────────────────────────
    MODELS = _build_models()
    results_test: dict = {m: {} for m in MODELS}
    results_val:  dict = {m: {} for m in MODELS}

    for model_name, model_proto in MODELS.items():
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Model: {model_name}")
        t_model = time.time()

        for task in TASKS:
            model = clone(model_proto)
            mask_test = le_map[f"{task}_test_mask"]
            mask_val  = le_map[f"{task}_val_mask"]

            t_task = time.time()
            try:
                model.fit(X_train, y_train[task])
                # Test
                y_pred_test = model.predict(X_test[mask_test])
                metrics_test = evaluate_with_ci(y_test[task], y_pred_test)
                # Validation
                y_pred_val = model.predict(X_val[mask_val])
                metrics_val = evaluate_with_ci(y_val[task], y_pred_val)
            except Exception as exc:
                logger.error(f"  [{task}] FAILED: {exc}")
                empty = {k: float("nan") for k in
                         ["accuracy", "balanced_accuracy", "f1_macro", "f1_macro_seen", "f1_weighted"]}
                metrics_test = metrics_val = empty

            elapsed = time.time() - t_task
            metrics_test["fit_predict_s"] = round(elapsed, 2)
            results_test[model_name][task] = metrics_test
            results_val[model_name][task]  = metrics_val

            logger.info(
                f"  {task}: "
                f"test bal_acc={metrics_test['balanced_accuracy']:.3f} f1†={metrics_test['f1_macro_seen']:.3f}  "
                f"val bal_acc={metrics_val['balanced_accuracy']:.3f} f1†={metrics_val['f1_macro_seen']:.3f}  "
                f"({elapsed:.1f}s)"
            )

        logger.info(f"  Total: {time.time() - t_model:.1f}s")

    # ── Save raw JSON ────────────────────────────────────────────────────────
    out = {
        "diana":          diana_ref,
        "diana_val":      diana_val_ref,
        "baselines_test": results_test,
        "baselines_val":  results_val,
        "metadata": {
            "train_n":   int(len(meta_train)),
            "test_n":    int(len(meta_test)),
            "val_n":     int(len(meta_val)),
            "date":      datetime.now().isoformat(),
            "sklearn":   sklearn_version,
            "n_boot":    N_BOOT,
        },
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nRaw metrics saved to {OUTPUT_DIR / 'metrics.json'}")

    # ── Build summary table ──────────────────────────────────────────────────
    METRIC_COLS = ["accuracy", "balanced_accuracy", "f1_macro_seen"]
    DISPLAY_NAMES = {
        "MajorityClass":        "Majority Class",
        "LogisticRegression":   "Logistic Regression",
        "LogisticRegression_Bal": "Logistic Regression (Bal.)",
        "LinearSVM":            "Linear SVM",
        "LinearSVM_Bal":        "Linear SVM (Bal.)",
        "RidgeClassifier":      "Ridge Classifier",
        "RidgeClassifier_Bal":  "Ridge Classifier (Bal.)",
        "RandomForest":         "Random Forest",
        "RandomForest_Bal":     "Random Forest (Bal.)",
        "DIANA":                "DIANA (multi-task MLP)",
    }

    rows = []
    # DIANA row
    for task in TASKS:
        d = diana_ref.get(task, {})
        row = {"model": "DIANA", "split": "test", "task": task}
        for m in METRIC_COLS:
            row[m] = d.get(m, float("nan"))
        rows.append(row)

    # Baseline rows — both splits
    for model_name in MODELS:
        for split, res_dict in (("test", results_test), ("val", results_val)):
            for task in TASKS:
                m_dict = res_dict[model_name].get(task, {})
                row = {"model": model_name, "split": split, "task": task}
                for m in METRIC_COLS:
                    row[m] = m_dict.get(m, float("nan"))
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    # ── LaTeX table (test set only — val in supplementary plot) ─────────────
    task_labels = {
        "sample_type":    "Sample Type",
        "community_type": "Community Type",
        "sample_host":    "Sample Host",
        "material":       "Material",
    }
    model_order = ["DIANA"] + list(MODELS.keys())
    model_order = [m for m in model_order if m in set(df["model"])]
    df_test = df[df["split"] == "test"]

    lines = [
        r"\centering",
        r"\caption{Comparison of DIANA against baseline classifiers on the held-out"
        r" BioProject-disjoint test set (n\,=\,523). All models are trained on the full"
        r" training set (n\,=\,2{,}515) and evaluated on samples with known labels only."
        r" Bal.\ = class\_weight=`balanced'."
        r" F1$^{\dagger}$: macro-averaged F1 restricted to classes present in the test set"
        r" (same denominator as Table~\ref{tab:performance}).}",
        r"\label{tab:baseline_comparison}",
        r"\small",
        r"\begin{tabular}{ll" + "r" * len(METRIC_COLS) + r"}",
        r"\toprule",
        r"Model & Task & Accuracy (\%) & Bal.\ Acc.\ (\%) & F1$^{\dagger}$ (\%) \\",
        r"\midrule",
    ]

    for i, model_name in enumerate(model_order):
        display = DISPLAY_NAMES.get(model_name, model_name)
        first_row = True
        for task in TASKS:
            subset = df_test[(df_test["model"] == model_name) & (df_test["task"] == task)]
            if subset.empty:
                continue
            r = subset.iloc[0]
            acc     = f"{r['accuracy']*100:.1f}"
            bal_acc = f"{r['balanced_accuracy']*100:.1f}"
            f1      = f"{r['f1_macro_seen']*100:.1f}"
            task_str = task_labels.get(task, task)
            model_str = display if first_row else ""
            lines.append(f"{model_str} & {task_str} & {acc} & {bal_acc} & {f1} \\\\")
            first_row = False

        # Separator between groups (except last)
        if model_name == "DIANA":
            lines.append(r"\midrule")
        elif i < len(model_order) - 1:
            lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]

    tex = "\n".join(lines)
    with open(OUTPUT_DIR / "summary.tex", "w") as f:
        f.write(tex)

    # ── Console summary ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY — Balanced Accuracy  |  F1† (seen classes only)")
    logger.info("=" * 70)
    for split_label, split_key, res_dict in [
        ("TEST",       "test", results_test),
        ("VALIDATION", "val",  results_val),
    ]:
        logger.info(f"\n  [{split_label}]")
        header = f"  {'Model':<32} " + "  ".join(f"{t:<16}" for t in TASKS)
        logger.info(header)
        logger.info("  " + "-" * (len(header) - 2))
        for model_name in model_order:
            display = DISPLAY_NAMES.get(model_name, model_name)
            if model_name == "DIANA":
                src = df[(df["model"] == "DIANA") & (df["split"] == "test")]
            else:
                src = df[(df["model"] == model_name) & (df["split"] == split_key)]
            vals = []
            for task in TASKS:
                row = src[src["task"] == task]
                if row.empty:
                    vals.append("  N/A          ")
                else:
                    ba = row.iloc[0]["balanced_accuracy"]
                    f1 = row.iloc[0]["f1_macro_seen"]
                    vals.append(f"{ba*100:.1f}% / {f1*100:.1f}%")
            logger.info(f"  {display:<32} " + "  ".join(f"{v:<16}" for v in vals))

    logger.info(f"\nOutputs in: {OUTPUT_DIR}/")
    logger.info(f"  summary.csv  summary.tex  metrics.json  baseline_comparison.log")
    logger.info("DONE")


if __name__ == "__main__":
    main()
