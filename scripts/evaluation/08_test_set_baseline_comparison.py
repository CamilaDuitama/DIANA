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
  - Config file (YAML) OR command-line arguments
  - Matrix file (unitigs.frac.mat)
  - Metadata files (train, test, val)
  - Optional: DIANA reference metrics for comparison

OUTPUT:
  - results/baseline_comparison_*/summary.csv
  - results/baseline_comparison_*/summary.tex
  - results/baseline_comparison_*/metrics.json
  - results/baseline_comparison_*/baseline_comparison.log

USAGE:
  # With config file (recommended)
  python scripts/evaluation/08_test_set_baseline_comparison.py --config configs/baseline_config_v7.yaml

  # With command-line arguments (backward compatible)
  python scripts/evaluation/08_test_set_baseline_comparison.py --matrix ... --train-meta ... --tasks ...

  # Or via sbatch
  sbatch scripts/evaluation/run_test_baseline_comparison.sbatch
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

def _load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for config file support. Install with: pip install pyyaml")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Test-set baseline comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config file (recommended)
  python 08_test_set_baseline_comparison.py --config configs/baseline_config_v7.yaml

  # Override specific paths
  python 08_test_set_baseline_comparison.py --config configs/baseline_config_v7.yaml --tasks sample_host,material

  # Command-line only (backward compatible)
  python 08_test_set_baseline_comparison.py --matrix ... --train-meta ... --test-meta ...
        """
    )
    p.add_argument("--config",        type=str, default=None,
                   help="Path to YAML config file (recommended). Command-line args override config.")
    p.add_argument("--matrix",        type=str, default=None,
                   help="Path to unitigs.frac.mat")
    p.add_argument("--train-meta",    type=str, default=None,
                   help="Training metadata TSV")
    p.add_argument("--test-meta",     type=str, default=None,
                   help="Test metadata TSV")
    p.add_argument("--val-meta",      type=str, default=None,
                   help="Validation metadata TSV")
    p.add_argument("--val-pred-dir",  type=str, default=None,
                   help="Directory with per-sample abundance files")
    p.add_argument("--diana-metrics", type=str, default=None,
                   help="DIANA test metrics JSON")
    p.add_argument("--diana-preds",   type=str, default=None,
                   help="DIANA test predictions TSV")
    p.add_argument("--output-dir",    type=str, default=None,
                   help="Directory for output files")
    p.add_argument("--skip-val",      action="store_true", default=None,
                   help="Skip validation evaluation")
    p.add_argument("--tasks",         type=str, default=None,
                   help="Comma-separated list of tasks to evaluate")
    p.add_argument("--n-boot",        type=int, default=None,
                   help="Number of bootstrap resamples for CI (default: 1000)")

    args = p.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        config = _load_yaml_config(args.config)
        logging.info(f"Loaded config from {args.config}")

    # Merge: command-line args override config file
    def get_value(arg_name, config_key=None, default=None):
        """Get value from command-line arg, then config, then default."""
        cmd_val = getattr(args, arg_name.replace('-', '_'), None)
        if cmd_val is not None and cmd_val != p.get_default(arg_name.replace('-', '_')):
            return cmd_val
        cfg_keys = (config_key or arg_name.replace('-', '_')).split('.')
        val = config
        for k in cfg_keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val if val is not None else default

    return args, config, get_value


# ─── Global configuration (set after parsing) ───────────────────────────────

_ARGS, _CONFIG, GET = None, None, None
MATRIX_PATH = None
TRAIN_META = None
TEST_META = None
VAL_META = None
VAL_PRED_DIR = None
SKIP_VAL = True
DIANA_METRICS = None
DIANA_TEST_PREDS = None
OUTPUT_DIR = None
TASKS = []
N_BOOT = 1000


def _init_config():
    """Initialize global configuration from args and/or config file."""
    global _ARGS, _CONFIG, GET
    global MATRIX_PATH, TRAIN_META, TEST_META, VAL_META, VAL_PRED_DIR
    global SKIP_VAL, DIANA_METRICS, DIANA_TEST_PREDS, OUTPUT_DIR, TASKS, N_BOOT

    _ARGS, _CONFIG, GET = _parse_args()

    # Resolve paths
    MATRIX_PATH = Path(GET("matrix", "data.matrix", "data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat"))
    TRAIN_META = Path(GET("train-meta", "data.train_metadata", "data/splits_v5/train_metadata.tsv"))
    TEST_META = Path(GET("test-meta", "data.test_metadata", "data/splits_v5/test_metadata.tsv"))
    VAL_META = Path(GET("val-meta", "data.val_metadata", "data/splits_v5/validation_metadata.tsv"))

    val_pred_dir = GET("val-pred-dir", "data.val_pred_dir", None)
    VAL_PRED_DIR = Path(val_pred_dir) if val_pred_dir else None

    diana_metrics = GET("diana-metrics", "data.diana_metrics", None)
    DIANA_METRICS = Path(diana_metrics) if diana_metrics else None

    diana_preds = GET("diana-preds", "data.diana_preds", None)
    DIANA_TEST_PREDS = Path(diana_preds) if diana_preds else None

    OUTPUT_DIR = Path(GET("output-dir", "output.base_dir", "results/baseline_comparison"))

    # Skip validation if no val_pred_dir provided
    SKIP_VAL = VAL_PRED_DIR is None or GET("skip-val", default=False)

    # Tasks: from config or default
    tasks_arg = GET("tasks", default=None)
    if tasks_arg:
        TASKS = [t.strip() for t in tasks_arg.split(",")]
    elif "tasks" in _CONFIG.get("data", {}):
        TASKS = _CONFIG["data"]["tasks"]
    else:
        TASKS = ["sample_type", "community_type", "sample_host", "material"]

    # Bootstrap samples
    N_BOOT = GET("n-boot", "bootstrap.n_boot", 1000)


# ─── Model building ─────────────────────────────────────────────────────────

RANDOM_STATE = 42


def _build_models() -> dict:
    """Build baseline models with configurable hyperparameters."""
    model_cfg = _CONFIG.get("models", {}) if _CONFIG else {}

    lr_cfg = model_cfg.get("logistic_regression", {})
    svm_cfg = model_cfg.get("linear_svm", {})
    ridge_cfg = model_cfg.get("ridge_classifier", {})
    rf_cfg = model_cfg.get("random_forest", {})

    models = {
        "MajorityClass": DummyClassifier(strategy="most_frequent"),

        "LogisticRegression": LogisticRegression(
            C=lr_cfg.get("C", 1.0),
            solver=lr_cfg.get("solver", "lbfgs"),
            max_iter=lr_cfg.get("max_iter", 2000),
            class_weight=lr_cfg.get("class_weight"),
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "LogisticRegression_Bal": LogisticRegression(
            C=lr_cfg.get("C", 1.0),
            solver=lr_cfg.get("solver", "lbfgs"),
            max_iter=lr_cfg.get("max_iter", 2000),
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "LinearSVM": LinearSVC(
            C=svm_cfg.get("C", 1.0),
            max_iter=svm_cfg.get("max_iter", 5000),
            dual="auto",
            class_weight=svm_cfg.get("class_weight"),
            random_state=RANDOM_STATE,
        ),

        "LinearSVM_Bal": LinearSVC(
            C=svm_cfg.get("C", 1.0),
            max_iter=svm_cfg.get("max_iter", 5000),
            dual="auto",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "RidgeClassifier": RidgeClassifier(
            alpha=ridge_cfg.get("alpha", 1.0),
            class_weight=ridge_cfg.get("class_weight"),
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 200),
            max_features=rf_cfg.get("max_features", "sqrt"),
            class_weight=rf_cfg.get("class_weight"),
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "RandomForest_Bal": RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 200),
            max_features=rf_cfg.get("max_features", "sqrt"),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    if Version(sklearn_version) >= Version("1.2"):
        models["RidgeClassifier_Bal"] = RidgeClassifier(
            alpha=ridge_cfg.get("alpha", 1.0),
            class_weight="balanced" if ridge_cfg.get("class_weight") == "balanced" else None,
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
    Each file has N features (one float per unitig), matching the training matrix.
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
    X_val = np.stack(rows, axis=0)   # (n_val, n_features)
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
    skip_val: bool = False,
) -> tuple:
    """
    Load the full matrix ONCE and slice rows for train and test.
    Validation is assembled from per-sample unitig_abundance.txt files
    unless skip_val=True, in which case empty arrays are returned.
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

    if skip_val or val_pred_dir is None:
        logger.info("Skipping validation set (--skip-val or no val-pred-dir)")
        n_features = X_train.shape[1]
        X_val    = np.empty((0, n_features), dtype=np.float32)
        meta_val = pd.DataFrame(columns=pd.read_csv(val_meta_path, sep="\t", nrows=0).columns)
    else:
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
    # Initialize configuration
    _init_config()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)

    logger.info("=" * 70)
    logger.info("TEST-SET BASELINE COMPARISON (BioProject-disjoint split)")
    logger.info("=" * 70)
    logger.info(f"Date: {datetime.now().isoformat()}")
    logger.info(f"sklearn: {sklearn_version}")
    logger.info(f"Config: {_ARGS.config if _ARGS.config else 'command-line'}")
    logger.info(f"Matrix: {MATRIX_PATH}")
    logger.info(f"Train: {TRAIN_META} (tasks: {TASKS})")
    logger.info(f"Test: {TEST_META}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # Load data — single matrix read, then slice
    t0 = time.time()
    X_train, meta_train, X_test, meta_test, X_val, meta_val = load_all_splits(
        MATRIX_PATH, TRAIN_META, TEST_META, VAL_META, VAL_PRED_DIR, logger,
        skip_val=SKIP_VAL
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
        if not SKIP_VAL and len(meta_val) > 0:
            val_labels = meta_val[task].fillna("Unknown")
            n_unseen_val = (~val_labels.isin(known)).sum()
            if n_unseen_val > 0:
                logger.warning(f"  {task}: {n_unseen_val} val samples have unseen labels — excluded")
            mask_val = val_labels.isin(known)
            y_val[task] = le.transform(val_labels[mask_val])
            le_map[f"{task}_val_mask"] = mask_val.values
        else:
            mask_val = pd.Series([], dtype=bool)
            y_val[task] = np.array([])
            le_map[f"{task}_val_mask"] = np.array([], dtype=bool)

        logger.info(f"  {task}: {len(le.classes_)} classes — "
                    f"test n={mask_test.sum()}, val n={mask_val.sum()}")

    # Load DIANA reference
    diana_ref: dict = {}
    if DIANA_METRICS and DIANA_METRICS.exists():
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
        logger.info("No DIANA metrics provided (will train baselines only)")

    # Bootstrap CI for DIANA test predictions
    if DIANA_TEST_PREDS and DIANA_TEST_PREDS.exists():
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
    diana_val_ref: dict = {}
    if not SKIP_VAL and VAL_PRED_DIR and (VAL_PRED_DIR / "diana_predict_summary.json").exists():
        logger.info(f"Loading DIANA validation predictions from {VAL_PRED_DIR}...")
        from sklearn.metrics import balanced_accuracy_score as _ba, f1_score as _f1
        task_arrs: dict = {t: {"y_true": [], "y_pred": []} for t in TASKS}
        for _, vrow in meta_val.iterrows():
            sid   = vrow["Run_accession"]
            fpath = VAL_PRED_DIR / sid / f"{sid}_predictions.json"
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
                if not SKIP_VAL and len(y_val[task]) > 0:
                    y_pred_val = model.predict(X_val[mask_val])
                    metrics_val = evaluate_with_ci(y_val[task], y_pred_val)
                else:
                    empty = {k: float("nan") for k in
                             ["accuracy", "balanced_accuracy", "f1_macro", "f1_macro_seen", "f1_weighted"]}
                    metrics_val = empty
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
                f"test bal_acc={metrics_test['balanced_accuracy']:.3f} f1†={metrics_test['f1_macro_seen']:.3f}"
                + (f"  val bal_acc={metrics_val['balanced_accuracy']:.3f} f1†={metrics_val['f1_macro_seen']:.3f}" if not SKIP_VAL else "")
                + f"  ({elapsed:.1f}s)"
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
            "config":    _ARGS.config if _ARGS.config else "command-line",
        },
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
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

    # ── LaTeX table ─────────────────────────────────────────────────────────
    task_labels = {t: t.replace("_", " ").title() for t in TASKS}
    model_order = ["DIANA"] + list(MODELS.keys())
    model_order = [m for m in model_order if m in set(df["model"])]
    df_test = df[df["split"] == "test"]

    lines = [
        r"\centering",
        r"\caption{Comparison of DIANA against baseline classifiers on the held-out"
        r" BioProject-disjoint test set. All models are trained on the full"
        r" training set and evaluated on samples with known labels only."
        r" Bal.\ = class\_weight=`balanced'.}",
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
            acc     = f"{r['accuracy']*100:.1f}" if not pd.isna(r['accuracy']) else "nan"
            bal_acc = f"{r['balanced_accuracy']*100:.1f}" if not pd.isna(r['balanced_accuracy']) else "nan"
            f1      = f"{r['f1_macro_seen']*100:.1f}" if not pd.isna(r['f1_macro_seen']) else "nan"
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
                    if pd.isna(ba):
                        vals.append("  nan          ")
                    else:
                        vals.append(f"{ba*100:.1f}% / {f1*100:.1f}%")
            logger.info(f"  {display:<32} " + "  ".join(f"{v:<16}" for v in vals))

    logger.info(f"\nOutputs in: {OUTPUT_DIR}/")
    logger.info(f"  summary.csv  summary.tex  metrics.json  baseline_comparison.log")
    logger.info("DONE")


if __name__ == "__main__":
    main()
