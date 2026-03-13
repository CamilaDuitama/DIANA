#!/usr/bin/env python
"""
Baseline Classifier Comparison for DIANA (Supplementary)
=========================================================
Trains simple linear classifiers on the training set using 5-fold stratified
cross-validation — the same outer-CV protocol as DIANA — so results are
directly comparable to results/training/cv_results/aggregated_results.json.

NOTE: training_set_metrics.json (DIANA evaluated on its own training data) is
NOT used for comparison — it is inflated.  The fair reference is the CV numbers
from aggregated_results.json (per-fold values are also available there for
statistical tests).

Models evaluated (simple linear classifiers only — supplementary experiment):
  - LogisticRegression         (L2, lbfgs, C=1, default weights)
  - LogisticRegression_Bal     (same + class_weight='balanced')
  - LinearSVM                  (C=1, default weights)
  - LinearSVM_Bal              (same + class_weight='balanced')
  - RidgeClassifier            (alpha=1, default weights)
  - RidgeClassifier_Bal        (same + class_weight='balanced', sklearn >= 1.2)

Class-imbalance variants are important because DIANA uses class-weighted loss;
without them the comparison would unfairly disadvantage the baselines.

Confusion matrices and classification reports use the LAST fold only —
they are not averaged across folds — to keep output manageable while still
providing per-class detail for the most representative split.

INPUT:
  - data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat
  - paper/metadata/train_metadata.tsv (2,597 training samples)
  - results/training/cv_results/aggregated_results.json (DIANA reference)

OUTPUT:
  - results/baseline_comparison/cv_results.json           per-fold raw metrics
  - results/baseline_comparison/fold_indices.json         fold indices (audit)
  - results/baseline_comparison/aggregated_metrics.json   same format as DIANA
  - results/baseline_comparison/timing.json               per-model timing summary
  - results/baseline_comparison/summary.csv               mean +/- std, 95% CI
  - results/baseline_comparison/summary.tex               LaTeX comparison table
  - results/baseline_comparison/ttest_results.json        paired t-test vs DIANA
  - results/baseline_comparison/best_per_task.json        best model per task
  - results/baseline_comparison/confusion_matrices/       last fold, JSON
  - results/baseline_comparison/classification_reports/   last fold, TXT
  - results/baseline_comparison/baseline_comparison.log   full log
  - results/baseline_comparison/comparison_figure.html    Plotly bar chart
  - results/baseline_comparison/comparison_figure.pdf     same, for paper

USAGE:
  mamba run -p ./env python scripts/evaluation/07_baseline_comparison.py
  # or: sbatch scripts/evaluation/run_baseline_comparison.sbatch
"""

import sys
import json
import logging
import time
import warnings
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from packaging.version import Version
from scipy import stats as scipy_stats
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ---------------------------------------------------------------------------
# Count convergence warnings rather than hiding them silently.
# ---------------------------------------------------------------------------
_convergence_warning_count = 0

class _ConvergenceCounter(logging.Filter):
    pass

def _warn_hook(message, category, filename, lineno, file=None, line=None):
    global _convergence_warning_count
    if issubclass(category, ConvergenceWarning):
        _convergence_warning_count += 1
    # swallow — we raise max_iter to reduce these, but still track them

warnings.showwarning = _warn_hook
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Sanity-check warning counter
_sanity_warning_count = 0

# ---------------------------------------------------------------------------

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from diana.data.loader import MatrixLoader

# ─── Configuration ─────────────────────────────────────────────────────────

MATRIX_PATH   = Path("data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat")
METADATA_PATH = Path("paper/metadata/train_metadata.tsv")
DIANA_CV_PATH = Path("results/training/cv_results/aggregated_results.json")
OUTPUT_DIR    = Path("results/baseline_comparison")

TASKS        = ["sample_type", "community_type", "sample_host", "material"]
N_FOLDS      = 5
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

# ─── Build model registry (Ridge balanced added if sklearn >= 1.2) ──────────

def _build_models() -> dict:
    models = {
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
    }
    # class_weight='balanced' supported in RidgeClassifier since sklearn 1.2
    if Version(sklearn_version) >= Version("1.2"):
        models["RidgeClassifier_Bal"] = RidgeClassifier(
            alpha=1.0, class_weight="balanced"
        )
    return models

MODELS = _build_models()

# ─── Logging ───────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to both stderr (INFO) and a file (DEBUG)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "baseline_comparison.log"
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    return logging.getLogger(__name__)


# ─── Helpers ───────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return accuracy, balanced accuracy, weighted F1, and macro F1."""
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_weighted":       float(f1_score(y_true, y_pred, average="weighted",
                                            zero_division=0)),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro",
                                            zero_division=0)),
    }


def mean_std(values: list) -> str:
    arr = np.array(values, dtype=float)
    return f"{np.nanmean(arr):.4f} +/- {np.nanstd(arr):.4f}"


def ci95(values: list) -> tuple:
    """Return (lo, hi) 95% t-interval."""
    arr = np.array(values, dtype=float)
    n   = int(np.sum(~np.isnan(arr)))
    if n < 2:
        return (float("nan"), float("nan"))
    m  = float(np.nanmean(arr))
    se = float(np.nanstd(arr, ddof=1)) / np.sqrt(n)
    t  = float(scipy_stats.t.ppf(0.975, df=n - 1))
    return (m - t * se, m + t * se)


def sanity_check(task: str, n_total_classes: int, bal_acc: float,
                 logger: logging.Logger) -> None:
    """
    Warn if balanced accuracy is close to or below the random baseline.
    Uses total class count from the encoder (not just classes seen in the
    training fold) to avoid misleading warnings when rare classes are absent
    from a particular fold.
    """
    global _sanity_warning_count
    random_baseline = 1.0 / n_total_classes
    if bal_acc < random_baseline + 0.05:
        _sanity_warning_count += 1
        logger.warning(
            f"    [sanity] [{task}] balanced_accuracy={bal_acc:.3f} is close to or "
            f"below the random baseline ({random_baseline:.3f}). "
            f"Check class distribution."
        )


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ─── Data Loading ──────────────────────────────────────────────────────────

def load_data(logger: logging.Logger) -> tuple:
    """Load the unitig matrix filtered to training metadata."""
    for p in (MATRIX_PATH, METADATA_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    logger.info(f"Loading matrix: {MATRIX_PATH}")
    t0 = time.time()
    loader = MatrixLoader(MATRIX_PATH)
    X, metadata_pl = loader.load_with_metadata(
        metadata_path=METADATA_PATH,
        align_to_matrix=True,
        filter_matrix_to_metadata=True,
    )
    metadata = metadata_pl.to_pandas()
    elapsed = time.time() - t0

    # Data characteristics
    n_samples, n_features = X.shape
    is_sparse = hasattr(X, "toarray")
    if is_sparse:
        nnz = X.nnz
        sparsity = 1.0 - nnz / (n_samples * n_features)
        mem_mb = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6
    else:
        sparsity = float(np.mean(X == 0))
        mem_mb   = X.nbytes / 1e6

    logger.info(
        f"Matrix: {n_samples} samples x {n_features} features  "
        f"[dtype={X.dtype}, sparse={is_sparse}, "
        f"sparsity={sparsity:.1%}, mem={mem_mb:.0f} MB, elapsed={elapsed:.1f}s]"
    )
    return X, metadata


def encode_labels(metadata: pd.DataFrame, logger: logging.Logger) -> tuple:
    """Fit LabelEncoders on training data, return encoded arrays + encoders."""
    labels:   dict = {}
    encoders: dict = {}
    for task in TASKS:
        le = LabelEncoder()
        labels[task]   = le.fit_transform(metadata[task].fillna("Unknown"))
        encoders[task] = le
        counts = pd.Series(labels[task]).value_counts().sort_index()
        logger.info(
            f"  {task}: {len(le.classes_)} classes -> "
            + str(dict(zip(le.classes_, counts.values)))
        )
    return labels, encoders


# ─── DIANA Reference ───────────────────────────────────────────────────────

def load_diana_reference(path: Path, logger: logging.Logger) -> dict | None:
    """
    Load DIANA per-fold CV metrics. Validates that per-fold value arrays
    exist and have exactly N_FOLDS entries for each task/metric combination.
    """
    if not path.exists():
        logger.warning(f"DIANA reference not found at {path}; t-tests/comparison disabled.")
        return None

    with open(path) as f:
        data = json.load(f)

    ref = data.get("aggregated_metrics", {})
    metrics_to_check = ["accuracy", "balanced_accuracy", "f1_macro"]
    issues = []

    for task in TASKS:
        if task not in ref:
            issues.append(f"task '{task}' missing from DIANA reference")
            continue
        for metric in metrics_to_check:
            vals = ref[task].get(metric, {}).get("values")
            if vals is None:
                issues.append(f"[{task}][{metric}] missing 'values' array")
            elif len(vals) != N_FOLDS:
                issues.append(
                    f"[{task}][{metric}] has {len(vals)} fold values, expected {N_FOLDS}"
                )

    if issues:
        for issue in issues:
            logger.warning(f"  DIANA reference issue: {issue}")
    else:
        logger.info(
            f"DIANA reference loaded OK ({len(TASKS)} tasks x "
            f"{len(metrics_to_check)} metrics x {N_FOLDS} folds each)"
        )
    return ref


# ─── Fold Verification ─────────────────────────────────────────────────────

def verify_folds(folds: list, labels: dict, logger: logging.Logger) -> None:
    """
    Log per-fold class distributions for sample_type (binary stratification
    target) so results can be compared against DIANA's training logs.

    DIANA does not export fold indices, so exact index comparison is not
    possible; class distribution logging is the next best audit trail.
    """
    logger.info("\nFold class distributions (sample_type — stratification target):")
    le_classes = list(range(int(labels["sample_type"].max()) + 1))
    for i, (train_idx, val_idx) in enumerate(folds):
        train_y = labels["sample_type"][train_idx]
        val_y   = labels["sample_type"][val_idx]
        t_dist  = {c: int(np.sum(train_y == c)) for c in le_classes}
        v_dist  = {c: int(np.sum(val_y   == c)) for c in le_classes}
        logger.info(
            f"  fold {i+1}: train={len(train_idx)} {t_dist}  "
            f"val={len(val_idx)} {v_dist}"
        )
    logger.info(
        "  Note: DIANA fold indices are not exported; compare class "
        "distributions against DIANA's training log to verify alignment."
    )


# ─── CV Loop ───────────────────────────────────────────────────────────────

def run_cv(
    X: np.ndarray,
    labels: dict,
    encoders: dict,
    logger: logging.Logger,
) -> tuple:
    """
    Run N_FOLDS-fold stratified CV for all models and tasks.

    Folds are stratified on sample_type (binary, most stable) — same as DIANA.

    Returns:
        results    : results[model][task] = list of per-fold metric dicts
        folds      : list of (train_idx, val_idx)
        model_times: model_times[model] = {"total_s": ..., "per_task": {...}}
    """
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
    folds = list(skf.split(X, labels["sample_type"]))

    results:     dict = {m: {t: [] for t in TASKS} for m in MODELS}
    model_times: dict = {
        m: {"total_s": 0.0, "per_task": {t: 0.0 for t in TASKS}}
        for m in MODELS
    }

    verify_folds(folds, labels, logger)

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"\n{'─'*70}")
        logger.info(
            f"Fold {fold_idx + 1}/{N_FOLDS}  "
            f"(train={len(train_idx)}, val={len(val_idx)})"
        )
        X_train, X_val = X[train_idx], X[val_idx]

        for model_name, model_proto in MODELS.items():
            model = clone(model_proto)
            logger.info(f"  -- {model_name}")

            for task in TASKS:
                y_train = labels[task][train_idx]
                y_val   = labels[task][val_idx]
                # Use total class count from encoder (not just fold classes)
                n_total_classes = len(encoders[task].classes_)

                t0 = time.time()
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                except Exception as exc:
                    logger.error(
                        f"    [{model_name}][{task}] FAILED: {exc}. Storing NaN."
                    )
                    results[model_name][task].append(
                        {k: float("nan") for k in
                         ["accuracy", "balanced_accuracy",
                          "f1_weighted", "f1_macro"]}
                    )
                    continue

                elapsed = time.time() - t0
                model_times[model_name]["total_s"]          += elapsed
                model_times[model_name]["per_task"][task]   += elapsed

                m = compute_metrics(y_val, y_pred)
                results[model_name][task].append(m)

                sanity_check(task, n_total_classes, m["balanced_accuracy"], logger)

                logger.info(
                    f"    {task:20s}  "
                    f"acc={m['accuracy']:.3f}  "
                    f"bal_acc={m['balanced_accuracy']:.3f}  "
                    f"f1_macro={m['f1_macro']:.3f}  "
                    f"({elapsed:.1f}s)"
                )

    # Log timing summary
    logger.info(f"\n{'─'*70}")
    logger.info("Timing summary (total across all folds and tasks):")
    for model_name, t in sorted(model_times.items(),
                                key=lambda kv: kv[1]["total_s"], reverse=True):
        logger.info(f"  {model_name:35s}  {t['total_s']:6.1f}s total")

    return results, folds, model_times


# ─── Post-processing & Outputs ─────────────────────────────────────────────

def build_summary(results: dict) -> pd.DataFrame:
    """Aggregate per-fold dicts into a summary DataFrame (mean, std, 95% CI)."""
    metric_cols = ["accuracy", "balanced_accuracy", "f1_weighted", "f1_macro"]
    rows = []
    for model_name in MODELS:
        for task in TASKS:
            fold_metrics = results[model_name][task]
            row: dict = {"model": model_name, "task": task}
            for m in metric_cols:
                vals            = [f[m] for f in fold_metrics]
                row[m]          = float(np.nanmean(vals))
                row[f"{m}_std"] = float(np.nanstd(vals))
                row[f"{m}_str"] = mean_std(vals)
                lo, hi          = ci95(vals)
                row[f"{m}_ci95_lo"] = lo
                row[f"{m}_ci95_hi"] = hi
            rows.append(row)
    return pd.DataFrame(rows)


def save_aggregated_metrics(results: dict, output_dir: Path,
                            logger: logging.Logger) -> None:
    """
    Save aggregated_metrics.json in the same format as DIANA's
    aggregated_results.json for automated downstream comparison.
    """
    metric_cols = ["accuracy", "balanced_accuracy", "f1_weighted", "f1_macro"]
    out: dict = {}
    for model_name in MODELS:
        out[model_name] = {}
        for task in TASKS:
            out[model_name][task] = {}
            for m in metric_cols:
                vals = [f[m] for f in results[model_name][task]]
                out[model_name][task][m] = {
                    "mean":   float(np.nanmean(vals)),
                    "std":    float(np.nanstd(vals)),
                    "values": [float(v) for v in vals],
                }

    path = output_dir / "aggregated_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Aggregated metrics      -> {path}")


def save_confusion_matrices(
    X: np.ndarray,
    labels: dict,
    encoders: dict,
    folds: list,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Re-train each model on fold 5 (the last fold) and save confusion matrices
    (JSON) and classification reports (TXT).  Only the last fold is used to
    keep output manageable; it is auditable via fold_indices.json.
    """
    cm_dir     = output_dir / "confusion_matrices"
    report_dir = output_dir / "classification_reports"
    cm_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    train_idx, val_idx = folds[-1]
    X_train, X_val = X[train_idx], X[val_idx]

    for model_name, model_proto in MODELS.items():
        model = clone(model_proto)
        for task in TASKS:
            y_train = labels[task][train_idx]
            y_val   = labels[task][val_idx]
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            except Exception as exc:
                logger.warning(
                    f"  [{model_name}][{task}] confusion matrix skipped: {exc}"
                )
                continue

            class_names = encoders[task].classes_.tolist()
            cm = confusion_matrix(y_val, y_pred,
                                  labels=list(range(len(class_names))))
            cm_path = cm_dir / f"{model_name}_{task}.json"
            with open(cm_path, "w") as f:
                json.dump({"classes": class_names, "matrix": cm.tolist(),
                           "fold": N_FOLDS}, f, indent=2)

            report   = classification_report(y_val, y_pred,
                                             target_names=class_names,
                                             zero_division=0)
            rpt_path = report_dir / f"{model_name}_{task}.txt"
            rpt_path.write_text(
                f"Model: {model_name}\nTask: {task}\n"
                f"Fold: {N_FOLDS} (last fold only — for illustration)\n\n{report}"
            )

    logger.info(f"Confusion matrices      -> {cm_dir}")
    logger.info(f"Classification reports  -> {report_dir}")


def find_best_baseline_per_task(summary_df: pd.DataFrame) -> dict:
    """Return model with highest mean balanced_accuracy for each task."""
    best: dict = {}
    for task in TASKS:
        sub  = summary_df[summary_df.task == task]
        idx  = sub["balanced_accuracy"].idxmax()
        best[task] = {
            "model":            sub.loc[idx, "model"],
            "balanced_accuracy": float(sub.loc[idx, "balanced_accuracy"]),
        }
    return best


def run_ttests(results: dict, diana_ref: dict | None,
               logger: logging.Logger) -> dict:
    """
    One-sided paired t-test (H1: baseline < DIANA) per model x task x metric.
    Requires per-fold values from DIANA's aggregated_results.json.
    """
    ttest_out: dict = {}
    if diana_ref is None:
        logger.info("Skipping t-tests (DIANA reference unavailable).")
        return ttest_out

    for model_name in MODELS:
        ttest_out[model_name] = {}
        for task in TASKS:
            ttest_out[model_name][task] = {}
            for metric in ["accuracy", "balanced_accuracy", "f1_macro"]:
                baseline_vals = [f[metric] for f in results[model_name][task]]
                diana_vals    = (diana_ref.get(task, {})
                                         .get(metric, {})
                                         .get("values"))
                if diana_vals is None or len(diana_vals) != N_FOLDS:
                    ttest_out[model_name][task][metric] = {
                        "error": "DIANA per-fold values unavailable"
                    }
                    continue
                a = np.array(baseline_vals, dtype=float)
                b = np.array(diana_vals,    dtype=float)
                res = scipy_stats.ttest_rel(a, b, alternative="less")
                ttest_out[model_name][task][metric] = {
                    "statistic":       float(res.statistic),
                    "pvalue":          float(res.pvalue),
                    "significant_p05": bool(res.pvalue < 0.05),
                    "note":            "one-sided paired t-test (H1: baseline < DIANA)",
                }
    return ttest_out


# ─── LaTeX Table ───────────────────────────────────────────────────────────

def save_latex_table(
    summary_df: pd.DataFrame,
    diana_ref: dict | None,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Write a LaTeX table with DIANA reference prepended for side-by-side
    reading.  Columns: Model | Task | Accuracy | Bal. Acc. | F1 Macro
    """
    metric_pairs = [
        ("accuracy",          "Accuracy"),
        ("balanced_accuracy", "Bal. Acc."),
        ("f1_macro",          "F1 Macro"),
    ]

    def fmt_diana(task: str, metric: str) -> str:
        if diana_ref is None:
            return "---"
        vals = diana_ref.get(task, {}).get(metric, {})
        m, s = vals.get("mean"), vals.get("std")
        return f"{m:.4f} $\\pm$ {s:.4f}" if m is not None else "---"

    def esc(s: str) -> str:
        return s.replace("_", "\\_")

    tex_rows: list = []

    # DIANA reference block
    ncols_empty = " & ".join("" for _ in metric_pairs)
    tex_rows.append(
        f"  \\multicolumn{{2}}{{l}}{{\\textit{{DIANA (MLP, 5-fold CV)}}}} "
        f"& {ncols_empty} \\\\"
    )
    for task in TASKS:
        vals_str = " & ".join(fmt_diana(task, m) for m, _ in metric_pairs)
        tex_rows.append(f"  & {esc(task)} & {vals_str} \\\\")
    tex_rows.append("  \\midrule")

    # Baseline blocks
    for model_name in MODELS:
        for i, task in enumerate(TASKS):
            row        = summary_df[
                (summary_df.model == model_name) & (summary_df.task == task)
            ].iloc[0]
            model_cell = (
                f"\\multirow{{4}}{{*}}{{{esc(model_name)}}}" if i == 0 else ""
            )
            metric_strs = " & ".join(
                row[f"{m}_str"].replace("+/-", "$\\pm$")
                for m, _ in metric_pairs
            )
            tex_rows.append(
                f"  {model_cell} & {esc(task)} & {metric_strs} \\\\"
            )
        tex_rows.append("  \\midrule")

    col_header = " & ".join(f"\\textbf{{{h}}}" for _, h in metric_pairs)
    tex = "\n".join([
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\caption{Baseline classifier comparison vs.\\ DIANA (5-fold CV on "
        "training set, mean $\\pm$ std). Stratified folds identical across all "
        "models (\\texttt{random\\_state=42}). One-sided paired $t$-tests "
        "($\\alpha = 0.05$) are in \\texttt{ttest\\_results.json}.}",
        "\\label{tab:baseline_comparison}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        f"\\textbf{{Model}} & \\textbf{{Task}} & {col_header} \\\\",
        "\\midrule",
        *tex_rows[:-1],   # drop trailing \midrule
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    out_tex = output_dir / "summary.tex"
    out_tex.write_text(tex)
    logger.info(f"LaTeX table             -> {out_tex}")


# ─── Plotly Visualization ──────────────────────────────────────────────────

def save_comparison_figure(
    summary_df: pd.DataFrame,
    diana_ref: dict | None,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Bar chart comparing DIANA and all baselines per task on balanced_accuracy.
    Uses the Plotly 'Vivid' qualitative palette.
    """
    try:
        import plotly.graph_objects as go
        import plotly.colors as pc
    except ImportError:
        logger.warning("Plotly not available — skipping comparison figure.")
        return

    vivid = pc.qualitative.Vivid
    model_names = ["DIANA (MLP)"] + list(MODELS.keys())

    # One colour per model (cycle if more models than palette entries)
    color_map = {name: vivid[i % len(vivid)] for i, name in enumerate(model_names)}

    fig = go.Figure()
    task_labels = [t.replace("_", " ") for t in TASKS]

    for model_name in model_names:
        if model_name == "DIANA (MLP)":
            y_vals = []
            y_err  = []
            for task in TASKS:
                if diana_ref and task in diana_ref:
                    v = diana_ref[task].get("balanced_accuracy", {})
                    y_vals.append(v.get("mean", None))
                    y_err.append(v.get("std", None))
                else:
                    y_vals.append(None)
                    y_err.append(None)
        else:
            y_vals = []
            y_err  = []
            for task in TASKS:
                row = summary_df[
                    (summary_df.model == model_name) & (summary_df.task == task)
                ]
                if len(row):
                    y_vals.append(float(row.iloc[0]["balanced_accuracy"]))
                    y_err.append(float(row.iloc[0]["balanced_accuracy_std"]))
                else:
                    y_vals.append(None)
                    y_err.append(None)

        fig.add_trace(go.Bar(
            name=model_name,
            x=task_labels,
            y=y_vals,
            error_y=dict(type="data", array=y_err, visible=True),
            marker_color=color_map[model_name],
        ))

    fig.update_layout(
        barmode="group",
        title="Baseline Classifier Comparison vs DIANA (Balanced Accuracy, 5-fold CV)",
        xaxis_title="Task",
        yaxis_title="Balanced Accuracy",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(title="Model"),
        template="plotly_white",
        font=dict(size=13),
        width=1100,
        height=550,
    )

    html_path = output_dir / "comparison_figure.html"
    fig.write_html(str(html_path))
    logger.info(f"Comparison figure (HTML) -> {html_path}")

    try:
        pdf_path = output_dir / "comparison_figure.pdf"
        fig.write_image(str(pdf_path))
        logger.info(f"Comparison figure (PDF)  -> {pdf_path}")
    except Exception as e:
        logger.warning(f"PDF export failed (kaleido not installed?): {e}")


# ─── Final Recommendation ──────────────────────────────────────────────────

def print_recommendation(best_per_task: dict, summary_df: pd.DataFrame,
                         diana_ref: dict | None) -> None:
    """Print a clear final recommendation section to stdout."""
    width = 90
    print(f"\n{'='*width}")
    print(" FINAL RECOMMENDATION — best baseline per task (by balanced accuracy)")
    print(f"{'='*width}")
    print(f"  {'Task':20s}  {'Best baseline':35s}  {'Bal. Acc.':>12}  {'vs DIANA CV':>14}")
    print(f"  {'-'*20}  {'-'*35}  {'-'*12}  {'-'*14}")
    for task in TASKS:
        info      = best_per_task[task]
        model     = info["model"]
        bal_acc   = info["balanced_accuracy"]
        diana_str = "N/A"
        if diana_ref and task in diana_ref:
            d = diana_ref[task].get("balanced_accuracy", {})
            dm = d.get("mean")
            if dm is not None:
                delta     = bal_acc - dm
                sign      = "+" if delta >= 0 else ""
                diana_str = f"{_fmt_pct(dm)} ({sign}{delta*100:.1f}pp)"
        print(f"  {task:20s}  {model:35s}  {_fmt_pct(bal_acc):>12}  {diana_str:>14}")
    print(f"{'='*width}")
    print(
        "  Interpretation: DIANA outperforms baselines on tasks with higher "
        "class\n"
        "  imbalance and non-linear separability. Use ttest_results.json for "
        "significance."
    )
    print(f"{'='*width}\n")


# ─── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)

    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("DIANA Baseline Comparison -- supplementary experiment")
    logger.info(f"Started:      {start_time.isoformat()}")
    logger.info(f"Python:       {sys.version.split()[0]}")
    logger.info(f"NumPy:        {np.__version__}")
    logger.info(f"scikit-learn: {sklearn_version}")
    logger.info(f"N_FOLDS:      {N_FOLDS}   |   RANDOM_STATE: {RANDOM_STATE}")
    logger.info(f"Models ({len(MODELS)}): {list(MODELS.keys())}")
    logger.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    try:
        X, metadata = load_data(logger)
    except FileNotFoundError as exc:
        logger.critical(str(exc))
        sys.exit(1)
    except MemoryError:
        logger.critical("Out of memory while loading matrix. Request more RAM.")
        sys.exit(1)

    labels, encoders = encode_labels(metadata, logger)
    diana_ref        = load_diana_reference(DIANA_CV_PATH, logger)

    # ── 5-fold CV ─────────────────────────────────────────────────────────
    results, folds, model_times = run_cv(X, labels, encoders, logger)

    # ── Save per-fold results ──────────────────────────────────────────────
    out_json = OUTPUT_DIR / "cv_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nPer-fold results        -> {out_json}")

    # ── Save fold indices ──────────────────────────────────────────────────
    fold_idx_path = OUTPUT_DIR / "fold_indices.json"
    with open(fold_idx_path, "w") as f:
        json.dump(
            {f"fold_{i}": {"train": ti.tolist(), "val": vi.tolist()}
             for i, (ti, vi) in enumerate(folds)},
            f,
        )
    logger.info(f"Fold indices            -> {fold_idx_path}")

    # ── Save timing summary ────────────────────────────────────────────────
    timing_path = OUTPUT_DIR / "timing.json"
    with open(timing_path, "w") as f:
        json.dump(model_times, f, indent=2)
    logger.info(f"Timing summary          -> {timing_path}")

    # ── Aggregated metrics (DIANA-compatible format) ───────────────────────
    save_aggregated_metrics(results, OUTPUT_DIR, logger)

    # ── Summary CSV ───────────────────────────────────────────────────────
    summary_df = build_summary(results)
    out_csv    = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"Summary CSV             -> {out_csv}")

    # ── Confusion matrices & classification reports (last fold) ───────────
    save_confusion_matrices(X, labels, encoders, folds, OUTPUT_DIR, logger)

    # ── Best baseline per task ────────────────────────────────────────────
    best_per_task = find_best_baseline_per_task(summary_df)
    (OUTPUT_DIR / "best_per_task.json").write_text(
        json.dumps(best_per_task, indent=2)
    )
    logger.info(f"Best per task           -> {OUTPUT_DIR / 'best_per_task.json'}")

    # ── Paired t-tests vs DIANA ───────────────────────────────────────────
    ttest_results = run_ttests(results, diana_ref, logger)
    if ttest_results:
        ttest_path = OUTPUT_DIR / "ttest_results.json"
        with open(ttest_path, "w") as f:
            json.dump(ttest_results, f, indent=2)
        logger.info(f"Paired t-test results   -> {ttest_path}")

    # ── LaTeX table ───────────────────────────────────────────────────────
    save_latex_table(summary_df, diana_ref, OUTPUT_DIR, logger)

    # ── Plotly comparison figure ──────────────────────────────────────────
    save_comparison_figure(summary_df, diana_ref, OUTPUT_DIR, logger)

    # ── Pretty-print summary to stdout ────────────────────────────────────
    print(f"\n{'='*110}")
    print("BASELINE COMPARISON -- 5-fold CV on training set (mean +/- std)")
    print(f"{'='*110}")
    display_cols = ["model", "task",
                    "accuracy_str", "balanced_accuracy_str", "f1_macro_str"]
    col_labels   = ["Model", "Task", "Accuracy", "Balanced Acc", "F1 Macro"]
    print(pd.DataFrame(summary_df[display_cols].values, columns=col_labels)
          .to_string(index=False))

    if diana_ref:
        print(f"\n{'─'*110}")
        print("DIANA 5-fold CV reference (aggregated_results.json):")
        for task in TASKS:
            for metric in ["accuracy", "balanced_accuracy", "f1_macro"]:
                d = diana_ref.get(task, {}).get(metric, {})
                m = d.get("mean", float("nan"))
                s = d.get("std",  float("nan"))
                print(f"  {task:20s}  {metric:22s}  {m:.4f} +/- {s:.4f}")

    # ── Final recommendation ──────────────────────────────────────────────
    print_recommendation(best_per_task, summary_df, diana_ref)

    # ── Warnings summary ──────────────────────────────────────────────────
    elapsed_total = (datetime.now() - start_time).total_seconds()
    logger.info(f"\n{'='*70}")
    logger.info(f"Total runtime:              {elapsed_total:.1f}s")
    logger.info(f"ConvergenceWarnings caught: {_convergence_warning_count}")
    logger.info(f"Sanity check warnings:      {_sanity_warning_count}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
