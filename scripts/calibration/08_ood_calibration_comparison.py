#!/usr/bin/env python3
"""
OOD Calibration Comparison — v5 DIANA model

Compares four confidence metrics for calibration quality and
wrong-prediction detection on the validation set:
  1. Softmax max-probability  (from predictions JSON, existing pipeline)
  2. Energy score             log Σ_c exp(logit_c)  — higher = more in-distribution
  3. Mahalanobis distance     min_c d_M(z, μ_c)     — lower = more in-distribution
  4. MC Dropout               mean variance of softmax over 30 stochastic passes

Generates per-task 4-panel figures:
  A — Reliability diagram (ECE)
  B — AUROC for wrong-prediction detection
  C — Score distribution: correct vs incorrect
  D — Summary heatmap (Metric × Task ECE / AUROC)

Usage:
  python 36_ood_calibration_comparison.py \\
      --val-embeddings  results/validation_predictions_bioproject_v5_full/val_embeddings.npz \\
      --train-embeddings results/training_bioproject_v5/train_embeddings.npz \\
      --mahal-params    results/training_bioproject_v5/mahalanobis_params.npz \\
      --val-pred-dir    results/validation_predictions_bioproject_v5_full \\
      --val-metadata    data/splits_v5/validation_metadata.tsv \\
      --model           results/training_bioproject_v5/best_model.pth \\
      --label-encoders  results/training_bioproject_v5/label_encoders.json \\
      --out-dir         results/calibration_analysis \\
      --figure-dir      paper/figures/final

Notes:
  - Energy score is normalised to [0,1] via min-max over the val set
    before computing ECE, so it is comparable to probabilities.
  - Mahalanobis distance is negated (−d) and normalised to [0,1].
  - MC Dropout uses model.train() mode (activates dropout) but keeps
    parameters fixed; 30 stochastic forward passes.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from diana.inference.predictor import Predictor
from diana.data.loader import MatrixLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TASKS = ["sample_type", "community_type", "sample_host", "material"]
N_MC  = 30          # Monte Carlo Dropout passes
N_BINS = 10         # reliability diagram bins
N_FEATURES = 107_480  # Expected number of unitig features


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def expected_calibration_error(confidences: np.ndarray,
                                correct: np.ndarray,
                                n_bins: int = N_BINS) -> float:
    """ECE computed over equal-width confidence bins [0,1]."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(confidences)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return ece


def reliability_diagram_data(confidences: np.ndarray,
                              correct: np.ndarray,
                              n_bins: int = N_BINS):
    """Returns bin centres, mean accuracy per bin, mean confidence per bin, counts."""
    bins = np.linspace(0, 1, n_bins + 1)
    centres, acc_per_bin, conf_per_bin, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        counts.append(mask.sum())
        centres.append((lo + hi) / 2)
        if mask.sum() == 0:
            acc_per_bin.append(0.0)
            conf_per_bin.append((lo + hi) / 2)
        else:
            acc_per_bin.append(correct[mask].mean())
            conf_per_bin.append(confidences[mask].mean())
    return (np.array(centres), np.array(acc_per_bin),
            np.array(conf_per_bin), np.array(counts))


def auroc_wrong_prediction(scores: np.ndarray, correct: np.ndarray) -> float:
    """AUROC for detecting wrong predictions: score should be LOW when wrong."""
    try:
        return roc_auc_score(correct.astype(int), scores)
    except ValueError:
        return 0.5  # not enough positives/negatives


def load_predictions_from_tsv(tsv_path: Path, task: str, label_encoder: dict) -> dict:
    """
    Load predictions from test_predictions.tsv format.
    
    Returns dict with keys: sample_ids, predicted_classes, probabilities
    """
    df = pd.read_csv(tsv_path, sep="\t")
    df = df.set_index("Run_accession")
    
    # Get number of classes from probability columns
    prob_cols = [c for c in df.columns if c.startswith(f"{task}_prob_")]
    n_classes = len(prob_cols)
    
    # Extract predictions and probabilities
    sample_ids = df.index.values
    predicted_classes = df[f"{task}_pred"].values
    
    # Build probability matrix
    probs = np.zeros((len(sample_ids), n_classes))
    for i, col in enumerate(sorted(prob_cols)):
        probs[:, i] = df[col].values
    
    return {
        "sample_ids": sample_ids,
        "predicted_classes": predicted_classes,
        "probabilities": probs
    }


def auroc_wrong_prediction(scores: np.ndarray, correct: np.ndarray) -> float:
    """AUROC for detecting wrong predictions: score should be LOW when wrong."""
    try:
        return roc_auc_score(correct.astype(int), scores)
    except ValueError:
        return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_energy_scores(logits: np.ndarray) -> np.ndarray:
    """
    Energy score: log Σ_c exp(logit_c)
    Higher = more in-distribution (model is more 'certain' overall).
    Uses scipy.special.logsumexp for numerical stability.
    """
    return logsumexp(logits, axis=1)


def compute_mahalanobis_scores(embeddings: np.ndarray, mu_arr: np.ndarray,
                                Sigma_inv: np.ndarray) -> np.ndarray:
    """
    Minimum Mahalanobis distance to any class centroid.
    Returns distances (lower = more in-distribution).
    """
    N = embeddings.shape[0]
    scores = np.full(N, np.inf)
    for mu in mu_arr:
        diff = embeddings - mu[np.newaxis, :]   # (N, D)
        # d² = diff @ Sigma_inv @ diff.T  diag only
        tmp  = diff @ Sigma_inv              # (N, D)
        d2   = (tmp * diff).sum(axis=1)     # (N,)
        d    = np.sqrt(np.maximum(d2, 0))
        scores = np.minimum(scores, d)
    return scores


def _load_mc_model(model_path: Path, device: str = "cpu") -> torch.nn.Module:
    """Load model in MC Dropout mode: train() with BatchNorm layers frozen."""
    predictor = Predictor(model_path, device=device)
    model = predictor.model
    model.train()
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.eval()
    return model


def compute_mc_dropout(model: torch.nn.Module, frac_files: List[Path],
                        task: str, n_passes: int = N_MC,
                        device: str = "cpu") -> dict:
    """
    MC Dropout: run n_passes stochastic forward passes per sample.
    Returns dict sid → mean_variance (scalar, lower = more certain).
    Expects model already in train() mode with BN frozen (use _load_mc_model).
    """
    results = {}
    for frac_path in frac_files:
        sid = frac_path.parent.name
        try:
            x_data = np.fromfile(frac_path, sep="\n", dtype=np.float32)
            assert len(x_data) == N_FEATURES, f"Wrong feature count: {len(x_data)}"
            x = torch.tensor(x_data, dtype=torch.float32).unsqueeze(0).to(device)
        except Exception as e:
            logger.warning(f"MC Dropout: failed to load {frac_path}: {e}")
            continue

        pass_probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                out  = model(x)
                prob = F.softmax(out[task].squeeze(0), dim=0).cpu().numpy()
                pass_probs.append(prob)

        # Mean variance across classes (scalar uncertainty measure)
        pass_probs = np.array(pass_probs)          # (n_passes, C)
        variance   = pass_probs.var(axis=0).mean() # mean over classes
        results[sid] = variance

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

METRIC_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]


def plot_task(task: str, metrics: dict,
              out_dir: Path, fig_dir: Path):
    """
    4-panel figure for one task.
    metrics: {metric_name: {"conf": np.ndarray, "correct": np.ndarray}}
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Calibration Analysis — task: {task}", fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    ax_rel  = fig.add_subplot(gs[0, 0])   # reliability diagram
    ax_roc  = fig.add_subplot(gs[0, 1])   # AUROC bar
    ax_dist = fig.add_subplot(gs[1, 0])   # score distributions
    ax_summ = fig.add_subplot(gs[1, 1])   # summary table

    metric_names = list(metrics.keys())
    ece_vals  = []
    auroc_vals = []

    # ── Panel A: Reliability diagrams ────────────────────────────────────────
    ax_rel.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect")
    for name, color, mdata in zip(metric_names, METRIC_COLORS, metrics.values()):
        conf = mdata["conf"]
        crr  = mdata["correct"]
        ece  = expected_calibration_error(conf, crr)
        ece_vals.append(ece)
        centres, acc, _, counts = reliability_diagram_data(conf, crr)
        ax_rel.plot(centres, acc, "-o", color=color, linewidth=1.5,
                    label=f"{name.replace(chr(10),' ')} (ECE={ece:.3f})", markersize=4)

    ax_rel.set_xlim(0, 1); ax_rel.set_ylim(0, 1)
    ax_rel.set_xlabel("Confidence"); ax_rel.set_ylabel("Accuracy")
    ax_rel.set_title("A — Reliability Diagram")
    ax_rel.legend(fontsize=7)

    # ── Panel B: AUROC bars ───────────────────────────────────────────────────
    for name, color, mdata in zip(metric_names, METRIC_COLORS, metrics.values()):
        auc = auroc_wrong_prediction(mdata["conf"], mdata["correct"])
        auroc_vals.append(auc)

    bars = ax_roc.bar(metric_names, auroc_vals, color=METRIC_COLORS, width=0.6)
    ax_roc.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="Random")
    ax_roc.set_ylim(0, 1.05)
    ax_roc.set_ylabel("AUROC")
    ax_roc.set_title("B — AUROC: Correct vs Wrong Prediction")
    ax_roc.tick_params(axis="x", labelsize=7)
    for bar, val in zip(bars, auroc_vals):
        if np.isfinite(val):
            ax_roc.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # ── Panel C: Score distributions ─────────────────────────────────────────
    offsets = np.linspace(-0.3, 0.3, len(metric_names))
    for offset, name, color, mdata in zip(offsets, metric_names, METRIC_COLORS, metrics.values()):
        conf     = mdata["conf"]
        crr      = mdata["correct"]
        correct_scores   = conf[crr  == 1]
        incorrect_scores = conf[crr  == 0]
        
        # Guard against insufficient data for violinplot
        if len(correct_scores) > 1 and len(incorrect_scores) > 1:
            # Violin-style approximation with boxplot
            parts = ax_dist.violinplot(
                [correct_scores, incorrect_scores],
                positions=[0 + offset, 1 + offset],
                widths=0.15, showmedians=True
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.5)
        else:
            # Insufficient data for violin plot
            ax_dist.text(0.5, 0.5, 
                        f"Insufficient data\n(n_correct={len(correct_scores)}, n_incorrect={len(incorrect_scores)})",
                        ha='center', va='center', transform=ax_dist.transAxes,
                        fontsize=8, color='gray')

    ax_dist.set_xticks([0, 1])
    ax_dist.set_xticklabels(["Correct", "Incorrect"])
    ax_dist.set_ylabel("Confidence score")
    ax_dist.set_title("C — Score Distributions")

    # ── Panel D: Summary heatmap (ECE ↓ and AUROC ↑ with correct colormaps) ────
    short_names = [n.replace("\n", " ") for n in metric_names]
    n_m = len(ece_vals)
    # ECE: low is good → RdYlGn_r (green=low, red=high)
    im_ece = ax_summ.imshow(
        np.array(ece_vals).reshape(n_m, 1), aspect="auto", cmap="RdYlGn_r",
        vmin=0, vmax=1, extent=[-0.5, 0.5, n_m - 0.5, -0.5]
    )
    # AUROC: high is good → RdYlGn (green=high, red=low)
    im_auc = ax_summ.imshow(
        np.array([a if np.isfinite(a) else 0.5 for a in auroc_vals]).reshape(n_m, 1),
        aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
        extent=[0.5, 1.5, n_m - 0.5, -0.5]
    )
    ax_summ.set_xlim(-0.5, 1.5)
    ax_summ.set_xticks([0, 1])
    ax_summ.set_xticklabels(["ECE ↓", "AUROC ↑"])
    ax_summ.set_yticks(range(n_m))
    ax_summ.set_yticklabels(short_names, fontsize=8)
    ax_summ.set_title("D — Summary (ECE / AUROC)")
    for i, (e, a) in enumerate(zip(ece_vals, auroc_vals)):
        ax_summ.text(0, i, f"{e:.3f}", ha="center", va="center", fontsize=9,
                     fontweight="bold", color="black")
        ax_summ.text(1, i, f"{a:.3f}" if np.isfinite(a) else "n/a",
                     ha="center", va="center", fontsize=9, fontweight="bold", color="black")
    plt.colorbar(im_ece, ax=ax_summ, fraction=0.023, pad=0.04, label="ECE")
    plt.colorbar(im_auc, ax=ax_summ, fraction=0.023, pad=0.12, label="AUROC")

    fname = f"calibration_{task}.png"
    for d in dict.fromkeys([out_dir, fig_dir]):
        d.mkdir(parents=True, exist_ok=True)
        dpi = 300 if d == fig_dir else 150
        fig.savefig(d / fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Stale-path guard (added 2026-09-09).
#
# This script's defaults pointed at v5 artefacts. Run without explicit paths
# during v9 work it would silently produce v5 numbers under a v9 filename, which
# is the exact failure mode we spent the day removing elsewhere. Rather than
# repoint the defaults -- which would leave the same trap for whatever comes
# after v9 -- required paths must now be given explicitly.
#
# Current version and its artefacts are listed at the top of README.md.
# ---------------------------------------------------------------------------
_SUPERSEDED = ("splits_v5", "splits_v7", "large_matrix_3070", "matrix_v7_3190",
               "training_bioproject_v5", "training_bioproject_v7",
               "validation_vectors_v7")


def _refuse_superseded_paths(args) -> None:
    """Abort if any path argument still points at a superseded artefact."""
    hits = []
    for name, value in vars(args).items():
        if value is None:
            continue
        text = str(value)
        for marker in _SUPERSEDED:
            if marker in text:
                hits.append(f"  --{name.replace('_', '-')} = {text}   ({marker})")
    if hits:
        raise SystemExit(
            "Refusing to run: these arguments point at superseded artefacts.\n"
            + "\n".join(hits)
            + "\n\nv9 is current. See README.md for the paths. Pass them explicitly."
        )

def main():
    parser = argparse.ArgumentParser(
        description="OOD calibration comparison for DIANA v5 validation set"
    )
    parser.add_argument("--val-embeddings",  required=True,
                        help="val_embeddings.npz from extract_embeddings_and_logits.py")
    parser.add_argument("--train-embeddings", required=True,
                        help="train_embeddings.npz — used for energy/Mahalanobis score normalisation")
    parser.add_argument("--mahal-params",    required=True,
                        help="mahalanobis_params.npz from extract_embeddings_and_logits.py")
    parser.add_argument("--val-pred-dir",    required=True,
                        help="Dir with per-sample prediction subdirs (or path to predictions TSV for test set)")
    parser.add_argument("--predictions-tsv", default=None,
                        help="Optional: TSV file with predictions (for test set). If provided, overrides JSON loading.")
    parser.add_argument("--val-metadata",    required=True,
                        help="validation_metadata.tsv (ground truth labels)")
    parser.add_argument("--model",           required=True,
                        help="best_model.pth (for MC Dropout)")
    parser.add_argument("--label-encoders",  required=True,
                        help="label_encoders.json")
    parser.add_argument("--out-dir",         default="results/calibration_analysis")
    parser.add_argument("--figure-dir",      default="paper/figures/final")
    parser.add_argument("--skip-mc-dropout", action="store_true",
                        help="Skip MC Dropout (faster, no model re-loading)")
    parser.add_argument("--matrix",          default=None,
                        help="Optional: Matrix directory for loading fraction data (test set MC Dropout)")
    args = parser.parse_args()
    _refuse_superseded_paths(args)

    out_dir  = Path(args.out_dir)
    fig_dir  = Path(args.figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading val embeddings …")
    val_npz = np.load(args.val_embeddings, allow_pickle=True)
    val_ids = val_npz["sample_ids"].astype(str)  # Ensure string type to match metadata index
    val_emb = val_npz["embeddings"]                  # (N, 384)
    logger.info(f"  {len(val_ids)} val samples, embedding dim {val_emb.shape[1]}")

    logger.info("Loading Mahalanobis params …")
    mparams = np.load(args.mahal_params, allow_pickle=True)
    Sigma_inv = mparams["Sigma_inv"]                 # (384, 384)

    logger.info("Loading validation metadata …")
    meta = pd.read_csv(args.val_metadata, sep="\t").set_index("Run_accession")

    with open(args.label_encoders) as f:
        label_encoders = json.load(f)

    val_pred_dir = Path(args.val_pred_dir)
    predictions_tsv = Path(args.predictions_tsv) if args.predictions_tsv else None
    model_path   = Path(args.model)
    logger.info("Loading train embeddings (energy/Mahalanobis normalisation) …")
    train_npz = np.load(args.train_embeddings, allow_pickle=True)
    
    # Pre-load TSV predictions if provided (for test set)
    tsv_predictions = {}
    if predictions_tsv:
        logger.info(f"Loading predictions from TSV: {predictions_tsv}")
        for task in TASKS:
            tsv_predictions[task] = load_predictions_from_tsv(predictions_tsv, task, label_encoders[task])

    # Load MC Dropout model once (avoids reloading from disk once per task)
    mc_model = None
    if not args.skip_mc_dropout:
        logger.info("Loading model for MC Dropout …")
        mc_model = _load_mc_model(model_path)
    # ── Per-task analysis ─────────────────────────────────────────────────────
    summary_rows = []

    for task in TASKS:
        if task not in label_encoders:
            logger.info(f"Skipping {task} (not in label_encoders)")
            continue
        if f"logits_{task}" not in val_npz:
            logger.info(f"Skipping {task} (logits not in val_embeddings.npz)")
            continue

        logger.info(f"\n── Task: {task} ────────────────────────────────────────")

        # Ground-truth labels for val samples that are in metadata
        gt_labels = []
        valid_mask = []
        for sid in val_ids:
            if sid in meta.index and pd.notna(meta.loc[sid, task]):
                gt_labels.append(meta.loc[sid, task])
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        valid_mask = np.array(valid_mask)
        gt_labels  = np.array(gt_labels)
        val_ids_v  = val_ids[valid_mask]
        val_emb_v  = val_emb[valid_mask]

        # Predicted class from JSON or TSV
        pred_classes = []
        if predictions_tsv:
            # Load from TSV (test set)
            tsv_data = tsv_predictions[task]
            tsv_idx = {sid: i for i, sid in enumerate(tsv_data["sample_ids"])}
            for sid in val_ids_v:
                if sid in tsv_idx:
                    pred_classes.append(tsv_data["predicted_classes"][tsv_idx[sid]])
                else:
                    pred_classes.append("__MISSING__")
        else:
            # Load from JSON (validation set)
            for sid in val_ids_v:
                p = val_pred_dir / sid / f"{sid}_predictions.json"
                if p.exists():
                    with open(p) as f:
                        d = json.load(f)
                    pred_classes.append(d["predictions"].get(task, {}).get("predicted_class", "__MISSING__"))
                else:
                    pred_classes.append("__MISSING__")
        pred_classes = np.array(pred_classes)
        n_missing = int((pred_classes == "__MISSING__").sum())
        if n_missing > 0:
            logger.warning(
                f"  Task {task}: {n_missing}/{len(pred_classes)} samples have missing "
                f"prediction JSONs — treated as incorrect"
            )
        n_actual = int((pred_classes != "__MISSING__").sum())
        correct = (pred_classes == gt_labels).astype(float)
        logger.info(f"  {valid_mask.sum()} labeled, {n_actual} with predictions, "
                    f"accuracy {correct.mean():.3f}")

        metrics = {}

        # Metric 1: Softmax max-prob
        logits_task   = val_npz[f"logits_{task}"][valid_mask]
        probs_task    = val_npz[f"probs_{task}"][valid_mask]
        softmax_conf  = probs_task.max(axis=1)
        metrics["Softmax\nmax-prob"] = {"conf": softmax_conf, "correct": correct}

        # Metric 2: Energy score — normalised using training distribution bounds
        energy  = compute_energy_scores(logits_task)
        if f"energy_min_{task}" in train_npz and f"energy_max_{task}" in train_npz:
            e_min = float(train_npz[f"energy_min_{task}"])
            e_max = float(train_npz[f"energy_max_{task}"])
        else:
            logger.warning(
                f"energy_min/max_{task} not in train_npz — falling back to val-set "
                f"normalization. Re-run extract_embeddings.py in train mode."
            )
            e_min = float(energy.min())
            e_max = float(energy.max())
        energy_norm = np.clip((energy - e_min) / (e_max - e_min + 1e-8), 0, 1)
        metrics["Energy\nscore"] = {"conf": energy_norm, "correct": correct}

        # Metric 3: Mahalanobis — normalised using training p95 threshold
        unreliable_key = f"mahal_unreliable_{task}"
        if unreliable_key in mparams and bool(mparams[unreliable_key]):
            logger.warning(f"  Task {task}: Mahalanobis scores flagged UNRELIABLE by extract_embeddings.py")
        mu_key = f"mu_{task}"
        if mu_key in mparams:
            mu_arr     = mparams[mu_key]
            mahal_dist = compute_mahalanobis_scores(val_emb_v, mu_arr, Sigma_inv)
            p95_key    = f"threshold_95_{task}"
            train_p95  = float(mparams[p95_key]) if p95_key in mparams else float(np.percentile(mahal_dist, 95))
            # dist=0 → conf=1; dist=train_p95 → conf≈0; clip to [0,1]
            mahal_conf = np.clip(1.0 - mahal_dist / (train_p95 + 1e-8), 0, 1)
        else:
            logger.warning(f"  {mu_key} not in mahal params — using zeros")
            mahal_conf = np.zeros(len(val_ids_v))
        metrics["Mahalanobis\n(−dist)"] = {"conf": mahal_conf, "correct": correct}

        # Metric 4: MC Dropout (negate variance + normalise)
        n_mc = 0  # Track actual MC Dropout sample count
        if not args.skip_mc_dropout and mc_model is not None:
            logger.info("  Running MC Dropout …")
            
            # For test set (TSV + matrix): load fraction data from matrix
            if predictions_tsv and args.matrix:
                logger.info("  Loading fraction data from matrix for test set …")
                matrix_path = Path(args.matrix) / "unitigs.frac.mat"
                loader = MatrixLoader(matrix_path)
                features, _ = loader.load_with_metadata(
                    args.val_metadata,
                    align_to_matrix=True
                )
                n_mc = len(features)
                # Compute MC Dropout directly from features array
                mc_var_dict = {}
                for i, sid in enumerate(val_ids_v):
                    x = torch.tensor(features[i], dtype=torch.float32).unsqueeze(0)
                    variances = []
                    for _ in range(30):  # N_MC = 30
                        with torch.no_grad():
                            out_dict = mc_model(x)
                            prob = torch.softmax(out_dict[task], dim=1).cpu().numpy()[0]
                            variances.append(np.var(prob))
                    mc_var_dict[sid] = np.mean(variances)
                mc_var = mc_var_dict
            else:
                # For validation set: use individual unitig_fraction.txt files
                frac_files = [
                    val_pred_dir / sid / f"{sid}_unitig_fraction.txt"
                    for sid in val_ids_v
                    if (val_pred_dir / sid / f"{sid}_unitig_fraction.txt").exists()
                ]
                n_mc = len(frac_files)
                mc_var = compute_mc_dropout(mc_model, frac_files, task)
            mc_conf_raw = np.array([mc_var.get(sid, np.nan) for sid in val_ids_v])
            # Replace NaN with max variance (= least confident)
            nan_mask = np.isnan(mc_conf_raw)
            mc_conf_raw[nan_mask] = np.nanmax(mc_conf_raw) if not nan_mask.all() else 1.0
            
            # Normalize using training bounds if available
            if f"mc_var_min_{task}" in train_npz and f"mc_var_max_{task}" in train_npz:
                mc_min = float(train_npz[f"mc_var_min_{task}"])
                mc_max = float(train_npz[f"mc_var_max_{task}"])
                mc_conf = np.clip((-mc_conf_raw - (-mc_max)) / ((-mc_min) - (-mc_max) + 1e-8), 0, 1)
                logger.info(f"  MC Dropout normalized using training bounds (min={mc_min:.4f}, max={mc_max:.4f})")
            else:
                mc_conf = minmax(-mc_conf_raw)
                logger.warning(
                    f"MC Dropout normalised on val set — scores not comparable across runs. "
                    f"Consider running extract_embeddings.py with --compute-mc-bounds in train mode."
                )
        else:
            mc_conf = np.zeros(len(val_ids_v))
            logger.info("  MC Dropout skipped (--skip-mc-dropout)")
        
        # Only add MC Dropout to metrics if it was actually computed
        if n_mc > 0:
            metrics["MC Dropout\n(−variance)"] = {"conf": mc_conf, "correct": correct}

        # ── Plot ──────────────────────────────────────────────────────────────
        plot_task(task, metrics, out_dir, fig_dir)

        # ── Collect summary ─────────────────────────────────────────────────
        for metric_name, mdata in metrics.items():
            ece   = expected_calibration_error(mdata["conf"], mdata["correct"])
            auroc = auroc_wrong_prediction(mdata["conf"], mdata["correct"])
            # Use n_mc for MC Dropout, n_actual for others
            n_samples = n_mc if "MC Dropout" in metric_name else n_actual
            summary_rows.append({
                "task":     task,
                "metric":   metric_name.replace("\n", " "),
                "n":        n_samples,
                "accuracy": correct.mean(),
                "ECE":      ece,
                "AUROC":    auroc,
            })

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = pd.DataFrame(summary_rows)
    csv_path = out_dir / "calibration_summary.csv"
    summary.to_csv(csv_path, index=False)
    logger.info(f"\nSaved summary → {csv_path}")
    logger.info("\n" + summary.pivot_table(
        index="metric", columns="task", values="ECE"
    ).to_string())

    # ── Global summary figure ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Calibration Summary — all tasks", fontsize=13, fontweight="bold")

    for ax, metric_name, title in [
        (axes[0], "ECE", "Expected Calibration Error ↓"),
        (axes[1], "AUROC", "AUROC (wrong-prediction detection) ↑"),
    ]:
        pivot = summary.pivot_table(index="metric", columns="task", values=metric_name)
        im = ax.imshow(pivot.values, aspect="auto",
                       cmap="RdYlGn_r" if metric_name == "ECE" else "RdYlGn",
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([m.replace("\n", " ") for m in pivot.index], fontsize=8)
        ax.set_title(title)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=8, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    for d in dict.fromkeys([out_dir, fig_dir]):
        d.mkdir(parents=True, exist_ok=True)
        dpi = 300 if d == fig_dir else 150
        fig.savefig(d / "calibration_summary.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved calibration_summary.png")


if __name__ == "__main__":
    main()
