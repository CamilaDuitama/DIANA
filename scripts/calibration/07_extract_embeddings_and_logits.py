#!/usr/bin/env python3
"""
Extract backbone embeddings and raw logits from the v5 DIANA model.

Runs each sample's unitig_fraction.txt through the model backbone and
all task heads, storing:
  - 384-dim embedding z from the backbone
  - raw logits (pre-softmax) for each task head
  - softmax probabilities for each task head

Used as input for:
  - Fitting Mahalanobis parameters on training set
  - Computing energy scores on any set
  - MC Dropout uncertainty estimation

Usage:
  # Training set (fits Mahalanobis params too)
  python extract_embeddings_and_logits.py \\
      --mode train \\
      --model results/training_bioproject_v5/best_model.pth \\
      --metadata data/splits_v5/train_metadata.tsv \\
      --matrix data/matrices/large_matrix_3070_with_frac \\
      --output results/training_bioproject_v5/train_embeddings.npz \\
      --fit-mahalanobis results/training_bioproject_v5/mahalanobis_params.npz

  # Validation set
  python extract_embeddings_and_logits.py \\
      --mode val \\
      --model results/training_bioproject_v5/best_model.pth \\
      --val-pred-dir results/validation_predictions_bioproject_v5_full \\
      --output results/validation_predictions_bioproject_v5_full/val_embeddings.npz
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import chi2
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from diana.inference.predictor import Predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_FEATURES = 107_480  # Expected number of unitig features


# ─── Model loading with embedding hook ───────────────────────────────────────

def load_model_with_hooks(model_path: Path, device: str = "cpu"):
    """Load model and register a hook to capture backbone output."""
    predictor = Predictor(model_path, device=device)
    model = predictor.model
    model.eval()

    if not hasattr(model, "backbone"):
        available = [n for n, _ in model.named_children()]
        raise AttributeError(
            f"Model has no 'backbone' attribute. Available children: {available}"
        )

    captured = {}

    def hook_fn(module, input, output):
        captured["embedding"] = output.detach().cpu()

    # Register hook on the backbone (last module before heads)
    handle = model.backbone.register_forward_hook(hook_fn)

    return model, predictor, captured, handle


def load_fraction_file(path: Path) -> np.ndarray:
    """Load a unitig_fraction.txt file → float32 array of shape (N_FEATURES,)."""
    data = np.fromfile(path, sep="\n", dtype=np.float32)
    assert len(data) == N_FEATURES, f"Expected {N_FEATURES} features, got {len(data)} for {path}"
    return data


def run_inference_with_embedding(model, captured, x_tensor, device):
    """
    Forward pass returning:
      - embedding: (D,) backbone output
      - logits: dict[task -> (C,) array]
      - probs:  dict[task -> (C,) array]
    """
    captured.clear()  # Clear stale embeddings from previous samples
    with torch.no_grad():
        outputs = model(x_tensor.to(device))

    if "embedding" not in captured:
        available = [n for n, _ in model.named_children()]
        raise RuntimeError(
            "Backbone hook did not fire. Verify that model.backbone is directly "
            f"called during forward(). Available children: {available}"
        )
    embedding = captured["embedding"].squeeze(0).numpy()  # (D,)

    logits = {}
    probs  = {}
    for task, logit_tensor in outputs.items():
        l = logit_tensor.squeeze(0).cpu()
        logits[task] = l.numpy()
        probs[task]  = F.softmax(l, dim=0).numpy()

    return embedding, logits, probs


# ─── Training mode ────────────────────────────────────────────────────────────

def extract_train(args):
    """Extract embeddings from training set using the frac.mat matrix columns."""
    from diana.data.loader import MatrixLoader

    logger.info("Loading training matrix …")
    # MatrixLoader expects path to .mat file, not directory
    matrix_file = Path(args.matrix) / "unitigs.frac.mat"
    loader = MatrixLoader(matrix_file)
    # Use load_with_metadata to filter matrix to only samples in metadata (like training script)
    features, meta_pl = loader.load_with_metadata(
        metadata_path=Path(args.metadata),
        align_to_matrix=True
    )
    sample_ids = meta_pl["Run_accession"].to_list()
    logger.info(f"Matrix: {features.shape[0]} samples × {features.shape[1]} features")

    # Convert metadata to pandas for compatibility
    meta = meta_pl.to_pandas().set_index("Run_accession")
    label_cols = [c for c in ["sample_type","community_type","sample_host","material"]
                  if c in meta.columns]
    logger.info(f"Label columns: {label_cols}")

    model, predictor, captured, hook_handle = load_model_with_hooks(Path(args.model))
    device = "cpu"
    task_n_classes  = {t: model.heads[t][-1].out_features for t in model.heads}
    fallback_counts = {t: 0 for t in model.heads}

    embeddings  = []
    all_logits  = {task: [] for task in model.heads.keys()}
    all_probs   = {task: [] for task in model.heads.keys()}
    all_labels  = {col: []  for col in label_cols}
    valid_ids   = []

    for i, sid in enumerate(tqdm(sample_ids, desc="Train embeddings")):
        x = torch.tensor(features[i], dtype=torch.float32).unsqueeze(0)
        emb, logits, probs = run_inference_with_embedding(model, captured, x, device)

        embeddings.append(emb)
        for task in all_logits:
            if task in logits:
                all_logits[task].append(logits[task])
                all_probs[task].append(probs[task])
            else:
                all_logits[task].append(np.zeros(task_n_classes[task]))
                all_probs[task].append(np.zeros(task_n_classes[task]))
                fallback_counts[task] += 1

        for col in label_cols:
            val = meta.loc[sid, col] if sid in meta.index else np.nan
            all_labels[col].append(val)

        valid_ids.append(sid)

    hook_handle.remove()
    if len(embeddings) == 0:
        logger.error("No valid embeddings extracted — all samples failed")
        embeddings = np.empty((0, model.backbone.out_features if hasattr(model.backbone, 'out_features') else 384))
    else:
        embeddings = np.array(embeddings)  # (N, D)
    logger.info(f"Extracted {len(valid_ids)} embeddings, shape {embeddings.shape}")
    for task, count in fallback_counts.items():
        if count > 0:
            logger.warning(f"  Task {task}: fallback zero-logits used for {count} samples")

    # Build save dict
    save_dict = {
        "sample_ids": np.array(valid_ids),
        "embeddings": embeddings,
    }
    for task in all_logits:
        save_dict[f"logits_{task}"] = np.array(all_logits[task])
        save_dict[f"probs_{task}"]  = np.array(all_probs[task])
    for col in label_cols:
        save_dict[f"label_{col}"] = np.array(all_labels[col])

    # Save training energy min/max per task for consistent normalisation downstream
    for task in all_logits:
        train_energy = logsumexp(save_dict[f"logits_{task}"], axis=1)
        save_dict[f"energy_min_{task}"] = float(train_energy.min())
        save_dict[f"energy_max_{task}"] = float(train_energy.max())

    # Optionally compute MC Dropout variance bounds on training set
    # This allows consistent normalization in calibration analysis (script 08)
    if hasattr(args, 'compute_mc_bounds') and args.compute_mc_bounds:
        logger.info("Computing MC Dropout variance bounds on training set (30 passes per sample) …")
        from diana.models.multitask import DianaMultiTaskModel
        
        # Create MC model (eval mode but with dropout active)
        mc_model = DianaMultiTaskModel(
            n_features=N_FEATURES,
            backbone_dim=predictor.model.backbone.out_features,
            task_configs=predictor.model.task_configs,
            dropout_p=0.2
        ).to(device)
        mc_model.load_state_dict(model.state_dict())
        mc_model.eval()
        
        # Enable dropout during inference
        for module in mc_model.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()
        
        task_variances = {task: [] for task in model.heads.keys()}
        for i in tqdm(range(len(features)), desc="MC Dropout on training"):
            x = torch.tensor(features[i], dtype=torch.float32).unsqueeze(0).to(device)
            task_samples = {task: [] for task in model.heads.keys()}
            
            with torch.no_grad():
                for _ in range(30):  # N_MC passes
                    outputs = mc_model(x)
                    for task in model.heads.keys():
                        probs = F.softmax(outputs[task], dim=1).cpu().numpy()[0]
                        task_samples[task].append(probs)
            
            for task in model.heads.keys():
                variance = np.var(task_samples[task], axis=0).max()  # Max across classes
                task_variances[task].append(variance)
        
        # Save variance min/max per task
        for task in task_variances:
            variances_arr = np.array(task_variances[task])
            save_dict[f"mc_var_min_{task}"] = float(variances_arr.min())
            save_dict[f"mc_var_max_{task}"] = float(variances_arr.max())
        logger.info("MC Dropout variance bounds saved")
    else:
        logger.info("Skipping MC Dropout variance bounds (use --compute-mc-bounds to enable)")

    np.savez_compressed(args.output, **save_dict)
    logger.info(f"Saved train embeddings → {args.output}")

    # Fit Mahalanobis parameters
    if args.fit_mahalanobis:
        fit_mahalanobis(embeddings, all_labels, label_cols, args.fit_mahalanobis)


# ─── Validation mode ──────────────────────────────────────────────────────────

def extract_val(args):
    """Extract embeddings from completed validation prediction dirs."""
    val_pred_dir = Path(args.val_pred_dir)
    pred_jsons = sorted(val_pred_dir.glob("*/*_predictions.json"))
    logger.info(f"Found {len(pred_jsons)} completed prediction JSONs")

    model, predictor, captured, hook_handle = load_model_with_hooks(Path(args.model))
    device = "cpu"
    task_n_classes  = {t: model.heads[t][-1].out_features for t in model.heads}
    fallback_counts = {t: 0 for t in model.heads}

    embeddings = []
    all_logits  = {task: [] for task in model.heads.keys()}
    all_probs   = {task: [] for task in model.heads.keys()}
    valid_ids   = []

    for pred_json in tqdm(pred_jsons, desc="Val embeddings"):
        sid = pred_json.parent.name
        frac_file = pred_json.parent / f"{sid}_unitig_fraction.txt"

        if not frac_file.exists():
            logger.warning(f"Fraction file missing: {frac_file}")
            continue

        try:
            x = torch.tensor(load_fraction_file(frac_file), dtype=torch.float32).unsqueeze(0)
            emb, logits, probs = run_inference_with_embedding(model, captured, x, device)
        except Exception as e:
            logger.warning(f"Failed {sid}: {e}")
            continue

        embeddings.append(emb)
        for task in all_logits:
            if task in logits:
                all_logits[task].append(logits[task])
                all_probs[task].append(probs[task])
            else:
                all_logits[task].append(np.zeros(task_n_classes[task]))
                all_probs[task].append(np.zeros(task_n_classes[task]))
                fallback_counts[task] += 1
        valid_ids.append(sid)

    hook_handle.remove()
    if len(embeddings) == 0:
        logger.error("No valid val embeddings extracted — all samples failed")
        embeddings = np.empty((0, model.backbone.out_features if hasattr(model.backbone, 'out_features') else 384))
    else:
        embeddings = np.array(embeddings)
    logger.info(f"Extracted {len(valid_ids)} val embeddings, shape {embeddings.shape}")
    for task, count in fallback_counts.items():
        if count > 0:
            logger.warning(f"  Task {task}: fallback zero-logits used for {count} samples")

    save_dict = {
        "sample_ids": np.array(valid_ids),
        "embeddings": embeddings,
    }
    for task in all_logits:
        save_dict[f"logits_{task}"] = np.array(all_logits[task])
        save_dict[f"probs_{task}"]  = np.array(all_probs[task])

    np.savez_compressed(args.output, **save_dict)
    logger.info(f"Saved val embeddings → {args.output}")


# ─── Mahalanobis fitting ──────────────────────────────────────────────────────

def _is_valid_label(lbl) -> bool:
    """Return True if lbl is a real (non-NaN, non-'nan') label."""
    return pd.notna(lbl) and str(lbl).lower() != "nan" and str(lbl).strip() != ""


def fit_mahalanobis(embeddings: np.ndarray, all_labels: dict, label_cols: List[str],
                    output_path: str):
    """
    Fit Mahalanobis parameters on training embeddings.

    Covariance is fitted using ONE anchor task (sample_type or first available)
    to avoid biasing the pooled covariance by mixing centerings from different
    class structures across tasks.

    For each task:
      - Compute per-class centroids (mu[task][class_idx])
      - Vectorised minimum Mahalanobis distance to nearest centroid
      - Empirical 95th-percentile threshold + chi² df-based threshold

    Saves: mu_<task>, classes_<task>, Sigma_inv, threshold_95_<task>,
           chi2_threshold_<task>, mahal_unreliable_<task>
    """
    logger.info("Fitting Mahalanobis parameters …")
    save_dict = {}

    # ── Per-task centroids ────────────────────────────────────────────────────
    for col in label_cols:
        labels  = np.array(all_labels[col])
        valid   = np.array([_is_valid_label(l) for l in labels])
        classes = sorted(set(labels[valid]))
        logger.info(f"  Task {col}: {len(classes)} classes, {valid.sum()} labeled samples")

        centroids  = {}
        n_skipped  = 0
        for cls in classes:
            mask = (labels == cls) & valid
            if mask.sum() < 2:
                logger.warning(f"    Skipping class '{cls}' (n={mask.sum()})")
                n_skipped += 1
                continue
            centroids[cls] = embeddings[mask].mean(axis=0)

        unreliable = bool(len(classes) > 0 and n_skipped / len(classes) > 0.5)
        if unreliable:
            logger.warning(
                f"  Task {col}: {n_skipped}/{len(classes)} classes skipped — "
                f"Mahalanobis scores for this task are UNRELIABLE"
            )
        save_dict[f"mahal_unreliable_{col}"] = np.bool_(unreliable)

        if len(centroids) == 0:
            save_dict[f"mu_{col}"]      = np.empty((0, embeddings.shape[1]))
            save_dict[f"classes_{col}"] = np.array([], dtype=object)
        else:
            save_dict[f"mu_{col}"]      = np.array([centroids[c] for c in sorted(centroids)])
            save_dict[f"classes_{col}"] = np.array(sorted(centroids.keys()))

    # ── Pooled covariance: fit on ONE anchor task only ────────────────────────
    anchor_task = "sample_type" if "sample_type" in label_cols else label_cols[0]
    logger.info(f"  Fitting pooled covariance using anchor task: '{anchor_task}'")

    labels   = np.array(all_labels[anchor_task])
    valid    = np.array([_is_valid_label(l) for l in labels])
    mu_arr   = save_dict[f"mu_{anchor_task}"]

    all_centered = []
    for cls_name, mu in zip(save_dict[f"classes_{anchor_task}"], mu_arr):
        mask = (labels == cls_name) & valid
        if mask.sum() >= 2:
            all_centered.append(embeddings[mask] - mu)

    if not all_centered:
        logger.error("No labeled samples found for anchor task — cannot fit covariance")
        return

    Z_centered = np.vstack(all_centered)
    logger.info(f"  LedoitWolf on {Z_centered.shape[0]} centered embeddings …")
    lw = LedoitWolf(assume_centered=True)
    lw.fit(Z_centered)
    Sigma_inv = lw.precision_
    save_dict["Sigma_inv"]      = Sigma_inv
    save_dict["anchor_task"]    = anchor_task
    logger.info(f"  Shrinkage coefficient: {lw.shrinkage_:.4f}")

    # ── Per-task thresholds (vectorised) ─────────────────────────────────────
    D = embeddings.shape[1]
    for col in label_cols:
        labels  = np.array(all_labels[col])
        valid   = np.array([_is_valid_label(l) for l in labels])
        mu_arr  = save_dict[f"mu_{col}"]

        # Vectorised min-distance over all centroids
        scores_all = np.full(embeddings.shape[0], np.inf)
        for mu in mu_arr:
            diff = embeddings - mu[np.newaxis, :]
            tmp  = diff @ Sigma_inv
            d2   = (tmp * diff).sum(axis=1)
            scores_all = np.minimum(scores_all, np.sqrt(np.maximum(d2, 0)))

        scores = scores_all[valid]
        if np.all(np.isinf(scores)):
            logger.error(
                f"Task {col}: all Mahalanobis scores are inf — no valid centroids. "
                f"Skipping threshold."
            )
            save_dict[f"threshold_95_{col}"]   = float("inf")
            # chi2_threshold in distance units (sqrt of d²)
            save_dict[f"chi2_threshold_{col}"] = float(np.sqrt(chi2.ppf(0.95, df=D)))
            continue
        # Both thresholds are in distance units (sqrt of d²), not squared distance
        threshold_95   = float(np.percentile(scores, 95))
        chi2_threshold = float(np.sqrt(chi2.ppf(0.95, df=D)))

        save_dict[f"threshold_95_{col}"]   = threshold_95
        save_dict[f"chi2_threshold_{col}"] = chi2_threshold
        logger.info(
            f"  {col}: empirical_95={threshold_95:.4f}, "
            f"chi2_95(df={D}, distance)={chi2_threshold:.4f}"
        )

    np.savez_compressed(output_path, **save_dict)
    logger.info(f"Saved Mahalanobis params → {output_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────


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
    parser = argparse.ArgumentParser(description="Extract DIANA model embeddings and logits")
    parser.add_argument("--mode", choices=["train","val"], required=True)
    parser.add_argument("--model", required=True, help="Path to best_model.pth")
    parser.add_argument("--output", required=True, help="Output .npz file")

    # Train mode
    parser.add_argument("--metadata", help="[train] TSV with Run_accession + label cols")
    parser.add_argument("--matrix",   help="[train] MUSET matrix dir (unitigs.frac.mat)")
    parser.add_argument("--fit-mahalanobis", dest="fit_mahalanobis",
                        help="[train] Save Mahalanobis params to this .npz path")
    parser.add_argument("--compute-mc-bounds", dest="compute_mc_bounds", action="store_true",
                        help="[train] Compute MC Dropout variance bounds (30 passes/sample, slow)")

    # Val mode
    parser.add_argument("--val-pred-dir", dest="val_pred_dir",
                        help="[val] Dir containing per-sample prediction subdirs")

    args = parser.parse_args()

    _refuse_superseded_paths(args)

    if args.mode == "train":
        if not args.metadata or not args.matrix:
            parser.error("--metadata and --matrix required for train mode")
        extract_train(args)
    else:
        if not args.val_pred_dir:
            parser.error("--val-pred-dir required for val mode")
        extract_val(args)


if __name__ == "__main__":
    main()
