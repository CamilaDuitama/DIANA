#!/usr/bin/env python3
"""
28_train_sample_inference.py

INVESTIGATION SCRIPT — NOT FOR PAPER

Purpose:
    Run the trained DIANA model directly on Philips2017 (PRJNA354503),
    Bozzi2024 (PRJEB18722), and Jackson2024 (PRJEB64128) samples, using:
      - train/test samples: feature vectors extracted from the large matrix
      - validation samples: pre-computed predictions from results/validation_predictions/
        (validation FASTAs were not included in the 3070-sample matrix)

    Diagnostic question:
        Does the model correctly predict its OWN training samples from these
        studies, or does it also fail?

        If it fails on training samples → strong evidence of label noise /
        contamination in the AMD metadata (samples are genuinely unusual).
        If it succeeds on training but fails on val/test → the val/test subset
        contains different or contaminated samples.

Usage:
    python scripts/paper/28_train_sample_inference.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from diana.models.multitask_mlp import MultiTaskMLP

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MATRIX_DIR  = REPO / "data/matrices/large_matrix_3070_with_frac"
MATRIX_FILE = MATRIX_DIR / "unitigs.frac.mat"
FOF_FILE    = MATRIX_DIR / "kmer_matrix/kmtricks.fof"
CHECKPOINT  = REPO / "results/training/best_model.pth"
LE_FILE     = REPO / "results/training/label_encoders.json"
TRAIN_META  = REPO / "paper/metadata/train_metadata.tsv"
VAL_META    = REPO / "paper/metadata/val_metadata.tsv"
TEST_META   = REPO / "paper/metadata/test_metadata.tsv"

TASKS = ["sample_type", "community_type", "sample_host", "material"]

TARGET_BIOPROJECTS = {
    "PRJNA354503": "Philips2017",
    "PRJEB18722":  "Bozzi2024",
    "PRJEB64128":  "Jackson2024",
}
# Jackson2024 has NO training samples and validation samples are not in the
# 3070-sample matrix — handled separately via pre-computed val predictions.
MATRIX_ONLY_BIOPROJECTS = {"PRJNA354503", "PRJEB18722"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_label_encoders():
    with open(LE_FILE) as f:
        raw = json.load(f)
    return {task: raw[task]["classes"] for task in TASKS}


def load_fof_sample_ids(fof_path: Path) -> list[str]:
    """Parse kmtricks.fof → ordered list of sample IDs."""
    ids = []
    with open(fof_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split(" : ")[0].strip())
    return ids


def find_metadata(sample_ids: list[str]) -> pd.DataFrame:
    """Load train/val/test metadata and return rows for given sample IDs."""
    rows = []
    for path, split in [
        (TRAIN_META, "train"),
        (REPO / "paper/metadata/validation_metadata.tsv", "val"),
        (REPO / "paper/metadata/test_metadata.tsv", "test"),
    ]:
        if path.exists():
            df = pd.read_csv(path, sep="\t", low_memory=False)
            df["split"] = split
            rows.append(df)
        else:
            print(f"  [warn] metadata not found: {path}")
    meta = pd.concat(rows, ignore_index=True)
    return meta[meta["Run_accession"].isin(sample_ids)].drop_duplicates("Run_accession")


def load_model(le: dict) -> MultiTaskMLP:
    """Reconstruct and load the trained model from checkpoint."""
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    # Infer architecture from state dict
    input_dim   = state["backbone.0.weight"].shape[1]
    # Find all backbone linear layer weight keys (no BN keys present → use_batch_norm=False)
    bb_linear_keys = sorted(
        [k for k in state if k.startswith("backbone.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1])
    )
    hidden_dims = [state[k].shape[0] for k in bb_linear_keys]

    # Detect if BN was used: BN keys have "running_mean"
    use_bn = any("running_mean" in k for k in state if k.startswith("backbone."))

    # Head output layer index depends on whether BN is used in heads
    head_out_idx = 3  # heads: Linear, act, dropout, Linear
    num_classes = {task: state[f"heads.{task}.{head_out_idx}.weight"].shape[0] for task in TASKS}

    print(f"  Architecture: input={input_dim}, hidden={hidden_dims}, use_bn={use_bn}, num_classes={num_classes}")
    print(f"  Checkpoint epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    model = MultiTaskMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=0.5,          # value at inference doesn't matter (eval mode)
        use_batch_norm=use_bn,
        activation="relu",
    )
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(model: MultiTaskMLP,
                  X: np.ndarray,
                  le: dict) -> dict[str, np.ndarray]:
    """Run forward pass, return {task: array of predicted class names}."""
    tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = model(tensor)
    preds = {}
    for task in TASKS:
        probs = torch.softmax(logits[task], dim=1).numpy()
        idx   = probs.argmax(axis=1)
        names = np.array(le[task])
        preds[task]           = names[idx]
        preds[f"{task}_conf"] = probs.max(axis=1)
    return preds


def load_validation_predictions_for_bioprojects(bioprojects: set) -> pd.DataFrame:
    """
    Load pre-computed per-sample validation predictions from
    results/validation_predictions/ and filter to target BioProjects.
    Returns a long-form DataFrame with columns:
        Run_accession, BioProject, task, true_label, predicted, confidence, split
    """  
    sys.path.insert(0, str(REPO / "scripts" / "validation"))
    from load_validation_data import load_validation_predictions

    df = load_validation_predictions(quiet=True)
    # df columns: sample_id, task, true_label, pred_label, confidence, is_correct

    val_meta = pd.read_csv(REPO / "paper/metadata/validation_metadata.tsv",
                           sep="\t", low_memory=False)
    meta_slim = val_meta[["Run_accession", "BioProject", "project_name",
                          "sample_type", "community_type", "sample_host", "material"
                          ]].drop_duplicates("Run_accession")

    df = df.rename(columns={"sample_id": "Run_accession", "pred_label": "predicted"})
    df = df.merge(meta_slim, on="Run_accession", how="left")
    df = df[df["BioProject"].isin(bioprojects)].copy()
    df["split"] = "val"
    return df[["Run_accession", "BioProject", "project_name", "split",
               "task", "true_label", "predicted", "confidence", "is_correct"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_split_results(sp_df: pd.DataFrame, split: str,
                         pred_suffix: str = "_pred", conf_suffix: str = "_conf",
                         wide_format: bool = True):
    """
    Print per-task accuracy for one split.
    wide_format=True  → columns {task}, {task}_pred, {task}_conf  (matrix inference)
    wide_format=False → long format: task / true_label / predicted / confidence (val predictions)
    """
    n_total = sp_df["Run_accession"].nunique() if not wide_format else len(sp_df)
    print(f"\n  --- {split.upper()} ({n_total} samples) ---")

    for task in TASKS:
        if wide_format:
            pred_col = f"{task}{pred_suffix}"
            conf_col = f"{task}{conf_suffix}"
            true_col = task
            if pred_col not in sp_df.columns:
                continue
            correct_mask = sp_df[pred_col] == sp_df[true_col]
            n_correct = correct_mask.sum()
            n_task    = len(sp_df)
            confs     = sp_df[conf_col]
            errors    = sp_df[~correct_mask].copy()
            err_group_cols = [true_col, pred_col]
            err_true_col   = true_col
            err_pred_col   = pred_col
            err_conf_col   = conf_col
        else:
            task_df   = sp_df[sp_df["task"] == task].copy()
            if task_df.empty:
                continue
            correct_mask = task_df["is_correct"]
            n_correct = correct_mask.sum()
            n_task    = len(task_df)
            confs     = task_df["confidence"]
            errors    = task_df[~correct_mask].copy()
            err_group_cols = ["true_label", "predicted"]
            err_true_col   = "true_label"
            err_pred_col   = "predicted"
            err_conf_col   = "confidence"

        acc = n_correct / n_task * 100 if n_task else 0
        print(f"\n    Task: {task}  [{n_correct}/{n_task}  {acc:.0f}% correct]")
        if errors.empty:
            print(f"      ✓ All correct")
        else:
            conf_pairs = (
                errors.groupby(err_group_cols)
                .agg(n=("Run_accession", "count"),
                     mean_conf=(err_conf_col, "mean"))
                .reset_index()
                .sort_values("n", ascending=False)
            )
            for _, row in conf_pairs.iterrows():
                print(f"      TRUE={str(row[err_true_col]):<30}  "
                      f"PRED={str(row[err_pred_col]):<30}  "
                      f"n={row['n']}  conf={row['mean_conf']:.2f}")


def main():
    print("=" * 80)
    print("DIANA — Inference for Philips2017, Bozzi2024, Jackson2024")
    print("=" * 80)

    le = load_label_encoders()

    # ── 1. Read sample IDs from FOF ──────────────────────────────────────────
    print("\n[1] Loading sample ID order from kmtricks.fof …")
    fof_ids = load_fof_sample_ids(FOF_FILE)
    print(f"    {len(fof_ids)} samples in matrix")
    id_to_col = {sid: i for i, sid in enumerate(fof_ids)}

    # ── 2. Load metadata for matrix-available BioProjects (train/test) ────────
    print("\n[2] Loading metadata …")
    meta = find_metadata(fof_ids)
    target_meta = meta[meta["BioProject"].isin(MATRIX_ONLY_BIOPROJECTS)].copy()
    print(f"    Found {len(target_meta)} train/test rows in matrix:")
    print(target_meta.groupby(["BioProject", "split"]).size().to_string())

    # ── 3. Extract feature matrix rows for target samples ────────────────────
    target_ids   = target_meta["Run_accession"].tolist()
    target_cols  = [id_to_col[s] for s in target_ids if s in id_to_col]
    missing      = [s for s in target_ids if s not in id_to_col]

    if missing:
        print(f"\n    [warn] {len(missing)} target samples NOT in matrix: {missing[:5]}")

    found_ids = [s for s in target_ids if s in id_to_col]
    found_cols = [id_to_col[s] for s in found_ids]

    print(f"\n[3] Reading feature matrix for {len(found_ids)} samples …")
    print("    (reading full matrix then selecting columns — may take ~1-2 min)")

    # Read with polars — matrix is features × samples (transpose needed)
    import polars as pl
    mat = pl.read_csv(
        MATRIX_FILE,
        separator=" ",
        has_header=False,
        infer_schema_length=0,  # read as strings first → faster schema
    )
    # col 0 = feature ID (string), cols 1..N = sample values
    # select only the columns we need (1-indexed in polars, but col names are "column_1" etc.)
    # polars names: column_1=feature_id, column_2=sample_1,  column_{col+2}=sample_{col}
    needed_pl_cols = [f"column_{c + 2}" for c in found_cols]  # +2: col 0-indexed but polars is 1-indexed for naming, +1 for feature col
    sub = mat.select(needed_pl_cols)

    # Convert to float numpy, shape = (n_features, n_target_samples) → transpose
    X = sub.cast(pl.Float32).to_numpy().T  # shape: (n_target_samples, n_features)
    print(f"    X shape: {X.shape}")

    # ── 4. Load model ─────────────────────────────────────────────────────────
    print("\n[4] Loading model …")
    model = load_model(le)

    # ── 5. Run inference ──────────────────────────────────────────────────────
    print("\n[5] Running inference …")
    preds = run_inference(model, X, le)

    # ── 6. Build results table ────────────────────────────────────────────────
    results = target_meta[target_meta["Run_accession"].isin(found_ids)].copy()
    results = results.set_index("Run_accession").loc[found_ids].reset_index()

    for task in TASKS:
        results[f"{task}_pred"] = preds[task]
        results[f"{task}_conf"] = preds[f"{task}_conf"]

    # ── 7. Load pre-computed validation predictions ───────────────────────────
    print("\n[7] Loading pre-computed validation predictions …")
    val_preds = load_validation_predictions_for_bioprojects(set(TARGET_BIOPROJECTS.keys()))
    print(f"    Found {val_preds['Run_accession'].nunique()} validation samples across BioProjects:")
    print(val_preds.groupby("BioProject")["Run_accession"].nunique().to_string())

    # ── 8. Print results: all studies, all splits ─────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for bp, bp_name in TARGET_BIOPROJECTS.items():
        n_train_test = len(results[results["BioProject"] == bp]) if bp in MATRIX_ONLY_BIOPROJECTS else 0
        n_val        = val_preds[val_preds["BioProject"] == bp]["Run_accession"].nunique()
        print(f"\n{'=' * 70}")
        print(f"  {bp_name} ({bp})")
        parts = []
        if n_train_test: parts.append(f"{n_train_test} train/test (matrix inference)")
        if n_val:        parts.append(f"{n_val} val (pre-computed predictions)")
        print("  " + "  |  ".join(parts))
        print(f"{'=' * 70}")

        # --- train / test splits (matrix inference, wide format) ---
        if bp in MATRIX_ONLY_BIOPROJECTS:
            bp_df = results[results["BioProject"] == bp].copy()
            for split in ["train", "test"]:
                sp_df = bp_df[bp_df["split"] == split]
                if sp_df.empty:
                    continue
                print_split_results(sp_df, split, wide_format=True)

        # --- validation split (pre-computed, long format) ---
        val_bp = val_preds[val_preds["BioProject"] == bp]
        if not val_bp.empty:
            print_split_results(val_bp, "val", wide_format=False)
        elif bp not in MATRIX_ONLY_BIOPROJECTS:
            print("\n  --- VAL — no pre-computed predictions found ---")

    # ── 9. Full per-sample CSV (matrix-inferred samples only) ─────────────────
    out_path = REPO / "results" / "train_sample_inference_results.tsv"
    cols_out = ["Run_accession", "BioProject", "split"] + TASKS + [
        f"{t}_pred" for t in TASKS
    ] + [f"{t}_conf" for t in TASKS]
    cols_out = [c for c in cols_out if c in results.columns]
    results[cols_out].to_csv(out_path, sep="\t", index=False)
    print(f"\n\nFull results (train/test) written to: {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
