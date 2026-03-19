#!/usr/bin/env python3
"""
29_train_predictions_for_table.py

Run the trained DIANA model on the training samples that belong to
BioProjects shown in sup_table_09 (only 5 out of 15 table BioProjects
have any training samples at all).

All 621 samples across these 5 BioProjects are in the 3070-sample matrix.
153 of them (PRJNA354503) were already inferred; this script re-infers all
621 for consistency and saves to:

    results/train_evaluation/train_predictions.tsv

Output format matches results/test_evaluation/test_predictions.tsv:
    Run_accession, BioProject, project_name,
    {task}_true, {task}_pred, {task}_conf  (one column per task)
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

MATRIX_FILE = REPO / "data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat"
FOF_FILE    = REPO / "data/matrices/large_matrix_3070_with_frac/kmer_matrix/kmtricks.fof"
CHECKPOINT  = REPO / "results/training/best_model.pth"
LE_FILE     = REPO / "results/training/label_encoders.json"
TRAIN_META  = REPO / "paper/metadata/train_metadata.tsv"
OUT_TSV     = REPO / "results/train_evaluation/train_predictions.tsv"

TASKS = ["sample_type", "community_type", "sample_host", "material"]

# Only the BioProjects from the sup_table_09 top-15 that have training samples
TABLE_BIOPROJECTS = {
    "PRJEB33848",    # Neukamm2020    — 113 train samples
    "PRJEB34569",    # FellowsYates2021 — 134 train samples
    "PRJEB41240",    # SeguinOrlando2021 — 185 train samples
    "PRJEB49638",    # Moraitou2022    — 56 train samples
    "PRJNA354503",   # Philips2017     — 133 train samples
}


def load_label_encoders():
    with open(LE_FILE) as f:
        raw = json.load(f)
    return {task: raw[task]["classes"] for task in TASKS}


def load_fof_sample_ids(fof_path: Path) -> list:
    ids = []
    with open(fof_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split(" : ")[0].strip())
    return ids


def load_model(le: dict) -> MultiTaskMLP:
    ckpt  = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    bb_keys     = sorted([k for k in state if k.startswith("backbone.") and k.endswith(".weight")],
                         key=lambda k: int(k.split(".")[1]))
    hidden_dims = [state[k].shape[0] for k in bb_keys]
    input_dim   = state["backbone.0.weight"].shape[1]
    use_bn      = any("running_mean" in k for k in state if k.startswith("backbone."))
    num_classes = {t: state[f"heads.{t}.3.weight"].shape[0] for t in TASKS}

    print(f"  input={input_dim}, hidden={hidden_dims}, use_bn={use_bn}")
    print(f"  epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    model = MultiTaskMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=0.5,
        use_batch_norm=use_bn,
        activation="relu",
    )
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(model: MultiTaskMLP, X: np.ndarray, le: dict) -> dict:
    tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = model(tensor)
    preds = {}
    for task in TASKS:
        probs = torch.softmax(logits[task], dim=1).numpy()
        idx   = probs.argmax(axis=1)
        preds[task]           = np.array(le[task])[idx]
        preds[f"{task}_conf"] = probs.max(axis=1)
    return preds


def main():
    print("=" * 70)
    print("DIANA — Training-set inference for sup_table_09 BioProjects")
    print("=" * 70)

    le = load_label_encoders()

    print("\n[1] Loading sample ID order from kmtricks.fof …")
    fof_ids   = load_fof_sample_ids(FOF_FILE)
    id_to_col = {sid: i for i, sid in enumerate(fof_ids)}
    print(f"    {len(fof_ids)} samples in matrix")

    print("\n[2] Loading training metadata for target BioProjects …")
    train_meta = pd.read_csv(TRAIN_META, sep="\t", low_memory=False)
    target     = train_meta[train_meta["BioProject"].isin(TABLE_BIOPROJECTS)].copy()
    target     = target[target["Run_accession"].isin(id_to_col)].copy()
    print(target.groupby("BioProject")[["Run_accession","project_name"]]
          .agg({"Run_accession": "count", "project_name": "first"})
          .rename(columns={"Run_accession": "n"}).to_string())
    print(f"    Total: {len(target)} samples")

    target_ids  = target["Run_accession"].tolist()
    target_cols = [id_to_col[s] for s in target_ids]

    print(f"\n[3] Reading feature matrix (selecting {len(target_ids)} columns from 1.6 GB file) …")
    import polars as pl
    mat            = pl.read_csv(str(MATRIX_FILE), separator=" ",
                                 has_header=False, infer_schema_length=0)
    needed_pl_cols = [f"column_{c + 2}" for c in target_cols]
    X              = mat.select(needed_pl_cols).cast(pl.Float32).to_numpy().T
    del mat
    print(f"    X shape: {X.shape}")

    print("\n[4] Loading model …")
    model = load_model(le)

    print("\n[5] Running inference …")
    preds = run_inference(model, X, le)

    print("\n[6] Per-task accuracy:")
    results = target.reset_index(drop=True).copy()
    for task in TASKS:
        results[f"{task}_true"] = results[task]
        results[f"{task}_pred"] = preds[task]
        results[f"{task}_conf"] = preds[f"{task}_conf"]
        correct = (results[f"{task}_pred"] == results[f"{task}_true"]).sum()
        print(f"    {task:<20}: {correct}/{len(results)}  ({100*correct/len(results):.1f}%)")

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    cols_out = (["Run_accession", "BioProject", "project_name"] +
                [f"{t}_true" for t in TASKS] +
                [f"{t}_pred" for t in TASKS] +
                [f"{t}_conf" for t in TASKS])
    cols_out = [c for c in cols_out if c in results.columns]
    results[cols_out].to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\n✓  Written {OUT_TSV.relative_to(REPO)}")


if __name__ == "__main__":
    main()
