#!/usr/bin/env python3
"""
25_offender_deep_dive.py

Six-question diagnostic for the top BioProject offenders in DIANA validation.

Q1. Fraction of total / confident errors explained by key BioProjects (per task).
Q2. Whether key BioProjects appear in train / val / test (run counts per split).
Q3. Raw AMD label → DIANA label evidence (root cause table).
Q4. Full Austin2024 (PRJNA1056444) breakdown.
Q5. Replicate / run inflation: per-run vs per-physical-sample evaluation.
Q6. Metrics with offenders excluded (sanity check).
Q7. Per-split error rates (train / test / validation) for each offender BioProject.
    Train/test predictions come from results/train_evaluation/ and
    results/test_evaluation/ (wide TSV format); validation predictions come from
    results/validation_predictions/ (per-run JSON). Only PRJEB34569 appears in
    training — the other offenders are exclusively seen in validation.

How the OFFENDERS list was built
---------------------------------
Script 24_bioproject_error_analysis.py ranked every BioProject in validation
by total confident errors (confidence >= 0.9 and predicted != true_label)
aggregated across all four tasks. Top performers from that ranking:

  Rank  N_cerr  BioProject    Project
  1     141     PRJNA994900   Kirdok2024
  2      36     PRJEB34569    FellowsYates2021
  3      35     PRJEB80877    vonHippel2025
  4      34     PRJNA1211513  Schreiber2025
  5      24     PRJEB49638    Moraitou2022
  6      17     PRJEB74036    Liu2024
  ...
  10     10     PRJNA1056444  Austin2024

All six OFFENDERS are included. Austin2024 (PRJNA1056444) is kept separately
as AUSTIN because its error mode is distinct (see Q4).

Note on is_correct logic
-------------------------
Prediction correctness is evaluated as a raw string comparison:
  is_correct = (pred_label == true_label)
where true_label comes from paper/metadata/validation_metadata.tsv and
pred_label is decoded from the label encoder classes. No case normalisation
is applied. For PRJEB34569 and PRJEB49638 the errors are due to taxonomic
resolution mismatch: validation metadata stores subspecies-level labels
(e.g. 'gorilla beringei beringei') while the label encoder only has
genus-level classes ('Gorilla sp.', 'Pan troglodytes'). The model is
structurally incapable of predicting at subspecies level, so every such
run is always counted as wrong — the capitalisation difference between
AMD v25.09.0 and paper/metadata is irrelevant to is_correct.

Output files (paper/tables/final/):
  offender_error_fraction.tsv
  offender_split_counts.tsv
  offender_root_cause.tsv
  austin2024_breakdown.tsv
  replicate_inflation.tsv
  metrics_excl_offenders.tsv
  offender_per_split_errors.tsv

Key finding
-----------
Most confident errors are explainable by label resolution/coverage mismatches
concentrated in a few BioProjects: OOD material classes (Kirdok2024 birch
pitch), label granularity (vonHippel2025/Schreiber2025/Liu2024 lake/marine
sediment not disaggregated in training), and taxonomic resolution
(FellowsYates2021/Moraitou2022 subspecies labels vs genus-level label encoder).
After excluding or collapsing these to DIANA resolution, the remaining confident
error rate — especially for sample_host (12.4% → 3.7%) and material
(25.1% → 16.5%) — drops sharply, leaving Austin2024 as the main genuine
confusion case (Homo sapiens confidently predicted as Pan troglodytes).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO      = Path(__file__).resolve().parents[2]
PRED_DIR       = REPO / "results" / "validation_predictions"
LE_PATH        = REPO / "results" / "training" / "label_encoders.json"
VAL_META       = REPO / "paper" / "metadata" / "validation_metadata.tsv"
TRAIN_META     = REPO / "paper" / "metadata" / "train_metadata.tsv"
TEST_META      = REPO / "paper" / "metadata" / "test_metadata.tsv"
TRAIN_PREDS    = REPO / "results" / "train_evaluation" / "test_predictions.tsv"
TEST_PREDS     = REPO / "results" / "test_evaluation"  / "test_predictions.tsv"
AMD_HOST  = REPO / "data" / "metadata" / "ancientmetagenome-hostassociated_samples.tsv"
AMD_ENV   = REPO / "data" / "metadata" / "ancientmetagenome-environmental_samples.tsv"
OUT_DIR   = REPO / "paper" / "tables" / "final"

TASKS = ["sample_type", "community_type", "sample_host", "material"]
CONF_THRESH = 0.9

# Top offender BioProjects ranked by total confident errors across all tasks.
# PRJNA1056444 (Austin2024) is kept separately (AUSTIN) because its error
# mode is distinct — label is in training but model confuses it (see Q4).
OFFENDERS = [
    "PRJNA994900",   # Kirdok2024       — rank 1, 141 cerr, OOD material (birch pitch)
    "PRJEB34569",    # FellowsYates2021 — rank 2,  36 cerr, taxonomic resolution (subspecies)
    "PRJEB80877",    # vonHippel2025    — rank 3,  35 cerr, label granularity (lake sediment)
    "PRJNA1211513",  # Schreiber2025    — rank 4,  34 cerr, label granularity (marine sediment)
    "PRJEB49638",    # Moraitou2022     — rank 5,  24 cerr, taxonomic resolution (subspecies)
    "PRJEB74036",    # Liu2024          — rank 6,  17 cerr, label granularity (lake sediment)
]
AUSTIN    = "PRJNA1056444"   # rank 10, 10 cerr, investigated separately in Q4

# DIANA label encoder classes
LE: dict[str, list[str]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_le() -> None:
    global LE
    raw = json.loads(LE_PATH.read_text())
    LE = {t: raw[t]["classes"] for t in TASKS}


def load_predictions() -> pd.DataFrame:
    rows = []
    for sd in sorted(PRED_DIR.iterdir()):
        if not sd.is_dir():
            continue
        pfs = list(sd.glob("*_predictions.json"))
        if not pfs:
            continue
        data = json.loads(pfs[0].read_text())
        run  = data.get("sample_id", sd.name)
        for task in TASKS:
            tp = data.get("predictions", {}).get(task)
            if not tp:
                continue
            cls       = LE[task]
            probs_raw = tp.get("probabilities", {})
            pred_class = tp.get("predicted_class")
            pred_idx   = tp.get("class_index")
            if pred_class and pred_class in cls:
                predicted = pred_class
            elif pred_idx is not None and pred_idx < len(cls):
                predicted = cls[pred_idx]
            else:
                best = max(probs_raw, key=probs_raw.__getitem__)
                predicted = cls[int(best)] if best.isdigit() else best
            rows.append({
                "run": run, "task": task,
                "predicted": predicted,
                "confidence": float(tp["confidence"]),
            })
    return pd.DataFrame(rows)


def load_meta(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.rename(columns={"Run_accession": "run"})
    df["BioProject"] = df["BioProject"].astype(str).str.strip()
    df.loc[~df["BioProject"].str.match(r"PRJ[A-Z]+\d+"), "BioProject"] = "Unknown"
    return df


def build_master(preds: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df = preds.merge(
        meta[["run", "BioProject", "project_name", "archive_accession",
              "sample_name", "sample_type", "community_type",
              "sample_host", "material"]],
        on="run", how="left",
    )
    true_map = {"sample_type": "sample_type", "community_type": "community_type",
                "sample_host": "sample_host", "material": "material"}
    df["true_label"] = [row.get(true_map[row["task"]], pd.NA)
                        for _, row in df.iterrows()]
    df["is_correct"]  = df["predicted"] == df["true_label"]
    df["conf_wrong"]  = ~df["is_correct"] & (df["confidence"] >= CONF_THRESH)
    return df


# ---------------------------------------------------------------------------
# Q1: Fraction of errors explained by key BioProjects
# ---------------------------------------------------------------------------

def q1_error_fraction(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, tdf in df.groupby("task"):
        total_wrong      = (~tdf["is_correct"]).sum()
        total_conf_wrong = tdf["conf_wrong"].sum()
        for bp in OFFENDERS + [AUSTIN]:
            bdf = tdf[tdf["BioProject"] == bp]
            if bdf.empty:
                continue
            n_wrong      = (~bdf["is_correct"]).sum()
            n_conf_wrong = bdf["conf_wrong"].sum()
            rows.append({
                "task": task,
                "BioProject": bp,
                "project_name": bdf["project_name"].mode()[0] if bdf["project_name"].notna().any() else "",
                "N_val_runs": len(bdf),
                "N_wrong":      int(n_wrong),
                "pct_of_task_errors":      round(100 * n_wrong / total_wrong, 1) if total_wrong else float("nan"),
                "N_conf_wrong": int(n_conf_wrong),
                "pct_of_task_conf_errors": round(100 * n_conf_wrong / total_conf_wrong, 1) if total_conf_wrong else float("nan"),
                "own_error_rate":      round(100 * n_wrong / len(bdf), 1) if len(bdf) else float("nan"),
                "own_conf_error_rate": round(100 * n_conf_wrong / len(bdf), 1) if len(bdf) else float("nan"),
            })
    return pd.DataFrame(rows).sort_values(["task", "pct_of_task_conf_errors"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Q2: BioProject appearance in train / val / test
# ---------------------------------------------------------------------------

def q2_split_counts(val_meta: pd.DataFrame,
                    train_meta: pd.DataFrame,
                    test_meta: pd.DataFrame) -> pd.DataFrame:
    all_bps = OFFENDERS + [AUSTIN]
    rows = []
    for bp in all_bps:
        n_train = (train_meta["BioProject"] == bp).sum()
        n_test  = (test_meta["BioProject"]  == bp).sum()
        n_val   = (val_meta["BioProject"]   == bp).sum()
        # Unique physical samples (archive_accession) per split
        def uniq_samples(meta):
            sub = meta[meta["BioProject"] == bp]["archive_accession"].dropna()
            return sub.nunique()
        rows.append({
            "BioProject": bp,
            "project_name": val_meta.loc[val_meta["BioProject"] == bp, "project_name"].mode()[0]
                if (val_meta["BioProject"] == bp).any() else
                train_meta.loc[train_meta["BioProject"] == bp, "project_name"].mode()[0]
                if (train_meta["BioProject"] == bp).any() else "",
            "train_runs": int(n_train),
            "train_unique_samples": int(uniq_samples(train_meta)),
            "test_runs":  int(n_test),
            "test_unique_samples":  int(uniq_samples(test_meta)),
            "val_runs":   int(n_val),
            "val_unique_samples":   int(uniq_samples(val_meta)),
            "in_train": "YES" if n_train > 0 else "NO",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Q3: Raw AMD label → DIANA class evidence
# ---------------------------------------------------------------------------

def q3_root_cause() -> pd.DataFrame:
    env  = pd.read_csv(AMD_ENV,  sep="\t", low_memory=False)
    host = pd.read_csv(AMD_HOST, sep="\t", low_memory=False)
    amd  = pd.concat([env, host], ignore_index=True)

    evidence = [
        # (BioProject, project_name, task, raw_label_in_AMD, diana_training_classes, predicted_by_diana, root_cause)
        ("PRJNA994900", "Kirdok2024",       "material",
         "birch pitch",
         "NOT IN TRAINING SET (bone,dental calculus,plaque,saliva,sediment,soil,skin,tooth,…)",
         "sediment",
         "Out-of-distribution material class: birch pitch absent from training data"),

        ("PRJNA994900", "Kirdok2024",       "community_type",
         "oral",
         "oral IS a training class",
         "Not applicable - env sample",
         "Spill-over from material OOD: model associates birch pitch k-mer profile with env signal"),

        ("PRJNA994900", "Kirdok2024",       "sample_host",
         "Homo sapiens",
         "Homo sapiens IS a training class",
         "Not applicable - env sample",
         "Spill-over from material OOD (same cause as community_type)"),

        ("PRJEB80877",  "vonHippel2025",    "material",
         "lake sediment",
         "Training class is 'sediment' (no lake/marine distinction)",
         "sediment",
         "Label granularity mismatch: 'lake sediment' not a DIANA class, correct superclass predicted"),

        ("PRJNA1211513","Schreiber2025",    "material",
         "marine sediment",
         "Training class is 'sediment' (no lake/marine distinction)",
         "sediment",
         "Label granularity mismatch: 'marine sediment' not a DIANA class, correct superclass predicted"),

        ("PRJEB34569",  "FellowsYates2021", "sample_host",
         "gorilla beringei beringei / gorilla beringei graueri / gorilla gorilla gorilla / "
         "alouatta palliata / pan troglodytes schweinfurthii / pan troglodytes ellioti",
         "Training classes are genus-level: 'Gorilla sp.', 'Pan troglodytes' — "
         "subspecies not in label encoder. is_correct is a raw string comparison "
         "so pred='Gorilla sp.' vs true='gorilla beringei beringei' is always False.",
         "Gorilla sp. / Pan troglodytes / Other mammal",
         "Taxonomic resolution mismatch: validation labels are subspecies-level "
         "(e.g. gorilla beringei beringei) but label encoder only has genus-level "
         "classes — model structurally cannot be correct on these runs"),

        ("PRJEB49638",  "Moraitou2022",     "sample_host",
         "gorilla beringei beringei / gorilla beringei graueri / gorilla gorilla gorilla",
         "Training classes are genus-level: 'Gorilla sp.' — subspecies not in label encoder.",
         "Gorilla sp.",
         "Taxonomic resolution mismatch (same as FellowsYates2021): all 20 runs have "
         "subspecies-level labels, model can only predict 'Gorilla sp.', "
         "so is_correct is always False regardless of model quality"),

        ("PRJEB74036",  "Liu2024",           "material",
         "lake sediment",
         "Training class is 'sediment' (no lake/marine/river distinction)",
         "sediment",
         "Label granularity mismatch: 'lake sediment' not a DIANA class, "
         "correct superclass predicted (same root cause as vonHippel2025)"),

        ("PRJNA1056444","Austin2024",        "sample_host",
         "Homo sapiens",
         "Homo sapiens IS a training class",
         "Pan troglodytes",
         "Investigated in Q4"),
    ]

    return pd.DataFrame(evidence, columns=[
        "BioProject","project_name","task",
        "raw_label_in_AMD","diana_training_classes",
        "diana_prediction","root_cause",
    ])


# ---------------------------------------------------------------------------
# Q4: Full Austin2024 breakdown
# ---------------------------------------------------------------------------

def q4_austin(df: pd.DataFrame, val_meta: pd.DataFrame) -> pd.DataFrame:
    adf = df[df["BioProject"] == AUSTIN].copy()
    ameta = val_meta[val_meta["BioProject"] == AUSTIN]

    rows = []
    for task, tdf in adf.groupby("task"):
        n = len(tdf)
        err = (~tdf["is_correct"]).sum()
        cerr = tdf["conf_wrong"].sum()
        # top confusion
        ewrong = tdf[~tdf["is_correct"]]
        top_pair = ""
        if len(ewrong):
            top_pair = (ewrong["true_label"].astype(str) + " → " + ewrong["predicted"].astype(str)).value_counts().index[0]
        rows.append({
            "task": task,
            "N_runs": n,
            "N_unique_samples": ameta["archive_accession"].nunique(),
            "N_wrong": int(err),
            "error_rate_pct": round(100 * err / n, 1) if n else float("nan"),
            "N_conf_wrong": int(cerr),
            "conf_error_rate_pct": round(100 * cerr / n, 1) if n else float("nan"),
            "top_confusion": top_pair,
        })

    # Metadata summary
    print(f"\n=== Q4: Austin2024 (PRJNA1056444) metadata ===")
    print(f"  N runs in validation : {len(ameta)}")
    print(f"  Unique samples       : {ameta['archive_accession'].nunique()}")
    print(f"  material             : {dict(ameta['material'].value_counts())}")
    print(f"  sample_host          : {dict(ameta['sample_host'].value_counts())}")
    print(f"  community_type       : {dict(ameta['community_type'].value_counts())}")

    # Check if Austin2024 is in training
    tr = load_meta(TRAIN_META)
    n_train = (tr["BioProject"] == AUSTIN).sum()
    print(f"  In TRAINING?         : {'YES (' + str(n_train) + ' runs)' if n_train else 'NO'}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Q5: Replicate / run inflation
# ---------------------------------------------------------------------------

def q5_inflation(df: pd.DataFrame, val_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bp in OFFENDERS + [AUSTIN]:
        bdf = df[df["BioProject"] == bp]
        bmeta = val_meta[val_meta["BioProject"] == bp]
        if bdf.empty:
            continue
        n_runs    = bmeta["run"].nunique() if "run" in bmeta.columns else len(bmeta)
        n_samples = bmeta["archive_accession"].nunique()
        runs_per_sample = round(n_runs / n_samples, 1) if n_samples else float("nan")
        # Per-task errors at run level
        for task in TASKS:
            tdf = bdf[bdf["task"] == task]
            if tdf.empty:
                continue
            # per-run
            run_err_rate = round(100 * (~tdf["is_correct"]).mean(), 1)
            # per-sample: evaluate by majority vote per archive_accession
            # (archive_accession already in tdf from build_master)
            if tdf["archive_accession"].isna().all():
                sample_err_rate = float("nan")
                n_sample_errors = float("nan")
            else:
                # majority vote: predicted = most frequent prediction per sample
                def majority(grp):
                    pred = grp["predicted"].mode()[0]
                    true = grp["true_label"].mode()[0]
                    return pred != true
                sample_errors = tdf.groupby("archive_accession").apply(majority)
                sample_err_rate = round(100 * sample_errors.mean(), 1)
                n_sample_errors = int(sample_errors.sum())

            rows.append({
                "BioProject": bp,
                "project_name": bdf["project_name"].mode()[0] if bdf["project_name"].notna().any() else "",
                "task": task,
                "N_runs": len(tdf),
                "N_unique_samples": n_samples,
                "runs_per_sample": runs_per_sample,
                "error_rate_per_run_pct": run_err_rate,
                "error_rate_per_sample_pct": sample_err_rate,
                "n_sample_errors": n_sample_errors,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Q7: Per-split error rates for offender BioProjects
# ---------------------------------------------------------------------------

def load_split_preds_tsv(tsv_path: Path, meta: pd.DataFrame) -> pd.DataFrame:
    """Read the wide predictions TSV (train_evaluation / test_evaluation format)
    and return a long DataFrame matching the shape of build_master() output,
    merged with BioProject/project_name from *meta* (train or test metadata)."""
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    df = df.rename(columns={"Run_accession": "run"})

    prob_cols = {t: [c for c in df.columns if c.startswith(f"{t}_prob_")] for t in TASKS}

    rows = []
    for _, row in df.iterrows():
        for task in TASKS:
            pred  = row.get(f"{task}_pred", pd.NA)
            true  = row.get(f"{task}_true", pd.NA)
            probs = [row[c] for c in prob_cols[task] if pd.notna(row[c])]
            conf  = float(max(probs)) if probs else float("nan")
            is_correct = (pred == true)
            rows.append({
                "run":         row["run"],
                "task":        task,
                "predicted":   pred,
                "true_label":  true,
                "confidence":  conf,
                "is_correct":  is_correct,
                "conf_wrong":  (not is_correct) and (conf >= CONF_THRESH),
            })

    long = pd.DataFrame(rows)
    long = long.merge(
        meta[["run", "BioProject", "project_name"]],
        on="run", how="left",
    )
    return long


def q7_per_split_errors(
    val_df:   pd.DataFrame,
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
) -> pd.DataFrame:
    """For each offender BioProject × task show error rates in train/test/val."""
    rows = []
    for bp in OFFENDERS + [AUSTIN]:
        for task in TASKS:
            for split_label, sdf in [("train", train_df), ("test", test_df), ("val", val_df)]:
                tdf = sdf[(sdf["BioProject"] == bp) & (sdf["task"] == task)]
                n = len(tdf)
                if n == 0:
                    rows.append({
                        "BioProject": bp,
                        "task": task,
                        "split": split_label,
                        "N_runs": 0,
                        "N_wrong": 0,
                        "error_rate_pct": float("nan"),
                        "N_conf_wrong": 0,
                        "conf_error_rate_pct": float("nan"),
                    })
                else:
                    n_wrong  = int((~tdf["is_correct"]).sum())
                    n_cwrong = int(tdf["conf_wrong"].sum())
                    rows.append({
                        "BioProject": bp,
                        "task": task,
                        "split": split_label,
                        "N_runs": n,
                        "N_wrong": n_wrong,
                        "error_rate_pct": round(100 * n_wrong / n, 1),
                        "N_conf_wrong": n_cwrong,
                        "conf_error_rate_pct": round(100 * n_cwrong / n, 1),
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Q6: Metrics with offenders excluded
# ---------------------------------------------------------------------------

def q6_excl_offenders(df: pd.DataFrame) -> pd.DataFrame:
    excl = OFFENDERS + [AUSTIN]
    rows = []
    for task in TASKS:
        tdf_all  = df[df["task"] == task]
        tdf_excl = tdf_all[~tdf_all["BioProject"].isin(excl)]

        def stats(d, label):
            n = len(d)
            wrong = (~d["is_correct"]).sum()
            cwrong = d["conf_wrong"].sum()
            return {
                "task": task,
                "cohort": label,
                "N_runs": n,
                "N_unique_BPs": d["BioProject"].nunique(),
                "N_wrong": int(wrong),
                "error_rate_pct": round(100 * wrong / n, 1) if n else float("nan"),
                "N_conf_wrong": int(cwrong),
                "conf_error_rate_pct": round(100 * cwrong / n, 1) if n else float("nan"),
            }

        rows.append(stats(tdf_all,  "full_validation"))
        rows.append(stats(tdf_excl, "excl_offenders"))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_q1(t: pd.DataFrame) -> None:
    print(f"\n{'='*72}")
    print("Q1  Fraction of errors explained by key BioProjects (per task)")
    print(f"{'='*72}")
    for task, tdf in t.groupby("task"):
        print(f"\n  Task: {task}")
        print(f"  {'BioProject':<16}  {'Project':<20}  {'N':>5}  {'err%':>5}  {'cerr%':>6}  {'%all_err':>8}  {'%all_cerr':>9}")
        print(f"  {'-'*16}  {'-'*20}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*9}")
        for _, r in tdf.iterrows():
            print(f"  {r['BioProject']:<16}  {str(r['project_name'])[:19]:<20}  "
                  f"{r['N_val_runs']:>5}  {r['own_error_rate']:>4.0f}%  {r['own_conf_error_rate']:>5.0f}%  "
                  f"{r['pct_of_task_errors']:>7.1f}%  {r['pct_of_task_conf_errors']:>8.1f}%")


def print_q2(t: pd.DataFrame) -> None:
    print(f"\n{'='*72}")
    print("Q2  BioProject presence in train / val / test splits")
    print(f"{'='*72}")
    print(f"  {'BioProject':<16}  {'Project':<20}  {'Train runs':>10}  {'Tr_samp':>7}  {'Test runs':>9}  {'Te_samp':>7}  {'Val runs':>8}  {'Val_samp':>8}  in_train")
    print(f"  {'-'*16}  {'-'*20}  {'-'*10}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")
    for _, r in t.iterrows():
        print(f"  {r['BioProject']:<16}  {str(r['project_name'])[:19]:<20}  "
              f"{r['train_runs']:>10}  {r['train_unique_samples']:>7}  "
              f"{r['test_runs']:>9}  {r['test_unique_samples']:>7}  "
              f"{r['val_runs']:>8}  {r['val_unique_samples']:>8}  {r['in_train']:>8}")


def print_q3(t: pd.DataFrame) -> None:
    print(f"\n{'='*72}")
    print("Q3  Raw AMD label → DIANA class: root cause evidence")
    print(f"{'='*72}")
    for _, r in t.iterrows():
        print(f"\n  {r['BioProject']} ({r['project_name']}) — task: {r['task']}")
        print(f"    AMD raw label    : {r['raw_label_in_AMD']}")
        print(f"    DIANA classes    : {r['diana_training_classes']}")
        print(f"    DIANA predicts   : {r['diana_prediction']}")
        print(f"    Root cause       : {r['root_cause']}")


def print_q4(t: pd.DataFrame) -> None:
    print(f"\n{'='*72}")
    print("Q4  Austin2024 (PRJNA1056444) per-task breakdown")
    print(f"{'='*72}")
    print(f"\n  {'Task':<18}  {'N_runs':>6}  {'N_samp':>6}  {'err%':>5}  {'cerr%':>6}  Top confusion")
    print(f"  {'-'*18}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*35}")
    for _, r in t.iterrows():
        print(f"  {r['task']:<18}  {r['N_runs']:>6}  {r['N_unique_samples']:>6}  "
              f"{r['error_rate_pct']:>4.0f}%  {r['conf_error_rate_pct']:>5.0f}%  {r['top_confusion']}")


def print_q5(t: pd.DataFrame) -> None:
    print(f"\n{'='*72}")
    print("Q5  Run inflation: per-run vs per-physical-sample error rate")
    print(f"{'='*72}")
    print(f"\n  {'BioProject':<16}  {'task':<18}  {'runs':>5}  {'samp':>5}  {'runs/samp':>9}  {'err%/run':>9}  {'err%/samp':>10}")
    print(f"  {'-'*16}  {'-'*18}  {'-'*5}  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*10}")
    for _, r in t.iterrows():
        print(f"  {r['BioProject']:<16}  {r['task']:<18}  {r['N_runs']:>5}  {r['N_unique_samples']:>5}  "
              f"{r['runs_per_sample']:>9.1f}  {r['error_rate_per_run_pct']:>8.1f}%  "
              f"{str(r['error_rate_per_sample_pct']):>9}%")


def print_q7(t: pd.DataFrame) -> None:
    print(f"\n{'='*72}")
    print("Q7  Per-split error rates for offender BioProjects")
    print(f"    (only BioProjects with ≥1 run in that split are shown)")
    print(f"{'='*72}")
    for bp, bdf in t.groupby("BioProject", sort=False):
        # Find project name from the row with most runs
        proj = ""
        has_data = bdf[bdf["N_runs"] > 0]
        if not has_data.empty:
            proj = ""  # project_name not in this table; look up from OFFENDERS context
        print(f"\n  {bp}")
        print(f"  {'task':<18}  {'split':<5}  {'N':>5}  {'err%':>6}  {'cerr%':>6}")
        print(f"  {'-'*18}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}")
        for _, r in bdf.iterrows():
            if r["N_runs"] == 0:
                print(f"  {r['task']:<18}  {r['split']:<5}  {'—':>5}  {'—':>6}  {'—':>6}")
            else:
                print(f"  {r['task']:<18}  {r['split']:<5}  {r['N_runs']:>5}  "
                      f"{r['error_rate_pct']:>5.1f}%  {r['conf_error_rate_pct']:>5.1f}%")


def print_q6(t: pd.DataFrame) -> None:
    print(f"\n{'='*72}")
    print(f"Q6  Sanity check: metrics with offenders excluded")
    print(f"    (excluded: {', '.join(OFFENDERS + [AUSTIN])})")
    print(f"{'='*72}")
    print(f"\n  {'Task':<18}  {'Cohort':<20}  {'N':>5}  {'BPs':>4}  {'err%':>6}  {'cerr%':>7}")
    print(f"  {'-'*18}  {'-'*20}  {'-'*5}  {'-'*4}  {'-'*6}  {'-'*7}")
    for _, r in t.iterrows():
        print(f"  {r['task']:<18}  {r['cohort']:<20}  {r['N_runs']:>5}  {r['N_unique_BPs']:>4}  "
              f"{r['error_rate_pct']:>5.1f}%  {r['conf_error_rate_pct']:>6.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading label encoders...")
    load_le()

    print("Loading predictions...")
    preds = load_predictions()

    print("Loading metadata...")
    val_meta   = load_meta(VAL_META)
    train_meta = load_meta(TRAIN_META)
    test_meta  = load_meta(TEST_META)

    print("Building master table...")
    df = build_master(preds, val_meta)

    print("Q1: error fractions per BioProject...")
    t1 = q1_error_fraction(df)
    t1.to_csv(OUT_DIR / "offender_error_fraction.tsv", sep="\t", index=False)

    print("Q2: split counts...")
    t2 = q2_split_counts(val_meta, train_meta, test_meta)
    t2.to_csv(OUT_DIR / "offender_split_counts.tsv", sep="\t", index=False)

    print("Q3: root cause evidence...")
    t3 = q3_root_cause()
    t3.to_csv(OUT_DIR / "offender_root_cause.tsv", sep="\t", index=False)

    print("Q4: Austin2024 breakdown...")
    t4 = q4_austin(df, val_meta)
    t4.to_csv(OUT_DIR / "austin2024_breakdown.tsv", sep="\t", index=False)

    print("Q5: replicate inflation...")
    t5 = q5_inflation(df, val_meta)
    t5.to_csv(OUT_DIR / "replicate_inflation.tsv", sep="\t", index=False)

    print("Q6: metrics excluding offenders...")
    t6 = q6_excl_offenders(df)
    t6.to_csv(OUT_DIR / "metrics_excl_offenders.tsv", sep="\t", index=False)

    print("Q7: per-split error rates for offenders...")
    train_preds = load_split_preds_tsv(TRAIN_PREDS, train_meta)
    test_preds  = load_split_preds_tsv(TEST_PREDS,  test_meta)
    t7 = q7_per_split_errors(df, train_preds, test_preds)
    t7.to_csv(OUT_DIR / "offender_per_split_errors.tsv", sep="\t", index=False)

    # Print all summaries
    print_q1(t1)
    print_q2(t2)
    print_q3(t3)
    print_q4(t4)
    print_q5(t5)
    print_q6(t6)
    print_q7(t7)

    print(f"\n{'='*72}")
    print("Output files written to paper/tables/final/")


if __name__ == "__main__":
    main()
