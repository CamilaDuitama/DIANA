#!/usr/bin/env python3
"""
Generate BioProject error-concentration supplementary table (sup_table_09)

Ranks BioProjects by total misclassifications (predicted != true, no
confidence threshold) aggregated across all four tasks and both the test
and validation sets.  The cumulative-percentage column makes it
immediately visible whether errors are concentrated in a few BioProjects.

Inputs:
- results/validation_predictions/  (per-sample JSON predictions)
- paper/metadata/validation_metadata.tsv
- results/test_evaluation/test_predictions.tsv
- paper/metadata/test_metadata.tsv

Output:
- paper/tables/final/sup_table_09_bioproject_offenders.tex

Root-cause categories
---------------------
  Absent class         – true label never seen in training (e.g. birch pitch)
  Label granularity    – true label absent from training but is a biological
                         subtype of a training class (e.g. lake sediment ⊂ sediment)
  Taxonomic granularity – true host is a subspecies whose parent taxon is the
                         training class (e.g. Gorilla beringei → Gorilla sp.)
  Genuine confusion    – true label was in training; model still predicts wrong
"""

import sys
from pathlib import Path
import pandas as pd

REPO    = Path(__file__).resolve().parents[2]
OUT_TEX = REPO / "paper" / "tables" / "final" / "sup_table_09_bioproject_offenders.tex"
sys.path.insert(0, str(REPO / "scripts" / "validation"))

MIN_SAMPLES = 5   # minimum run count (test + val) to include a BioProject
TOP_N       = 15  # show only the top-N rows (ranked by total_errors)

TASKS = ["sample_type", "community_type", "sample_host", "material"]

# Per-BioProject annotations: (study_name, dominant_tasks, error_type, dominant_errors)
# error_type categories:
#   Absent class          – true label (material or host) never seen in DIANA training
#   Label granularity     – true label absent from training, but is a biological subtype/synonym
#                           of a training class; model predicts the correct coarser class
#   Taxonomic granularity – true host is a species/subspecies whose genus is the training-level label
#   Genuine confusion     – all labels present in training; model still predicts the wrong class
ANNOTATIONS = {
    "PRJNA994900":   ("Kirdok2024",           "CT, Mat., SH",  "Absent class (Mat.)",
                      "birch pitch$\\to$sediment (Mat.); ancient$\\to$modern (ST)"),
    "PRJEB64128":    ("Jackson2024",           "Mat., CT, SH",  "Genuine confusion",
                      "tooth$\\to$dent.\\ calc. (Mat.); oral$\\to$skel.\\ tissue (CT)"),
    "PRJEB61887":    ("Velsko2024",            "CT, Mat., SH",  "Genuine confusion",
                      "oral$\\to$skel.\\ tissue (CT); bone$\\to$tooth (Mat.)"),
    "PRJNA1211513":  ("Schreiber2025",         "Mat.",           "Label granularity (Mat.)",
                      "marine sed.$\\to$sediment (Mat.)"),
    "PRJEB34569":    ("FellowsYates2021",      "SH",             "Taxonomic granularity (SH)",
                      "\\textit{G.\\ b.\\ beringei}$\\to$\\textit{Gorilla} sp.; "
                      "\\textit{P.\\ t.\\ schweinfurthii}$\\to$\\textit{P.\\ troglodytes}"),
    "PRJEB80877":    ("vonHippel2025",         "Mat.",           "Label granularity (Mat.)",
                      "lake sed.$\\to$sediment (Mat.)"),
    "PRJEB49638":    ("Moraitou2022",          "SH",             "Taxonomic granularity (SH)",
                      "\\textit{G.\\ b.\\ graueri}$\\to$\\textit{Gorilla} sp.; "
                      "\\textit{G.\\ gorilla gorilla}$\\to$\\textit{Gorilla} sp."),
    "PRJEB33848":    ("Neukamm2020",           "CT, Mat.",       "Genuine confusion; Absent class (Mat.)",
                      "soft tis.$\\to$skel.\\ tis. (CT); Unknown$\\to$bone (Mat.)"),
    "PRJNA354503":   ("Philips2017",           "CT, Mat.",       "Genuine confusion",
                      "oral$\\to$skel.\\ tissue (CT); tooth$\\to$bone (Mat.)"),
    "PRJNA836960":   ("Madison2023",           "SH, CT, Mat.",   "Absent class (Mat., SH)",
                      "intestine$\\to$skin (Mat.); gut$\\to$soft tissue (CT)"),
    "PRJEB67998":    ("Bozzi2024",             "Mat., CT, ST",   "Genuine confusion",
                      "tooth$\\to$bone/skin (Mat.); skel.\\ tis.$\\to$soft tis. (CT)"),
    "PRJEB41240":    ("SeguinOrlando2021",     "Mat., SH, CT",   "Genuine confusion",
                      "bone$\\to$tooth/sediment (Mat.); \\textit{H.\\ sapiens}$\\to$n.a. (SH)"),
    "PRJNA1056444":  ("Austin2024",            "SH, CT, Mat.",   "Genuine confusion",
                      "\\textit{H.\\ sapiens}$\\to$\\textit{Pan troglodytes} (SH)"),
    "PRJEB11419":    ("–",                     "Mat., ST, SH",   "Label granularity (Mat.)",
                      "faeces$\\to$dig.\\ tract (Mat.); modern$\\to$ancient (ST)"),
    "PRJNA48479":    ("–",                     "ST, CT",         "Genuine confusion",
                      "modern$\\to$ancient (ST); oral$\\to$skel.\\ tissue (CT)"),
    "PRJEB34875":    ("Ottoni2019",            "Mat., SH, CT",   "Absent class (SH)",
                      "dent.\\ calc.$\\to$tooth (Mat.); \\textit{Papio} sp.$\\to$\\textit{H.\\ sapiens} (SH)"),
    "PRJEB30280":    ("Jensen2019",            "Mat., CT, ST",   "Absent class (Mat.)",
                      "birch pitch$\\to$skin (Mat.); ancient$\\to$modern (ST)"),
    "PRJEB74036":    ("Liu2024",               "Mat.",           "Label granularity (Mat.)",
                      "lake sed.$\\to$sediment (Mat.)"),
    "PRJNA433935":   ("–",                     "CT, Mat., SH",   "Genuine confusion",
                      "soil$\\to$skin (Mat.); env$\\to$soft tissue (CT)"),
    "PRJNA320875":   ("Graham2016",            "Mat.",           "Label granularity (Mat.)",
                      "lake sed.$\\to$sediment (Mat.)"),
    "PRJEB46022":    ("Fagernas2022",          "CT, SH, ST",     "Genuine confusion",
                      "oral$\\to$skel.\\ tissue (CT); \\textit{H.\\ sapiens}$\\to$\\textit{Gorilla} sp. (SH)"),
    "PRJNA791766":   ("Quagliariello2022",     "Mat., SH",       "Genuine confusion",
                      "dent.\\ calc.$\\to$sediment (Mat.); \\textit{H.\\ sapiens}$\\to$n.a. (SH)"),
    "PRJNA445215":   ("Mann2018",              "Mat., CT",       "Genuine confusion",
                      "tooth$\\to$bone (Mat.); skel.\\ tis.$\\to$oral (CT)"),
    "PRJNA706195":   ("–",                     "ST, Mat.",       "Genuine confusion",
                      "modern$\\to$ancient (ST); soil$\\to$sediment (Mat.)"),
}


def load_val_errors() -> pd.DataFrame:
    """
    Load all validation misclassifications (predicted != true, no threshold).
    Returns long-form: Run_accession, BioProject, project_name, task, is_correct
    """
    from load_validation_data import load_validation_predictions
    df = load_validation_predictions(quiet=True)
    df = df.rename(columns={"sample_id": "Run_accession"})

    val_meta = pd.read_csv(
        REPO / "paper/metadata/validation_metadata.tsv", sep="\t", low_memory=False
    )
    meta_slim = val_meta[["Run_accession", "BioProject", "project_name"]].drop_duplicates()
    df = df.merge(meta_slim, on="Run_accession", how="left")
    df["split"] = "val"
    return df[["Run_accession", "BioProject", "project_name", "task", "is_correct", "split"]]


def load_train_errors() -> pd.DataFrame:
    """
    Load training-set predictions for BioProjects in the table.
    Returns long-form: Run_accession, BioProject, project_name, task, is_correct
    """
    train_tsv = REPO / "results/train_evaluation/train_predictions.tsv"
    if not train_tsv.exists():
        return pd.DataFrame(columns=["Run_accession", "BioProject", "project_name",
                                     "task", "is_correct", "split"])
    preds = pd.read_csv(train_tsv, sep="\t", low_memory=False)
    rows = []
    for task in TASKS:
        pred_col = f"{task}_pred"
        true_col = f"{task}_true"
        if pred_col not in preds.columns or true_col not in preds.columns:
            continue
        task_df = preds[["Run_accession", "BioProject", "project_name",
                          pred_col, true_col]].copy()
        task_df["task"]       = task
        task_df["is_correct"] = task_df[pred_col] == task_df[true_col]
        rows.append(task_df[["Run_accession", "BioProject", "project_name",
                              "task", "is_correct"]])
    if not rows:
        return pd.DataFrame(columns=["Run_accession", "BioProject", "project_name",
                                     "task", "is_correct", "split"])
    df = pd.concat(rows, ignore_index=True)
    df["split"] = "train"
    return df


def load_test_errors() -> pd.DataFrame:
    """
    Load all test misclassifications (predicted != true, no threshold).
    Returns long-form: Run_accession, BioProject, project_name, task, is_correct
    """
    preds = pd.read_csv(
        REPO / "results/test_evaluation/test_predictions.tsv", sep="\t", low_memory=False
    )
    test_meta = pd.read_csv(
        REPO / "paper/metadata/test_metadata.tsv", sep="\t", low_memory=False
    )
    meta_slim = test_meta[["Run_accession", "BioProject", "project_name"]].drop_duplicates()
    preds = preds.merge(meta_slim, on="Run_accession", how="left")

    rows = []
    for task in TASKS:
        pred_col = f"{task}_pred"
        true_col = f"{task}_true"
        if pred_col not in preds.columns or true_col not in preds.columns:
            continue
        task_df = preds[["Run_accession", "BioProject", "project_name",
                          pred_col, true_col]].copy()
        task_df["task"]       = task
        task_df["is_correct"] = task_df[pred_col] == task_df[true_col]
        rows.append(task_df[["Run_accession", "BioProject", "project_name",
                              "task", "is_correct"]])
    df = pd.concat(rows, ignore_index=True)
    df["split"] = "test"
    return df


def build_ranking() -> pd.DataFrame:
    val   = load_val_errors()
    test  = load_test_errors()
    train = load_train_errors()
    combined = pd.concat([val, test], ignore_index=True)  # ranking uses val+test only

    # Errors per BioProject per split
    def split_errors(df):
        return (
            df[~df["is_correct"]]
            .groupby("BioProject").size()
            .reset_index(name="n")
        )
    val_errors   = split_errors(val).rename(columns={"n": "val_errors"})
    test_errors  = split_errors(test).rename(columns={"n": "test_errors"})
    train_errors = split_errors(train).rename(columns={"n": "train_errors"})

    # Sample counts per split — MIN_SAMPLES filter uses val+test; err_pct uses all three
    val_samples   = val.groupby("BioProject")["Run_accession"].nunique().rename("n_val").reset_index()
    test_samples  = test.groupby("BioProject")["Run_accession"].nunique().rename("n_test").reset_index()
    train_samples = train.groupby("BioProject")["Run_accession"].nunique().rename("n_train").reset_index()

    # Total errors across val+test (ranking criterion)
    errors = (
        combined[~combined["is_correct"]]
        .groupby("BioProject")
        .size()
        .rename("total_errors")
        .reset_index()
    )

    # Canonical project_name: most frequent per BioProject
    project_names = (
        combined.dropna(subset=["project_name"])
        .groupby("BioProject")["project_name"]
        .agg(lambda x: x.value_counts().index[0])
        .rename("project_name")
        .reset_index()
    )

    agg = (
        errors
        .merge(project_names, on="BioProject", how="left")
        .merge(val_samples,   on="BioProject", how="left")
        .merge(test_samples,  on="BioProject", how="left")
        .merge(train_samples, on="BioProject", how="left")
        .merge(val_errors,    on="BioProject", how="left")
        .merge(test_errors,   on="BioProject", how="left")
        .merge(train_errors,  on="BioProject", how="left")
        .fillna({"n_val": 0, "n_test": 0, "n_train": 0,
                 "val_errors": 0, "test_errors": 0, "train_errors": 0})
        .assign(n_total=lambda d: d["n_val"] + d["n_test"])
    )
    agg = agg[agg["n_total"] >= MIN_SAMPLES].copy()
    # Total and err_pct now cover all three splits
    agg["total_errors"] = agg["val_errors"] + agg["test_errors"] + agg["train_errors"]
    agg["err_pct"] = 100.0 * agg["total_errors"] / ((agg["n_total"] + agg["n_train"]) * len(TASKS))
    agg = agg.sort_values("total_errors", ascending=False).reset_index(drop=True)

    # Cumulative % uses the grand total across ALL qualifying BioProjects (not just top-N)
    grand_total = agg["total_errors"].sum()
    agg["cum_pct"] = 100.0 * agg["total_errors"].cumsum() / grand_total

    return agg.head(TOP_N).copy()


def generate_latex(ranking: pd.DataFrame) -> str:
    lines = []
    lines.append(
        "\\caption{Top 15 BioProjects by total misclassifications (validation + test, "
        "all four tasks, no confidence threshold, $\\geq$5 samples). "
        "Train/Test/Val: number of \\textit{errors} (wrong predictions across all four tasks) per split; `--' means absent from that split. "
        "Total: wrong predictions across all three splits (train + test + val). "
        "Cum.: running percentage of all errors, showing concentration. "
        "\\label{tab:bioproject_offenders}}"
    )
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{@{}llrrrrrrp{3cm}p{4cm}@{}}")
    lines.append("\\toprule")
    lines.append(
        "BioProject & Study & Train err. & Test err. & Val err. & Total & Err.\\,(\\%)$^{a}$ & Cum.\\,(\\%)$^{b}$ "
        "& Error type$^{c}$ & Main errors$^{d}$ \\\\"
    )
    lines.append("\\midrule")

    for _, row in ranking.iterrows():
        bp    = row["BioProject"]
        study = str(row["project_name"]) if pd.notna(row["project_name"]) else ""

        if bp in ANNOTATIONS:
            ann_study, dominant, err_type, errors_str = ANNOTATIONS[bp]
            if not study:
                study = ann_study
        else:
            err_type   = "–"
            errors_str = "–"

        bp_tex     = bp.replace("_", "\\_")
        study_tex  = study if study else "–"
        train_err = int(row.get("train_errors", 0))
        test_err  = int(row.get("test_errors",  0))
        val_err   = int(row.get("val_errors",   0))
        n_train_tex = str(train_err) if train_err > 0 else "–"
        n_test_tex  = str(test_err)  if test_err  > 0 else "–"
        n_val_tex   = str(val_err)   if val_err   > 0 else "–"
        tot         = int(row["total_errors"])
        err_pct     = f"{row['err_pct']:.1f}"
        cum_pct     = f"{row['cum_pct']:.1f}"

        lines.append(
            f"\\texttt{{{bp_tex}}} & {study_tex} & {n_train_tex} & {n_test_tex} & {n_val_tex} & {tot} "
            f"& {err_pct} & {cum_pct} & {err_type} & {errors_str} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(
        "\\addcontentsline{toc}{subsection}"
        "{Supplementary Table 9: Top 15 BioProjects by total misclassifications}"
    )
    lines.append("\\\\[2mm]")
    lines.append(
        "{\\footnotesize "
        "$^{a}$~Percentage of wrong predictions out of all predictions made for that BioProject "
        "across all splits (each sample contributes four predictions, one per task). "
        "$^{b}$~Running total of errors as a percentage of all errors across every BioProject "
        "with at least 5 samples; shows error concentration. "
        "$^{c}$~Four error types are distinguished. "
        "\\textit{Absent class}: the true label never appeared in DIANA's training set. "
        "\\textit{Label granularity}: the true label is absent from training but is a biological "
        "subtype or synonym of a class that \\textit{is} in training---the model predicts the "
        "correct class at a coarser resolution "
        "(e.g.\\ \\texttt{lake sediment} absent from training; "
        "\\texttt{sediment} present and biologically equivalent). "
        "\\textit{Taxonomic granularity}: the true host is a subspecies whose parent taxon is the "
        "training-level label "
        "(e.g.\\ \\textit{Gorilla beringei beringei} labelled as \\textit{Gorilla} sp.\\ in training). "
        "\\textit{Genuine confusion}: all true labels were seen during training; "
        "the model still predicts the wrong class. "
        "$^{d}$~Top true$\\to$predicted label pairs by frequency; task abbreviation in parentheses. "
        "Task abbreviations: ST\\,=\\,Sample Type, CT\\,=\\,Community Type, "
        "SH\\,=\\,Sample Host, Mat.\\,=\\,Material.}"
    )
    return "\n".join(lines)


def main() -> None:
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)

    ranking = build_ranking()
    latex = generate_latex(ranking)
    OUT_TEX.write_text(latex)
    print(f"✓ Written {OUT_TEX}")

    total_shown = ranking["total_errors"].sum()
    last_cum    = ranking["cum_pct"].iloc[-1]
    print(
        f"\nTop {len(ranking)} BioProjects (≥{MIN_SAMPLES} samples), "
        f"sorted by total errors (no confidence threshold):\n"
        f"  Errors shown: {total_shown}  |  Cumulative coverage: {last_cum:.1f}%"
    )
    print(
        ranking[["BioProject", "project_name", "train_errors", "test_errors", "val_errors",
                  "total_errors", "err_pct", "cum_pct"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
