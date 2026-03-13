#!/usr/bin/env python3
"""
Generate BioProject offenders supplementary table (sup_table_09)

Ranks the top BioProjects in the validation set by total confident errors
(confidence >= 0.9, wrong prediction) aggregated across all four tasks,
and annotates each with a root-cause category. This table serves as a
"decoder ring" for Supplementary Table 5 (common misclassification patterns).

Input:
- paper/tables/final/bioproject_error_rates.tsv  (from 24_bioproject_error_analysis.py)

Output:
- paper/tables/final/sup_table_09_bioproject_offenders.tex

Root-cause categories
---------------------
  OOD material         – label class absent from training (e.g. birch pitch)
  Label granularity    – DIANA trains at coarser resolution than AMD label
                         (e.g. sediment vs lake sediment / marine sediment)
  Taxonomic resolution – DIANA trains at genus level, validation labels are
                         subspecies-level (e.g. gorilla beringei beringei vs
                         Gorilla sp.)
  Genuine confusion    – label is in training, model still errs confidently
"""

from pathlib import Path
import pandas as pd

REPO    = Path(__file__).resolve().parents[2]
IN_TSV  = REPO / "paper" / "tables" / "final" / "bioproject_error_rates.tsv"
OUT_TEX = REPO / "paper" / "tables" / "final" / "sup_table_09_bioproject_offenders.tex"

TOP_N = 10  # number of BioProjects to show

# Manually curated root-cause annotations for known offenders.
# BioProjects not in this dict get "–" in the root-cause column.
ROOT_CAUSE = {
    "PRJNA994900": ("Kirdok2024",       "CT, Mat., SH", "OOD material"),
    "PRJEB34569":  ("FellowsYates2021", "SH",           "Taxonomic resolution"),
    "PRJEB80877":  ("vonHippel2025",    "Mat.",          "Label granularity"),
    "PRJNA1211513":("Schreiber2025",    "Mat.",          "Label granularity"),
    "PRJEB49638":  ("Moraitou2022",     "SH",            "Taxonomic resolution"),
    "PRJEB74036":  ("Liu2024",          "Mat.",          "Label granularity"),
    "PRJEB33848":  ("Neukamm2020",      "CT, Mat.",      "Label granularity"),
    "PRJNA320875": ("Graham2016",       "Mat.",          "Label granularity"),
    "PRJEB11419":  ("",                 "Mat., CT",      "OOD material"),
    "PRJEB67998":  ("Bozzi2024",        "Mat., CT",      "Label ambiguity"),
    "PRJNA1056444":("Austin2024",       "SH",            "Genuine confusion"),
}


def build_ranking() -> pd.DataFrame:
    df = pd.read_csv(IN_TSV, sep="\t", low_memory=False)
    # Aggregate total confident errors and val runs per BioProject
    agg = (
        df.groupby("BioProject")
        .agg(
            project_name=("project_name", lambda x: x.dropna().mode()[0] if x.notna().any() else ""),
            N_val_runs=("N_total", "max"),          # same for all tasks per BP
            total_conf_wrong=("N_conf_wrong", "sum"),
        )
        .reset_index()
        .sort_values("total_conf_wrong", ascending=False)
    )
    return agg.head(TOP_N)


def generate_latex(ranking: pd.DataFrame) -> str:
    lines = []
    lines.append(
        "\\caption{BioProjects with the most high-confidence errors "
        "(confidence $\\geq 0.9$, predicted $\\neq$ true) in the full external "
        "validation set (987 runs), summed across all four classification tasks "
        "(i.e.\\ counting run--task misclassifications). "
        "This analysis diagnoses systematic, study-level sources of error."
        "\\label{tab:bioproject_offenders}}"
    )
    # resizebox scales the tabular to \linewidth so it never overflows margins
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{@{}llrrll@{}}")
    lines.append("\\toprule")
    lines.append(
        "BioProject & Study & Val runs & High-conf.\\ errors & Main tasks$^{a}$ & Root cause \\\\"
    )
    lines.append("\\midrule")

    for _, row in ranking.iterrows():
        bp = row["BioProject"]
        n_runs = int(row["N_val_runs"])
        n_cerr = int(row["total_conf_wrong"])

        if bp in ROOT_CAUSE:
            study, dominant, cause = ROOT_CAUSE[bp]
        else:
            study = str(row["project_name"]) if pd.notna(row["project_name"]) else ""
            dominant = "–"
            cause = "–"

        # Escape underscores in BioProject ID
        bp_tex = bp.replace("_", "\\_")
        study_tex = study if study else "–"
        dominant_tex = dominant if dominant else "–"
        cause_tex = cause if cause else "–"

        lines.append(
            f"\\texttt{{{bp_tex}}} & {study_tex} & {n_runs} & {n_cerr} "
            f"& {dominant_tex} & {cause_tex} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")  # end resizebox
    lines.append("\\addcontentsline{toc}{subsection}"
                 "{Supplementary Table 9: Top BioProjects dominating high-confidence validation errors (full validation set)}")
    lines.append("\\\\[2mm]")
    lines.append(
        "{\\footnotesize "
        "Counts use the full validation set (987 runs), including runs whose true labels are "
        "outside DIANA's label set. "
        "$^{a}$~Main tasks indicate the task(s) contributing the largest share of high-confidence errors for that BioProject. "
        "Task abbreviations: ST\\,=\\,Sample Type, CT\\,=\\,Community Type, "
        "SH\\,=\\,Sample Host, Mat.\\,=\\,Material. "
        "Root-cause categories --- "
        "\\textit{OOD material}: material class absent from DIANA training classes; "
        "\\textit{Label granularity}: external label is a subtype of a DIANA class "
        "(e.g.\\ lake/marine sediment vs.\\ sediment); "
        "\\textit{Taxonomic resolution}: subspecies-level true label vs.\\ genus/species-level DIANA outputs "
        "(e.g.\\ \\textit{gorilla beringei beringei} vs.\\ \\textit{Gorilla sp.}); "
        "\\textit{Label ambiguity}: confusion among similar classes all present in training "
        "(e.g.\\ tooth vs.\\ bone); "
        "\\textit{Genuine confusion}: label is in DIANA's label set but the model still predicts "
        "confidently incorrectly.}"
    )
    return "\n".join(lines)


def main() -> None:
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)

    if not IN_TSV.exists():
        raise FileNotFoundError(
            f"{IN_TSV} not found — run 24_bioproject_error_analysis.py first"
        )

    ranking = build_ranking()
    latex = generate_latex(ranking)
    OUT_TEX.write_text(latex)
    print(f"✓ Written {OUT_TEX}")

    # Print preview
    print("\nTop-10 BioProjects by confident errors:")
    print(ranking[["BioProject", "project_name", "N_val_runs", "total_conf_wrong"]].to_string(index=False))


if __name__ == "__main__":
    main()
