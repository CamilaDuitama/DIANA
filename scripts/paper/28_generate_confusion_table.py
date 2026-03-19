#!/usr/bin/env python3
"""
28_generate_confusion_table.py

Generate Supplementary Table 9: top misclassification patterns across all tasks,
combining validation and test sets, with an error-type annotation column.

Error type categories
─────────────────────
  Genuine confusion   – both true and predicted labels are in DIANA's training
                        set; the model assigns the wrong class.
  Absent class        – true label was never seen during training; the model
                        assigns the nearest known class, which may be wrong.
  Label granularity   – true label absent from training but is a biological
                        subtype or synonym of a training class; the model's
                        prediction is semantically correct at coarser resolution.
  Taxon granularity   – true host label is a species/subspecies absent from
                        training; the model correctly predicts the parent taxon
                        that is present in training.

Output
──────
  paper/tables/final/sup_table_10_confusion_patterns.tex
"""

import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "paper"))
sys.path.insert(0, str(REPO / "scripts" / "validation"))

TEST_PREDICTIONS = REPO / "results" / "test_evaluation" / "test_predictions.tsv"
LABEL_ENCODERS   = REPO / "results" / "training" / "label_encoders.json"
VAL_METADATA     = REPO / "paper" / "metadata" / "validation_metadata.tsv"
TEST_METADATA    = REPO / "paper" / "metadata" / "test_metadata.tsv"
OUTPUT           = REPO / "paper" / "tables" / "final" / "sup_table_10_confusion_patterns.tex"

TASKS = ["sample_type", "community_type", "sample_host", "material"]
TASK_LABELS = {
    "sample_type":    "Sample Type",
    "community_type": "Community Type",
    "sample_host":    "Sample Host",
    "material":       "Material",
}

# Minimum count (val + test combined) to include a pair in the table
MIN_COUNT = 5

# ---------------------------------------------------------------------------
# Semantically acceptable coarser predictions
# Key: (true_label_lower, task, predicted_label) → error type string
# Only pairs where the prediction is the correct broader-level class.
# ---------------------------------------------------------------------------
COARSER: dict[tuple[str, str, str], str] = {
    # Material — label granularity (subtype of a training class)
    ("lake sediment",               "material",    "sediment"):               "Label gran.",
    ("marine sediment",             "material",    "sediment"):               "Label gran.",
    ("faeces",                      "material",    "digestive tract contents"):"Label gran.",
    # Sample host — taxonomic granularity (subspecies → genus-level training entry)
    ("gorilla beringei beringei",   "sample_host", "Gorilla sp."):            "Taxon gran.",
    ("gorilla beringei graueri",    "sample_host", "Gorilla sp."):            "Taxon gran.",
    ("gorilla gorilla gorilla",     "sample_host", "Gorilla sp."):            "Taxon gran.",
    ("gorilla beringei",            "sample_host", "Gorilla sp."):            "Taxon gran.",
    ("pan troglodytes ellioti",     "sample_host", "Pan troglodytes"):        "Taxon gran.",
    ("pan troglodytes schweinfurthii","sample_host","Pan troglodytes"):       "Taxon gran.",
}


def classify(true_label: str, predicted: str, task: str, is_seen: bool) -> str:
    """Return the error type for a (true, predicted, task) triple."""
    tl_lower = str(true_label).lower().strip()
    pred_lower = str(predicted).lower().strip()
    # Data quality: capitalisation / format difference only (e.g. "Skin" vs "skin")
    if str(true_label) != str(predicted) and tl_lower == pred_lower:
        return "Data quality"
    # Semantically correct coarser prediction
    if (tl_lower, task, predicted) in COARSER:
        return COARSER[(tl_lower, task, predicted)]
    # Genuine confusion vs absent class — use is_seen from validation loader
    return "Genuine" if is_seen else "Absent class"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_validation_errors() -> pd.DataFrame:
    from load_validation_data import load_validation_predictions
    df = load_validation_predictions(quiet=True)
    err = df[~df["is_correct"]].copy()
    return err[["task", "true_label", "pred_label", "is_seen"]].rename(
        columns={"pred_label": "predicted"}
    )


def load_test_errors() -> pd.DataFrame:
    preds = pd.read_csv(TEST_PREDICTIONS, sep="\t", low_memory=False)
    enc   = json.loads(LABEL_ENCODERS.read_text())

    rows = []
    for task in TASKS:
        pc, tc = f"{task}_pred", f"{task}_true"
        if pc not in preds.columns:
            continue
        training_classes = set(enc[task]["classes"])
        sub = preds[preds[pc] != preds[tc]][[tc, pc]].rename(
            columns={tc: "true_label", pc: "predicted"}
        ).copy()
        sub["task"]    = task
        sub["is_seen"] = sub["true_label"].isin(training_classes)
        rows.append(sub)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["true_label", "predicted", "task", "is_seen"]
    )


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _itahost(label: str) -> str:
    """Italicise host labels (proper names)."""
    # Wrap in \textit{} — but only for non-trivial strings
    if not label or label in ("–", "Not applicable - env sample", "Other mammal"):
        return label
    return f"\\textit{{{label}}}"


def _fmt_label(label: str, task: str) -> str:
    label = str(label)
    if task == "sample_host":
        return _itahost(label)
    return label.replace("_", "\\_")


def generate_latex(top: pd.DataFrame) -> str:
    lines: list[str] = []

    lines.append(
        "\\caption{Frequent misclassification patterns in DIANA (test and validation sets "
        "combined, $n \\geq " + str(MIN_COUNT) + "$ occurrences), annotated by error type. "
        "\\label{tab:confusion_patterns}}"
    )
    lines.append("\\begin{longtable}{lp{3.5cm}p{3.5cm}rl}")
    lines.append("\\toprule")
    lines.append("Task & True label & Predicted label & $n$ & Error type$^{a}$ \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\multicolumn{5}{c}{\\textit{Table \\thetable{} continued from previous page}} \\\\")
    lines.append("\\toprule")
    lines.append("Task & True label & Predicted label & $n$ & Error type$^{a}$ \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{r}{\\textit{Continued on next page}} \\\\")
    lines.append("\\endfoot")
    lines.append("\\bottomrule")
    lines.append("\\endlastfoot")

    for task in TASKS:
        sub = top[top["task"] == task].sort_values("count", ascending=False)
        if sub.empty:
            continue
        first = True
        for _, row in sub.iterrows():
            task_cell = TASK_LABELS[task] if first else ""
            first = False
            true_tex  = _fmt_label(row["true_label"], task)
            pred_tex  = _fmt_label(row["predicted"],  task)
            etype = row["error_type"]
            # Italicise semantically acceptable types to visually distinguish them
            if etype in ("Label gran.", "Taxon gran."):
                etype_tex = f"\\textit{{{etype}}}"
            else:
                etype_tex = etype
            lines.append(
                f"{task_cell} & {true_tex} & {pred_tex} & {row['count']} & {etype_tex} \\\\"
            )
        lines.append("\\addlinespace")

    lines.append("\\end{longtable}")
    lines.append(
        "\\addcontentsline{toc}{subsection}"
        "{Supplementary Table 9: Misclassification patterns}"
    )
    lines.append("\\\\[2mm]")
    lines.append(
        "{\\footnotesize "
        "$^{a}$~Error type categories: "
        "\\textit{Genuine confusion} = both true and predicted labels are in DIANA's training set, "
        "model assigns the wrong class; "
        "\\textit{Absent class} = true label was never seen during training, model assigns the "
        "nearest known class (prediction may be unrelated); "
        "\\textit{Label gran.} = true label absent from training but is a biological subtype or "
        "synonym of a training class---model prediction is correct at coarser resolution "
        "(e.g.\\ \\texttt{lake sediment} correctly predicted as \\texttt{sediment}); "
        "\\textit{Taxon gran.} = true host is a species or subspecies absent from training, "
        "model correctly predicts the parent taxon "
        "(e.g.\\ \\textit{Gorilla gorilla gorilla} predicted as \\textit{Gorilla} sp.). "
        "Italicised error types indicate semantically acceptable predictions.}"
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    val_err  = load_validation_errors()
    test_err = load_test_errors()

    combined = pd.concat(
        [val_err, test_err[val_err.columns]], ignore_index=True
    )
    combined["error_type"] = combined.apply(
        lambda r: classify(r["true_label"], r["predicted"], r["task"], r["is_seen"]),
        axis=1,
    )

    top = (
        combined
        .groupby(["task", "true_label", "predicted", "error_type"])
        .size()
        .reset_index(name="count")
    )
    top = top[top["count"] >= MIN_COUNT].copy()

    # Drop data quality rows (capitalisation artefacts — not meaningful for paper)
    top = top[top["error_type"] != "Data quality"].copy()

    print(f"Rows in table (count >= {MIN_COUNT}, excluding data quality): {len(top)}")
    print("\nBreakdown by error type:")
    print(top.groupby("error_type")["count"].agg(["count", "sum"])
          .rename(columns={"count": "pairs", "sum": "total_instances"}).to_string())

    latex = generate_latex(top)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(latex, encoding="utf-8")
    print(f"\n✓ Written {OUTPUT}")


if __name__ == "__main__":
    main()
