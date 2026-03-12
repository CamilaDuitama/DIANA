#!/usr/bin/env python3
"""
Generate Seen vs Unseen Validation Table (Supplementary Table 7)

PURPOSE:
    Create LaTeX longtable showing validation set performance broken down
    by seen/unseen labels with top 10 misclassification patterns per task.

INPUTS (from config.py):
    Validation predictions loaded via load_validation_data.py

OUTPUTS:
    - paper/tables/final/sup_table_07_seen_unseen_validation.tex

USAGE:
    python scripts/paper/17_generate_seen_unseen_table.py
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "validation"))
from load_validation_data import load_validation_predictions

sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, TASKS


TASK_LABELS = {
    "sample_type": "Sample Type",
    "community_type": "Community Type",
    "sample_host": "Sample Host",
    "material": "Material",
}

PLAIN_TEXT_HOSTS = {"Not applicable - env sample", "Other mammal"}


def _fmt_host(label: str) -> str:
    s = str(label).replace("_", "\\_")
    if label in PLAIN_TEXT_HOSTS:
        return s
    return f"\\textit{{{s}}}"


def generate_seen_unseen_table(df: pd.DataFrame, output_path: Path) -> None:
    """Generate the seen/unseen validation longtable."""
    total_samples = len(df["sample_id"].unique())

    lines = []
    lines.append("\\small")
    lines.append("\\begin{longtable}{llp{3.5cm}p{3.5cm}rrr}")
    lines.append(
        "\\caption{Validation set performance: Seen vs unseen labels with top 10 most frequent "
        "misclassification patterns\\label{tab:seen_unseen_validation}}\\\\"
    )
    lines.append("\\toprule")
    lines.append("Task & Category & True Label & Predicted Label & Count & \\% Task & \\% Total \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("")
    lines.append("\\multicolumn{7}{c}{{\\tablename\\ \\thetable{} -- continued from previous page}} \\\\")
    lines.append("\\toprule")
    lines.append("Task & Category & True Label & Predicted Label & Count & \\% Task & \\% Total \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("")
    lines.append("\\midrule")
    lines.append("\\multicolumn{7}{r}{{Continued on next page}} \\\\")
    lines.append("\\endfoot")
    lines.append("")
    lines.append("\\bottomrule")
    lines.append("\\endlastfoot")

    for task in TASKS:
        task_df = df[df["task"] == task].copy()
        task_total = len(task_df)
        if task_total == 0:
            continue

        task_name = TASK_LABELS[task]

        # Seen correct
        seen_correct_count = int((task_df["is_seen"] & task_df["is_correct"]).sum())
        pct_task = seen_correct_count / task_total * 100
        pct_total = seen_correct_count / total_samples * 100
        lines.append(
            f"{task_name} & Seen - Correct & — & — & {seen_correct_count} "
            f"& {pct_task:.1f}\\% & {pct_total:.1f}\\% \\\\"
        )

        # Seen wrong (top 10)
        seen_wrong = task_df[task_df["is_seen"] & ~task_df["is_correct"]]
        if len(seen_wrong) > 0:
            confusion = (
                seen_wrong.groupby(["true_label", "pred_label"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(10)
            )
            for i, (_, row) in enumerate(confusion.iterrows()):
                tl = row["true_label"]
                pl = row["pred_label"]
                if task == "sample_host":
                    tl_fmt = _fmt_host(str(tl))
                    pl_fmt = _fmt_host(str(pl))
                else:
                    tl_fmt = str(tl).replace("_", "\\_")
                    pl_fmt = str(pl).replace("_", "\\_")
                count = int(row["count"])
                pt = count / task_total * 100
                ptt = count / total_samples * 100
                prefix = " & Seen - Wrong" if i == 0 else " & "
                lines.append(f" {prefix} & {tl_fmt} & {pl_fmt} & {count} & {pt:.1f}\\% & {ptt:.1f}\\% \\\\")

        # Unseen (top 10)
        unseen = task_df[~task_df["is_seen"]]
        if len(unseen) > 0:
            confusion_u = (
                unseen.groupby(["true_label", "pred_label"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(10)
            )
            for i, (_, row) in enumerate(confusion_u.iterrows()):
                tl = row["true_label"]
                pl = row["pred_label"]
                if task == "sample_host":
                    tl_fmt = _fmt_host(str(tl))
                    pl_fmt = _fmt_host(str(pl))
                else:
                    tl_fmt = str(tl).replace("_", "\\_")
                    pl_fmt = str(pl).replace("_", "\\_")
                count = int(row["count"])
                pt = count / task_total * 100
                ptt = count / total_samples * 100
                prefix = " & Unseen" if i == 0 else " & "
                lines.append(f" {prefix} & {tl_fmt} & {pl_fmt} & {count} & {pt:.1f}\\% & {ptt:.1f}\\% \\\\")

        lines.append("\\addlinespace")

    lines.append("\\end{longtable}")
    lines.append("\\addcontentsline{toc}{subsection}{Supplementary Table 7: Seen vs unseen validation performance}")
    lines.append("\\\\[2mm]")
    lines.append("{\\footnotesize")
    lines.append(
        "\\textbf{Category:} Seen labels were present in training; Unseen labels were not in training. "
    )
    lines.append(
        "\\textbf{Seen - Correct:} Samples correctly classified with labels seen during training. "
    )
    lines.append(
        "\\textbf{Seen - Wrong:} Top 10 most frequent misclassification patterns for seen labels "
        "(True Label $\\to$ Predicted Label)."
    )
    lines.append(
        "\\textbf{Unseen:} Top 10 most frequent predictions for unseen labels "
        "(model maps to semantically similar seen classes)."
    )
    lines.append(
        "\\textbf{\\% Task:} Percentage relative to total samples for that task in validation set. "
    )
    lines.append(
        "\\textbf{\\% Total:} Percentage relative to total validation samples across all tasks. "
    )
    lines.append(
        f"Total validation samples: {total_samples} (unique sample IDs across "
        f"{len(TASKS)} tasks = {len(df)} predictions)."
    )
    lines.append("}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  ✓ {output_path.name}")


def main():
    print("=" * 80)
    print("GENERATING SEEN/UNSEEN VALIDATION TABLE (SUPPLEMENTARY TABLE 7)")
    print("=" * 80)

    print("\nLoading validation predictions...")
    df = load_validation_predictions()
    print(f"  ✓ Loaded {len(df)} predictions")

    output_path = Path(PATHS["tables_dir"]) / "sup_table_07_seen_unseen_validation.tex"
    generate_seen_unseen_table(df, output_path)

    print("\n" + "=" * 80)
    print("✓ COMPLETE")


if __name__ == "__main__":
    main()
